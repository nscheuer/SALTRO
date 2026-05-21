"""
Property-based unit tests for SALTRO's optimizer.

Companion to ``test_alilqr.py`` / ``test_backwardpass.py`` / etc., which
test the optimizer primitives. The tests here verify *behavioural
properties* of ``saltro_py.trajOpt`` end-to-end:

* **Warmstart determinism**: two identical calls give bit-identical
  trajectories. SALTRO's warmstart options (``initcontroller`` =
  ZeroController / ExcitationController / IntegratedBdotController)
  are all deterministic by design, so any nondeterminism is a real
  regression.

* **Eigenaxis preservation**: with a single y-axis RW and an initial
  rotation purely about y, the trajectory MUST stay in the y-axial
  subspace (``omega_x = omega_z = q_x = q_z = 0``) to machine precision.
  Symmetry-protected -- robust to any convergence quirk.

* **At-goal-commands-zero**: if the initial state IS the goal with zero
  rate, the unique optimum is ``u = 0`` (no cost to minimise). SALTRO
  must converge to it.

These properties were ported from a Generalized_ADCS-side test file
that used the ``ADCS.controller.saltro.SALTRO`` Python wrapper. They
test properties of SALTRO itself, not of the bridge layer, so they
belong here (in the SALTRO repo) and use ``saltro_py`` directly. The
bridge-layer smoke test stays in Generalized_ADCS.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(ROOT / "tests" / "debug" / "optimizer" / "configs"))

import saltro_py
from sat_0_1_rw_y import create_satellite as create_satellite_y_rw


SEC_PER_CENTURY = 36525.0 * 86400.0


def create_default_planner_settings(dt_seconds: float) -> saltro_py.PlannerSettings:
    """Standard single-pass PlannerSettings tuned for the property tests.
    Uses the deterministic ``IntegratedBdotController`` warmstart and
    moderate convergence tolerances.
    """
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = 2          # IntegratedBdotController

    ps.num_passes = 1
    ps.passes[0].dt = dt_seconds
    ps.passes[0].ilqr.max_iters = 30
    ps.passes[0].ilqr.cost_tol = 1e-6
    ps.passes[0].auglag.max_outer_iters = 15
    ps.passes[0].auglag.constraint_tol = 1e-3

    c = ps.passes[0].cost
    c.angle = 1e3
    c.angle_N = 1e6
    c.ang_vel = 1e3
    c.ang_vel_N = 1e5
    c.ang_vel_mag = 0.0
    c.ang_vel_err_dir = 0.0
    c.control_mult = 1.0
    c.mtq_control_weight = 1e3
    c.rw_control_weight = 1.0
    c.magic_control_weight = 0.0
    c.rw_AM_weight = 0.0
    c.rw_stic_weight = 0.0
    c.RWh_max_mult = 0.0
    c.RWh_stiction_mult = 0.0
    c.RWh_ok_mult = 0.0
    c.ang_vel_mag_N = 0.0
    c.ang_vel_err_dir_N = 0.0
    c.ang_cost_func_type = 2
    c.use_cost_hess = True

    ps.disturbances.plan_for_aero = False
    ps.disturbances.plan_for_gg = False
    ps.disturbances.plan_for_srp = False
    ps.disturbances.plan_for_prop = False
    ps.disturbances.plan_for_gendist = False
    ps.disturbances.plan_for_resdipole = False

    ps.passes[0].reg.reg_init = 1e-6
    ps.passes[0].reg.reg_max = 1e10
    ps.passes[0].reg.reg_scale = 10.0
    ps.passes[0].reg.use_dynamics_hess = False
    ps.passes[0].reg.use_constraint_hess = False

    ps.passes[0].linesearch.max_iters = 24
    ps.passes[0].linesearch.beta1 = 1e-10
    ps.passes[0].linesearch.beta2 = 5000.0

    return ps


def _run_trajopt(initial_y_angle_rad: float, omega0_y: float = 0.0,
                  dt_seconds: float = 1.0, N: int = 31):
    """Construct a y-axis RW satellite + plan a maneuver from
    ``initial_y_angle_rad`` rotation about y to identity. Returns
    ``(ok, X, U)``.
    """
    ps = create_default_planner_settings(dt_seconds)
    sat = create_satellite_y_rw(ps)

    jtime = np.array([0.22, 0.22 + (N - 1) * dt_seconds / SEC_PER_CENTURY],
                     dtype=float)
    # Goal: identity attitude (q = [1, 0, 0, 0]) held throughout the
    # horizon. ``qgoal`` is (4, 2) shaped (start and end of jtime).
    qgoal = np.tile(np.array([1.0, 0.0, 0.0, 0.0]).reshape(4, 1), (1, 2))
    # Boresight = body z-axis, held in body frame.
    boresight = np.tile(np.array([0.0, 0.0, 1.0]).reshape(3, 1), (1, 2))

    w0 = np.array([0.0, omega0_y, 0.0])
    q0 = np.array([np.cos(initial_y_angle_rad / 2.0), 0.0,
                   np.sin(initial_y_angle_rad / 2.0), 0.0])
    q0 = q0 / np.linalg.norm(q0)
    h0 = np.array([0.0])    # 1 RW
    x0 = np.hstack((w0, q0, h0))

    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7.5e3, 0.0])

    ok, X, U, _ = saltro_py.trajOpt(ps, sat, x0, r0, v0, jtime, qgoal, boresight)
    return ok, np.asarray(X), np.asarray(U)


# ---------------------------------------------------------------------------
# Warmstart determinism
# ---------------------------------------------------------------------------

def test_warmstart_is_deterministic():
    """Two identical ``trajOpt`` calls must give bit-identical
    trajectories. SALTRO's warmstart controllers (Zero / Excitation /
    IntegratedBdot) are deterministic by design; this test catches any
    regression that introduces hidden randomness (e.g., unseeded
    ``randn``, a la OldPlanner's ``OldPlanner.cpp:276``).
    """
    ok1, X1, U1 = _run_trajopt(initial_y_angle_rad=0.1)
    ok2, X2, U2 = _run_trajopt(initial_y_angle_rad=0.1)
    assert ok1 and ok2, "trajOpt failed"
    assert np.array_equal(X1, X2), (
        f"State trajectory not bit-identical across identical runs: "
        f"max diff = {float(np.max(np.abs(X1 - X2))):.3e}"
    )
    assert np.array_equal(U1, U2), (
        f"Control trajectory not bit-identical: "
        f"max diff = {float(np.max(np.abs(U1 - U2))):.3e}"
    )


# ---------------------------------------------------------------------------
# Eigenaxis preservation (symmetry-protected)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("init_angle,N", [
    (0.1, 31),
    pytest.param(
        np.pi / 2, 121,
        marks=pytest.mark.xfail(
            strict=False,
            reason="AL-iLQR convergence: SALTRO with default-tuned costs "
                   "doesn't converge for a 90 deg rest-to-rest slew via "
                   "single y-axis RW; needs cost retuning or longer horizon. "
                   "Tracked separately; the symmetry property is still "
                   "exercised by the small-angle case.",
        ),
    ),
    pytest.param(
        np.pi - 0.05, 201,
        marks=pytest.mark.xfail(
            strict=False,
            reason="Same convergence issue as 90 deg case; pre-180 deg is "
                   "harder (cost-function near-singularity).",
        ),
    ),
])
def test_y_axis_rw_preserves_y_axial_subspace(init_angle, N):
    """Single y-axis RW + initial rotation purely about y MUST give a
    trajectory that stays in the y-axial subspace
    ``(omega_y, delta_q_y, h_RW)``. Off-axis omega_x, omega_z, q_x, q_z
    must be zero to machine precision throughout the horizon. The
    dynamics conserve this subspace exactly when only a y-axis torque is
    available and the cost is rotation-axis-agnostic.

    Only the small-angle (0.1 rad) case is currently passing -- 90 deg
    and ~180 deg are xfailed pending a cost-tuning investigation for
    large-angle rest-to-rest slews in SALTRO. The xfail tests are still
    valuable: if SALTRO ever converges these, they XPASS and we know to
    promote them to regular tests.
    """
    ok, X, U = _run_trajopt(initial_y_angle_rad=init_angle, N=N)
    assert ok, "trajOpt failed"

    # State layout: [omega_x, omega_y, omega_z, q_w, q_x, q_y, q_z, h_RW]
    off_axis_w_idx = [(0, "omega_x"), (2, "omega_z")]
    off_axis_q_idx = [(4, "q_x"), (6, "q_z")]
    for idx, name in off_axis_w_idx + off_axis_q_idx:
        max_off = float(np.max(np.abs(X[idx, :])))
        assert max_off < 1e-9, (
            f"{name} (off-axis) deviated from zero: max = {max_off:.3e} "
            f"(init angle = {init_angle:.4f} rad)"
        )


# ---------------------------------------------------------------------------
# At-goal commands zero
# ---------------------------------------------------------------------------

def test_at_goal_with_zero_rate_commands_near_zero():
    """If x(0) = goal with zero rate, the optimal control is u = 0 by
    construction (zero state cost everywhere along u = 0; zero control
    cost has the unique global minimum at u = 0). The planner must
    converge to this -- any sustained nonzero control is either a bug
    in goal handling or the optimiser is stuck on a spurious local
    solution.
    """
    ok, X, U = _run_trajopt(initial_y_angle_rad=0.0, omega0_y=0.0, N=20)
    assert ok, "trajOpt failed"
    max_u = float(np.max(np.abs(U)))
    rms_u = float(np.sqrt(np.mean(U ** 2)))
    assert max_u < 1e-3, (
        f"At-goal commanded nonzero control: max|u| = {max_u:.3e} "
        f"(rms|u| = {rms_u:.3e}); the optimum is u = 0."
    )
