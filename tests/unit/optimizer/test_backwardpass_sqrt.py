"""
Python tests for the square-root backward pass (use_sqrt_bp).
Mirrors test_backwardpass_sqrt.cpp and test_alilqr_sqrt.cpp with equivalent
test cases, driven through the saltro_py bindings (backward_pass dispatches
on settings.passes[0].reg.use_sqrt_bp; there is no direct sqrt binding).
"""

import numpy as np
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "build"))
import saltro_py


SEC_PER_CENTURY = 36525.0 * 86400.0


def make_attitude_traj(att, n_cols):
    traj = np.zeros((4, n_cols))
    for k in range(n_cols):
        traj[:, k] = att
    return traj


class BackwardPassSqrtFixture:
    """Twin of BackwardPassSqrtFixture in test_backwardpass_sqrt.cpp."""

    def __init__(self, n=6):
        self.N = n
        self.settings = saltro_py.PlannerSettings()

        J = np.diag([0.067, 0.071, 0.069])
        self.satellite = saltro_py.Satellite(J, self.settings)

        self.satellite.addMTQ(np.array([1.0, 0.0, 0.0]), 0.2)
        self.satellite.addMTQ(np.array([0.0, 1.0, 0.0]), 0.2)
        self.satellite.addMTQ(np.array([0.0, 0.0, 1.0]), 0.2)
        self.satellite.addRW(np.array([1.0, 0.0, 0.0]), 0.001, 1e-5, 0.0, 0.02)
        self.satellite.addRW(np.array([0.0, 1.0, 0.0]), 0.001, 1e-5, 0.0, 0.02)
        self.satellite.addRW(np.array([0.0, 0.0, 1.0]), 0.001, 1e-5, 0.0, 0.02)

        self.settings.disturbances.plan_for_aero = False
        self.settings.disturbances.plan_for_gg = False
        self.settings.disturbances.plan_for_srp = False
        self.settings.disturbances.plan_for_prop = False
        self.settings.disturbances.plan_for_gendist = False
        self.settings.disturbances.plan_for_resdipole = False
        self.settings.num_passes = 1
        self.settings.passes[0].dt = 0.5
        self.settings.passes[0].reg.reg_init = 1e-8
        self.settings.passes[0].reg.reg_scale = 10.0
        self.settings.passes[0].reg.reg_max = 1e4

        self.reg = 1e-8

    def make_trajectory(self, force_active=False):
        nx = self.satellite.stateDim
        nu = self.satellite.controlDim
        N = self.N

        X = np.zeros((nx, N))
        U = np.zeros((nu, max(0, N - 1)))
        R = np.zeros((3, N))
        V = np.zeros((3, N))
        B = np.zeros((3, N))
        S = np.zeros((3, N))
        rho = np.zeros((1, N))
        boresight = np.zeros((3, N))

        x0 = np.zeros(nx)
        x0[0:3] = np.array([0.01, -0.005, 0.008])
        x0[3:7] = np.array([1.0, 0.0, 0.0, 0.0])

        for k in range(N):
            X[:, k] = x0
            # Vary the state slightly along the trajectory so gains differ
            # between timesteps.
            X[0:3, k] += 0.001 * k * np.array([1.0, -0.5, 0.25])
            if k < N - 1:
                if force_active:
                    U[:, k] = np.array([0.15, -0.12, 0.10, 0.01, -0.01, 0.01])
                else:
                    U[:, k] = 0.001 * np.ones(nu) * (k + 1)

            R[:, k] = np.array([7000e3, 0.0, 0.0])
            V[:, k] = np.array([0.0, 7500.0, 0.0])
            B[:, k] = np.array([2.5e-5, -1.5e-5, 3.0e-5])
            S[:, k] = np.array([1.0, 0.1, -0.05])
            S[:, k] /= np.linalg.norm(S[:, k])
            boresight[:, k] = np.array([1.0, 0.0, 0.0])

        attitude_target = make_attitude_traj(np.array([np.nan, 0.0, 0.0, 0.0]), N)

        if force_active:
            self.settings.constraints.wmax = 1e-4
            self.settings.constraints.control_limit_scale = 1.0
            self.settings.constraints.sun_limit_angle = np.pi
            self.settings.constraints.u_max = np.full(nu, 1e-4)

        return X, U, R, V, B, S, rho, boresight, attitude_target

    def collect_constraints(self, X, U, S):
        sat = self.satellite
        cfg = self.settings.constraints
        N = X.shape[1]
        nu = sat.controlDim
        c_list = []
        for k in range(N):
            uk = U[:, k] if k < U.shape[1] else np.zeros(nu)
            ck = np.asarray(sat.constraints(k, N, X[:, k], uk, S[:, k], cfg), dtype=float)
            c_list.append(ck)
        return c_list

    @staticmethod
    def make_const_aug(c_list, lambda_val, mu_val):
        lambda_aug = [np.full(c.shape[0], lambda_val) for c in c_list]
        mu_aug = [np.full(c.shape[0], mu_val) for c in c_list]
        return lambda_aug, mu_aug

    def run(self, traj, lambda_aug, mu_aug, use_sqrt_bp):
        """Run saltro_py.backward_pass with the sqrt flag set as requested."""
        X, U, R, V, B, S, rho, boresight, attitude_target = traj
        self.settings.passes[0].reg.use_sqrt_bp = use_sqrt_bp
        return saltro_py.backward_pass(
            self.satellite,
            X,
            U,
            R,
            V,
            B,
            S,
            rho,
            boresight,
            attitude_target,
            self.settings,
            lambda_aug,
            mu_aug,
            self.reg,
        )


def rel_delta(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b) / max(1e-300, np.linalg.norm(a)))


def empty_aug():
    return [], []


# ---------------------------------------------------------------------------
# Twins of test_backwardpass_sqrt.cpp
# ---------------------------------------------------------------------------


def test_backward_pass_sqrt_matches_dense_pass_without_constraints():
    fixture = BackwardPassSqrtFixture(6)
    traj = fixture.make_trajectory(force_active=False)
    lambda_aug, mu_aug = empty_aug()

    ok_dense, K_dense, d_dense, dV_dense = fixture.run(traj, lambda_aug, mu_aug, use_sqrt_bp=False)
    ok_sqrt, K_sqrt, d_sqrt, dV_sqrt = fixture.run(traj, lambda_aug, mu_aug, use_sqrt_bp=True)

    assert ok_dense
    assert ok_sqrt
    assert K_sqrt.shape == K_dense.shape
    assert rel_delta(K_dense, K_sqrt) < 1e-8
    assert rel_delta(d_dense, d_sqrt) < 1e-8
    assert abs(dV_dense[0] - dV_sqrt[0]) <= 1e-8 * max(1.0, abs(dV_dense[0]))
    assert abs(dV_dense[1] - dV_sqrt[1]) <= 1e-8 * max(1.0, abs(dV_dense[1]))


def test_backward_pass_sqrt_matches_dense_pass_with_active_al_constraints():
    fixture = BackwardPassSqrtFixture(6)
    traj = fixture.make_trajectory(force_active=True)
    X, U = traj[0], traj[1]
    S = traj[5]
    c_list = fixture.collect_constraints(X, U, S)
    lambda_aug, mu_aug = BackwardPassSqrtFixture.make_const_aug(c_list, 1.0, 100.0)

    ok_dense, K_dense, d_dense, dV_dense = fixture.run(traj, lambda_aug, mu_aug, use_sqrt_bp=False)
    ok_sqrt, K_sqrt, d_sqrt, dV_sqrt = fixture.run(traj, lambda_aug, mu_aug, use_sqrt_bp=True)

    assert ok_dense
    assert ok_sqrt
    assert rel_delta(K_dense, K_sqrt) < 1e-8
    assert rel_delta(d_dense, d_sqrt) < 1e-8
    assert abs(dV_dense[0] - dV_sqrt[0]) <= 1e-8 * max(1.0, abs(dV_dense[0]))
    assert abs(dV_dense[1] - dV_sqrt[1]) <= 1e-8 * max(1.0, abs(dV_dense[1]))


def test_backward_pass_sqrt_stays_finite_with_extreme_al_penalties():
    fixture = BackwardPassSqrtFixture(6)
    traj = fixture.make_trajectory(force_active=True)
    X, U = traj[0], traj[1]
    S = traj[5]
    c_list = fixture.collect_constraints(X, U, S)

    # Penalty weights far past the point where forming mu * c^T c outer
    # products loses half the available precision.
    for mu_val in (1e8, 1e12, 1e16):
        lambda_aug, mu_aug = BackwardPassSqrtFixture.make_const_aug(c_list, 10.0, mu_val)

        ok, K_sqrt, d_sqrt, dV_sqrt = fixture.run(traj, lambda_aug, mu_aug, use_sqrt_bp=True)

        assert ok
        assert np.all(np.isfinite(dV_sqrt))
        assert np.all(np.isfinite(K_sqrt))
        assert np.all(np.isfinite(d_sqrt))


def test_backward_pass_sqrt_use_sqrt_bp_flag_routes_to_the_sqrt_pass():
    # No direct backwardPassSqrt binding exists, so the python twin of the
    # dispatch test checks that toggling use_sqrt_bp actually switches
    # algorithms: the flagged result agrees with the dense pass only to
    # floating-point parity (sqrt path taken), not bit-exactly, while
    # repeated flagged runs are bit-identical (deterministic dispatch).
    fixture = BackwardPassSqrtFixture(6)
    traj = fixture.make_trajectory(force_active=False)
    lambda_aug, mu_aug = empty_aug()

    ok_dense, K_dense, d_dense, dV_dense = fixture.run(traj, lambda_aug, mu_aug, use_sqrt_bp=False)
    ok_flag1, K_flag1, d_flag1, dV_flag1 = fixture.run(traj, lambda_aug, mu_aug, use_sqrt_bp=True)
    ok_flag2, K_flag2, d_flag2, dV_flag2 = fixture.run(traj, lambda_aug, mu_aug, use_sqrt_bp=True)

    assert ok_dense
    assert ok_flag1
    assert ok_flag2

    # Deterministic: two flagged runs are bit-identical.
    assert np.array_equal(K_flag1, K_flag2)
    assert np.array_equal(d_flag1, d_flag2)
    assert np.array_equal(dV_flag1, dV_flag2)

    # Parity with dense to fp tolerance...
    assert rel_delta(K_dense, K_flag1) < 1e-8
    assert rel_delta(d_dense, d_flag1) < 1e-8
    # ...but a genuinely different algorithm (QR square-root vs dense
    # Cholesky), so not bit-identical: the flag really re-routed the call.
    assert not (
        np.array_equal(K_dense, K_flag1)
        and np.array_equal(d_dense, d_flag1)
        and np.array_equal(dV_dense, dV_flag1)
    )


def test_backward_pass_sqrt_ddp_flags_force_the_dense_path():
    # Twin of "backward_pass sqrt: DDP flags force the dense path".
    # The direct backward_pass binding never runs validatePlannerSettings, so
    # use_sqrt_bp + use_dynamics_hess/use_constraint_hess can reach the
    # dispatch. The sqrt pass has no DDP second-order path; the dispatch must
    # fall back to the dense DDP pass (bit-identical results) instead of
    # silently dropping the curvature terms.
    fixture = BackwardPassSqrtFixture(6)
    traj = fixture.make_trajectory(force_active=True)
    X, U = traj[0], traj[1]
    S = traj[5]
    c_list = fixture.collect_constraints(X, U, S)
    lambda_aug, mu_aug = BackwardPassSqrtFixture.make_const_aug(c_list, 1.0, 100.0)

    fixture.settings.passes[0].reg.use_dynamics_hess = True
    fixture.settings.passes[0].reg.use_constraint_hess = True

    ok_dense, K_dense, d_dense, dV_dense = fixture.run(traj, lambda_aug, mu_aug, use_sqrt_bp=False)
    ok_flag, K_flag, d_flag, dV_flag = fixture.run(traj, lambda_aug, mu_aug, use_sqrt_bp=True)

    assert ok_dense
    assert ok_flag
    assert np.array_equal(K_dense, K_flag)
    assert np.array_equal(d_dense, d_flag)
    assert np.array_equal(dV_dense, dV_flag)


def test_backward_pass_sqrt_n1_edge_case():
    fixture = BackwardPassSqrtFixture(1)
    traj = fixture.make_trajectory(force_active=False)
    lambda_aug, mu_aug = empty_aug()

    ok, K, d, deltaV = fixture.run(traj, lambda_aug, mu_aug, use_sqrt_bp=True)

    assert ok
    assert K.shape[0] == 0
    assert d.shape[1] == 0


# ---------------------------------------------------------------------------
# Twins of test_alilqr_sqrt.cpp (end-to-end AL-iLQR with use_sqrt_bp)
# ---------------------------------------------------------------------------


def create_rw_planner_settings(dt_seconds, use_sqrt_bp):
    plannersettings = saltro_py.PlannerSettings()

    plannersettings.init_traj.initcontroller = 2

    plannersettings.num_passes = 1
    plannersettings.passes[0].dt = dt_seconds
    plannersettings.passes[0].ilqr.cost_tol = 1e-5
    plannersettings.passes[0].ilqr.max_iters = 20

    plannersettings.passes[0].auglag.max_outer_iters = 10
    plannersettings.passes[0].auglag.constraint_tol = 1e-3

    cost = plannersettings.passes[0].cost
    cost.angle = 1.0
    cost.ang_vel = 1e1
    cost.ang_vel_mag = 0.0
    cost.ang_vel_err_dir = 0.0
    cost.control_mult = 1.0
    cost.mtq_control_weight = 1e-2
    cost.rw_control_weight = 1.0
    cost.magic_control_weight = 0.0
    cost.rw_AM_weight = 0.0
    cost.rw_stic_weight = 0.0
    cost.RWh_stiction_mult = 0.0
    cost.angle_N = 0.0
    cost.ang_vel_N = 0.0
    cost.ang_vel_mag_N = 0.0
    cost.ang_vel_err_dir_N = 0.0
    cost.ang_cost_func_type = 3
    cost.use_cost_hess = True

    plannersettings.disturbances.plan_for_aero = False
    plannersettings.disturbances.plan_for_gg = False
    plannersettings.disturbances.plan_for_srp = False
    plannersettings.disturbances.plan_for_prop = False
    plannersettings.disturbances.plan_for_gendist = False
    plannersettings.disturbances.plan_for_resdipole = False

    plannersettings.passes[0].reg.reg_init = 1e-6
    plannersettings.passes[0].reg.reg_max = 1e10
    plannersettings.passes[0].reg.reg_scale = 10.0
    plannersettings.passes[0].reg.use_dynamics_hess = False
    plannersettings.passes[0].reg.use_constraint_hess = False
    plannersettings.passes[0].reg.use_sqrt_bp = use_sqrt_bp

    plannersettings.passes[0].linesearch.max_iters = 24
    plannersettings.passes[0].linesearch.beta1 = 1e-10
    plannersettings.passes[0].linesearch.beta2 = 5000.0

    return plannersettings


def create_rw_satellite(settings):
    J = np.diag([0.067, 0.071, 0.069])
    satellite = saltro_py.Satellite(J, settings)
    satellite.addRW(np.array([1.0, 0.0, 0.0]), 0.001, 1e-5, 0.0, 0.02)
    satellite.addRW(np.array([0.0, 1.0, 0.0]), 0.001, 1e-5, 0.0, 0.02)
    satellite.addRW(np.array([0.0, 0.0, 1.0]), 0.001, 1e-5, 0.0, 0.02)
    return satellite


def quat_pointing_error_deg(q, q_goal):
    q_goal_inv = np.array([q_goal[0], -q_goal[1], -q_goal[2], -q_goal[3]], dtype=float)
    q_err = np.array(
        [
            q_goal_inv[0] * q[0] - q_goal_inv[1] * q[1] - q_goal_inv[2] * q[2] - q_goal_inv[3] * q[3],
            q_goal_inv[0] * q[1] + q_goal_inv[1] * q[0] + q_goal_inv[2] * q[3] - q_goal_inv[3] * q[2],
            q_goal_inv[0] * q[2] - q_goal_inv[1] * q[3] + q_goal_inv[2] * q[0] + q_goal_inv[3] * q[1],
            q_goal_inv[0] * q[3] + q_goal_inv[1] * q[2] - q_goal_inv[2] * q[1] + q_goal_inv[3] * q[0],
        ],
        dtype=float,
    )
    return float(2.0 * np.arctan2(np.linalg.norm(q_err[1:]), abs(q_err[0])) * 180.0 / np.pi)


def run_rw_case(tf_seconds, dt_seconds, use_sqrt_bp):
    plannersettings = create_rw_planner_settings(dt_seconds, use_sqrt_bp)
    satellite = create_rw_satellite(plannersettings)

    w0 = np.array([-0.01, 0.02, 0.03], dtype=float)
    q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    h0 = np.array([0.0, 0.0, 0.0], dtype=float)
    x0 = np.hstack((w0, q0, h0))

    r0 = np.array([7000e3, 0.0, 0.0], dtype=float)
    v0 = np.array([0.0, 7.5e3, 0.0], dtype=float)

    jtime = np.array([0.22, 0.22 + tf_seconds / SEC_PER_CENTURY], dtype=float)
    q_goal = np.array(
        [
            [np.sqrt(2.0) / 2.0, np.sqrt(2.0) / 2.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [np.sqrt(2.0) / 2.0, np.sqrt(2.0) / 2.0],
        ],
        dtype=float,
    )
    boresight = np.array(
        [
            [1.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=float,
    )

    ok, X, U, _ = saltro_py.trajOpt(
        plannersettings,
        satellite,
        x0,
        r0,
        v0,
        jtime,
        q_goal,
        boresight,
    )
    return ok, X, U, q_goal


@pytest.mark.parametrize("tf_seconds", [200.0, 1000.0])
def test_alilqr_sqrt_bp_rw_case_converges(tf_seconds):
    # Twin of "AL-iLQR sqrt BP: RW case tf=200/1000 dt=10".
    ok, X, U, q_goal = run_rw_case(tf_seconds, 10.0, use_sqrt_bp=True)

    assert ok
    assert X.shape[1] > 0
    assert np.all(np.isfinite(X))
    assert np.all(np.isfinite(U))

    final_w_norm = float(np.linalg.norm(X[0:3, -1]))
    final_pointing_error_deg = quat_pointing_error_deg(X[3:7, -1], q_goal[:, -1])

    assert final_w_norm < 2e-2, f"final angular velocity too high: {final_w_norm:.3e} rad/s"
    assert final_pointing_error_deg < 5.0, (
        f"final pointing error too high: {final_pointing_error_deg:.3f} deg"
    )


def test_alilqr_sqrt_bp_matches_dense_results():
    # End-to-end sqrt-vs-dense: the sqrt backward pass is a reformulation of
    # the same Gauss-Newton recursion, so the full AL-iLQR solve should land
    # on the same trajectory as the dense pass on this small case.
    ok_sqrt, X_sqrt, U_sqrt, q_goal = run_rw_case(200.0, 10.0, use_sqrt_bp=True)
    ok_dense, X_dense, U_dense, _ = run_rw_case(200.0, 10.0, use_sqrt_bp=False)

    assert ok_sqrt
    assert ok_dense
    assert X_sqrt.shape == X_dense.shape
    assert U_sqrt.shape == U_dense.shape

    # Per-iteration backward-pass parity is ~1e-8 relative; iteration-to-
    # iteration accumulation across the solve keeps the end-to-end
    # trajectories close but not bit-identical.
    assert np.allclose(X_sqrt, X_dense, rtol=1e-5, atol=1e-8)
    assert np.allclose(U_sqrt, U_dense, rtol=1e-5, atol=1e-8)

    final_w_sqrt = float(np.linalg.norm(X_sqrt[0:3, -1]))
    final_w_dense = float(np.linalg.norm(X_dense[0:3, -1]))
    err_sqrt = quat_pointing_error_deg(X_sqrt[3:7, -1], q_goal[:, -1])
    err_dense = quat_pointing_error_deg(X_dense[3:7, -1], q_goal[:, -1])

    assert abs(final_w_sqrt - final_w_dense) < 1e-6
    assert abs(err_sqrt - err_dense) < 1e-4
