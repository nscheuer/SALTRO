import numpy as np
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'build'))
import saltro_py


SEC_PER_CENTURY = 36525.0 * 86400.0


class WarmStartFixture:
    N = 20

    def __init__(self):
        self.settings = saltro_py.PlannerSettings()

        J = np.diag([0.067, 0.071, 0.069]).astype(float)
        self.satellite = saltro_py.Satellite(J, self.settings)

        self.satellite.addMTQ(np.array([1.0, 0.0, 0.0]), 0.2)
        self.satellite.addMTQ(np.array([0.0, 1.0, 0.0]), 0.2)
        self.satellite.addMTQ(np.array([0.0, 0.0, 1.0]), 0.2)

        self.satellite.addRW(np.array([1.0, 0.0, 0.0]), 0.001, 1e-5, 0.0, 0.02)
        self.satellite.addRW(np.array([0.0, 1.0, 0.0]), 0.001, 1e-5, 0.0, 0.02)
        self.satellite.addRW(np.array([0.0, 0.0, 1.0]), 0.001, 1e-5, 0.0, 0.02)

        self.x0 = np.zeros(self.satellite.stateDim)
        self.x0[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3] = np.array([0.02, -0.01, 0.015])
        self.x0[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1.0, 0.0, 0.0, 0.0])

        dt_seconds = 0.5
        dt_centuries = dt_seconds / SEC_PER_CENTURY

        self.jtime = np.zeros(self.N)
        self.q_goal = np.zeros((4, self.N))
        self.boresight = np.zeros((3, self.N))
        self.R = np.zeros((3, self.N))
        self.V = np.zeros((3, self.N))
        self.B = np.zeros((3, self.N))
        self.S = np.zeros((3, self.N))
        self.rho = np.zeros(self.N)

        for k in range(self.N):
            self.jtime[k] = 0.25 + k * dt_centuries
            self.q_goal[:, k] = np.array([1.0, 0.0, 0.0, 0.0])
            self.boresight[:, k] = np.array([1.0, 0.0, 0.0])

            self.R[:, k] = np.array([7000e3, 100.0 * k, -50.0 * k])
            self.V[:, k] = np.array([0.0, 7500.0, 0.0])
            self.B[:, k] = np.array([2.5e-5, -1.5e-5, 3.0e-5])
            self.S[:, k] = np.array([1.0, 0.1, -0.05])
            self.rho[k] = 0.0

        self.settings.disturbances.plan_for_aero = False
        self.settings.disturbances.plan_for_gg = False
        self.settings.disturbances.plan_for_srp = False
        self.settings.disturbances.plan_for_prop = False
        self.settings.disturbances.plan_for_gendist = False
        self.settings.disturbances.plan_for_resdipole = False

        self.settings.num_passes = 1
        self.settings.passes[0].dt = dt_seconds


def rk4_step(f, x, t, dt, quat_idx=None):
    def renorm(state):
        state = state.copy()
        if quat_idx is not None:
            q = state[quat_idx:quat_idx + 4]
            state[quat_idx:quat_idx + 4] = q / np.linalg.norm(q)
        return state

    m = renorm(x)
    k1 = f(t, m)
    k2 = f(t + 0.5 * dt, renorm(m + 0.5 * dt * k1))
    k3 = f(t + 0.5 * dt, renorm(m + 0.5 * dt * k2))
    k4 = f(t + dt, renorm(m + dt * k3))
    return renorm(m + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))


@pytest.fixture
def fixture():
    return WarmStartFixture()


def test_warm_start_dimensions(fixture):
    fixture.settings.init_traj.initcontroller = 0

    ok, X, U = saltro_py.warm_start(
        fixture.settings,
        fixture.satellite,
        fixture.x0,
        fixture.jtime,
        fixture.q_goal,
        fixture.boresight,
        fixture.R,
        fixture.V,
        fixture.B,
        fixture.S,
        fixture.rho,
    )

    assert ok
    assert X.shape == (fixture.satellite.stateDim, fixture.N)
    assert U.shape == (fixture.satellite.controlDim, fixture.N)


def test_warm_start_rk4_consistency_zero_controller(fixture):
    fixture.settings.init_traj.initcontroller = 0

    ok, X, U = saltro_py.warm_start(
        fixture.settings,
        fixture.satellite,
        fixture.x0,
        fixture.jtime,
        fixture.q_goal,
        fixture.boresight,
        fixture.R,
        fixture.V,
        fixture.B,
        fixture.S,
        fixture.rho,
    )

    assert ok
    assert np.linalg.norm(U) < 1e-14

    dt = (fixture.jtime[1] - fixture.jtime[0]) * SEC_PER_CENTURY
    rho0 = int(max(0.0, np.round(fixture.rho[0])))
    u0 = np.zeros(fixture.satellite.controlDim)

    def dyn(_, x_state):
        return fixture.satellite.dynamics(
            x_state,
            u0,
            fixture.settings.disturbances,
            fixture.R[:, 0],
            fixture.B[:, 0],
            fixture.S[:, 0],
            fixture.V[:, 0],
            rho0,
        )

    x_manual_next = rk4_step(
        dyn, fixture.x0, 0.0, dt, quat_idx=saltro_py.Satellite.QUAT_INDEX
    )

    assert np.linalg.norm(X[:, 1] - x_manual_next) < 1e-11

    for k in range(fixture.N):
        qk = X[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4, k]
        assert np.isclose(np.linalg.norm(qk), 1.0, atol=1e-10)


def test_warm_start_controllers_behavior(fixture):
    fixture.settings.init_traj.initcontroller = 0
    ok_zero, X_zero, U_zero = saltro_py.warm_start(
        fixture.settings,
        fixture.satellite,
        fixture.x0,
        fixture.jtime,
        fixture.q_goal,
        fixture.boresight,
        fixture.R,
        fixture.V,
        fixture.B,
        fixture.S,
        fixture.rho,
    )

    fixture.settings.init_traj.initcontroller = 1
    ok_exc, X_exc, U_exc = saltro_py.warm_start(
        fixture.settings,
        fixture.satellite,
        fixture.x0,
        fixture.jtime,
        fixture.q_goal,
        fixture.boresight,
        fixture.R,
        fixture.V,
        fixture.B,
        fixture.S,
        fixture.rho,
    )

    assert ok_zero
    assert ok_exc
    assert np.linalg.norm(U_zero) < 1e-14
    assert np.linalg.norm(U_exc) > 1e-10
    assert np.linalg.norm(X_exc - X_zero) > 1e-10


# ---------------------------------------------------------------------------
# Magic-actuator PD warm start (post-#28 rebase).
#
# PDController.find_u now allocates across the FULL control dimension
# (nu = controlDim, Magic columns included in the actuator Jacobian), so a PD
# warm start must (a) pass warm_start's uk.size() == controlDim gate for mixed
# actuator sets, and (b) actively command Magic actuators -- including the
# Magic-ONLY case, which previously risked a silent U == 0 "success"
# indistinguishable from initcontroller=0.
# ---------------------------------------------------------------------------

def _make_custom_fixture(add_actuators, N=20, dt_seconds=0.5):
    settings = saltro_py.PlannerSettings()
    J = np.diag([0.067, 0.071, 0.069]).astype(float)
    satellite = saltro_py.Satellite(J, settings)
    add_actuators(satellite)

    settings.disturbances.plan_for_aero = False
    settings.disturbances.plan_for_gg = False
    settings.disturbances.plan_for_srp = False
    settings.disturbances.plan_for_prop = False
    settings.disturbances.plan_for_gendist = False
    settings.disturbances.plan_for_resdipole = False
    settings.num_passes = 1
    settings.passes[0].dt = dt_seconds
    settings.init_traj.initcontroller = 3

    jtime = 0.25 + np.arange(N) * dt_seconds / SEC_PER_CENTURY
    q_goal = np.zeros((4, N))
    q_goal[0, :] = 1.0
    boresight = np.tile(np.array([[1.0], [0.0], [0.0]]), (1, N))
    R = np.tile(np.array([[7000e3], [0.0], [0.0]]), (1, N))
    V = np.tile(np.array([[0.0], [7500.0], [0.0]]), (1, N))
    B = np.tile(np.array([[2.5e-5], [-1.5e-5], [3.0e-5]]), (1, N))
    S = np.tile(np.array([[1.0], [0.1], [-0.05]]), (1, N))
    rho = np.zeros(N)

    x0 = np.zeros(satellite.stateDim)
    x0[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3] = [0.02, -0.01, 0.015]
    x0[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = [1.0, 0.0, 0.0, 0.0]

    return settings, satellite, x0, jtime, q_goal, boresight, R, V, B, S, rho


def test_warm_start_pd_mixed_mtq_rw_magic_sizes_and_drives_magic():
    def actuators(sat):
        sat.addMTQ(np.array([1.0, 0.0, 0.0]), 0.2)
        sat.addMTQ(np.array([0.0, 1.0, 0.0]), 0.2)
        sat.addMTQ(np.array([0.0, 0.0, 1.0]), 0.2)
        sat.addRW(np.array([0.0, 0.0, 1.0]), 0.001, 1e-5, 0.0, 0.02)
        sat.addMagic(np.array([1.0, 0.0, 0.0]), 1.0e-3)

    settings, sat, x0, jtime, q_goal, boresight, R, V, B, S, rho = \
        _make_custom_fixture(actuators)
    assert sat.controlDim == 5

    ok, X, U = saltro_py.warm_start(
        settings, sat, x0, jtime, q_goal, boresight, R, V, B, S, rho)

    assert ok                                   # uk.size() gate accepts controlDim
    assert U.shape == (5, jtime.size)
    assert np.isfinite(U).all()
    assert np.linalg.norm(U) > 1e-10            # PD actually commands something
    # Post-#28 PD allocation spans Magic columns: the Magic row is driven too.
    assert np.abs(U[4, :]).max() > 1e-12


def test_warm_start_pd_magic_only_nonzero_control():
    def actuators(sat):
        sat.addMagic(np.array([1.0, 0.0, 0.0]), 1.0e-3)
        sat.addMagic(np.array([0.0, 1.0, 0.0]), 1.0e-3)
        sat.addMagic(np.array([0.0, 0.0, 1.0]), 1.0e-3)

    settings, sat, x0, jtime, q_goal, boresight, R, V, B, S, rho = \
        _make_custom_fixture(actuators)
    assert sat.controlDim == 3

    ok, X, U = saltro_py.warm_start(
        settings, sat, x0, jtime, q_goal, boresight, R, V, B, S, rho)

    assert ok
    assert np.isfinite(U).all()
    # The audited failure mode was a "successful" warm start with U == 0
    # (indistinguishable from initcontroller=0).  Post-#28, find_u allocates
    # over Magic columns, so the PD must command every Magic channel...
    assert np.linalg.norm(U) > 1e-10
    assert (np.abs(U).max(axis=1) > 1e-12).all()
    # ...and the rollout must actually damp the tumble (rate-to-zero PD).
    AV = saltro_py.Satellite.AV_INDEX
    omega0 = x0[AV:AV + 3]
    omegaN = X[AV:AV + 3, -1]
    assert np.linalg.norm(omegaN) < 0.5 * np.linalg.norm(omega0)


# ---------------------------------------------------------------------------
# Goal-rate feedforward: exercise the ACTUAL omega_des computation in
# warm_start (not setGoalRate directly) with a slowly rotating quaternion-goal
# sequence.  For goals q_g(k) = [cos(th_k/2), 0, 0, sin(th_k/2)],
# th_k = w*k*dt (constant body rate w about +z), the FF should command a
# spin-up toward w*z_hat and track far more tightly than the rate-to-zero PD.
# A sign or frame error in the omega_des finite difference makes the FF FIGHT
# the maneuver (tracks worse than no-FF), which this test catches.
# ---------------------------------------------------------------------------

def test_warm_start_pd_goal_rate_ff_tracks_rotating_quaternion_goal():
    N = 40
    dt = 1.0
    w_goal = 0.02  # rad/s about body +z

    def actuators(sat):
        sat.addRW(np.array([1.0, 0.0, 0.0]), 0.001, 1e-5, 0.0, 0.02)
        sat.addRW(np.array([0.0, 1.0, 0.0]), 0.001, 1e-5, 0.0, 0.02)
        sat.addRW(np.array([0.0, 0.0, 1.0]), 0.001, 1e-5, 0.0, 0.02)

    settings, sat, x0, jtime, q_goal, boresight, R, V, B, S, rho = \
        _make_custom_fixture(actuators, N=N, dt_seconds=dt)

    for k in range(N):
        th = w_goal * k * dt
        q_goal[:, k] = [np.cos(0.5 * th), 0.0, 0.0, np.sin(0.5 * th)]

    # Start ON the initial goal, at rest: all subsequent error comes from the
    # goal moving, which is exactly what the FF is supposed to anticipate.
    AV = saltro_py.Satellite.AV_INDEX
    QI = saltro_py.Satellite.QUAT_INDEX
    x0[:] = 0.0
    x0[QI:QI + 4] = [1.0, 0.0, 0.0, 0.0]

    def run(ff_enabled):
        settings.init_traj.pd_goal_rate_ff_enabled = ff_enabled
        return saltro_py.warm_start(
            settings, sat, x0, jtime, q_goal, boresight, R, V, B, S, rho)

    ok_off, X_off, U_off = run(False)
    ok_on, X_on, U_on = run(True)
    assert ok_off
    assert ok_on
    assert np.isfinite(X_on).all()
    assert np.isfinite(U_on).all()

    def attitude_err(X, k):
        q = X[QI:QI + 4, k]
        q = q / np.linalg.norm(q)
        d = min(1.0, abs(float(np.dot(q, q_goal[:, k]))))
        return 2.0 * np.arccos(d)

    err_off = np.array([attitude_err(X_off, k) for k in range(N)])
    err_on = np.array([attitude_err(X_on, k) for k in range(N)])

    # No-FF lags the moving goal substantially; FF must cut the mean tracking
    # error by well over 2x (measured: ~14.4 deg -> ~2.4 deg) and beat it at
    # the final knot too.  A sign-flipped omega_des FAILS these (error grows).
    assert err_off.mean() > 0.05                # sanity: goal does move
    assert err_on.mean() < 0.5 * err_off.mean()
    assert err_on[-1] < err_off[-1]

    # The FF rollout must acquire the analytic goal rate w*z_hat
    # (constant-rate rotation), not its negative.
    omegaN = X_on[AV:AV + 3, -1]
    assert omegaN[2] > 0.5 * w_goal
    assert abs(omegaN[2] - w_goal) < 0.25 * w_goal
