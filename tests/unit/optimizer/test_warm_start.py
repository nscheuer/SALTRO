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
