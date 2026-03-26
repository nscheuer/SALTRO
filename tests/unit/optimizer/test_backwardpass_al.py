"""
Python tests for backward pass with augmented Lagrangian inputs.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "build"))
import saltro_py


def make_attitude_traj(att, n_cols):
    traj = np.zeros((4, n_cols))
    for k in range(n_cols):
        traj[:, k] = att
    return traj


class BackwardPassALFixture:
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

        self.reg = self.settings.passes[0].reg.reg_init

    def make_trajectory(self, force_active=False):
        nx = self.satellite.stateDim
        nu = self.satellite.controlDim
        N = self.N

        X = np.zeros((nx, N))
        U = np.zeros((nu, N - 1))

        x0 = np.zeros(nx)
        x0[0:3] = np.array([0.01, -0.005, 0.008])
        x0[3:7] = np.array([1.0, 0.0, 0.0, 0.0])

        for k in range(N):
            X[:, k] = x0
            if force_active and k < N - 1:
                U[:, k] = np.array([0.15, -0.12, 0.10, 0.01, -0.01, 0.01])

        if force_active:
            self.settings.constraints.wmax = 1e-4
            self.settings.constraints.control_limit_scale = 1.0
            self.settings.constraints.sun_limit_angle = np.pi

        R = np.zeros((3, N))
        V = np.zeros((3, N))
        B = np.zeros((3, N))
        S = np.zeros((3, N))
        rho = np.zeros((1, N))
        boresight = np.zeros((3, N))

        for k in range(N):
            R[:, k] = np.array([7000e3, 0.0, 0.0])
            V[:, k] = np.array([0.0, 7500.0, 0.0])
            B[:, k] = np.array([2.5e-5, -1.5e-5, 3.0e-5])
            S[:, k] = np.array([1.0, 0.1, -0.05])
            S[:, k] /= np.linalg.norm(S[:, k])
            boresight[:, k] = np.array([1.0, 0.0, 0.0])

        attitude_target = np.array([np.nan, 0.0, 0.0, 0.0])
        attitude_target_traj = make_attitude_traj(attitude_target, N)

        return X, U, R, V, B, S, rho, boresight, attitude_target_traj


def collect_constraints(fixture, X, U, S):
    sat = fixture.satellite
    cfg = fixture.settings.constraints
    N = X.shape[1]
    nu = sat.controlDim
    c_list = []
    for k in range(N):
        uk = U[:, k] if k < U.shape[1] else np.zeros(nu)
        ck = np.asarray(sat.constraints(k, N, X[:, k], uk, S[:, k], cfg), dtype=float)
        c_list.append(ck)
    return c_list


def make_zero_aug(c_list):
    lambda_aug = [np.zeros_like(c) for c in c_list]
    mu_aug = [np.zeros_like(c) for c in c_list]
    return lambda_aug, mu_aug


def run_backward(fixture, X, U, R, V, B, S, rho, boresight, attitude_target, lambda_aug, mu_aug):
    return saltro_py.backward_pass(
        fixture.satellite,
        X,
        U,
        R,
        V,
        B,
        S,
        rho,
        boresight,
        attitude_target,
        fixture.settings,
        lambda_aug,
        mu_aug,
        fixture.reg,
    )


@pytest.fixture
def fixture():
    return BackwardPassALFixture(n=6)


def test_backward_pass_al_runs_with_constraint_sized_multipliers(fixture):
    X, U, R, V, B, S, rho, boresight, attitude_target = fixture.make_trajectory(force_active=False)
    c_list = collect_constraints(fixture, X, U, S)
    lambda_aug, mu_aug = make_zero_aug(c_list)

    ok, K, d, deltaV = run_backward(fixture, X, U, R, V, B, S, rho, boresight, attitude_target, lambda_aug, mu_aug)

    assert ok
    assert K.shape == (fixture.N - 1, fixture.satellite.controlDim, fixture.satellite.reducedStateDim)
    assert d.shape == (fixture.satellite.controlDim, fixture.N - 1)
    assert np.all(np.isfinite(K))
    assert np.all(np.isfinite(d))
    assert np.all(np.isfinite(deltaV))


def test_backward_pass_al_active_penalties_modify_solution(fixture):
    X, U, R, V, B, S, rho, boresight, attitude_target = fixture.make_trajectory(force_active=True)
    c_list = collect_constraints(fixture, X, U, S)
    zero_lambda, zero_mu = make_zero_aug(c_list)

    any_active = any(np.any(c > 0.0) for c in c_list[:-1])
    assert any_active

    lambda_aug = [np.where(c > 0.0, 5.0, 0.0) for c in c_list]
    mu_aug = [np.where(c > 0.0, 25.0, 0.0) for c in c_list]

    ok0, K0, d0, deltaV0 = run_backward(fixture, X, U, R, V, B, S, rho, boresight, attitude_target, zero_lambda, zero_mu)
    ok1, K1, d1, deltaV1 = run_backward(fixture, X, U, R, V, B, S, rho, boresight, attitude_target, lambda_aug, mu_aug)

    assert ok0 and ok1
    assert np.linalg.norm(K1 - K0) > 1e-12 or np.linalg.norm(d1 - d0) > 1e-12
    assert np.linalg.norm(deltaV1 - deltaV0) > 1e-12


def test_backward_pass_al_size_mismatch_is_ignored(fixture):
    X, U, R, V, B, S, rho, boresight, attitude_target = fixture.make_trajectory(force_active=True)
    c_list = collect_constraints(fixture, X, U, S)
    zero_lambda, zero_mu = make_zero_aug(c_list)

    bad_lambda = [np.zeros(c.shape[0] + 1) for c in c_list]
    bad_mu = [np.ones(c.shape[0] + 2) for c in c_list]

    ok0, K0, d0, deltaV0 = run_backward(fixture, X, U, R, V, B, S, rho, boresight, attitude_target, zero_lambda, zero_mu)
    okb, Kb, db, deltaVb = run_backward(fixture, X, U, R, V, B, S, rho, boresight, attitude_target, bad_lambda, bad_mu)

    assert ok0 and okb
    assert np.allclose(Kb, K0, atol=1e-12, rtol=1e-10)
    assert np.allclose(db, d0, atol=1e-12, rtol=1e-10)
    assert np.allclose(deltaVb, deltaV0, atol=1e-12, rtol=1e-10)


def test_backward_pass_al_partial_horizon_multiplier_lists(fixture):
    X, U, R, V, B, S, rho, boresight, attitude_target = fixture.make_trajectory(force_active=True)
    c_list = collect_constraints(fixture, X, U, S)

    lambda_short = [np.where(c_list[0] > 0.0, 2.0, 0.0)]
    mu_short = [np.where(c_list[0] > 0.0, 10.0, 0.0)]

    ok, K, d, deltaV = run_backward(
        fixture, X, U, R, V, B, S, rho, boresight, attitude_target, lambda_short, mu_short
    )

    assert ok
    assert np.all(np.isfinite(K))
    assert np.all(np.isfinite(d))
    assert np.all(np.isfinite(deltaV))


def test_backward_pass_al_long_horizon_random_multipliers_are_stable():
    fixture = BackwardPassALFixture(n=10)
    X, U, R, V, B, S, rho, boresight, attitude_target = fixture.make_trajectory(force_active=True)

    rng = np.random.default_rng(7)
    c_list = collect_constraints(fixture, X, U, S)

    lambda_aug = []
    mu_aug = []
    for c in c_list:
        lambda_aug.append(np.maximum(0.0, rng.normal(0.2, 0.1, size=c.shape[0])))
        mu_aug.append(np.maximum(0.0, rng.normal(5.0, 2.0, size=c.shape[0])))

    ok, K, d, deltaV = run_backward(
        fixture, X, U, R, V, B, S, rho, boresight, attitude_target, lambda_aug, mu_aug
    )

    assert ok
    assert K.shape[0] == fixture.N - 1
    assert d.shape[1] == fixture.N - 1
    assert np.all(np.isfinite(K))
    assert np.all(np.isfinite(d))
    assert np.all(np.isfinite(deltaV))
