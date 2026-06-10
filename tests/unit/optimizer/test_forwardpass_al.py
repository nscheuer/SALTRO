"""
Python tests for forward pass with augmented Lagrangian inputs.
These mirror test_forwardpass_al.cpp in structure and intent.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "build"))
import saltro_py

SEC_PER_CENTURY = 36525.0 * 86400.0


def make_zero_aug_terms(satellite, settings, X, U, S):
    N = X.shape[1]
    nu = satellite.controlDim
    lambda_aug = []
    mu_aug = []
    for k in range(N):
        uk = U[:, k] if k < U.shape[1] else np.zeros(nu)
        ck = np.asarray(
            satellite.constraints(k, N, X[:, k], uk, S[:, k], settings.constraints),
            dtype=float,
        )
        lambda_aug.append(np.zeros_like(ck))
        mu_aug.append(np.zeros_like(ck))
    return lambda_aug, mu_aug


def make_active_aug_terms(c_list, lambda_value=0.5, mu_value=1.0):
    lambda_aug = []
    mu_aug = []
    for c in c_list:
        lambda_aug.append(np.where(c > 0.0, lambda_value, 0.0))
        mu_aug.append(np.where(c > 0.0, mu_value, 0.0))
    return lambda_aug, mu_aug


def augmented_penalty_total(settings, satellite, X, U, S, lambda_aug, mu_aug):
    if not lambda_aug or not mu_aug:
        return 0.0

    N = X.shape[1]
    nu = satellite.controlDim
    total = 0.0
    for k in range(min(N, len(lambda_aug), len(mu_aug))):
        uk = U[:, k] if k < U.shape[1] else np.zeros(nu)
        ck = np.asarray(
            satellite.constraints(k, N, X[:, k], uk, S[:, k], settings.constraints),
            dtype=float,
        )
        lam = np.asarray(lambda_aug[k], dtype=float)
        mu = np.asarray(mu_aug[k], dtype=float)
        if lam.shape[0] != ck.shape[0] or mu.shape[0] != ck.shape[0]:
            continue
        # Mirror the forward-pass merit exactly: lambda term always active
        # with signed c; mu penalty when c > 0 OR lambda > 0.
        warm = (ck > 0.0) | (lam > 0.0)
        total += float(lam @ ck + 0.5 * np.sum(np.where(warm, mu * ck * ck, 0.0)))
    return total


class ForwardPassALFixture:
    def __init__(self, N=8):
        self.N = N

        self.settings = saltro_py.PlannerSettings()
        self.settings.num_passes = 1
        self.settings.passes[0].dt = 0.5
        self.settings.disturbances.plan_for_aero = False
        self.settings.disturbances.plan_for_gg = False
        self.settings.disturbances.plan_for_srp = False
        self.settings.disturbances.plan_for_prop = False
        self.settings.disturbances.plan_for_gendist = False
        self.settings.disturbances.plan_for_resdipole = False
        self.settings.passes[0].reg.reg_init = 1e-2

        J = np.diag([0.067, 0.071, 0.069])
        self.satellite = saltro_py.Satellite(J, self.settings)
        self.satellite.addMTQ(np.array([1.0, 0.0, 0.0]), 0.2)
        self.satellite.addMTQ(np.array([0.0, 1.0, 0.0]), 0.2)
        self.satellite.addMTQ(np.array([0.0, 0.0, 1.0]), 0.2)
        self.satellite.addRW(np.array([1.0, 0.0, 0.0]), 0.001, 1e-5, 0.0, 0.02)
        self.satellite.addRW(np.array([0.0, 1.0, 0.0]), 0.001, 1e-5, 0.0, 0.02)
        self.satellite.addRW(np.array([0.0, 0.0, 1.0]), 0.001, 1e-5, 0.0, 0.02)

        self.nx = self.satellite.stateDim
        self.nu = self.satellite.controlDim

        self.x0 = np.zeros(self.nx)
        self.x0[0:3] = np.array([0.02, -0.01, 0.015])
        self.x0[3:7] = np.array([1.0, 0.0, 0.0, 0.0])

        self.jtime = np.zeros(self.N)
        dt_centuries = self.settings.passes[0].dt / SEC_PER_CENTURY
        for k in range(self.N):
            self.jtime[k] = 0.25 + k * dt_centuries

        self.q_goal = np.zeros((4, self.N))
        self.q_goal[0, :] = 1.0
        self.boresight = np.zeros((3, self.N))
        self.boresight[0, :] = 1.0
        self.attitude_target_traj = self.q_goal.copy()

        r0 = np.array([7000e3, 0.0, 0.0])
        v0 = np.array([0.0, 7500.0, 0.0])
        ok, R, V, B, S, rho = saltro_py.generate_orbit(
            r0,
            v0,
            self.jtime,
            0,
            0,
            0,
            0,
            0,
        )
        if not ok:
            raise RuntimeError("generate_orbit failed")

        self.R = R
        self.V = V
        self.B = B
        self.S = S
        self.rho = rho.reshape(1, -1)

    def warm_start(self):
        ok, X, U = saltro_py.warm_start(
            self.settings,
            self.satellite,
            self.x0,
            self.jtime,
            self.q_goal,
            self.boresight,
            self.R,
            self.V,
            self.B,
            self.S,
            self.rho,
        )
        assert ok
        return X, U

    def initial_trajectory(self):
        ok, X, U = saltro_py.warm_start(
            self.settings,
            self.satellite,
            self.x0,
            self.jtime,
            self.q_goal,
            self.boresight,
            self.R,
            self.V,
            self.B,
            self.S,
            self.rho,
        )
        if not ok:
            X = np.repeat(self.x0[:, None], self.N, axis=1)
            U = np.zeros((self.nu, self.N))
        return X, U


def run_backward(fixture, X, U_trim, lambda_aug, mu_aug):
    return saltro_py.backward_pass(
        fixture.satellite,
        X,
        U_trim,
        fixture.R,
        fixture.V,
        fixture.B,
        fixture.S,
        fixture.rho,
        fixture.boresight,
        fixture.attitude_target_traj,
        fixture.settings,
        lambda_aug,
        mu_aug,
        fixture.settings.passes[0].reg.reg_init,
    )


def run_forward(fixture, X, U, K, d, deltaV, lambda_aug, mu_aug, J_prev):
    K_list = [K[k] for k in range(K.shape[0])]
    d_list = [d[:, k] for k in range(d.shape[1])]
    return saltro_py.forward_pass(
        fixture.satellite,
        X,
        U,
        K_list,
        d_list,
        deltaV,
        fixture.B,
        fixture.R,
        fixture.V,
        fixture.S,
        fixture.rho,
        fixture.boresight,
        fixture.attitude_target_traj,
        fixture.settings,
        lambda_aug,
        mu_aug,
        fixture.jtime,
        J_prev,
    )


@pytest.fixture
def fixture():
    return ForwardPassALFixture(N=8)


def test_forward_pass_al_runs_with_constraint_sized_multipliers(fixture):
    X, U = fixture.initial_trajectory()
    U_trim = U[:, : fixture.N - 1]

    lambda_aug, mu_aug = make_zero_aug_terms(fixture.satellite, fixture.settings, X, U_trim, fixture.S)
    ok_bp, K, d, deltaV = run_backward(fixture, X, U_trim, lambda_aug, mu_aug)
    assert ok_bp

    cost_cfg = fixture.settings.passes[0].cost
    J_prev = fixture.satellite.totalCost(X, U_trim, fixture.B, fixture.boresight, fixture.attitude_target_traj, cost_cfg)

    ok_fp, X_new, U_new, J_new = run_forward(fixture, X, U, K, d, deltaV, lambda_aug, mu_aug, J_prev)
    assert ok_fp
    assert np.all(np.isfinite(X_new))
    assert np.all(np.isfinite(U_new))
    assert np.isfinite(J_new)


def test_forward_pass_al_active_penalties_modify_cost(fixture):
    fixture.settings.constraints.wmax = 1e-4
    fixture.settings.constraints.control_limit_scale = 1.0
    fixture.settings.constraints.sun_limit_angle = np.pi
    fixture.settings.constraints.u_max = np.full(fixture.satellite.controlDim, 1e-4)

    X, U = fixture.initial_trajectory()
    U_trim = U[:, : fixture.N - 1]

    c_list = []
    for k in range(fixture.N):
        uk = U_trim[:, k] if k < U_trim.shape[1] else np.zeros(fixture.satellite.controlDim)
        c_list.append(np.asarray(fixture.satellite.constraints(k, fixture.N, X[:, k], uk, fixture.S[:, k], fixture.settings.constraints), dtype=float))

    assert any(np.any(c > 0.0) for c in c_list[:-1])

    lambda_zero, mu_zero = make_zero_aug_terms(fixture.satellite, fixture.settings, X, U_trim, fixture.S)
    lambda_active, mu_active = make_active_aug_terms(c_list, lambda_value=0.5, mu_value=1.0)

    ok_bp0, K0, d0, deltaV0 = run_backward(fixture, X, U_trim, lambda_zero, mu_zero)
    ok_bp1, K1, d1, deltaV1 = run_backward(fixture, X, U_trim, lambda_active, mu_active)
    assert ok_bp0 and ok_bp1

    cost_cfg = fixture.settings.passes[0].cost
    J_nom = fixture.satellite.totalCost(X, U_trim, fixture.B, fixture.boresight, fixture.attitude_target_traj, cost_cfg)
    J_prev0 = J_nom + augmented_penalty_total(fixture.settings, fixture.satellite, X, U_trim, fixture.S, lambda_zero, mu_zero)
    J_prev1 = J_nom + augmented_penalty_total(fixture.settings, fixture.satellite, X, U_trim, fixture.S, lambda_active, mu_active)

    ok_fp0, _, _, J_new0 = run_forward(fixture, X, U, K0, d0, deltaV0, lambda_zero, mu_zero, J_prev0)
    ok_fp1, _, _, J_new1 = run_forward(fixture, X, U, K1, d1, deltaV1, lambda_active, mu_active, J_prev1)
    assert ok_fp0 == ok_fp1
    if ok_fp0:
        assert J_new1 >= J_new0 - 1e-9
    else:
        assert abs(J_new0 - J_prev0) < 1e-12
        assert abs(J_new1 - J_prev1) < 1e-12


def test_forward_pass_al_size_mismatch_returns_false(fixture):
    X, U = fixture.initial_trajectory()
    U_trim = U[:, : fixture.N - 1]

    lambda_ok, mu_ok = make_zero_aug_terms(fixture.satellite, fixture.settings, X, U_trim, fixture.S)
    ok_bp, K, d, deltaV = run_backward(fixture, X, U_trim, lambda_ok, mu_ok)
    assert ok_bp

    lambda_bad = [np.zeros(c.shape[0] + 1) for c in lambda_ok]
    mu_bad = [np.zeros(c.shape[0] + 2) for c in mu_ok]

    cost_cfg = fixture.settings.passes[0].cost
    J_prev = fixture.satellite.totalCost(X, U_trim, fixture.B, fixture.boresight, fixture.attitude_target_traj, cost_cfg)

    ok_fp, _, _, J_new = run_forward(fixture, X, U, K, d, deltaV, lambda_bad, mu_bad, J_prev)
    assert not ok_fp
    assert abs(J_new - J_prev) < 1e-12


def test_forward_pass_al_partial_horizon_multiplier_lists(fixture):
    X, U = fixture.initial_trajectory()
    U_trim = U[:, : fixture.N - 1]

    lambda_aug, mu_aug = make_zero_aug_terms(fixture.satellite, fixture.settings, X, U_trim, fixture.S)
    ok_bp, K, d, deltaV = run_backward(fixture, X, U_trim, lambda_aug, mu_aug)
    assert ok_bp

    lambda_short = [lambda_aug[0]]
    mu_short = [mu_aug[0]]

    cost_cfg = fixture.settings.passes[0].cost
    J_prev = fixture.satellite.totalCost(X, U_trim, fixture.B, fixture.boresight, fixture.attitude_target_traj, cost_cfg)

    ok_fp, X_new, U_new, J_new = run_forward(fixture, X, U, K, d, deltaV, lambda_short, mu_short, J_prev)
    if ok_fp:
        assert np.all(np.isfinite(X_new))
        assert np.all(np.isfinite(U_new))
        assert np.isfinite(J_new)
    else:
        assert abs(J_new - J_prev) < 1e-12


def test_forward_pass_al_long_horizon_random_multipliers_stable():
    fixture = ForwardPassALFixture(N=10)
    X, U = fixture.initial_trajectory()
    U_trim = U[:, : fixture.N - 1]

    rng = np.random.default_rng(7)
    lambda_aug, mu_aug = make_zero_aug_terms(fixture.satellite, fixture.settings, X, U_trim, fixture.S)

    for k in range(len(lambda_aug)):
        lambda_aug[k] = np.maximum(0.0, rng.normal(0.2, 0.1, size=lambda_aug[k].shape[0]))
        mu_aug[k] = np.maximum(0.0, rng.normal(1.0, 0.3, size=mu_aug[k].shape[0]))

    ok_bp, K, d, deltaV = run_backward(fixture, X, U_trim, lambda_aug, mu_aug)
    assert ok_bp

    cost_cfg = fixture.settings.passes[0].cost
    J_nom = fixture.satellite.totalCost(X, U_trim, fixture.B, fixture.boresight, fixture.attitude_target_traj, cost_cfg)
    J_prev = J_nom + augmented_penalty_total(fixture.settings, fixture.satellite, X, U_trim, fixture.S, lambda_aug, mu_aug)

    ok_fp, X_new, U_new, J_new = run_forward(fixture, X, U, K, d, deltaV, lambda_aug, mu_aug, J_prev)
    assert ok_fp
    assert np.all(np.isfinite(X_new))
    assert np.all(np.isfinite(U_new))
    assert np.isfinite(J_new)
