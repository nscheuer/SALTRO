import numpy as np
import pytest
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


def rk4_step(f, x, dt):
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class ForwardPassFixture:
    N = 8

    def __init__(self):
        self.settings = saltro_py.PlannerSettings()
        self.settings.num_passes = 1
        self.settings.passes[0].dt = 0.5
        # Disable disturbances
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
        self.attitude_target_traj = self.q_goal

        r0 = np.array([7000e3, 0.0, 0.0])
        v0 = np.array([0.0, 7500.0, 0.0])
        ok, R, V, B, S, rho = saltro_py.generate_orbit(
            r0,
            v0,
            self.jtime,
            0,  # Keplerian
            0,  # tilted dipole
            0,  # NOAA sun
            0,  # cylindrical eclipse
            0   # Harris-Priester density
        )
        if not ok:
            raise RuntimeError("generate_orbit failed in fixture")

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
            self.rho
        )
        assert ok
        return X, U

    def rollout_with_alpha(self, alpha, K, d, X_ref, U_ref):
        X = np.zeros_like(X_ref)
        U = np.zeros((self.nu, self.N - 1))
        X[:, 0] = X_ref[:, 0]

        dist_cfg = self.settings.disturbances
        cost_cfg = self.settings.passes[0].cost

        for k in range(self.N - 1):
            dt = (self.jtime[k + 1] - self.jtime[k]) * SEC_PER_CENTURY
            u_bar = U_ref[:, k].copy()
            # Match C++ forward pass reduced-state error construction.
            xerr = np.zeros(self.satellite.reducedStateDim)
            xerr[0:3] = X[0:3, k] - X_ref[0:3, k]
            # Small-angle approximation for attitude error in reduced coordinates.
            xerr[3:6] = (
                X[self.satellite.QUAT_INDEX+1:self.satellite.QUAT_INDEX+4, k]
                - X_ref[self.satellite.QUAT_INDEX+1:self.satellite.QUAT_INDEX+4, k]
            )
            rw_idx = self.satellite.RW_MOMENTUM_INDEX
            n_rw = self.satellite.numRW
            xerr[6:6+n_rw] = X[rw_idx:rw_idx+n_rw, k] - X_ref[rw_idx:rw_idx+n_rw, k]

            u_bar += K[k] @ xerr
            u_bar += alpha * d[k]
            U[:, k] = u_bar

            def f(x_state):
                return self.satellite.dynamics(
                    x_state,
                    u_bar,
                    dist_cfg,
                    self.R[:, k],
                    self.B[:, k],
                    self.S[:, k],
                    self.V[:, k],
                    int(max(0.0, round(self.rho[0, k])))
                )

            X[:, k + 1] = rk4_step(f, X[:, k], dt)

        J = self.satellite.totalCost(X, U, self.B, self.boresight, self.attitude_target_traj, cost_cfg)
        return X, U, J


@pytest.fixture
def fixture():
    return ForwardPassFixture()


def _find_chosen_alpha(J_new, K_list, d_list, fixture, settings_ls, X_base, U_base):
    """Scan FP's deterministic halving alpha sequence and return the alpha
    whose manual rollout cost matches J_new at fp64 precision (1e-12 rel
    tol), or None if no match.  Mirrors the C++ helper of the same name.
    """
    max_iters = settings_ls.passes[0].linesearch.max_iters
    tol = 1e-12 * max(1.0, abs(J_new))
    for iter_idx in range(max_iters):
        alpha_cand = 2.0 ** (-iter_idx)
        rollout = fixture.rollout_with_alpha(
            alpha_cand, K_list, d_list, X_base, U_base
        )
        if abs(rollout[2] - J_new) <= tol:
            return alpha_cand
    return None


def test_forward_pass_reduces_cost_and_matches_dynamics(fixture):
    X, U = fixture.warm_start()

    # Backward pass to get K, d, deltaV
    U_trim = U[:, : fixture.N - 1]
    lambda_aug, mu_aug = make_zero_aug_terms(fixture.satellite, fixture.settings, X, U_trim, fixture.S)
    ok, K, d, deltaV = saltro_py.backward_pass(
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
        fixture.settings.passes[0].reg.reg_init
    )
    assert ok

    cost_cfg = fixture.settings.passes[0].cost
    J_prev = fixture.satellite.totalCost(X, U_trim, fixture.B, fixture.boresight, fixture.attitude_target_traj, cost_cfg)

    K_list = [K[k] for k in range(K.shape[0])]
    d_list = [d[:, k] for k in range(d.shape[1])]

    ok, X_new, U_new, J_new = saltro_py.forward_pass(
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
        J_prev
    )
    assert ok
    assert np.isfinite(J_new)
    assert J_new <= J_prev + 1e-6

    # Dynamics consistency via RK4
    for k in range(fixture.N - 1):
        dt = (fixture.jtime[k + 1] - fixture.jtime[k]) * SEC_PER_CENTURY
        def f(x_state):
            return fixture.satellite.dynamics(
                x_state,
                U_new[:, k],
                fixture.settings.disturbances,
                fixture.R[:, k],
                fixture.B[:, k],
                fixture.S[:, k],
                fixture.V[:, k],
                int(max(0.0, round(fixture.rho[0, k])))
            )
        x_next = rk4_step(f, X_new[:, k], dt)
        assert np.linalg.norm(x_next - X_new[:, k + 1]) < 1e-9


def test_forward_pass_J_new_matches_its_accepted_alphas_rollout_cost(fixture):
    """Self-consistency: FP's reported J_new equals the manual rollout cost
    at some alpha in {1, 1/2, 1/4, ...} that FP would try.  Engineers an
    overshooting scenario but does NOT assert FP backed off — just that
    FP's reported J_new is consistent with its own alpha-backtracking
    algorithm.
    """
    X_base, U_base = fixture.warm_start()

    U_trim = U_base[:, : fixture.N - 1]
    lambda_aug, mu_aug = make_zero_aug_terms(fixture.satellite, fixture.settings, X_base, U_trim, fixture.S)
    ok, K, d, deltaV = saltro_py.backward_pass(
        fixture.satellite,
        X_base,
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
        fixture.settings.passes[0].reg.reg_init
    )
    assert ok

    cost_cfg = fixture.settings.passes[0].cost
    J_prev = fixture.satellite.totalCost(X_base, U_trim, fixture.B, fixture.boresight, fixture.attitude_target_traj, cost_cfg)

    scales = [2.0, 4.0, 6.0, 8.0]
    chosen_scale = None
    alpha1 = None
    alpha_half = None

    for scale in scales:
        d_scaled = d.copy()
        d_scaled *= scale
        alpha1 = fixture.rollout_with_alpha(1.0, [K[k] for k in range(K.shape[0])], [d_scaled[:, k] for k in range(d_scaled.shape[1])], X_base, U_base)
        alpha_half = fixture.rollout_with_alpha(0.5, [K[k] for k in range(K.shape[0])], [d_scaled[:, k] for k in range(d_scaled.shape[1])], X_base, U_base)
        if alpha1[2] > J_prev and alpha_half[2] < J_prev:
            chosen_scale = scale
            break

    assert chosen_scale is not None

    d_scaled = d.copy()
    d_scaled *= chosen_scale
    deltaV_scaled = np.array([
        chosen_scale * deltaV[0],
        chosen_scale * chosen_scale * deltaV[1],
    ])

    # Slightly tighter beta2 as in C++ test
    settings_ls = fixture.settings
    settings_ls.passes[0].linesearch.beta2 = 1.5

    K_list = [K[k] for k in range(K.shape[0])]
    d_list = [d_scaled[:, k] for k in range(d_scaled.shape[1])]

    ok, X_forward, U_forward, J_new = saltro_py.forward_pass(
        fixture.satellite,
        X_base,
        U_base,
        K_list,
        d_list,
        deltaV_scaled,
        fixture.B,
        fixture.R,
        fixture.V,
        fixture.S,
        fixture.rho,
        fixture.boresight,
        fixture.attitude_target_traj,
        settings_ls,
        lambda_aug,
        mu_aug,
        fixture.jtime,
        J_prev
    )
    assert ok
    # Cost-decrease invariant: never worse than alpha=1.
    assert J_new <= alpha1[2] + 1e-8

    # FP backtracks deterministically through alpha = {1, 1/2, 1/4, ...}
    # (forwardpass.cpp: `alpha = std::ldexp(1.0, -iter)`).  Identify the
    # alpha FP actually accepted by matching J_new against the cost of
    # each candidate's rollout, then assert exact match at fp64 precision.
    chosen_alpha = _find_chosen_alpha(
        J_new, K_list, d_list, fixture, settings_ls, X_base, U_base
    )
    assert chosen_alpha is not None

    # Controls should not be empty
    assert U_forward.shape[1] >= fixture.N - 1


def test_forward_pass_accepts_alpha_1_when_full_step_already_descends(fixture):
    """Use the unscaled BP step (descends at alpha=1) and assert FP picks
    alpha=1 exactly — no backtracking needed.
    """
    X_base, U_base = fixture.warm_start()

    U_trim = U_base[:, : fixture.N - 1]
    lambda_aug, mu_aug = make_zero_aug_terms(fixture.satellite, fixture.settings, X_base, U_trim, fixture.S)
    ok, K, d, deltaV = saltro_py.backward_pass(
        fixture.satellite,
        X_base, U_trim,
        fixture.R, fixture.V, fixture.B, fixture.S, fixture.rho,
        fixture.boresight, fixture.attitude_target_traj,
        fixture.settings,
        lambda_aug, mu_aug,
        fixture.settings.passes[0].reg.reg_init
    )
    assert ok

    cost_cfg = fixture.settings.passes[0].cost
    J_prev = fixture.satellite.totalCost(
        X_base, U_trim, fixture.B, fixture.boresight,
        fixture.attitude_target_traj, cost_cfg
    )

    K_list = [K[k] for k in range(K.shape[0])]
    d_list = [d[:, k] for k in range(d.shape[1])]

    # Precondition: alpha=1 with the BP-computed d already descends.
    alpha1 = fixture.rollout_with_alpha(1.0, K_list, d_list, X_base, U_base)
    assert alpha1[2] < J_prev

    ok, X_forward, U_forward, J_new = saltro_py.forward_pass(
        fixture.satellite,
        X_base, U_base,
        K_list, d_list, deltaV,
        fixture.B, fixture.R, fixture.V, fixture.S, fixture.rho,
        fixture.boresight, fixture.attitude_target_traj,
        fixture.settings, lambda_aug, mu_aug,
        fixture.jtime, J_prev
    )
    assert ok
    assert J_new <= alpha1[2] + 1e-8

    chosen_alpha = _find_chosen_alpha(
        J_new, K_list, d_list, fixture, fixture.settings, X_base, U_base
    )
    assert chosen_alpha is not None
    assert chosen_alpha == 1.0
