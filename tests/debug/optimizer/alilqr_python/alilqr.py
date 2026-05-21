import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
import saltro_py
from ilqr import ilqr
from ilqr import compute_cost_components


def _collect_constraints(
    plannersettings: saltro_py.PlannerSettings,
    satellite: saltro_py.Satellite,
    X: np.ndarray,
    U: np.ndarray,
    S: np.ndarray,
) -> list[np.ndarray]:
    """Collect c_k for all timesteps using c_k(x_k, u_k) <= 0 convention."""
    N = X.shape[1]
    cnst_cfg = plannersettings.constraints
    clist: list[np.ndarray] = []

    for k in range(N):
        xk = X[:, k]
        if U.shape[1] == N - 1 and k < N - 1:
            uk = U[:, k]
        elif U.shape[1] == N and k < N:
            uk = U[:, k]
        else:
            uk = np.zeros(satellite.controlDim)
        clist.append(np.asarray(satellite.constraints(k, N, xk, uk, S[:, k], cnst_cfg), dtype=float))

    return clist


def _constraint_violation_timeseries(
    plannersettings: saltro_py.PlannerSettings,
    satellite: saltro_py.Satellite,
    X: np.ndarray,
    U: np.ndarray,
    S: np.ndarray,
) -> np.ndarray:
    """Return max positive constraint violation at each timestep."""
    clist = _collect_constraints(plannersettings, satellite, X, U, S)
    return np.asarray([float(np.max(np.maximum(0.0, ck))) for ck in clist], dtype=float)


def _constraint_component_timeseries(
    plannersettings: saltro_py.PlannerSettings,
    satellite: saltro_py.Satellite,
    X: np.ndarray,
    U: np.ndarray,
    S: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return per-type positive violation traces for each timestep."""
    clist = _collect_constraints(plannersettings, satellite, X, U, S)
    n = len(clist)

    components = {
        "angular_velocity": np.zeros(n, dtype=float),
        "sun_avoidance": np.zeros(n, dtype=float),
        "mtq_limits": np.zeros(n, dtype=float),
        "rw_torque_limits": np.zeros(n, dtype=float),
        "rw_momentum_limits": np.zeros(n, dtype=float),
        "rw_stiction": np.zeros(n, dtype=float),
        "other": np.zeros(n, dtype=float),
    }

    mtq_terms = 2 * satellite.numMTQ
    rw_torque_terms = 2 * satellite.numRW
    rw_momentum_terms = 2 * satellite.numRW
    rw_stiction_terms = satellite.numRW

    for k, ck in enumerate(clist):
        c_pos = np.maximum(0.0, np.asarray(ck, dtype=float).reshape(-1))
        idx = 0

        if idx < c_pos.size:
            components["angular_velocity"][k] = float(c_pos[idx])
            idx += 1
        if idx < c_pos.size:
            components["sun_avoidance"][k] = float(c_pos[idx])
            idx += 1

        remaining = c_pos.size - idx
        take = min(mtq_terms, remaining)
        if take > 0:
            components["mtq_limits"][k] = float(np.sum(c_pos[idx : idx + take]))
            idx += take

        remaining = c_pos.size - idx
        take = min(rw_torque_terms, remaining)
        if take > 0:
            components["rw_torque_limits"][k] = float(np.sum(c_pos[idx : idx + take]))
            idx += take

        remaining = c_pos.size - idx
        take = min(rw_momentum_terms, remaining)
        if take > 0:
            components["rw_momentum_limits"][k] = float(np.sum(c_pos[idx : idx + take]))
            idx += take

        remaining = c_pos.size - idx
        take = min(rw_stiction_terms, remaining)
        if take > 0:
            components["rw_stiction"][k] = float(np.sum(c_pos[idx : idx + take]))
            idx += take

        if idx < c_pos.size:
            components["other"][k] = float(np.sum(c_pos[idx:]))

    return components


def max_constraint_violation(
    plannersettings: saltro_py.PlannerSettings,
    satellite: saltro_py.Satellite,
    X: np.ndarray,
    U: np.ndarray,
    S: np.ndarray,
) -> float:
    """Return the maximum positive constraint violation across all timesteps."""
    N = X.shape[1]
    cnst_cfg = plannersettings.constraints
    max_viol = 0.0

    for k in range(N):
        xk = X[:, k]

        if U.shape[1] == N - 1 and k < N - 1:
            uk = U[:, k]
        elif U.shape[1] == N and k < N:
            uk = U[:, k]
        else:
            uk = np.zeros(satellite.controlDim)

        ck = satellite.constraints(k, N, xk, uk, S[:, k], cnst_cfg)
        max_viol = max(max_viol, float(np.max(np.maximum(0.0, ck))))

    return float(max_viol)

def alilqr(
    plannersettings: saltro_py.PlannerSettings,
    pass_idx: int,
    satellite: saltro_py.Satellite,
    X: np.ndarray,
    U: np.ndarray,
    R: np.ndarray,
    V: np.ndarray,
    B: np.ndarray,
    S: np.ndarray,
    rho: np.ndarray,
    jtime: np.ndarray,
    q_goal: np.ndarray,
    boresight: np.ndarray,
    debug: bool = False,
) -> tuple[np.ndarray, np.ndarray, str, list, list]:
    passsettings = plannersettings.passes[pass_idx]
    
    snapshots = []
    transitions = []

    clist0 = _collect_constraints(plannersettings, satellite, X, U, S)
    lambda_aug = [
        np.full_like(ck, fill_value=passsettings.auglag.lag_mult_init, dtype=float)
        for ck in clist0
    ]
    mu_aug = [
        np.full_like(ck, fill_value=passsettings.auglag.penalty_init / passsettings.auglag.penalty_scale, dtype=float)
        for ck in clist0
    ]
    phi_aug = passsettings.auglag.penalty_scale
    stop_reason = "max_outer_iters"
    
    for iteration in range(passsettings.auglag.max_outer_iters):
        # Solve AL subproblem with iLQR first (ALTRO style).
        X, U, stop_reason, snaps, trans = ilqr(
            plannersettings, pass_idx, satellite, X, U, R, V, B, S, rho,
            jtime, q_goal, boresight, lambda_aug=lambda_aug, mu_aug=mu_aug, debug=debug,
            outer_iter=iteration,
        )

        if debug:
            for snap in snaps:
                snap["outer_iter"] = iteration
                snap["constraint_violation_t"] = _constraint_violation_timeseries(
                    plannersettings,
                    satellite,
                    snap["X"],
                    snap["U"],
                    S,
                )
                snap["constraint_components_t"] = _constraint_component_timeseries(
                    plannersettings,
                    satellite,
                    snap["X"],
                    snap["U"],
                    S,
                )
            snapshots.extend(snaps)
            transitions.extend(trans)

        # Check convergence after solving the current AL subproblem.
        max_c = max_constraint_violation(plannersettings, satellite, X, U, S)
        if debug:
            transitions.append(
                {
                    "outer_iter": iteration,
                    "max_constraint_violation": max_c,
                    "lambda_max": float(max(np.max(lam) for lam in lambda_aug)) if lambda_aug else 0.0,
                    "mu_max": float(max(np.max(mu) for mu in mu_aug)) if mu_aug else 0.0,
                }
            )

        # AL outer convergence — match C++ alilqr.cpp:155-164.
        # Exit when constraints are satisfied AND either:
        #   (a) the inner iLQR reported Converged AND we have done at least
        #       `min_outer_iters` outer iters (so duals are not arbitrary), OR
        #   (b) `max_c <= constraint_tol_strict` — strict fast-path bypassing
        #       the maturity gate (disabled when constraint_tol_strict <= 0).
        # Previously python exited on `max_c <= constraint_tol` alone, which
        # could declare victory after an inner MaxIterations bailout or before
        # duals had settled — divergence from C++ behavior.
        min_outer = int(getattr(passsettings.auglag, "min_outer_iters", 1))
        constraint_tol_strict = float(getattr(passsettings.auglag, "constraint_tol_strict", 0.0))
        if max_c <= passsettings.auglag.constraint_tol:
            inner_ok = (stop_reason == "converged")
            outer_matured = (iteration + 1) >= min_outer
            strict_path = (constraint_tol_strict > 0.0) and (max_c <= constraint_tol_strict)
            if strict_path or (inner_ok and outer_matured):
                stop_reason = f"AL-iLQR converged: max constraint violation {max_c:.2e} <= {passsettings.auglag.constraint_tol:.2e}"
                break

        # Update lambda and mu matching C++ alilqr.cpp:
        # Lambda update uses RAW constraint value (not clamped to positive).
        # Lambda clamped to non-negative after update (inequality constraints).
        clist = _collect_constraints(plannersettings, satellite, X, U, S)
        any_mu_below_max = False
        for k, ck in enumerate(clist):
            lambda_aug[k] = np.minimum(
                passsettings.auglag.lag_mult_max,
                lambda_aug[k] + mu_aug[k] * ck,  # raw c, not max(0,c)
            )
            lambda_aug[k] = np.maximum(0.0, lambda_aug[k])  # non-negative for inequality
            mu_aug[k] = np.minimum(passsettings.auglag.penalty_max, phi_aug * mu_aug[k])
            if np.any(mu_aug[k] < passsettings.auglag.penalty_max):
                any_mu_below_max = True

        # Penalty saturation exit — match C++ alilqr.cpp:187-191.
        # If μ has saturated everywhere and we still can't drive max_c below
        # constraint_tol, further outer iterations cannot grow penalty.
        # Break out rather than burning remaining outer budget.
        if not any_mu_below_max and max_c > passsettings.auglag.constraint_tol:
            stop_reason = f"penalty_max_reached: max_c={max_c:.2e} > {passsettings.auglag.constraint_tol:.2e}"
            break

    return X, U, stop_reason, snapshots, transitions


        