import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import saltro_py
from spike_removal import apply_spike_removal


def _augmented_penalty_total(
    plannersettings: saltro_py.PlannerSettings,
    satellite: saltro_py.Satellite,
    X: np.ndarray,
    U: np.ndarray,
    S: np.ndarray,
    lambda_aug: list[np.ndarray] | None,
    mu_aug: list[np.ndarray] | None,
) -> float:
    """Compute Σ_k [lambda_k^T c_k^+ + 0.5 * c_k^{+T} diag(mu_k) c_k^+]."""
    if lambda_aug is None or mu_aug is None:
        return 0.0

    N = X.shape[1]
    cnst_cfg = plannersettings.constraints
    total = 0.0

    for k in range(N):
        xk = X[:, k]
        if U.shape[1] == N - 1 and k < N - 1:
            uk = U[:, k]
        elif U.shape[1] == N and k < N:
            uk = U[:, k]
        else:
            uk = np.zeros(satellite.controlDim)

        ck = np.asarray(satellite.constraints(k, N, xk, uk, S[:, k], cnst_cfg), dtype=float)
        lam_k = np.asarray(lambda_aug[k], dtype=float)
        mu_k = np.asarray(mu_aug[k], dtype=float)
        # Lambda term always active; mu penalty active when c>0 OR lambda>0
        for i in range(len(ck)):
            total += lam_k[i] * ck[i]
            if ck[i] > 0.0 or lam_k[i] > 0.0:
                total += 0.5 * mu_k[i] * ck[i] * ck[i]

    return total

def compute_cost_components(X, U, satellite, attitude_target_traj, boresight, B, cost_cfg):
    """Compute cost breakdown by component at each timestep."""
    N = X.shape[1]
    components = {
        "attitude": np.zeros(N),
        "angular_velocity": np.zeros(N),
        "control": np.zeros(N),
        "rw_momentum": np.zeros(N),
    }
    
    for comp_name in components.keys():
        cfg = saltro_py.CostConfig()
        cfg.angle = cost_cfg.angle if comp_name == "attitude" else 0.0
        cfg.angle_N = cost_cfg.angle_N if comp_name == "attitude" else 0.0
        cfg.ang_vel = cost_cfg.ang_vel if comp_name == "angular_velocity" else 0.0
        cfg.ang_vel_N = cost_cfg.ang_vel_N if comp_name == "angular_velocity" else 0.0
        cfg.ang_vel_mag = cost_cfg.ang_vel_mag if comp_name == "angular_velocity" else 0.0
        cfg.ang_vel_mag_N = cost_cfg.ang_vel_mag_N if comp_name == "angular_velocity" else 0.0
        cfg.ang_vel_err_dir = cost_cfg.ang_vel_err_dir if comp_name == "angular_velocity" else 0.0
        cfg.ang_vel_err_dir_N = cost_cfg.ang_vel_err_dir_N if comp_name == "angular_velocity" else 0.0
        cfg.control_mult = cost_cfg.control_mult if comp_name == "control" else 0.0
        cfg.mtq_control_weight = cost_cfg.mtq_control_weight if comp_name == "control" else 0.0
        cfg.rw_control_weight = cost_cfg.rw_control_weight if comp_name == "control" else 0.0
        cfg.magic_control_weight = cost_cfg.magic_control_weight if comp_name == "control" else 0.0
        cfg.rw_AM_weight = cost_cfg.rw_AM_weight if comp_name == "rw_momentum" else 0.0
        cfg.rw_stic_weight = cost_cfg.rw_stic_weight if comp_name == "rw_momentum" else 0.0
        cfg.RWh_max_mult = cost_cfg.RWh_max_mult if comp_name == "rw_momentum" else 0.0
        cfg.RWh_stiction_mult = cost_cfg.RWh_stiction_mult if comp_name == "rw_momentum" else 0.0
        cfg.RWh_ok_mult = cost_cfg.RWh_ok_mult if comp_name == "rw_momentum" else 0.0
        cfg.ang_cost_func_type = cost_cfg.ang_cost_func_type
        cfg.use_cost_hess = False
        
        for k in range(N):
            u_k = U[:, k] if k < U.shape[1] else np.zeros(satellite.controlDim)
            c = satellite.stageCost(
                k, N,
                X[:, k],
                u_k,
                boresight[:, k],
                attitude_target_traj[:, k],
                B[:, k],
                cfg,
            )
            components[comp_name][k] = max(0.0, c)
    
    return components

def ilqr(
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
    lambda_aug: list[np.ndarray],
    mu_aug: list[np.ndarray],
    debug: bool = False
) -> tuple[np.ndarray, np.ndarray, str, list, list, dict]:
    passsettings = plannersettings.passes[pass_idx]
    
    snapshots = []
    transitions = []
    info = {
        "accepted_steps": 0,
        "iterations": 0,
        "last_delta_J": np.inf,
        "final_cost": np.nan,
    }
    
    # Warm-start snapshot
    if debug:
        U_trim = U[:, :X.shape[1] - 1]
        J_nom = satellite.totalCost(X, U_trim, B, boresight, q_goal, passsettings.cost)
        J = J_nom + _augmented_penalty_total(plannersettings, satellite, X, U, S, lambda_aug, mu_aug)
        components = compute_cost_components(X, U, satellite, q_goal, boresight, B, passsettings.cost)
        snapshots.append(
            {
                "X": X.copy(),
                "U": U.copy(),
                "J": J,
                "q_goal": q_goal.copy(),
                "boresight": boresight.copy(),
                "components": components,
                "R": R.copy(),
                "B": B.copy(),
            }
        )

    for iteration in range(passsettings.ilqr.max_iters):
        info["iterations"] = iteration + 1
        reg = passsettings.reg.reg_init

    def increase_reg():
        nonlocal reg, dreg
        dreg = max(dreg * reg_scale, reg_scale)
        reg = max(reg * dreg, reg_min)

    def decrease_reg():
        nonlocal reg, dreg
        dreg = min(dreg / reg_scale, 1.0 / reg_scale)
        reg = max(dreg * reg, reg_min)  # match C++ iLQR.cpp:137-149 — clamp to reg_min, not snap to 0

    base_lsl = max(1, int(getattr(passsettings.ilqr, "ls_attempts_lim", 30)))
    # Post-spike iterations get a modest bump: the substitution perturbs
    # X, U, so the next BP+FP needs some room to rebuild around a new
    # trajectory.  If the rebuild requires more than a small multiple of
    # base_lsl, something else is wrong (bad substitution, stale linearization,
    # stiff integrator) and more attempts won't fix it — investigate instead.
    post_spike_lsl = max(base_lsl * 3, 50)
    spike_occurred_last_iter = False

    # Convergence machinery — match C++ iLQR.cpp:220-275.
    #   - Two-tier cost tolerance: `inner_tol = max(ilqr_cost_tol, cost_tol)`.
    #     Disjunctive convergence (default): exit on ANY of {inner cost, grad}.
    #     Conjunctive: require BOTH outer-tol cost AND grad.
    #   - Stagnation counter: increments on `delta_J <= cost_tol` (strict);
    #     resets otherwise.  Exit when count >= z_count_lim.
    ilqr_cost_tol_loose = float(getattr(passsettings.ilqr, "ilqr_cost_tol",
                                         passsettings.ilqr.cost_tol))
    inner_tol = max(ilqr_cost_tol_loose, passsettings.ilqr.cost_tol)
    grad_tol = float(getattr(passsettings.ilqr, "grad_tol", 0.0))
    z_count_lim = int(getattr(passsettings.ilqr, "z_count_lim", 0))
    conjunctive = bool(getattr(passsettings.ilqr, "conjunctive_convergence", False))
    stagnation_count = 0

    for iteration in range(passsettings.ilqr.max_iters):

        effective_lsl = post_spike_lsl if spike_occurred_last_iter else base_lsl
        spike_occurred_this_iter = False

        attempts = 0
        while reg <= reg_max and attempts < effective_lsl:
            attempts += 1
            U_trim = U[:, :X.shape[1] - 1]
            ok_bp, K, d, deltaV = saltro_py.backward_pass(
                satellite,
                X,
                U_trim,
                R,
                V,
                B,
                S,
                rho,
                boresight,
                q_goal,
                plannersettings,
                lambda_aug,
                mu_aug,
                reg,
            )
            if not ok_bp:
                increase_reg()
                continue

            # Decrease reg after successful BP (like original ALTRO)
            decrease_reg()

            K_list = [K[k] for k in range(K.shape[0])]
            d_list = [d[:, k] for k in range(d.shape[1])]

            U_trim = U[:, :X.shape[1] - 1]
            J_prev_nom = satellite.totalCost(X, U_trim, B, boresight, q_goal, passsettings.cost)
            J_prev = J_prev_nom + _augmented_penalty_total(plannersettings, satellite, X, U, S, lambda_aug, mu_aug)

            # Save nominal controls before forward pass modifies them (needed for spike removal blend)
            U_bar = U.copy()

            ok_fp, X_new, U_new, J_new = saltro_py.forward_pass(
                satellite,
                X,
                U,
                K_list,
                d_list,
                deltaV,
                B,
                R,
                V,
                S,
                rho,
                boresight,
                q_goal,
                plannersettings,
                lambda_aug,
                mu_aug,
                jtime,
                J_prev,
            )
            if not ok_fp:
                # Triple increase: increaseReg + bump + increaseReg
                increase_reg()
                reg += reg_bump
                increase_reg()
                continue

            X = X_new
            U = U_new

            # Spike removal: detect and replace homotopy artifacts after accepted step.
            # Gate: if the current trajectory is close to satisfying AL constraints
            # (max violation within gate_ratio × constraint_tol), skip.  A PD
            # substitution at this stage would perturb the trajectory enough that
            # high μ penalties make recovery hard.  Naturally adaptive — scales
            # with constraint_tol, no fixed iteration threshold.
            if spike_removal_cfg is not None:
                from alilqr import max_constraint_violation
                max_c = max_constraint_violation(plannersettings, satellite, X, U, S)
                gate_ratio = spike_removal_cfg.get("constraint_gate_ratio", 10.0)
                gate_thresh = gate_ratio * passsettings.auglag.constraint_tol
                if max_c < gate_thresh:
                    if spike_removal_cfg.get("verbose", False):
                        print(f"[SpikeRemoval] outer={outer_iter} iter={iteration}: "
                              f"skipping (max_c={max_c:.2e} < {gate_thresh:.2e}, "
                              f"{gate_ratio}× constraint_tol)")
                else:
                    # Strip the gate key before passing through (apply_spike_removal
                    # doesn't accept it).
                    cfg_inner = {k: v for k, v in spike_removal_cfg.items()
                                 if k != "constraint_gate_ratio"}
                    X, U, spike_happened = apply_spike_removal(
                        X, U, U_bar, K_list,
                        satellite, plannersettings, pass_idx,
                        R, V, B, S, rho, jtime, boresight, q_goal,
                        iteration=iteration,
                        **cfg_inner,
                    )
                    if spike_happened:
                        spike_occurred_this_iter = True

            delta_J = abs(J_prev - J_new)
            info["accepted_steps"] += 1
            info["last_delta_J"] = float(delta_J)
            info["final_cost"] = float(J_new)
            
            if debug:
                components = compute_cost_components(X, U, satellite, q_goal, boresight, B, passsettings.cost)
                snapshots.append(
                    {
                        "X": X.copy(),
                        "U": U.copy(),
                        "J": J_new,
                        "q_goal": q_goal.copy(),
                        "boresight": boresight.copy(),
                        "components": components,
                        "R": R.copy(),
                        "B": B.copy(),
                    }
                )
                transitions.append({
                    "bp_ok": True,
                    "fp_ok": True,
                    "accepted_steps": info["accepted_steps"],
                    "act_delta": delta_J,
                    "delta_tol_ok": delta_J <= passsettings.ilqr.cost_tol,
                    "inner_cost_converged": inner_cost_converged,
                    "grad_converged": grad_converged,
                    "stagnation_count": stagnation_count,
                })
            
            if delta_J <= passsettings.ilqr.cost_tol:
                return X, U, "converged", snapshots, transitions, info
            
            break

        if reg > passsettings.reg.reg_max:
            return X, U, "reg_exceeded", snapshots, transitions, info
        
    return X, U, "max_iters", snapshots, transitions, info
