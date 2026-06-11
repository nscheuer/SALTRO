import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
import saltro_py

LAMBDA_AUG_ZERO = [np.array([0.0])]
MU_AUG_ZERO = [np.array([0.0])]

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
        cfg.RWh_desat_mult = cost_cfg.RWh_desat_mult if comp_name == "rw_momentum" else 0.0
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
    debug: bool = False
) -> tuple[np.ndarray, np.ndarray, str, list, list]:
    passsettings = plannersettings.passes[pass_idx]
    
    snapshots = []
    transitions = []
    
    # Warm-start snapshot
    if debug:
        U_trim = U[:, :X.shape[1] - 1]
        J = satellite.totalCost(X, U_trim, B, boresight, q_goal, passsettings.cost)
        components = compute_cost_components(X, U, satellite, q_goal, boresight, B, passsettings.cost)
        snapshots.append({"X": X.copy(), "U": U.copy(), "J": J, "q_goal": q_goal.copy(), "components": components})

    for iteration in range(passsettings.ilqr.max_iters):
        reg = passsettings.reg.reg_init

        while reg <= passsettings.reg.reg_max:
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
                LAMBDA_AUG_ZERO,
                MU_AUG_ZERO,
                reg,
            )
            if not ok_bp:
                reg *= passsettings.reg.reg_scale
                continue

            K_list = [K[k] for k in range(K.shape[0])]
            d_list = [d[:, k] for k in range(d.shape[1])]

            U_trim = U[:, :X.shape[1] - 1]
            J_prev = satellite.totalCost(X, U_trim, B, boresight, q_goal, passsettings.cost)

            ok_fp, X_new, U_new, J_new = saltro_py.forward_pass(satellite, X, U, K_list, d_list, deltaV, B, R, V, S, rho, boresight, q_goal, plannersettings, jtime, J_prev)
            if not ok_fp:
                reg *= passsettings.reg.reg_scale
                continue

            X = X_new
            U = U_new

            delta_J = abs(J_prev - J_new)
            
            if debug:
                components = compute_cost_components(X, U, satellite, q_goal, boresight, B, passsettings.cost)
                snapshots.append({"X": X.copy(), "U": U.copy(), "J": J_new, "q_goal": q_goal.copy(), "components": components})
                transitions.append({
                    "bp_ok": True,
                    "fp_ok": True,
                    "act_delta": delta_J,
                    "delta_tol_ok": delta_J <= passsettings.ilqr.cost_tol
                })
            
            if delta_J <= passsettings.ilqr.cost_tol:
                return X, U, "converged", snapshots, transitions
            
            break

        if reg > passsettings.reg.reg_max:
            return X, U, "reg_exceeded", snapshots, transitions
        
    return X, U, "max_iters", snapshots, transitions