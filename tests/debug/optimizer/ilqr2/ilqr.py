import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
import saltro_py

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
        snapshots.append({"X": X.copy(), "U": U.copy(), "J": J, "q_goal": q_goal.copy()})

    for iteration in range(passsettings.ilqr.max_iters):
        reg = passsettings.reg.reg_init

        while reg <= passsettings.reg.reg_max:
            U_trim = U[:, :X.shape[1] - 1]
            ok_bp, K, d, deltaV = saltro_py.backward_pass(satellite, X, U_trim, R, V, B, S, rho, boresight, q_goal, plannersettings, reg)
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
                snapshots.append({"X": X.copy(), "U": U.copy(), "J": J_new, "q_goal": q_goal.copy()})
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