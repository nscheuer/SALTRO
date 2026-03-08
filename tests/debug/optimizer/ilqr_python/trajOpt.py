import sys
import numpy as np
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import saltro_py
from ilqr import ilqr

def _resample_zero_order_hold(jtime_coarse, q_goal_coarse, boresight_coarse, dt_seconds):
    dt_cent = dt_seconds / (36525.0 * 86400.0)
    t0, tN = jtime_coarse[0], jtime_coarse[-1]
    jtime_fine = np.arange(t0, tN + dt_cent/2, dt_cent)
    if abs(jtime_fine[-1] - tN) > 1e-12:
        jtime_fine = np.append(jtime_fine, tN)
    
    idx = np.searchsorted(jtime_coarse[1:], jtime_fine, side='right')
    q_goal_fine = q_goal_coarse[:, idx]
    boresight_fine = boresight_coarse[:, idx]
    
    return jtime_fine, q_goal_fine, boresight_fine

def trajOpt(
    plannersettings: saltro_py.PlannerSettings,
    satellite: saltro_py.Satellite,
    x0: np.ndarray,
    r0: np.ndarray,
    v0: np.ndarray,
    jtime: np.ndarray,
    q_goal: np.ndarray,
    boresight: np.ndarray,
    debug: bool = False
) -> tuple:
    # Discretization
    dt_sec = plannersettings.passes[0].dt
    jtime_flat, q_goal, boresight = _resample_zero_order_hold(jtime, q_goal, boresight, dt_sec)
    
    # Generate Orbit
    ok, R, V, B, S, rho = saltro_py.generate_orbit(r0, v0, jtime_flat, 0, 0, 0, 0, 0)
    if not ok:
        raise RuntimeError("generate_orbit failed")
    
    # Ensure Fortran-contiguous arrays for C++ binding
    q_goal = np.asfortranarray(q_goal)
    boresight = np.asfortranarray(boresight)
    R = np.asfortranarray(R)
    V = np.asfortranarray(V)
    B = np.asfortranarray(B)
    S = np.asfortranarray(S)
    rho = rho.reshape(1, -1) if rho.ndim == 1 else rho

    # Warm-Start
    ok, X, U = saltro_py.warm_start(plannersettings, satellite, x0, jtime_flat, q_goal, boresight, R, V, B, S, rho)
    if not ok:
        raise RuntimeError("warm_start failed")

    # Passes
    snapshots = []
    transitions = []
    stop_reason = "not_run"
    
    start_time = time.time()
    for pass_idx in range(plannersettings.num_passes):
        X, U, stop_reason, snaps, trans = ilqr(
            plannersettings, pass_idx, satellite, X, U, R, V, B, S, rho, 
            jtime_flat, q_goal, boresight, debug=debug
        )
        if debug:
            snapshots.extend(snaps)
            transitions.extend(trans)
    elapsed_time = time.time() - start_time
    
    if debug:
        return X, U, stop_reason, snapshots, transitions, dt_sec, plannersettings.passes[0].ilqr.cost_tol, elapsed_time
    else:
        return X, U, stop_reason
