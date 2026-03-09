"""Diagnose C++ crash for MTQ-only satellite (3_0_slew90_dt60)."""
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(ROOT / "tests" / "debug" / "optimizer" / "alilqr_python"))
sys.path.insert(0, str(ROOT / "tests" / "debug" / "optimizer" / "configs"))

import saltro_py
from sat_3_0_mtq import create_satellite
from trajOpt import trajOpt


def create_planner_settings():
    plannersettings = saltro_py.PlannerSettings()
    plannersettings.init_traj.initcontroller = 2
    
    plannersettings.num_passes = 1
    plannersettings.passes[0].dt = 60.0
    plannersettings.passes[0].ilqr.cost_tol = 1e-5
    plannersettings.passes[0].ilqr.max_iters = 20

    plannersettings.passes[0].auglag.max_outer_iters = 10
    plannersettings.passes[0].auglag.constraint_tol = 1e-3

    cost = plannersettings.passes[0].cost
    cost.angle = 1.0
    cost.ang_vel = 1e1
    cost.ang_vel_mag = 0.0
    cost.ang_vel_err_dir = 0.0
    cost.control_mult = 1.0
    cost.mtq_control_weight = 1e-2
    cost.rw_control_weight = 1.0
    cost.magic_control_weight = 0.0
    cost.rw_AM_weight = 0.0
    cost.rw_stic_weight = 0.0
    cost.RWh_max_mult = 0.0
    cost.RWh_stiction_mult = 0.0
    cost.RWh_ok_mult = 0.0
    cost.angle_N = 0.0
    cost.ang_vel_N = 0.0
    cost.ang_vel_mag_N = 0.0
    cost.ang_vel_err_dir_N = 0.0
    cost.ang_cost_func_type = 3
    cost.use_cost_hess = True

    plannersettings.disturbances.plan_for_aero = False
    plannersettings.disturbances.plan_for_gg = False
    plannersettings.disturbances.plan_for_srp = False
    plannersettings.disturbances.plan_for_prop = False
    plannersettings.disturbances.plan_for_gendist = False
    plannersettings.disturbances.plan_for_resdipole = False

    plannersettings.passes[0].reg.reg_init = 1e-6
    plannersettings.passes[0].reg.reg_max = 1e10
    plannersettings.passes[0].reg.reg_scale = 10.0
    plannersettings.passes[0].reg.use_dynamics_hess = False
    plannersettings.passes[0].reg.use_constraint_hess = False

    plannersettings.passes[0].linesearch.max_iters = 24
    plannersettings.passes[0].linesearch.beta1 = 1e-10
    plannersettings.passes[0].linesearch.beta2 = 5000.0
    
    return plannersettings


def main():
    plannersettings = create_planner_settings()
    satellite = create_satellite(plannersettings)

    jtime = np.array([0.22, 0.22 + 5400/(36525 * 86400)])
    qgoal = np.array([
        [np.sqrt(2)/2, np.sqrt(2)/2],
        [0.0, 0.0],           
        [0.0, 0.0],            
        [np.sqrt(2)/2, np.sqrt(2)/2]
    ])
    boresight = np.array([
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0]
    ])

    w0 = np.array([0.0, 0.0, 0.0])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    x0 = np.hstack((w0, q0))

    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7.5e3, 0.0])

    print("="*70)
    print("LAYER-BY-LAYER DIAGNOSIS: MTQ-only satellite crash")
    print("="*70)
    
    # Test 1: Python trajOpt (known to work)
    print("\n[1/5] Running Python trajOpt...")
    try:
        X_py, U_py, stop_py, _, _, _, _, elapsed = trajOpt(
            plannersettings, satellite, x0, r0, v0, jtime, qgoal, boresight, debug=True
        )
        print(f"  ✓ SUCCESS: {stop_py}")
        print(f"  Shape: X={X_py.shape}, U={U_py.shape}")
        print(f"  Elapsed: {elapsed:.3f}s")
        py_success = True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        py_success = False
        return
    
    # Test 2: C++ trajOpt (known to crash)
    print("\n[2/5] Running C++ trajOpt...")
    try:
        ok, X_cpp, U_cpp, _K = saltro_py.trajOpt(
            plannersettings, satellite, x0, r0, v0, jtime, qgoal, boresight
        )
        if ok:
            print(f"  ✓ SUCCESS")
            print(f"  Shape: X={X_cpp.shape}, U={U_cpp.shape}")
        else:
            print(f"  ✗ FAILED: trajOpt returned ok=False")
    except Exception as e:
        print(f"  ✗ CRASHED: {e}")
    
    # Test 3: Just warm_start layer
    print("\n[3/5] Testing C++ warm_start layer...")
    try:
        # Resample goals
        from trajOpt import _resample_zero_order_hold
        dt_sec = plannersettings.passes[0].dt
        jtime_flat, q_goal_fine, boresight_fine = _resample_zero_order_hold(
            jtime, qgoal, boresight, dt_sec
        )
        
        # Generate orbit
        ok_orbit, R, V, B, S, rho = saltro_py.generate_orbit(r0, v0, jtime_flat, 0, 0, 0, 0, 0)
        if not ok_orbit:
            print(f"  ✗ generate_orbit failed")
            return
        
        # Ensure Fortran-contiguous
        q_goal_fine = np.asfortranarray(q_goal_fine)
        boresight_fine = np.asfortranarray(boresight_fine)
        R = np.asfortranarray(R)
        V = np.asfortranarray(V)
        B = np.asfortranarray(B)
        S = np.asfortranarray(S)
        rho = rho.reshape(1, -1) if rho.ndim == 1 else rho
        
        ok_ws, X0, U0 = saltro_py.warm_start(
            plannersettings, satellite, x0, jtime_flat, q_goal_fine, boresight_fine, R, V, B, S, rho
        )
        
        if ok_ws:
            print(f"  ✓ warm_start succeeded")
            print(f"  Shape: X0={X0.shape}, U0={U0.shape}")
            print(f"  n_timesteps = {X0.shape[1]}")
            ws_success = True
        else:
            print(f"  ✗ warm_start failed")
            ws_success = False
            return
    except Exception as e:
        print(f"  ✗ warm_start crashed: {e}")
        import traceback
        traceback.print_exc()
        ws_success = False
        return
    
    # Test 4: Try one iteration of alilqr
    print("\n[4/5] Testing C++ alilqr (1 iteration)...")
    alilqr_success = False
    try:
        # Temporarily reduce iterations
        original_max_iters = plannersettings.passes[0].ilqr.max_iters
        plannersettings.passes[0].ilqr.max_iters = 1
        
        ok_alilqr, X_alilqr, U_alilqr, status_str, max_c = saltro_py.alilqr(
            plannersettings, 0, satellite, X0, U0, R, V, B, S, rho,
            jtime_flat, boresight_fine, q_goal_fine
        )
        
        plannersettings.passes[0].ilqr.max_iters = original_max_iters
        
        if ok_alilqr:
            print(f"  ✓ alilqr (1 iter) succeeded: {status_str}, max_c={max_c:.3e}")
            alilqr_success = True
        else:
            print(f"  ✗ alilqr (1 iter) returned ok=False: {status_str}, max_c={max_c:.3e}")
            alilqr_success = False
    except Exception as e:
        print(f"  ✗ alilqr (1 iter) crashed: {e}")
        import traceback
        traceback.print_exc()
        alilqr_success = False
    
    # Test 5: Try backward_pass directly
    print("\n[5/5] Testing C++ backward_pass directly...")
    bp_success = False
    try:
        ok_bp, K, d = saltro_py.backward_pass(
            plannersettings, 0, satellite, X0, U0, R, V, B, S, rho,
            jtime_flat, boresight_fine, q_goal_fine
        )
        
        if ok_bp:
            print(f"  ✓ backward_pass succeeded")
            bp_success = True
        else:
            print(f"  ✗ backward_pass returned ok=False")
            bp_success = False
    except Exception as e:
        print(f"  ✗ backward_pass crashed: {e}")
        import traceback
        traceback.print_exc()
        bp_success = False
    
    print("\n" + "="*70)
    print("DIAGNOSIS COMPLETE")
    print(f"Python trajOpt: {'✓' if py_success else '✗'}")
    print(f"C++ warm_start: {'✓' if ws_success else '✗'}")
    print(f"C++ alilqr:     {'✓' if alilqr_success else '✗'}")
    print(f"C++ backward_pass: {'✓' if bp_success else '✗'}")
    print("="*70)


if __name__ == "__main__":
    main()
