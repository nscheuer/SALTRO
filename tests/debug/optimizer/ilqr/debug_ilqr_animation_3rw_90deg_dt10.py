"""
Interactive iLQR debugger with per-iteration visualization.
"""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
import saltro_py

from ilqr_animation import launch_iteration_viewer
from trajopt import trajOpt

SEC_PER_CENTURY = 36525.0 * 86400.0


def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def _is_eci_format(attitude_target_k):
    return np.isnan(attitude_target_k[0])


def _quat_from_two_vectors(v_from, v_to):
    a = _normalize(v_from)
    b = _normalize(v_to)
    dot = np.clip(np.dot(a, b), -1.0, 1.0)

    if dot > 1.0 - 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])

    if dot < -1.0 + 1e-10:
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-10:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis = _normalize(axis)
        return np.array([0.0, axis[0], axis[1], axis[2]])

    c = np.cross(a, b)
    q = np.array([1.0 + dot, c[0], c[1], c[2]])
    return _normalize(q)


def _goal_quaternion_for_plot(attitude_target_k, boresight_k, q_current):
    if _is_eci_format(attitude_target_k):
        target_vec = attitude_target_k[1:4]
        if np.linalg.norm(target_vec) < 1e-9:
            return q_current.copy()
        return _quat_from_two_vectors(boresight_k, target_vec)
    return _normalize(attitude_target_k)


def _angle_error_deg(q, q_goal):
    qd = np.clip(np.abs(np.dot(_normalize(q), _normalize(q_goal))), -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(qd))


def _sanitize_quaternion_state_inplace(X, quat_start=3, eps=1e-10):
    """Ensure quaternion columns are normalizable before calling C++ passes."""
    if X.shape[0] < quat_start + 4:
        return
    for k in range(X.shape[1]):
        q = X[quat_start:quat_start + 4, k]
        n = np.linalg.norm(q)
        if n < eps or not np.isfinite(n):
            X[quat_start:quat_start + 4, k] = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            X[quat_start:quat_start + 4, k] = q / n


def setup_satellite():
    line_search_max_iters = 24
    line_search_beta1 = 1e-10
    line_search_beta2 = 5000.0
    ang_cost_func_type = 4

    settings = saltro_py.PlannerSettings()
    settings.num_passes = 1
    settings.passes[0].dt = 10.0
    settings.passes[0].ilqr.cost_tol = 1e-5
    settings.passes[0].ilqr.max_iters = 20
    settings.init_traj.initcontroller = 2
    settings.passes[0].linesearch.max_iters = line_search_max_iters
    settings.passes[0].linesearch.beta1 = line_search_beta1
    settings.passes[0].linesearch.beta2 = line_search_beta2
    
    settings.passes[0].reg.reg_init = 1e-6
    settings.passes[0].reg.reg_max = 1e10
    settings.passes[0].reg.reg_scale = 10.0

    settings.constraints.control_limit_scale = 0.0
    settings.constraints.wmax = 1e9
    settings.constraints.sun_limit_angle = 0.0

    cost = settings.passes[0].cost
    cost.angle = 1e5
    cost.ang_vel = 1e3
    cost.ang_vel_mag = 0.0
    cost.ang_vel_err_dir = 0.0
    cost.control_mult = 1.0
    cost.mtq_control_weight = 1.0
    cost.rw_control_weight = 1e7
    cost.magic_control_weight = 0.0
    cost.rw_AM_weight = 0.0
    cost.rw_stic_weight = 0.0
    cost.RWh_max_mult = 0.0
    cost.RWh_stiction_mult = 0.0
    cost.RWh_ok_mult = 0.0
    cost.angle_N = 1e6
    cost.ang_vel_N = 1e6
    cost.ang_vel_mag_N = 0.0
    cost.ang_vel_err_dir_N = 0.0
    cost.ang_cost_func_type = ang_cost_func_type
    cost.use_cost_hess = True

    settings.disturbances.plan_for_aero = False
    settings.disturbances.plan_for_gg = False
    settings.disturbances.plan_for_srp = False
    settings.disturbances.plan_for_prop = False
    settings.disturbances.plan_for_gendist = False
    settings.disturbances.plan_for_resdipole = False

    J = np.diag([0.067, 0.071, 0.069])
    satellite = saltro_py.Satellite(J, settings)

    satellite.addRW(np.array([1.0, 0.0, 0.0]), 0.001, 1e-5, 0.0, 0.02)
    satellite.addRW(np.array([0.0, 1.0, 0.0]), 0.001, 1e-5, 0.0, 0.02)
    satellite.addRW(np.array([0.0, 0.0, 1.0]), 0.001, 1e-5, 0.0, 0.02)

    return satellite, settings


def make_time_grid(N, dt):
    jtime = np.zeros(N)
    dt_centuries = dt / SEC_PER_CENTURY
    for k in range(N):
        jtime[k] = 0.25 + k * dt_centuries
    return jtime


def make_initial_state(satellite):
    x0 = np.zeros(satellite.stateDim)
    x0[0:3] = np.array([0.0, 0.0, 0.0])
    x0[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    return x0


def compute_cost_components(X, U, satellite, qgoal, boresight, B, cost_cfg):
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
            c = satellite.stageCost(k, N, X[:, k], u_k, boresight[:, k], qgoal[:, k], B[:, k], cfg)
            components[comp_name][k] = max(0.0, c)

    return components


def collect_snapshot(X, U, satellite, B, boresight, qgoal, cost_cfg):
    N = X.shape[1]
    U_trim = U[:, :N - 1]
    J = satellite.totalCost(X, U_trim, B, boresight, qgoal, cost_cfg)
    components = compute_cost_components(X, U, satellite, qgoal, boresight, B, cost_cfg)

    q = X[3:7, :]
    q_goal = np.zeros_like(q)
    pointing_err_deg = np.zeros(N)
    for k in range(N):
        q_goal[:, k] = _goal_quaternion_for_plot(qgoal[:, k], boresight[:, k], q[:, k])
        pointing_err_deg[k] = _angle_error_deg(q[:, k], q_goal[:, k])

    return {
        "X": X.copy(),
        "U": U.copy(),
        "J": float(J),
        "components": components,
        "q_goal": q_goal,
        "pointing_err_deg": pointing_err_deg,
    }


def run_ilqr_python_loop(satellite, settings, X0, U0, jtime, R, V, B, S, rho, boresight, qgoal):
    from trajopt import ilqr
    
    cost_cfg = settings.passes[0].cost
    reg_cfg = settings.passes[0].reg
    ilqr_cfg = settings.passes[0].ilqr
    
    snapshots = [collect_snapshot(X0, U0, satellite, B, boresight, qgoal, cost_cfg)]
    transitions = []
    
    reg = reg_cfg.reg_init
    X = X0.copy()
    U = U0.copy()
    
    for iteration in range(ilqr_cfg.max_iters):
        U_trim = U[:, :X.shape[1] - 1]
        
        bp_succeeded = False
        fp_succeeded = False
        
        while reg <= reg_cfg.reg_max:
            ok_bp, K_arr, d_arr, deltaV = saltro_py.backward_pass(
                satellite, X, U_trim, R, V, B, S, rho, boresight, qgoal, settings, reg
            )
            
            if not ok_bp:
                reg *= reg_cfg.reg_scale
                continue
            
            bp_succeeded = True
            K_list = [K_arr[k] for k in range(K_arr.shape[0])]
            d_list = [d_arr[:, k] for k in range(d_arr.shape[1])]
            
            J_prev = snapshots[-1]["J"]
            
            ok_fp, X_new, U_new, J_new = saltro_py.forward_pass(
                satellite, X, U, K_list, d_list, deltaV, B, R, V, S, rho,
                boresight, qgoal, settings, jtime, J_prev
            )
            
            if not ok_fp:
                reg *= reg_cfg.reg_scale
                continue
            
            fp_succeeded = True
            
            pred_delta = max(0.0, -(deltaV[0] + deltaV[1]))
            act_delta = J_prev - J_new
            tol_ok = abs(act_delta) <= ilqr_cfg.cost_tol
            
            transition = {
                "bp_ok": True,
                "fp_ok": True,
                "pred_delta": float(pred_delta),
                "act_delta": float(act_delta),
                "cost_decrease_ok": bool(J_new <= J_prev + 1e-12),
                "delta_tol_ok": bool(tol_ok),
            }
            transitions.append(transition)
            
            X = X_new
            U = U_new
            snapshots.append(collect_snapshot(X, U, satellite, B, boresight, qgoal, cost_cfg))
            
            reg = max(reg_cfg.reg_init, reg / reg_cfg.reg_scale)
            
            if tol_ok:
                return snapshots, transitions, "converged"
            
            break
        
        if not (bp_succeeded and fp_succeeded):
            return snapshots, transitions, "reg_exceeded"
    
    return snapshots, transitions, "max_iters"


def main():
    dt = 10.0
    
    satellite, settings = setup_satellite()
    x0 = make_initial_state(satellite)
    
    boresight_sparse = np.zeros((3, 2))
    boresight_sparse[0, :] = 1.0
    
    qgoal_sparse = np.zeros((4, 2))
    qgoal_sparse[0, 0] = 1.0
    qgoal_sparse[0, 1] = np.sqrt(2) / 2
    qgoal_sparse[3, 1] = np.sqrt(2) / 2
    
    time_sparse = np.array([[0.25, 0.25 + 200.0 * dt / SEC_PER_CENTURY]])
    
    jtime_full = make_time_grid(21, dt)
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7500.0, 0.0])
    ok, R, V, B, S, rho = saltro_py.generate_orbit(r0, v0, jtime_full, 0, 0, 0, 0, 0)
    if not ok:
        raise RuntimeError("generate_orbit failed")
    rho = rho.reshape(1, -1)
    
    t0 = time.time()
    X_opt, U_opt, reason, jtime, boresight, qgoal = trajOpt(
        satellite, settings, x0, boresight_sparse, qgoal_sparse, time_sparse, R, V, B, S, rho
    )
    opt_time = time.time() - t0
    
    snapshots, transitions, stop_reason = run_ilqr_python_loop(
        satellite, settings, X_opt, U_opt, jtime, R, V, B, S, rho, boresight, qgoal
    )
    
    print(f"Optimization time: {opt_time*1000:.2f} ms")
    print(f"Stop reason: {stop_reason}")
    print(f"Final angular rates: {X_opt[0:3, -1]}")
    print(f"Snapshots: {len(snapshots)}")
    
    launch_iteration_viewer(snapshots, transitions, stop_reason, dt, settings.passes[0].ilqr.cost_tol)


if __name__ == "__main__":
    main()
