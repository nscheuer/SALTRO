"""
Deeper diagnostic: trace P_k evolution step-by-step to understand
how it goes indefinite. Compare regularized vs unregularized P_k update.
"""
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "build"))
ILQR_DIR = str(Path(__file__).resolve().parents[0].parent / "optimizer" / "ilqr")
sys.path.insert(0, ILQR_DIR)

import saltro_py
from create_3rw_sat import create_3rw_satellite


def create_planner_settings():
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = 2
    ps.num_passes = 1
    ps.passes[0].dt = 10.0
    ps.passes[0].ilqr.cost_tol = 1e-5
    ps.passes[0].ilqr.max_iters = 20
    cost = ps.passes[0].cost
    cost.angle = 1.0
    cost.ang_vel = 1e2
    cost.ang_vel_mag = 0.0
    cost.ang_vel_err_dir = 0.0
    cost.control_mult = 1.0
    cost.mtq_control_weight = 1.0
    cost.rw_control_weight = 1e1
    cost.magic_control_weight = 0.0
    cost.rw_AM_weight = 0.0
    cost.rw_stic_weight = 0.0
    cost.RWh_max_mult = 0.0
    cost.RWh_stiction_mult = 0.0
    cost.RWh_ok_mult = 0.0
    cost.angle_N = 0.0
    cost.ang_vel_N = 1.0
    cost.ang_vel_mag_N = 0.0
    cost.ang_vel_err_dir_N = 0.0
    cost.ang_cost_func_type = 4
    cost.use_cost_hess = True
    ps.disturbances.plan_for_aero = False
    ps.disturbances.plan_for_gg = False
    ps.disturbances.plan_for_srp = False
    ps.disturbances.plan_for_prop = False
    ps.disturbances.plan_for_gendist = False
    ps.disturbances.plan_for_resdipole = False
    ps.passes[0].reg.reg_init = 1e-6
    ps.passes[0].reg.reg_max = 1e10
    ps.passes[0].reg.reg_scale = 10.0
    ps.passes[0].reg.use_dynamics_hess = True
    ps.passes[0].reg.use_constraint_hess = False
    ps.passes[0].linesearch.max_iters = 24
    ps.passes[0].linesearch.beta1 = 1e-10
    ps.passes[0].linesearch.beta2 = 5000.0
    return ps


def findWMat(q):
    W = np.zeros((4, 3))
    W[0, :] = [-q[1], -q[2], -q[3]]
    W[1, :] = [ q[0], -q[3],  q[2]]
    W[2, :] = [ q[3],  q[0], -q[1]]
    W[3, :] = [-q[2],  q[1],  q[0]]
    return W


def findGMat(q, nRW):
    nx = 7 + nRW
    nxr = 6 + nRW
    G = np.zeros((nxr, nx))
    G[:3, :3] = np.eye(3)
    W = findWMat(q)
    G[3:6, 3:7] = W.T
    for i in range(nRW):
        G[6+i, 7+i] = 1.0
    return G


def rk4_jacobians_python(satellite, x, u, dt, dist_cfg, R_k, B_k, S_k, V_k):
    nx = len(x)
    nu = len(u)
    I = np.eye(nx)

    def eval_jac(x_loc):
        A_c, B_c, _ = satellite.dynamicsJacobians(x_loc, u, dist_cfg, R_k, B_k, S_k, V_k)
        f = np.array(satellite.dynamics(x_loc, u, dist_cfg, R_k, B_k, S_k, V_k, 0))
        return np.array(A_c), np.array(B_c), f

    A_c1, B_c1, k1 = eval_jac(x)
    dk1_dx, dk1_du = A_c1, B_c1
    x2 = x + 0.5*dt*k1
    A_c2, B_c2, k2 = eval_jac(x2)
    dk2_dx = A_c2 @ (I + 0.5*dt*dk1_dx)
    dk2_du = A_c2 @ (0.5*dt*dk1_du) + B_c2
    x3 = x + 0.5*dt*k2
    A_c3, B_c3, k3 = eval_jac(x3)
    dk3_dx = A_c3 @ (I + 0.5*dt*dk2_dx)
    dk3_du = A_c3 @ (0.5*dt*dk2_du) + B_c3
    x4 = x + dt*k3
    A_c4, B_c4, k4 = eval_jac(x4)
    dk4_dx = A_c4 @ (I + dt*dk3_dx)
    dk4_du = A_c4 @ (dt*dk3_du) + B_c4
    A_d = I + (dt/6.0)*(dk1_dx + 2*dk2_dx + 2*dk3_dx + dk4_dx)
    B_d = (dt/6.0)*(dk1_du + 2*dk2_du + 2*dk3_du + dk4_du)
    return A_d, B_d


def run_backward_pass(total_time_sec, reg, use_unreg_for_pk):
    """
    Run backward pass with either regularized or unregularized Q_uu for P_k.
    Returns the step at which it fails (or -1 if it succeeds).
    """
    ps = create_planner_settings()
    satellite = create_3rw_satellite(ps)
    dt = ps.passes[0].dt
    cost_cfg = ps.passes[0].cost
    nRW = satellite.numRW
    nx = satellite.stateDim
    nu = satellite.controlDim
    nxr = satellite.reducedStateDim

    jtime = np.array([0.22, 0.22 + total_time_sec / (36525 * 86400)])
    qgoal = np.array([[np.sqrt(2)/2, np.sqrt(2)/2], [0.0, 0.0], [0.0, 0.0], [np.sqrt(2)/2, np.sqrt(2)/2]])
    boresight = np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    w0 = np.array([-0.01, 0.02, 0.03])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    h0 = np.array([0.0, 0.0, 0.0])
    x0 = np.hstack((w0, q0, h0))
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7.5e3, 0.0])

    dt_cent = dt / (36525.0 * 86400.0)
    jtime_flat = np.arange(jtime[0], jtime[1] + dt_cent/2, dt_cent)
    if abs(jtime_flat[-1] - jtime[1]) > 1e-12:
        jtime_flat = np.append(jtime_flat, jtime[1])
    N = len(jtime_flat)

    idx = np.searchsorted(jtime[1:], jtime_flat, side='right')
    q_goal_traj = qgoal[:, idx]
    boresight_traj = boresight[:, idx]

    ok, R, V, B, S, rho = saltro_py.generate_orbit(r0, v0, jtime_flat, 0, 0, 0, 0, 0)
    rho = rho.reshape(1, -1) if rho.ndim == 1 else rho
    ok, X, U = saltro_py.warm_start(ps, satellite, x0, jtime_flat,
                                     np.asfortranarray(q_goal_traj),
                                     np.asfortranarray(boresight_traj),
                                     np.asfortranarray(R), np.asfortranarray(V),
                                     np.asfortranarray(B), np.asfortranarray(S), rho)

    dist_cfg = saltro_py.DisturbanceConfig()

    # Terminal
    x_final = X[:, -1]
    p_N_full, _, _ = satellite.terminalCostJacobians(x_final, boresight_traj[:, -1], q_goal_traj[:, -1], B[:, -1], cost_cfg)
    P_N_full, _, _ = satellite.terminalCostHessians(x_final, boresight_traj[:, -1], q_goal_traj[:, -1], B[:, -1], cost_cfg)
    p_N_full = np.array(p_N_full).flatten()
    P_N_full = np.array(P_N_full)

    G_N = findGMat(x_final[3:7], nRW)
    p_k = G_N @ p_N_full
    P_k = G_N @ P_N_full @ G_N.T

    fail_k = -1
    for k in range(N-2, -1, -1):
        x_k = X[:, k]
        u_k = U[:, k] if k < U.shape[1] else np.zeros(nu)
        B_k = B[:, k]

        lx_full, lu_mat, _ = satellite.stageCostJacobians(k, N, x_k, u_k, boresight_traj[:, k], q_goal_traj[:, k], B_k, cost_cfg)
        lxx_full, luu, lux_full = satellite.stageCostHessians(k, N, x_k, u_k, boresight_traj[:, k], q_goal_traj[:, k], B_k, cost_cfg)
        lx_full = np.array(lx_full).flatten() * dt
        lu = np.array(lu_mat).flatten() * dt
        lxx_full = np.array(lxx_full) * dt
        luu = np.array(luu) * dt
        lux_full = np.array(lux_full) * dt

        q_k = x_k[3:7]
        q_kp1 = X[3:7, k+1]
        G_k = findGMat(q_k, nRW)
        G_kp1 = findGMat(q_kp1, nRW)

        lx = G_k @ lx_full
        lxx = G_k @ lxx_full @ G_k.T
        lux = lux_full @ G_k.T

        A_full, B_full = rk4_jacobians_python(satellite, x_k, u_k, dt, dist_cfg, R[:, k], B_k, S[:, k], V[:, k])
        A_r = G_kp1 @ A_full @ G_k.T
        B_r = G_kp1 @ B_full

        Q_xx = lxx + A_r.T @ P_k @ A_r
        Q_uu = luu + B_r.T @ P_k @ B_r
        Q_ux = lux + B_r.T @ P_k @ A_r
        Q_x = lx + A_r.T @ p_k
        Q_u = lu + B_r.T @ p_k

        Q_uu_reg = Q_uu + reg * np.eye(nu)
        try:
            np.linalg.cholesky(Q_uu_reg)
        except np.linalg.LinAlgError:
            fail_k = k
            break

        K_k = -np.linalg.solve(Q_uu_reg, Q_ux)
        d_k = -np.linalg.solve(Q_uu_reg, Q_u)

        if use_unreg_for_pk:
            # Use unregularized Q_uu for P_k (current code)
            P_k = Q_xx + K_k.T @ Q_uu @ K_k + K_k.T @ Q_ux + Q_ux.T @ K_k
            p_k = Q_x + K_k.T @ Q_uu @ d_k + K_k.T @ Q_u + Q_ux.T @ d_k
        else:
            # Use regularized Q_uu for P_k (guaranteed PD)
            P_k = Q_xx + K_k.T @ Q_uu_reg @ K_k + K_k.T @ Q_ux + Q_ux.T @ K_k
            p_k = Q_x + K_k.T @ Q_uu_reg @ d_k + K_k.T @ Q_u + Q_ux.T @ d_k

    return fail_k, N


def main():
    total_times = [200, 210, 220, 250, 300]
    regs = [1e-6, 1e-4, 1e-2, 1.0, 1e2]
    
    print("=" * 100)
    print("COMPARISON: Unregularized vs Regularized P_k update")
    print("=" * 100)
    
    for total_time in total_times:
        print(f"\n--- total_time = {total_time}s ---")
        print(f"{'reg':>10} | {'unreg P_k':>20} | {'reg P_k':>20}")
        print("-" * 55)
        for reg in regs:
            fail_unreg, N = run_backward_pass(total_time, reg, use_unreg_for_pk=True)
            fail_reg, _ = run_backward_pass(total_time, reg, use_unreg_for_pk=False)
            
            unreg_str = f"FAIL at k={fail_unreg}" if fail_unreg >= 0 else f"OK (all {N-1} steps)"
            reg_str = f"FAIL at k={fail_reg}" if fail_reg >= 0 else f"OK (all {N-1} steps)"
            print(f"{reg:10.1e} | {unreg_str:>20} | {reg_str:>20}")


if __name__ == "__main__":
    main()
