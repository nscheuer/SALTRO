"""
Test the most robust fix: lxx PSD clamp + P_k PSD enforcement after each Riccati step.
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


def clamp_psd(M):
    eigvals, eigvecs = np.linalg.eigh(M)
    if eigvals[0] >= 0:
        return M
    eigvals_clamped = np.maximum(eigvals, 0.0)
    return eigvecs @ np.diag(eigvals_clamped) @ eigvecs.T


def rk4_jacobians_python(satellite, x, u, dt, dist_cfg, R_k, B_k, S_k, V_k):
    nx = len(x)
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


def run_backward_pass(total_time_sec, reg, clamp_lxx, clamp_Pk):
    ps = create_planner_settings()
    satellite = create_3rw_satellite(ps)
    dt = ps.passes[0].dt
    cost_cfg = ps.passes[0].cost
    nRW = satellite.numRW
    nx = satellite.stateDim
    nu = satellite.controlDim

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

    ok, R, V_orbit, B_field, S_sun, rho = saltro_py.generate_orbit(r0, v0, jtime_flat, 0, 0, 0, 0, 0)
    rho = rho.reshape(1, -1) if rho.ndim == 1 else rho
    ok, X, U = saltro_py.warm_start(ps, satellite, x0, jtime_flat,
                                     np.asfortranarray(q_goal_traj),
                                     np.asfortranarray(boresight_traj),
                                     np.asfortranarray(R), np.asfortranarray(V_orbit),
                                     np.asfortranarray(B_field), np.asfortranarray(S_sun), rho)
    dist_cfg = saltro_py.DisturbanceConfig()

    x_final = X[:, -1]
    p_N_full, _, _ = satellite.terminalCostJacobians(x_final, boresight_traj[:, -1], q_goal_traj[:, -1], B_field[:, -1], cost_cfg)
    P_N_full, _, _ = satellite.terminalCostHessians(x_final, boresight_traj[:, -1], q_goal_traj[:, -1], B_field[:, -1], cost_cfg)
    p_N_full = np.array(p_N_full).flatten()
    P_N_full = np.array(P_N_full)

    G_N = findGMat(x_final[3:7], nRW)
    p_k = G_N @ p_N_full
    P_k = G_N @ P_N_full @ G_N.T

    fail_k = -1
    max_Pk_norm = 0
    min_Pk_eig = np.inf
    min_Quu_eig = np.inf

    for k in range(N-2, -1, -1):
        x_k = X[:, k]
        u_k = U[:, k] if k < U.shape[1] else np.zeros(nu)
        B_k = B_field[:, k]

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

        if clamp_lxx:
            lxx = clamp_psd(lxx)

        A_full, B_full = rk4_jacobians_python(satellite, x_k, u_k, dt, dist_cfg, R[:, k], B_k, S_sun[:, k], V_orbit[:, k])
        A_r = G_kp1 @ A_full @ G_k.T
        B_r = G_kp1 @ B_full

        Q_xx = lxx + A_r.T @ P_k @ A_r
        Q_uu = luu + B_r.T @ P_k @ B_r
        Q_ux = lux + B_r.T @ P_k @ A_r
        Q_x = lx + A_r.T @ p_k
        Q_u = lu + B_r.T @ p_k

        Q_uu_reg = Q_uu + reg * np.eye(nu)
        P_eigs = np.linalg.eigvalsh(P_k)
        Q_uu_eigs = np.linalg.eigvalsh(Q_uu)
        min_Pk_eig = min(min_Pk_eig, P_eigs[0])
        min_Quu_eig = min(min_Quu_eig, Q_uu_eigs[0])

        try:
            np.linalg.cholesky(Q_uu_reg)
        except np.linalg.LinAlgError:
            fail_k = k
            break

        K_k = -np.linalg.solve(Q_uu_reg, Q_ux)
        d_k = -np.linalg.solve(Q_uu_reg, Q_u)

        # Use unregularized Q_uu for P_k (standard iLQR)
        P_k = Q_xx + K_k.T @ Q_uu @ K_k + K_k.T @ Q_ux + Q_ux.T @ K_k
        p_k = Q_x + K_k.T @ Q_uu @ d_k + K_k.T @ Q_u + Q_ux.T @ d_k
        
        # Symmetrize P_k to prevent asymmetry buildup
        P_k = 0.5 * (P_k + P_k.T)
        
        # Clamp P_k to PSD if requested (safety net)
        if clamp_Pk:
            P_k = clamp_psd(P_k)

        max_Pk_norm = max(max_Pk_norm, np.linalg.norm(P_k))

    return fail_k, N, max_Pk_norm, min_Pk_eig, min_Quu_eig


def main():
    reg = 1e-6
    total_times = [200, 220, 300, 500, 1000, 2000, 5000, 10000]
    
    configs = [
        ("No fix",                False, False),
        ("Clamp lxx",             True,  False),
        ("Clamp lxx + Clamp Pk",  True,  True),
    ]
    
    for label, cl, cp in configs:
        print(f"\n{'='*80}")
        print(f"{label} (reg={reg:.0e})")
        print(f"{'='*80}")
        print(f"{'time':>7} | {'result':>20} | {'min P_k eig':>14} | {'min Q_uu eig':>14} | {'max ||P||':>12}")
        print("-" * 75)
        for t in total_times:
            fail, N, maxP, minP, minQ = run_backward_pass(t, reg, cl, cp)
            res = f"FAIL k={fail}" if fail >= 0 else f"OK ({N-1} steps)"
            print(f"{t:7d} | {res:>20} | {minP:14.4e} | {minQ:14.4e} | {maxP:12.4e}")


if __name__ == "__main__":
    main()
