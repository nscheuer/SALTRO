"""Verify analytic cost Hessian vs finite-difference Hessian."""
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from sat_3_1_hybrid import create_satellite


def main():
    ps = saltro_py.PlannerSettings()
    ps.num_passes = 1
    ps.passes[0].cost.angle = 1e4
    ps.passes[0].cost.ang_vel = 1e2
    ps.passes[0].cost.angle_N = 1e4
    ps.passes[0].cost.ang_vel_N = 1e2
    ps.passes[0].cost.ang_cost_func_type = 3
    ps.passes[0].cost.use_cost_hess = True
    ps.passes[0].cost.ang_vel_err_dir = 0.0
    ps.passes[0].cost.ang_vel_mag = 0.0
    ps.passes[0].cost.control_mult = 1.0
    ps.passes[0].cost.mtq_control_weight = 1e-1
    ps.passes[0].cost.rw_control_weight = 1.0

    sat = create_satellite(ps)
    cost_cfg = ps.passes[0].cost

    # Test state: 45 deg off from goal
    q_goal = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
    q_test = np.array([0.9239, 0.0, 0.3827, 0.0])  # ~45 deg rotation about y
    q_test /= np.linalg.norm(q_test)
    omega = np.array([0.01, -0.005, 0.02])
    h_rw = np.array([0.0])
    x = np.hstack([omega, q_test, h_rw])
    u = np.zeros(sat.controlDim)
    bs = np.array([1.0, 0.0, 0.0])
    B = np.array([3e-5, 1e-5, -2e-5])

    N = 10
    k = 5

    # Get analytic Hessian
    lxx_a, luu_a, lux_a = sat.stageCostHessians(k, N, x, u, bs, q_goal, B, cost_cfg)
    lxx_a = np.asarray(lxx_a)

    # Finite-difference Hessian of the cost (FD of cost function itself, not gradient)
    eps = 1e-5
    nx = len(x)

    # Method 1: FD of gradient (what the old code was comparing against)
    lxx_fd_grad = np.zeros((nx, nx))
    lx_base = np.asarray(sat.stageCostJacobians(k, N, x, u, bs, q_goal, B, cost_cfg)[0])
    for j in range(nx):
        x_p = x.copy()
        x_p[j] += eps
        if 3 <= j < 7:
            x_p[3:7] /= np.linalg.norm(x_p[3:7])
        lx_p = np.asarray(sat.stageCostJacobians(k, N, x_p, u, bs, q_goal, B, cost_cfg)[0])
        lxx_fd_grad[:, j] = (lx_p - lx_base) / eps

    # Method 2: Direct FD of the Hessians function output (what we're actually comparing)
    # The analytic Hessian is returned by stageCostHessians, already projected.
    # Compare directly against FD of the cost value.
    lxx_fd_cost = np.zeros((nx, nx))
    cost_base = float(sat.stageCost(k, N, x, u, bs, q_goal, B, cost_cfg))
    lx_fd = np.zeros(nx)
    for i in range(nx):
        x_p = x.copy()
        x_p[i] += eps
        if 3 <= i < 7:
            x_p[3:7] /= np.linalg.norm(x_p[3:7])
        lx_fd[i] = (float(sat.stageCost(k, N, x_p, u, bs, q_goal, B, cost_cfg)) - cost_base) / eps

    for i in range(nx):
        for j in range(nx):
            x_pp = x.copy()
            x_pp[i] += eps
            x_pp[j] += eps
            if 3 <= i < 7 or 3 <= j < 7:
                x_pp[3:7] /= np.linalg.norm(x_pp[3:7])
            x_pi = x.copy()
            x_pi[i] += eps
            if 3 <= i < 7:
                x_pi[3:7] /= np.linalg.norm(x_pi[3:7])
            x_pj = x.copy()
            x_pj[j] += eps
            if 3 <= j < 7:
                x_pj[3:7] /= np.linalg.norm(x_pj[3:7])
            f_pp = float(sat.stageCost(k, N, x_pp, u, bs, q_goal, B, cost_cfg))
            f_pi = float(sat.stageCost(k, N, x_pi, u, bs, q_goal, B, cost_cfg))
            f_pj = float(sat.stageCost(k, N, x_pj, u, bs, q_goal, B, cost_cfg))
            lxx_fd_cost[i, j] = (f_pp - f_pi - f_pj + cost_base) / (eps * eps)
    lxx_fd = lxx_fd_grad  # use gradient FD for now

    print("=== Quaternion block of Hessian (rows 3:7, cols 3:7) ===")
    print("Analytic:")
    print(lxx_a[3:7, 3:7])
    print("\nFinite difference:")
    print(lxx_fd[3:7, 3:7])
    print("\nDifference:")
    diff = lxx_a[3:7, 3:7] - lxx_fd[3:7, 3:7]
    print(diff)
    print(f"\nMax abs diff (quat block): {np.max(np.abs(diff)):.6e}")
    print(f"Max abs analytic:         {np.max(np.abs(lxx_a[3:7, 3:7])):.6e}")
    print(f"Relative error:           {np.max(np.abs(diff)) / (np.max(np.abs(lxx_fd[3:7, 3:7])) + 1e-20):.6e}")

    print("\n=== Angular velocity block (rows 0:3, cols 0:3) ===")
    print("Analytic:")
    print(lxx_a[0:3, 0:3])
    print("FD:")
    print(lxx_fd[0:3, 0:3])

    print("\n=== Full Hessian comparison ===")
    full_diff = lxx_a - lxx_fd
    print(f"Max abs diff (full):  {np.max(np.abs(full_diff)):.6e}")
    print(f"Max abs FD:           {np.max(np.abs(lxx_fd)):.6e}")

    # Also test with different cost func types
    for cft in [0, 1, 2, 3, 4]:
        cost_cfg2 = ps.passes[0].cost
        cost_cfg2.ang_cost_func_type = cft
        lxx_a2 = np.asarray(sat.stageCostHessians(k, N, x, u, bs, q_goal, B, cost_cfg2)[0])
        lxx_fd2 = np.zeros((nx, nx))
        lx_base2 = np.asarray(sat.stageCostJacobians(k, N, x, u, bs, q_goal, B, cost_cfg2)[0])
        for j in range(nx):
            x_p = x.copy()
            x_p[j] += eps
            if 3 <= j < 7:
                x_p[3:7] /= np.linalg.norm(x_p[3:7])
            lx_p2 = np.asarray(sat.stageCostJacobians(k, N, x_p, u, bs, q_goal, B, cost_cfg2)[0])
            lxx_fd2[:, j] = (lx_p2 - lx_base2) / eps
        diff2 = lxx_a2[3:7, 3:7] - lxx_fd2[3:7, 3:7]
        max_err = np.max(np.abs(diff2))
        max_ref = np.max(np.abs(lxx_fd2[3:7, 3:7])) + 1e-20
        print(f"  type={cft}: max_abs_diff={max_err:.4e}  rel={max_err/max_ref:.4e}")


if __name__ == "__main__":
    main()
