"""Track regularization, BP/FP success, and step quality at each iteration."""
import sys, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from sat_3_1_hybrid import create_satellite

qg = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])

def make_ps(angle_w):
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = 1; ps.num_passes = 1
    ps.passes[0].dt = 10.0; ps.passes[0].ilqr.cost_tol = 1e-6
    ps.passes[0].ilqr.max_iters = 20; ps.passes[0].ilqr.grad_tol = 0.0
    ps.passes[0].auglag.max_outer_iters = 3
    ps.passes[0].auglag.constraint_tol = 1e-3
    c = ps.passes[0].cost
    c.angle = angle_w; c.ang_vel = angle_w / 100
    c.control_mult = 1.0; c.mtq_control_weight = 1e-1; c.rw_control_weight = 1.0
    c.angle_N = angle_w; c.ang_vel_N = angle_w / 100
    c.ang_cost_func_type = 3; c.use_cost_hess = True
    for a in ["aero","gg","srp","prop","gendist","resdipole"]:
        setattr(ps.disturbances, "plan_for_"+a, False)
    ps.passes[0].reg.reg_init = 1e-6; ps.passes[0].reg.reg_max = 1e30
    ps.passes[0].reg.reg_scale = 10.0
    ps.passes[0].linesearch.max_iters = 24
    ps.passes[0].linesearch.beta1 = 1e-10; ps.passes[0].linesearch.beta2 = 5000.0
    return ps

# Manually run iLQR to track reg and BP/FP outcomes
for angle_w in [1e4, 1e6]:
    ps = make_ps(angle_w)
    sat = create_satellite(ps)

    x0 = np.array([0.01, 0.01, 0.01, 1, 0, 0, 0, 0.0])
    r0 = np.array([7e6, 0, 0]); v0 = np.array([0, 7.5e3, 0])
    jtime_ep = np.array([0.22, 0.22 + 1000/(36525*86400)])
    qgoal = np.tile(qg[:, None], (1, 2))
    bs = np.array([[1,1],[0,0],[0,0]], dtype=float)

    dt_c = ps.passes[0].dt / (36525*86400)
    N = int((jtime_ep[1]-jtime_ep[0])/dt_c) + 1
    jtime_fine = np.array([jtime_ep[0] + i*dt_c for i in range(N)])
    ok, R, V, B, S, rho = saltro_py.generate_orbit(r0, v0, jtime_fine, 0,0,0,0,0)
    qg_n = np.tile(qgoal[:,0:1], (1,N))
    bs_n = np.tile(bs[:,0:1], (1,N))

    ok_ws, X, U = saltro_py.warm_start(ps, sat, x0, jtime_fine, qg_n, bs_n, R, V, B, S, rho)

    print(f"\n{'='*80}")
    print(f"angle={angle_w:.0e}  ExcCtrl")
    print(f"{'='*80}")
    print(f"{'iter':>4s} {'reg':>10s} {'BP':>4s} {'FP':>4s} {'alpha':>8s} {'J':>14s} {'dJ':>12s} {'max|d|':>10s} {'min_q0':>8s}")
    print("-" * 90)

    passsettings = ps.passes[0]
    # No AL penalties for simplicity
    lambda_aug = None
    mu_aug = None

    reg = passsettings.reg.reg_init
    J_prev = None

    for iteration in range(20):
        U_trim = U[:, :X.shape[1]-1]

        # Backward pass
        ok_bp, K, d, deltaV = saltro_py.backward_pass(
            sat, X, U_trim, R, V, B, S, rho, bs_n, qg_n,
            ps, lambda_aug, mu_aug, reg
        )

        if not ok_bp:
            reg *= passsettings.reg.reg_scale
            print(f"{iteration:4d} {reg:10.2e}  FAIL   -                                        BP failed")
            if reg > passsettings.reg.reg_max:
                print("  REG EXCEEDED")
                break
            continue

        # Compute max feedforward norm
        max_d = max(np.linalg.norm(d[:, k]) for k in range(d.shape[1]))

        # Compute J
        J_nom = sat.totalCost(X, U_trim, B, bs_n, qg_n, passsettings.cost)

        # Forward pass
        ok_fp, X_new, U_new, J_new = saltro_py.forward_pass(
            sat, X, U, [K[k] for k in range(K.shape[0])],
            [d[:, k] for k in range(d.shape[1])],
            deltaV, B, R, V, S, rho, bs_n, qg_n,
            ps, lambda_aug, mu_aug, jtime_fine, J_nom
        )

        if not ok_fp:
            reg *= passsettings.reg.reg_scale
            dJ_str = ""
            min_q0 = min(X[3, k] for k in range(X.shape[1]))
            print(f"{iteration:4d} {reg:10.2e}    ok  FAIL                   {J_nom:14.4e}              {max_d:10.2f} {min_q0:+8.4f}  FP failed")
            if reg > passsettings.reg.reg_max:
                print("  REG EXCEEDED")
                break
            continue

        X = X_new
        U = U_new
        dJ = J_nom - J_new if J_prev is not None else 0
        J_prev = J_new
        min_q0 = min(X[3, k] for k in range(X.shape[1]))
        print(f"{iteration:4d} {reg:10.2e}    ok    ok          {J_new:14.4e} {dJ:+12.4e} {max_d:10.2f} {min_q0:+8.4f}")
