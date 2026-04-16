"""Isolate which change caused the ExcCtrl regression.

Run through the Python iLQR (which calls C++ BP/FP) with NO AL penalties
(lambda=0, mu=0) so the constraint change doesn't matter.
This isolates the effect of:
  - Analytic Hessian (always on now)
  - Normalization Jacobians in RK4 (in the C++ backward pass)
  - (rho,drho) regularization (in the C++ iLQR — but we're calling
    BP/FP manually here, so reg is controlled by us)

By controlling reg ourselves, we can test:
  A) Fixed reg=1e-6 (original behavior)
  B) Fixed reg=1e-2
  C) Fixed reg=1.0
and see how the trajectory evolves.
"""
import sys, os, numpy as np, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from sat_3_1_hybrid import create_satellite

qg = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
x0 = np.array([0.01, 0.01, 0.01, 1, 0, 0, 0, 0.0])
r0 = np.array([7e6, 0, 0]); v0 = np.array([0, 7.5e3, 0])
jtime_ep = np.array([0.22, 0.22 + 1000/(36525*86400)])

ANGLE_W = 1e4

ps = saltro_py.PlannerSettings()
ps.init_traj.initcontroller = 1; ps.num_passes = 1
ps.passes[0].dt = 10.0; ps.passes[0].ilqr.cost_tol = 1e-6
ps.passes[0].ilqr.max_iters = 50
ps.passes[0].auglag.max_outer_iters = 1
ps.passes[0].auglag.constraint_tol = 1e-3
c = ps.passes[0].cost
c.angle = ANGLE_W; c.ang_vel = ANGLE_W / 100
c.control_mult = 1.0; c.mtq_control_weight = 1e-1; c.rw_control_weight = 1.0
c.angle_N = ANGLE_W; c.ang_vel_N = ANGLE_W / 100
c.ang_cost_func_type = 3; c.use_cost_hess = True
for a in ["aero","gg","srp","prop","gendist","resdipole"]:
    setattr(ps.disturbances, "plan_for_"+a, False)
ps.passes[0].reg.reg_init = 1e-6; ps.passes[0].reg.reg_max = 1e30
ps.passes[0].reg.reg_scale = 1.6; ps.passes[0].reg.reg_bump = 10.0
ps.passes[0].linesearch.max_iters = 24
ps.passes[0].linesearch.beta1 = 1e-10; ps.passes[0].linesearch.beta2 = 5000.0

sat = create_satellite(ps)

# Generate orbit and warm-start
dt_c = ps.passes[0].dt / (36525*86400)
N = int((jtime_ep[1]-jtime_ep[0])/dt_c) + 1
jtime_fine = np.array([jtime_ep[0] + i*dt_c for i in range(N)])
ok, R, V, B, S, rho = saltro_py.generate_orbit(r0, v0, jtime_fine, 0,0,0,0,0)
qg_n = np.tile(np.array([[np.sqrt(2)/2],[0],[0],[np.sqrt(2)/2]]), (1,N))
bs_n = np.tile(np.array([[1],[0],[0]], dtype=float), (1,N))

ok_ws, X, U = saltro_py.warm_start(ps, sat, x0, jtime_fine, qg_n, bs_n, R, V, B, S, rho)

def pe_profile(X):
    return np.array([2*np.degrees(np.arccos(min(abs(float(np.dot(X[3:7,k], qg))), 1)))
                      for k in range(X.shape[1])])

def run_manual_ilqr(X_in, U_in, reg_fixed, max_iters=30):
    """Run iLQR manually with fixed regularization, no AL penalties."""
    X = X_in.copy()
    U = U_in.copy()

    for iteration in range(max_iters):
        U_trim = U[:, :X.shape[1]-1]

        # Backward pass with fixed reg
        ok_bp, K, d, deltaV = saltro_py.backward_pass(
            sat, X, U_trim, R, V, B, S, rho, bs_n, qg_n,
            ps, [], [], reg_fixed  # empty lambda/mu = no AL
        )

        if not ok_bp:
            return X, U, iteration, "bp_fail"

        # Cost before
        J_prev = sat.totalCost(X, U_trim, B, bs_n, qg_n, ps.passes[0].cost)

        # Forward pass
        ok_fp, X_new, U_new, J_new = saltro_py.forward_pass(
            sat, X, U,
            [K[k] for k in range(K.shape[0])],
            [d[:, k] for k in range(d.shape[1])],
            deltaV, B, R, V, S, rho, bs_n, qg_n,
            ps, [], [], jtime_fine, J_prev
        )

        if not ok_fp:
            return X, U, iteration, "fp_fail"

        X = X_new
        U = U_new

        dJ = abs(J_prev - J_new)
        if dJ <= 1e-6:
            return X, U, iteration, "converged"

    return X, U, max_iters, "max_iters"

print(f"Warm-start: N={X.shape[1]}  final_pe={pe_profile(X)[-1]:.1f}°")
print()
print(f"{'reg':>10s}  {'iters':>5s}  {'reason':>12s}  {'max_pe':>8s}  {'final_pe':>8s}  {'max|u|':>8s}  {'time':>6s}")
print("-" * 70)

for reg in [0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 100.0]:
    t0 = time.time()
    X_out, U_out, iters, reason = run_manual_ilqr(X.copy(), U.copy(), reg, max_iters=30)
    elapsed = time.time() - t0
    pe = pe_profile(X_out)
    max_u = np.max(np.abs(U_out))
    print(f"{reg:10.1e}  {iters:5d}  {reason:>12s}  {pe.max():8.1f}  {pe[-1]:8.2f}  {max_u:8.2f}  {elapsed:6.2f}s")
