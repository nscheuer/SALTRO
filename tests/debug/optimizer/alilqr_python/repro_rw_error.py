"""Trace the sat_0_3_rw quaternion-norm error: is it warm_start or iLQR?"""
import sys, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from sat_0_3_rw import create_satellite

ps = saltro_py.PlannerSettings()
ps.init_traj.initcontroller = 1
ps.num_passes = 1
ps.passes[0].dt = 10.0
ps.passes[0].ilqr.max_iters = 200
ps.passes[0].auglag.max_outer_iters = 30
ps.passes[0].ilqr.cost_tol = 1e-6
# Match wide_test_runner baseline exactly
c = ps.passes[0].cost
c.angle = 1e4; c.ang_vel = 1e2
c.control_mult = 1.0
c.mtq_control_weight = 1e-1; c.rw_control_weight = 1.0
c.ang_cost_func_type = 3; c.use_cost_hess = True
c.setTerminalEmphasis(100.0)
ps.passes[0].reg.reg_init = 1e-6; ps.passes[0].reg.reg_max = 1e30
ps.passes[0].reg.reg_scale = 1.6
ps.passes[0].linesearch.max_iters = 24
ps.passes[0].linesearch.beta1 = 1e-10; ps.passes[0].linesearch.beta2 = 5000.0
ps.passes[0].auglag.constraint_tol = 1e-3

sat = create_satellite(ps)
print(f"sat: numMTQ={sat.numMTQ}, numRW={sat.numRW}, stateDim={sat.stateDim}, controlDim={sat.controlDim}")

x0 = np.array([0.01, 0.01, 0.01, 1, 0, 0, 0, 0.0, 0.0, 0.0])
r0 = np.array([7e6, 0, 0]); v0 = np.array([0, 7.5e3, 0])
jtime = np.array([0.22, 0.22 + 1000.0/(36525*86400)])
qg = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])

# Match trajOpt's pre-processing
from trajOpt import _resample_zero_order_hold
jtime_flat, q_goal_rs, bs_rs = _resample_zero_order_hold(
    jtime, np.tile(qg[:, None], (1, 2)),
    np.array([[1,1],[0,0],[0,0]], dtype=float), 10.0)

ok, R, V, B, S, rho = saltro_py.generate_orbit(r0, v0, jtime_flat, 0, 0, 0, 0, 0)
assert ok
print(f"Orbit ok, N={jtime_flat.size}, B[:,0]={B[:,0]}, B[:,-1]={B[:,-1]}")

q_goal_f = np.asfortranarray(q_goal_rs)
bs_f = np.asfortranarray(bs_rs)
R_f = np.asfortranarray(R); V_f = np.asfortranarray(V); B_f = np.asfortranarray(B); S_f = np.asfortranarray(S)
rho = rho.reshape(1, -1) if rho.ndim == 1 else rho

print("\n--- warm_start ---")
try:
    ok, X, U = saltro_py.warm_start(ps, sat, x0, jtime_flat, q_goal_f, bs_f, R_f, V_f, B_f, S_f, rho)
    print(f"warm_start ok={ok}, X shape={X.shape}, U shape={U.shape}")
    if ok:
        norms = np.linalg.norm(X[3:7, :], axis=0)
        bad = np.where(np.abs(norms - 1) > 1e-3)[0]
        if len(bad):
            print(f"  BAD quat knots: {bad[:5]}... first norm={norms[bad[0]]:.4f}")
        else:
            print(f"  quats all normalized (min norm={norms.min():.6f}, max={norms.max():.6f})")
        u_max = np.max(np.abs(U))
        print(f"  |U| max = {u_max:.6f}  (RW u_max={sat.getRW(0).u_max})")
        w_max = np.max(np.abs(X[0:3, :]))
        print(f"  |ω| max = {np.degrees(w_max):.2f} °/s")
        h_max = np.max(np.abs(X[7:10, :]))
        print(f"  |h_rw| max = {h_max:.6f}")
except Exception as e:
    print(f"warm_start FAILED: {type(e).__name__}: {e}")
    sys.exit(1)

print("\n--- full alilqr WITHOUT spike removal ---")
from alilqr import alilqr
try:
    X2, U2, stop, snaps, trans = alilqr(
        ps, 0, sat, X, U, R, V, B, S, rho, jtime_flat, q_goal_f, bs_f,
        debug=True, spike_removal_cfg=None,
    )
    print(f"alilqr stop: {stop}, iters: {len(snaps)}")
except Exception as e:
    print(f"alilqr FAILED: {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()

print("\n--- full alilqr WITH spike removal ---")
spike_cfg = {
    "start_at_iter": 2, "max_intervention_iters": 20,
    "blend_len": 30, "goal_switch_buffer": 15, "min_consecutive": 7,
    "exit_fudge": 2.0, "min_prior_decrease_knots": 5, "min_spike_ratio": 2.0,
    "kp_q": 0.3, "kd_w": 2.0, "rw_scale": 1.0, "omega_max": 0.30, "verbose": True,
}
# re-warm-start fresh
ok, X, U = saltro_py.warm_start(ps, sat, x0, jtime_flat, q_goal_f, bs_f, R_f, V_f, B_f, S_f, rho)
try:
    X3, U3, stop3, snaps3, trans3 = alilqr(
        ps, 0, sat, X, U, R, V, B, S, rho, jtime_flat, q_goal_f, bs_f,
        debug=True, spike_removal_cfg=spike_cfg,
    )
    print(f"alilqr w/ spike stop: {stop3}, iters: {len(snaps3)}")
except Exception as e:
    print(f"alilqr w/ spike FAILED: {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()
