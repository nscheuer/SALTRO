"""Check if 28deg final PE is a time-horizon issue."""
import os, sys, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))
import saltro_py
from sat_3_1_hybrid import create_satellite

qg = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
x0 = np.array([0.01, 0.01, 0.01, 1, 0, 0, 0, 0.0])
r0 = np.array([7e6, 0, 0]); v0 = np.array([0, 7.5e3, 0])

ps0 = saltro_py.PlannerSettings(); ps0.num_passes = 1
sat0 = create_satellite(ps0)
print(f"MTQ: {[(sat0.getMTQ(i).u_max, list(sat0.getMTQ(i).axis)) for i in range(sat0.numMTQ)]}")
print(f"RW:  {[(sat0.getRW(i).u_max, list(sat0.getRW(i).axis)) for i in range(sat0.numRW)]}")
print(f"wmax = {np.degrees(ps0.constraints.wmax):.1f} deg/s")
print()

def make_ps(spike):
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = 1; ps.num_passes = 1
    ps.passes[0].dt = 10.0; ps.passes[0].ilqr.cost_tol = 1e-3
    ps.passes[0].ilqr.max_iters = 250; ps.passes[0].auglag.max_outer_iters = 30
    ps.passes[0].auglag.constraint_tol = 1e-3
    c = ps.passes[0].cost
    c.angle = 1e2; c.ang_vel = 1e1; c.control_mult = 1.0
    c.mtq_control_weight = 1e-1; c.rw_control_weight = 1.0
    c.angle_N = 1e2; c.ang_vel_N = 1e1; c.ang_cost_func_type = 3; c.use_cost_hess = True
    for a in ["aero","gg","srp","prop","gendist","resdipole"]:
        setattr(ps.disturbances, "plan_for_" + a, False)
    ps.passes[0].reg.reg_init = 1e-6; ps.passes[0].reg.reg_max = 1e10; ps.passes[0].reg.reg_scale = 10.0
    ps.passes[0].linesearch.max_iters = 20; ps.passes[0].linesearch.beta1 = 1e-10; ps.passes[0].linesearch.beta2 = 5000.0
    if spike:
        sr = ps.passes[0].spike_removal
        sr.enabled = True; sr.start_at_iter = 2; sr.max_intervention_iters = 8
        sr.blend_len = 30; sr.goal_switch_buffer = 15; sr.min_consecutive = 7
        sr.exit_fudge = 2.0; sr.min_prior_decrease_knots = 10; sr.min_spike_ratio = 3.0
        sr.max_spike_knots = 55; sr.kp_q = 0.3; sr.kd_w = 2.0; sr.rw_scale = 0.0; sr.omega_max = 0.30
    return ps

for T in [1000, 2000, 5000]:
    jt = np.array([0.22, 0.22 + T / (36525.0 * 86400.0)])
    qgoal = np.tile(qg[:, None], (1, 2))
    bs = np.array([[1,1],[0,0],[0,0]], dtype=float)
    ps = make_ps(True); sat = create_satellite(ps)
    fd = os.open(os.devnull, os.O_WRONLY); old = os.dup(1)
    os.dup2(fd, 1); sys.stdout.flush()
    ok, X, U, K = saltro_py.trajOpt(ps, sat, x0, r0, v0, jt, qgoal, bs)
    os.dup2(old, 1); os.close(fd); os.close(old)
    pe = 2*np.degrees(np.arccos(min(abs(float(np.dot(X[3:7,-1], qg))), 1)))
    print(f"T={T:5d}s  N={X.shape[1]:4d}  ok={str(ok):5s}  final_pe={pe:.1f} deg")
