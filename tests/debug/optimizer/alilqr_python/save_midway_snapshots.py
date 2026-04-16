"""Save snapshots at key iterations from the last run's data."""
import sys, numpy as np, pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from sat_3_1_hybrid import create_satellite
from trajOpt import trajOpt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

qg = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
x0 = np.array([0.01, 0.01, 0.01, 1, 0, 0, 0, 0.0])
r0 = np.array([7e6, 0, 0]); v0 = np.array([0, 7.5e3, 0])
jtime = np.array([0.22, 0.22 + 1000.0 / (36525.0 * 86400.0)])
qgoal = np.tile(qg[:, None], (1, 2))
bs = np.array([[1,1],[0,0],[0,0]], dtype=float)

ps = saltro_py.PlannerSettings()
ps.init_traj.initcontroller = 1; ps.num_passes = 1
ps.passes[0].dt = 10.0; ps.passes[0].ilqr.cost_tol = 1e-6
ps.passes[0].ilqr.max_iters = 200; ps.passes[0].ilqr.grad_tol = 0.0
ps.passes[0].auglag.max_outer_iters = 30
ps.passes[0].auglag.constraint_tol = 1e-3
c = ps.passes[0].cost
c.angle = 1e4; c.ang_vel = 1e2
c.control_mult = 1.0; c.mtq_control_weight = 1e-1; c.rw_control_weight = 1.0
c.angle_N = 1e4; c.ang_vel_N = 1e2
c.ang_cost_func_type = 3; c.use_cost_hess = True
for a in ["aero","gg","srp","prop","gendist","resdipole"]:
    setattr(ps.disturbances, "plan_for_"+a, False)
ps.passes[0].reg.reg_init = 1e-6; ps.passes[0].reg.reg_max = 1e30
ps.passes[0].reg.reg_scale = 1.6
ps.passes[0].linesearch.max_iters = 24
ps.passes[0].linesearch.beta1 = 1e-10; ps.passes[0].linesearch.beta2 = 5000.0

sat = create_satellite(ps)

spike_cfg = {
    "start_at_iter": 2, "max_intervention_iters": 20,
    "blend_len": 30, "goal_switch_buffer": 15, "min_consecutive": 7,
    "exit_fudge": 2.0, "min_prior_decrease_knots": 5, "min_spike_ratio": 2.0,
    "kp_q": 0.3, "kd_w": 2.0, "rw_scale": 0.0, "omega_max": 0.30, "verbose": False,
}

X, U, stop, snaps, trans, dt, ctol, elapsed = trajOpt(
    ps, sat, x0, r0, v0, jtime, qgoal, bs, debug=True, spike_removal_cfg=spike_cfg,
)

print(f"Stop: {stop}  {len(snaps)} snapshots  elapsed: {elapsed:.1f}s")

def pe_profile(X):
    return np.array([2*np.degrees(np.arccos(min(abs(float(np.dot(X[3:7,k], qg))), 1)))
                      for k in range(X.shape[1])])

dt_val = ps.passes[0].dt
N = snaps[0]['X'].shape[1]
t_arr = np.arange(N) * dt_val

# Save snapshots at key points
key_iters = [0, 5, 10, 20, 50, 100, 200, 500, 1000, len(snaps)//2, len(snaps)-1]
key_iters = sorted(set(i for i in key_iters if 0 <= i < len(snaps)))

fig, axes = plt.subplots(len(key_iters), 2, figsize=(16, 4*len(key_iters)))
fig.suptitle(f"Convergence progression — angle=1e4 ExcCtrl + spike removal", fontsize=14)

for row, idx in enumerate(key_iters):
    snap = snaps[idx]
    pe = pe_profile(snap['X'])
    U_s = snap['U']
    max_u = np.max(np.abs(U_s)) if U_s.shape[1] > 0 else 0
    outer = snap.get('outer_iter', '?')

    ax_pe = axes[row, 0]
    ax_pe.plot(t_arr, pe, 'b-', linewidth=1.5)
    ax_pe.axhline(1.0, color='g', linestyle='--', alpha=0.5)
    ax_pe.set_ylim(-5, 185)
    ax_pe.set_ylabel("PE (deg)")
    ax_pe.set_title(f"iter={idx}  outer={outer}  J={snap['J']:.2e}  max_pe={pe.max():.1f}°  mean={pe.mean():.1f}°")
    ax_pe.grid(True, alpha=0.3)

    ax_ctrl = axes[row, 1]
    t_u = t_arr[:U_s.shape[1]]
    for i in range(U_s.shape[0]):
        ax_ctrl.plot(t_u, U_s[i, :], linewidth=0.8)
    u_lim = sat.getMTQ(0).u_max
    ax_ctrl.axhline(u_lim, color='r', linestyle='--', alpha=0.5)
    ax_ctrl.axhline(-u_lim, color='r', linestyle='--', alpha=0.5)
    ax_ctrl.set_ylabel("Control")
    ax_ctrl.set_title(f"max|u|={max_u:.3f}  (lim={u_lim:.3f})")
    ax_ctrl.grid(True, alpha=0.3)

axes[-1, 0].set_xlabel("Time (s)")
axes[-1, 1].set_xlabel("Time (s)")
plt.tight_layout()
out_path = Path(__file__).parent / "convergence_midway_snapshots.png"
fig.savefig(str(out_path), dpi=120)
print(f"Saved: {out_path}")
