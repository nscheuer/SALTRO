"""Run ONCE and save both the GIF and midway snapshots from the same data."""
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
from matplotlib.animation import FuncAnimation, PillowWriter

ANGLE_W = float(sys.argv[1]) if len(sys.argv) > 1 else 1e4
IC = int(sys.argv[2]) if len(sys.argv) > 2 else 1
MAX_ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 200
MAX_OUTER = int(sys.argv[4]) if len(sys.argv) > 4 else 30
USE_SPIKE = (sys.argv[5].lower() == "spike") if len(sys.argv) > 5 else False

qg = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
x0 = np.array([0.01, 0.01, 0.01, 1, 0, 0, 0, 0.0])
r0 = np.array([7e6, 0, 0]); v0 = np.array([0, 7.5e3, 0])
jtime = np.array([0.22, 0.22 + 1000.0 / (36525.0 * 86400.0)])
qgoal = np.tile(qg[:, None], (1, 2))
bs = np.array([[1,1],[0,0],[0,0]], dtype=float)

ps = saltro_py.PlannerSettings()
ps.init_traj.initcontroller = IC; ps.num_passes = 1
ps.passes[0].dt = 10.0; ps.passes[0].ilqr.cost_tol = 1e-6
ps.passes[0].ilqr.max_iters = MAX_ITERS; ps.passes[0].ilqr.grad_tol = 0.0
ps.passes[0].auglag.max_outer_iters = MAX_OUTER
ps.passes[0].auglag.constraint_tol = 1e-3
c = ps.passes[0].cost
c.angle = ANGLE_W; c.ang_vel = ANGLE_W / 100
c.control_mult = 1.0; c.mtq_control_weight = 1e-1; c.rw_control_weight = 1.0
c.ang_cost_func_type = 3; c.use_cost_hess = True
# Terminal emphasis: scale all terminal weights uniformly against stage
# to avoid weight-ratio pathology (see CostConfig docs).
c.setTerminalEmphasis(100.0)
for a in ["aero","gg","srp","prop","gendist","resdipole"]:
    setattr(ps.disturbances, "plan_for_"+a, False)
ps.passes[0].reg.reg_init = 1e-6; ps.passes[0].reg.reg_max = 1e30
ps.passes[0].reg.reg_scale = 1.6
ps.passes[0].linesearch.max_iters = 24
ps.passes[0].linesearch.beta1 = 1e-10; ps.passes[0].linesearch.beta2 = 5000.0

sat = create_satellite(ps)
spike_label = "_spike" if USE_SPIKE else ""
spike_cfg = None
if USE_SPIKE:
    spike_cfg = {
        "start_at_iter": 2, "max_intervention_iters": 20,
        "blend_len": 30, "goal_switch_buffer": 15, "min_consecutive": 7,
        "exit_fudge": 2.0, "min_prior_decrease_knots": 5, "min_spike_ratio": 2.0,
        "kp_q": 0.3, "kd_w": 2.0, "rw_scale": -1.0, "omega_max": 0.30, "verbose": True,
    }
    print("Spike removal ENABLED")

X, U, stop, snaps, trans, dt_val, ctol, elapsed = trajOpt(
    ps, sat, x0, r0, v0, jtime, qgoal, bs, debug=True, spike_removal_cfg=spike_cfg,
)
print(f"Stop: {stop}  {len(snaps)} snapshots  elapsed: {elapsed:.1f}s")

def pe_profile(X):
    return np.array([2*np.degrees(np.arccos(min(abs(float(np.dot(X[3:7,k], qg))), 1)))
                      for k in range(X.shape[1])])

N = snaps[0]['X'].shape[1]
t_arr = np.arange(N) * ps.passes[0].dt

# ===== Midway snapshots =====
key_iters = [0, 5, 10, 20, 50, 100, 200, 500, len(snaps)//4, len(snaps)//2, 3*len(snaps)//4, len(snaps)-1]
key_iters = sorted(set(i for i in key_iters if 0 <= i < len(snaps)))

fig, axes = plt.subplots(len(key_iters), 2, figsize=(16, 4*len(key_iters)))
fig.suptitle(f"angle={ANGLE_W:.0e} ic={IC} {('+ spike' if USE_SPIKE else 'baseline')} — {len(snaps)} iters", fontsize=14)

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
    ax_pe.set_title(f"iter={idx}  outer={outer}  J={snap['J']:.2e}  mean={pe.mean():.1f}°  max={pe.max():.1f}°")
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
snap_path = Path(__file__).parent / f"convergence_midway{spike_label}.png"
fig.savefig(str(snap_path), dpi=120)
print(f"Saved: {snap_path}")
plt.close()

# ===== GIF =====
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle(f"iLQR Convergence — angle={ANGLE_W:.0e}, ic={IC}", fontsize=14)

all_pe = [pe_profile(s['X']) for s in snaps]

def animate(frame):
    for ax in axes2.flat:
        ax.clear()
    snap = snaps[frame]
    pe = all_pe[frame]
    X_f = snap['X']; U_f = snap['U']
    J = snap['J']; oi = snap.get('outer_iter', '?')
    max_u = np.max(np.abs(U_f)) if U_f.shape[1] > 0 else 0

    ax = axes2[0, 0]
    for i in range(max(0, frame-10), frame):
        axes2[0, 0].plot(t_arr, all_pe[i], 'gray', alpha=0.2, linewidth=0.5)
    ax.plot(t_arr, pe, 'b-', linewidth=2)
    ax.axhline(1.0, color='g', linestyle='--', alpha=0.5)
    ax.set_ylabel("PE (deg)"); ax.set_xlabel("Time (s)"); ax.set_ylim(-5, 185)
    ax.set_title(f"iter={frame}/{len(snaps)-1}  outer={oi}  J={J:.4e}")
    ax.grid(True, alpha=0.3)

    ax = axes2[0, 1]
    for i in range(4):
        ax.plot(t_arr, X_f[3+i, :], linewidth=1.5, label=f"q{i}")
    ax.set_ylabel("q"); ax.set_xlabel("Time (s)"); ax.set_ylim(-1.1, 1.1)
    ax.set_title("Quaternion"); ax.legend(loc='upper right', fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes2[1, 0]
    t_u = t_arr[:U_f.shape[1]]
    for i in range(U_f.shape[0]):
        ax.plot(t_u, U_f[i, :], linewidth=1)
    ax.axhline(sat.getMTQ(0).u_max, color='r', linestyle='--', alpha=0.5)
    ax.axhline(-sat.getMTQ(0).u_max, color='r', linestyle='--', alpha=0.5)
    ax.set_ylabel("Control"); ax.set_xlabel("Time (s)")
    ax.set_title(f"Controls (max|u|={max_u:.2f})"); ax.grid(True, alpha=0.3)

    ax = axes2[1, 1]
    for i in range(3):
        ax.plot(t_arr, np.degrees(X_f[i, :]), linewidth=1.5, label=f"w{i}")
    wmax = np.degrees(ps.constraints.wmax)
    ax.axhline(wmax, color='r', linestyle='--', alpha=0.5)
    ax.axhline(-wmax, color='r', linestyle='--', alpha=0.5)
    ax.set_ylabel("Ang Vel (deg/s)"); ax.set_xlabel("Time (s)")
    ax.set_title(f"Angular Velocity (max={np.degrees(np.max(np.abs(X_f[0:3, :]))):.1f} deg/s)")
    ax.legend(loc='upper right', fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return axes2.flat

n_frames = len(snaps)
step = max(1, n_frames // 100)
frames = list(range(0, n_frames, step))
if frames[-1] != n_frames - 1:
    frames.append(n_frames - 1)

# Hold the final frame for 10 seconds at end of GIF.
FPS = 4
HOLD_SECONDS = 10
frames = frames + [n_frames - 1] * (FPS * HOLD_SECONDS)

print(f"Generating GIF with {len(frames)} frames...")
anim = FuncAnimation(fig2, animate, frames=frames, repeat=False)
gif_path = Path(__file__).parent / f"convergence_angle{ANGLE_W:.0e}_ic{IC}{spike_label}.gif"
anim.save(str(gif_path), writer=PillowWriter(fps=FPS))
print(f"Saved: {gif_path}")

png_path = Path(__file__).parent / f"convergence_angle{ANGLE_W:.0e}_ic{IC}{spike_label}_final.png"
animate(n_frames - 1)
fig2.savefig(str(png_path), dpi=150)
print(f"Saved: {png_path}")
