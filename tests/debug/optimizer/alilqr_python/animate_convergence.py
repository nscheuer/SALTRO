"""Animate iLQR convergence: PE profile + quaternion + controls at each iteration."""
import sys, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from sat_3_1_hybrid import create_satellite
from trajOpt import trajOpt

qg = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
x0 = np.array([0.01, 0.01, 0.01, 1, 0, 0, 0, 0.0])
r0 = np.array([7e6, 0, 0]); v0 = np.array([0, 7.5e3, 0])
jtime = np.array([0.22, 0.22 + 1000.0 / (36525.0 * 86400.0)])
qgoal = np.tile(qg[:, None], (1, 2))
bs = np.array([[1,1],[0,0],[0,0]], dtype=float)

def pe_profile(X):
    N = X.shape[1]
    return np.array([2*np.degrees(np.arccos(min(abs(float(np.dot(X[3:7,k], qg))), 1)))
                      for k in range(N)])

ANGLE_W = float(sys.argv[1]) if len(sys.argv) > 1 else 1e4
IC = int(sys.argv[2]) if len(sys.argv) > 2 else 1
MAX_ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 20
MAX_OUTER = int(sys.argv[4]) if len(sys.argv) > 4 else 5

print(f"angle={ANGLE_W:.0e}  ic={IC}  max_iters={MAX_ITERS}  max_outer={MAX_OUTER}")

ps = saltro_py.PlannerSettings()
ps.init_traj.initcontroller = IC; ps.num_passes = 1
ps.passes[0].dt = 10.0; ps.passes[0].ilqr.cost_tol = 1e-6
ps.passes[0].ilqr.max_iters = MAX_ITERS; ps.passes[0].ilqr.grad_tol = 0.0
ps.passes[0].auglag.max_outer_iters = MAX_OUTER
ps.passes[0].auglag.constraint_tol = 1e-3
c = ps.passes[0].cost
c.angle = ANGLE_W; c.ang_vel = ANGLE_W / 100
c.control_mult = 1.0; c.mtq_control_weight = 1e-1; c.rw_control_weight = 1.0
c.angle_N = ANGLE_W; c.ang_vel_N = ANGLE_W / 100
COST_TYPE = int(sys.argv[5]) if len(sys.argv) > 5 else 3
c.ang_cost_func_type = COST_TYPE; c.use_cost_hess = True
c.rw_AM_weight = 0.0; c.rw_stic_weight = 0.0
c.RWh_max_mult = 0.0; c.RWh_stiction_mult = 0.0; c.RWh_ok_mult = 0.0
for a in ["aero","gg","srp","prop","gendist","resdipole"]:
    setattr(ps.disturbances, "plan_for_"+a, False)
ps.passes[0].reg.reg_init = 1e-6; ps.passes[0].reg.reg_max = 1e30
ps.passes[0].reg.reg_scale = 1.6
ps.passes[0].linesearch.max_iters = 24
ps.passes[0].linesearch.beta1 = 1e-10; ps.passes[0].linesearch.beta2 = 5000.0

sat = create_satellite(ps)

X, U, stop, snaps, trans, dt, ctol, elapsed = trajOpt(
    ps, sat, x0, r0, v0, jtime, qgoal, bs, debug=True
)

print(f"Stop: {stop}  {len(snaps)} snapshots  elapsed: {elapsed:.1f}s")
if not snaps:
    print("No snapshots!")
    sys.exit(1)

# Compute PE, control, omega for each snapshot
dt_val = ps.passes[0].dt
N = snaps[0]['X'].shape[1]
t_arr = np.arange(N) * dt_val

all_pe = [pe_profile(s['X']) for s in snaps]
all_max_u = [np.max(np.abs(s['U']), axis=0) if s['U'].shape[1] > 0 else np.zeros(N-1) for s in snaps]
all_omega = [np.array([np.linalg.norm(s['X'][0:3, k]) for k in range(N)]) for s in snaps]
all_J = [s['J'] for s in snaps]

# Determine outer iteration for each snapshot
outer_iters = []
oi = 0
for i, s in enumerate(snaps):
    outer_iters.append(s.get('outer_iter', oi))

# Create animation
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f"iLQR Convergence — angle={ANGLE_W:.0e}, ic={IC}", fontsize=14)

def animate(frame):
    for ax in axes.flat:
        ax.clear()

    pe = all_pe[frame]
    X_f = snaps[frame]['X']
    U_f = snaps[frame]['U']
    J = all_J[frame]
    oi = outer_iters[frame]

    # Top-left: Pointing error
    ax = axes[0, 0]
    # Show all previous iterations as faint lines
    for i in range(max(0, frame-10), frame):
        ax.plot(t_arr, all_pe[i], 'gray', alpha=0.2, linewidth=0.5)
    ax.plot(t_arr, pe, 'b-', linewidth=2)
    ax.axhline(1.0, color='g', linestyle='--', alpha=0.5)
    ax.set_ylabel("Pointing Error (deg)")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(-5, 185)
    ax.set_title(f"iter={frame}/{len(snaps)-1}  outer={oi}  J={J:.4e}")
    ax.grid(True, alpha=0.3)

    # Top-right: Quaternion
    ax = axes[0, 1]
    for i in range(4):
        ax.plot(t_arr, X_f[3+i, :], linewidth=1.5, label=f"q{i}")
    ax.axhline(qg[0], color='r', linestyle='--', alpha=0.3)
    ax.axhline(qg[3], color='purple', linestyle='--', alpha=0.3)
    ax.set_ylabel("q")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(-1.1, 1.1)
    ax.set_title("Quaternion")
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-left: Controls
    ax = axes[1, 0]
    nu = U_f.shape[0]
    t_u = t_arr[:U_f.shape[1]]
    for i in range(nu):
        ax.plot(t_u, U_f[i, :], linewidth=1, label=f"u{i}")
    u_max_mtq = sat.getMTQ(0).u_max
    ax.axhline(u_max_mtq, color='r', linestyle='--', alpha=0.5, label=f"MTQ lim={u_max_mtq:.2f}")
    ax.axhline(-u_max_mtq, color='r', linestyle='--', alpha=0.5)
    ax.set_ylabel("Control")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Controls (max|u|={np.max(np.abs(U_f)):.2f})")
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)

    # Bottom-right: Angular velocity
    ax = axes[1, 1]
    for i in range(3):
        ax.plot(t_arr, np.degrees(X_f[i, :]), linewidth=1.5, label=f"w{i}")
    wmax = np.degrees(ps.constraints.wmax)
    ax.axhline(wmax, color='r', linestyle='--', alpha=0.5)
    ax.axhline(-wmax, color='r', linestyle='--', alpha=0.5)
    ax.set_ylabel("Angular Velocity (deg/s)")
    ax.set_xlabel("Time (s)")
    omega_max = np.degrees(np.max(np.abs(X_f[0:3, :])))
    ax.set_title(f"Angular Velocity (max={omega_max:.1f} deg/s)")
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return axes.flat

n_frames = len(snaps)
# Subsample if too many frames
step = max(1, n_frames // 100)
frames = list(range(0, n_frames, step))
if frames[-1] != n_frames - 1:
    frames.append(n_frames - 1)

print(f"Generating animation with {len(frames)} frames...")
anim = FuncAnimation(fig, animate, frames=frames, repeat=False)
out_path = Path(__file__).parent / f"convergence_angle{ANGLE_W:.0e}_ic{IC}.gif"
anim.save(str(out_path), writer=PillowWriter(fps=4))
print(f"Saved: {out_path}")

# Also save final frame as PNG
animate(n_frames - 1)
png_path = Path(__file__).parent / f"convergence_angle{ANGLE_W:.0e}_ic{IC}_final.png"
fig.savefig(str(png_path), dpi=150)
print(f"Saved: {png_path}")
