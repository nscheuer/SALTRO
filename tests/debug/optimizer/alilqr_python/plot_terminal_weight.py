"""Plot PE and controls for different terminal weight ratios."""
import sys, os, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from sat_3_1_hybrid import create_satellite

qg = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
x0 = np.array([0.01, 0.01, 0.01, 1, 0, 0, 0, 0.0])
r0 = np.array([7e6, 0, 0]); v0 = np.array([0, 7.5e3, 0])
jtime = np.array([0.22, 0.22 + 1000.0 / (36525.0 * 86400.0)])
qgoal = np.array([[np.sqrt(2)/2, np.sqrt(2)/2],[0,0],[0,0],[np.sqrt(2)/2, np.sqrt(2)/2]])
bs = np.array([[1,1],[0,0],[0,0]], dtype=float)

cases = [
    (1e4, 1e4, "1e4 / 1e4"),
    (1e4, 1e6, "1e4 / 1e6"),
    (1e6, 1e6, "1e6 / 1e6"),
    (1e6, 1e8, "1e6 / 1e8"),
]

fig, axes = plt.subplots(len(cases), 3, figsize=(18, 4*len(cases)))
fig.suptitle("BdotCtrl 90° Slew — Effect of Terminal Weight (angle / angle_N)", fontsize=14)

for row, (angle_w, angle_N, label) in enumerate(cases):
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = 2; ps.num_passes = 1
    ps.passes[0].dt = 10.0; ps.passes[0].ilqr.cost_tol = 1e-6
    ps.passes[0].ilqr.max_iters = 200; ps.passes[0].auglag.max_outer_iters = 30
    ps.passes[0].auglag.constraint_tol = 1e-3
    c = ps.passes[0].cost
    c.angle = angle_w; c.ang_vel = angle_w/100; c.control_mult = 1.0
    c.mtq_control_weight = 1e-1; c.rw_control_weight = 1.0
    c.angle_N = angle_N; c.ang_vel_N = angle_N/100
    c.ang_cost_func_type = 3; c.use_cost_hess = True
    for a in ["aero","gg","srp","prop","gendist","resdipole"]:
        setattr(ps.disturbances, "plan_for_"+a, False)
    ps.passes[0].reg.reg_init = 1e-6; ps.passes[0].reg.reg_max = 1e30; ps.passes[0].reg.reg_scale = 1.6
    ps.passes[0].linesearch.max_iters = 24; ps.passes[0].linesearch.beta1 = 1e-10; ps.passes[0].linesearch.beta2 = 5000.0
    sat = create_satellite(ps)

    fd = os.open(os.devnull, os.O_WRONLY); old = os.dup(1)
    os.dup2(fd, 1); sys.stdout.flush()
    ok, X, U, K = saltro_py.trajOpt(ps, sat, x0, r0, v0, jtime, qgoal, bs)
    os.dup2(old, 1); os.close(fd); os.close(old)

    N = X.shape[1]
    dt = ps.passes[0].dt
    t_arr = np.arange(N) * dt
    pe = np.array([2*np.degrees(np.arccos(min(abs(float(np.dot(X[3:7,k], qg))), 1))) for k in range(N)])

    conv = "CONVERGED" if ok else "not converged"

    # PE
    ax = axes[row, 0]
    ax.plot(t_arr, pe, 'b-', linewidth=1.5)
    ax.axhline(1.0, color='g', linestyle='--', alpha=0.5, label="1°")
    ax.set_ylabel("PE (deg)")
    ax.set_ylim(-2, max(95, pe.max()+5))
    ax.set_title(f"{label} — {conv}  final={pe[-1]:.2f}°  mean={pe.mean():.1f}°")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Controls
    ax = axes[row, 1]
    t_u = t_arr[:U.shape[1]]
    for i in range(U.shape[0]):
        ax.plot(t_u, U[i, :], linewidth=0.8)
    ax.axhline(sat.getMTQ(0).u_max, color='r', linestyle='--', alpha=0.5)
    ax.axhline(-sat.getMTQ(0).u_max, color='r', linestyle='--', alpha=0.5)
    ax.set_ylabel("Control")
    ax.set_title(f"max|u|={np.max(np.abs(U)):.3f}")
    ax.grid(True, alpha=0.3)

    # Quaternion
    ax = axes[row, 2]
    for i in range(4):
        ax.plot(t_arr, X[3+i, :], linewidth=1.5, label=f"q{i}")
    ax.axhline(qg[0], color='r', linestyle=':', alpha=0.3)
    ax.axhline(qg[3], color='purple', linestyle=':', alpha=0.3)
    ax.set_ylim(-1.1, 1.1)
    ax.set_ylabel("q")
    ax.set_title("Quaternion")
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)

for ax in axes[-1, :]:
    ax.set_xlabel("Time (s)")

plt.tight_layout()
out = Path(__file__).parent / "terminal_weight_comparison.png"
fig.savefig(str(out), dpi=150)
print(f"Saved: {out}")
