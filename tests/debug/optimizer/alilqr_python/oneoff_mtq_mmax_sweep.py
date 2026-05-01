"""One-off: MTQ-only, ω0=0, sweep MTQ max dipole m_max ∈ {0.2, 0.5, 1.0, 2.0}.
Tests whether MTQ-only failure is really hardware undersizing (1U-class MTQ
on 3-6U-class sat) vs. algorithmic / geometric limitation."""
import sys, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import saltro_py
from trajOpt import trajOpt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "oneoff_mtq_mmax_sweep"
OUT.mkdir(exist_ok=True)

MMAX_VALUES = [0.2, 0.5, 1.0, 2.0]  # A.m^2 per MTQ

qg = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
x0 = np.array([0.0, 0.0, 0.0, 1, 0, 0, 0])
r0 = np.array([7e6, 0, 0]); v0 = np.array([0, 7.5e3, 0])
jtime = np.array([0.22, 0.22 + 1000.0 / (36525.0 * 86400.0)])
qgoal = np.tile(qg[:, None], (1, 2))
bs = np.array([[1,1],[0,0],[0,0]], dtype=float)

def build_sat(ps, m_max):
    J = np.diag([0.067, 0.071, 0.069])
    sat = saltro_py.Satellite(J, ps)
    sat.addMTQ(np.array([1.0, 0.0, 0.0]), m_max)
    sat.addMTQ(np.array([0.0, 1.0, 0.0]), m_max)
    sat.addMTQ(np.array([0.0, 0.0, 1.0]), m_max)
    return sat

def build_ps():
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = 1; ps.num_passes = 1
    ps.passes[0].dt = 10.0; ps.passes[0].ilqr.cost_tol = 1e-6
    ps.passes[0].ilqr.max_iters = 200; ps.passes[0].ilqr.grad_tol = 0.0
    ps.passes[0].auglag.max_outer_iters = 30
    ps.passes[0].auglag.constraint_tol = 1e-3
    c = ps.passes[0].cost
    c.angle = 1e4; c.ang_vel = 1e2
    c.control_mult = 1.0; c.mtq_control_weight = 0.1
    c.ang_cost_func_type = 3; c.use_cost_hess = True
    c.setTerminalEmphasis(100.0)
    for a in ["aero","gg","srp","prop","gendist","resdipole"]:
        setattr(ps.disturbances, "plan_for_"+a, False)
    ps.passes[0].reg.reg_init = 1e-6; ps.passes[0].reg.reg_max = 1e30
    ps.passes[0].reg.reg_scale = 1.6
    ps.passes[0].linesearch.max_iters = 24
    ps.passes[0].linesearch.beta1 = 1e-10; ps.passes[0].linesearch.beta2 = 5000.0
    return ps

def pe_profile(X, qg):
    return np.array([2*np.degrees(np.arccos(min(abs(float(np.dot(X[3:7,k], qg))), 1)))
                      for k in range(X.shape[1])])

J_mean = (0.067+0.071+0.069)/3
B_LEO = 3e-5

results = []
for m_max in MMAX_VALUES:
    ps = build_ps()
    sat = build_sat(ps, m_max)
    alpha_theory = m_max * B_LEO * np.sqrt(2) / J_mean * 180/np.pi * 1000  # mdeg/s^2
    print(f"\n=== m_max = {m_max} A.m^2  (α_max_theory ≈ {alpha_theory:.1f} mdeg/s²) ===")
    X, U, stop, snaps, trans, dt_val, ctol, elapsed = trajOpt(
        ps, sat, x0.copy(), r0, v0, jtime, qgoal, bs, debug=True, spike_removal_cfg=None,
    )
    pe = pe_profile(X, qg)
    omega_norm = np.linalg.norm(X[0:3, :], axis=0)
    u_sat_frac = np.mean(np.abs(U) / sat.getMTQ(0).u_max) * 100
    print(f"  stop={stop}  it={len(snaps)}  PE_fin={pe[-1]:.3f}°  PE_mean={pe.mean():.1f}°  "
          f"|ω|_max={np.degrees(omega_norm.max()):.2f}°/s  util={u_sat_frac:.0f}%  t={elapsed:.1f}s")
    results.append(dict(
        m_max=m_max, alpha_theory=alpha_theory, X=X, U=U, snaps=snaps, stop=stop, pe=pe, dt=dt_val,
        sat=sat, omega_norm=omega_norm, u_sat=u_sat_frac,
    ))

fig, axes = plt.subplots(len(MMAX_VALUES), 3, figsize=(18, 3.5*len(MMAX_VALUES)))
for i, r in enumerate(results):
    X, U, pe = r["X"], r["U"], r["pe"]
    n_mtq = r["sat"].numMTQ
    N = X.shape[1]; t = np.arange(N) * r["dt"]
    mtq_lim = r["sat"].getMTQ(0).u_max

    ax = axes[i, 0]
    ax.plot(t, pe, 'b-', lw=1.5)
    ax.axhline(1.0, color='g', ls='--', alpha=0.5)
    ax.set_ylim(-5, 185)
    ax.set_title(f"m_max={r['m_max']} (α_th={r['alpha_theory']:.1f}mdeg/s²) "
                 f"it={len(r['snaps'])} PE_fin={pe[-1]:.2f}° mean={pe.mean():.1f}°")
    ax.set_ylabel("PE (deg)")
    ax.grid(True, alpha=0.3)

    ax = axes[i, 1]
    ax.plot(t, np.degrees(r["omega_norm"]), 'k-', lw=1.2, label="|ω|")
    for j in range(3):
        ax.plot(t, np.degrees(X[j, :]), lw=0.8, alpha=0.7, label=f"ω{j}")
    ax.set_title(f"Ang vel — |ω|_max={np.degrees(r['omega_norm'].max()):.2f}°/s")
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[i, 2]
    t_u = t[:U.shape[1]]
    for j in range(n_mtq):
        ax.plot(t_u, U[j, :], lw=0.8)
    ax.axhline(mtq_lim, color='r', ls='--', alpha=0.4)
    ax.axhline(-mtq_lim, color='r', ls='--', alpha=0.4)
    ax.set_title(f"MTQ ctrl — util={r['u_sat']:.0f}% of lim")
    ax.grid(True, alpha=0.3)

axes[-1, 0].set_xlabel("Time (s)")
axes[-1, 1].set_xlabel("Time (s)")
axes[-1, 2].set_xlabel("Time (s)")
plt.tight_layout()
out = OUT / "mtq_mmax_sweep.png"
fig.savefig(str(out), dpi=120)
print(f"\nSaved: {out}")
plt.close()
