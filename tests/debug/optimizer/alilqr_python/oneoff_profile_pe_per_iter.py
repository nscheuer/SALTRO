"""Profile PE_fin per iteration on vec 00_baseline (GN on, type 3 + new path).

Goal: see whether iter count is dominated by tail-polishing past mission-relevant
accuracy (e.g., PE < 1°), or whether convergence is genuinely slow throughout.
"""
import sys, time, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from sat_3_1_hybrid import create_satellite
from trajOpt import trajOpt


def _quat_rotmat(q):
    q0, qv = q[0], q[1:]
    qx, qy, qz = qv
    skew = np.array([[0, -qz, qy], [qz, 0, -qx], [-qy, qx, 0]])
    return (q0*q0 - qv.dot(qv)) * np.eye(3) + 2*np.outer(qv, qv) + 2*q0*skew


def pe_fin_at(X, r_eci, bs=np.array([1.0, 0.0, 0.0])):
    q = X[3:7, -1]; qn = q / max(np.linalg.norm(q), 1e-12)
    bs_eci = _quat_rotmat(qn) @ bs
    return np.degrees(np.arccos(min(max(float(bs_eci.dot(r_eci)), -1.0), 1.0)))


ps = saltro_py.PlannerSettings()
ps.init_traj.initcontroller = 1
ps.num_passes = 1
ps.passes[0].dt = 10.0
ps.passes[0].ilqr.cost_tol = 1e-6
ps.passes[0].ilqr.max_iters = 5000
ps.passes[0].ilqr.grad_tol = 0.0
ps.passes[0].auglag.max_outer_iters = 30
ps.passes[0].auglag.constraint_tol = 1e-3
c = ps.passes[0].cost
c.angle = 1e4; c.ang_vel = 100.0
c.control_mult = 1.0; c.mtq_control_weight = 0.1; c.rw_control_weight = 1.0
c.ang_cost_func_type = 3
c.use_cost_hess = True
c.cost_hess_gauss_newton = True
c.setTerminalEmphasis(100.0)
c.ang_vel_err_dir = 0.0
c.ang_vel_err_dir_ratio = 0.3
c.ang_vel_roll_ratio = 0.05
ps.passes[0].reg.reg_init = 1e-12; ps.passes[0].reg.reg_max = 1e30
ps.passes[0].reg.reg_scale = 1.6
ps.passes[0].linesearch.max_iters = 24
ps.passes[0].linesearch.beta1 = 1e-10; ps.passes[0].linesearch.beta2 = 5000.0

sat = create_satellite(ps)
nRW = sat.numRW
x0 = np.hstack([np.array([0.01, 0.01, 0.01]),
                np.array([1.0, 0.0, 0.0, 0.0]),
                np.zeros(nRW)])
ang = np.radians(90.0)
eci_target = np.array([np.cos(ang), np.sin(ang), 0.0])
qg = np.array([np.nan, eci_target[0], eci_target[1], eci_target[2]])
qgoal = np.tile(qg[:, None], (1, 2))
bs = np.array([[1, 1], [0, 0], [0, 0]], dtype=float)
r0 = np.array([7e6, 0, 0]); v0 = np.array([0, 7.5e3, 0])
jtime = np.array([0.22, 0.22 + 1000.0 / (36525.0 * 86400.0)])

cfg = {
    "start_at_iter": 2, "max_intervention_iters": 10000,
    "blend_len": 30, "goal_switch_buffer": 15, "min_consecutive": 7,
    "exit_fudge": 2.0, "min_prior_decrease_knots": 5, "min_spike_ratio": 2.0,
    "kp_q": 0.3, "kd_w": 2.0, "rw_scale": -1.0, "omega_max": 0.30,
    "verbose": False, "constraint_gate_ratio": 0.0,
}

print("Running vec 00_baseline (type=3 + GN + new-path crossterm)...")
t0 = time.time()
X, U, stop, snaps, trans, dt_val, ctol_cfg, elapsed = trajOpt(
    ps, sat, x0, r0, v0, jtime, qgoal, bs, debug=True, spike_removal_cfg=cfg,
)
wall = time.time() - t0

# Per-snapshot PE_fin and J
pe_fins = np.array([pe_fin_at(s["X"], eci_target) for s in snaps])
Js = np.array([s["J"] for s in snaps])
outer_iters = np.array([s.get("outer_iter", 0) for s in snaps])
n = len(snaps)

print(f"\nFinal: stop={stop[:60]}, total snaps={n}, wall={wall:.2f}s")
print(f"Final PE_fin = {pe_fins[-1]:.4f}°, J = {Js[-1]:.6e}")
print()

# Iter at which PE_fin first crosses thresholds (descending)
thresholds = [10, 5, 2, 1, 0.5, 0.1, 0.05, 0.01]
print(f"{'threshold':>10} {'first_iter':>12} {'frac_total':>10}")
print("-" * 36)
for thresh in thresholds:
    below = np.where(pe_fins < thresh)[0]
    if len(below) == 0:
        print(f"{thresh:>10.2f} {'never':>12} {'-':>10}")
    else:
        first = int(below[0])
        print(f"{thresh:>10.2f} {first:>12d} {first/n*100:>9.1f}%")

print()
print("PE_fin trajectory (sampled):")
sample_idx = sorted(set([0, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, n//2, n-1]) & set(range(n)))
print(f"{'iter':>6} {'outer':>4} {'PE_fin (deg)':>14} {'J':>14}")
for i in sample_idx:
    print(f"{i:>6d} {outer_iters[i]:>4d} {pe_fins[i]:>14.4f} {Js[i]:>14.4e}")

# Also save raw data
np.savez(Path(__file__).parent / "profile_pe_per_iter.npz",
         pe_fins=pe_fins, Js=Js, outer_iters=outer_iters)
print(f"\nRaw data saved to profile_pe_per_iter.npz")
