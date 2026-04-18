"""Test matrix for winding-number detector vs three-pass vs no-spike-removal.

Runs 6 cases + a threshold sweep and reports: convergence, iters, time,
final PE, final SO(3) excess, constraint violation.
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

qg = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
x0 = np.array([0.01, 0.01, 0.01, 1, 0, 0, 0, 0.0])
r0 = np.array([7e6, 0, 0]); v0 = np.array([0, 7.5e3, 0])
jtime = np.array([0.22, 0.22 + 1000.0 / (36525.0 * 86400.0)])
qgoal = np.tile(qg[:, None], (1, 2))
bs = np.array([[1,1],[0,0],[0,0]], dtype=float)


def step_angle(q1, q2):
    return 2 * np.arccos(min(abs(float(np.dot(q1, q2))), 1.0))


def excess_full(X):
    q = X[3:7, :]
    N = q.shape[1]
    steps = [step_angle(q[:, k], q[:, k+1]) for k in range(N-1)]
    traveled = sum(steps)
    direct = step_angle(q[:, 0], q[:, -1])
    return traveled, direct


def pe_profile(X):
    return np.array([2*np.degrees(np.arccos(min(abs(float(np.dot(X[3:7,k], qg))), 1)))
                     for k in range(X.shape[1])])


def build_settings(angle_w):
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = 1; ps.num_passes = 1
    ps.passes[0].dt = 10.0; ps.passes[0].ilqr.cost_tol = 1e-6
    ps.passes[0].ilqr.max_iters = 200; ps.passes[0].ilqr.grad_tol = 0.0
    ps.passes[0].auglag.max_outer_iters = 30
    ps.passes[0].auglag.constraint_tol = 1e-3
    c = ps.passes[0].cost
    c.angle = angle_w; c.ang_vel = angle_w / 100
    c.control_mult = 1.0; c.mtq_control_weight = 1e-1; c.rw_control_weight = 1.0
    c.angle_N = angle_w; c.ang_vel_N = angle_w / 100
    c.ang_cost_func_type = 3; c.use_cost_hess = True
    for a in ["aero","gg","srp","prop","gendist","resdipole"]:
        setattr(ps.disturbances, "plan_for_"+a, False)
    ps.passes[0].reg.reg_init = 1e-6; ps.passes[0].reg.reg_max = 1e30
    ps.passes[0].reg.reg_scale = 1.6
    ps.passes[0].linesearch.max_iters = 24
    ps.passes[0].linesearch.beta1 = 1e-10; ps.passes[0].linesearch.beta2 = 5000.0
    return ps


def run_case(name, angle_w, detector, threshold=None):
    ps = build_settings(angle_w)
    sat = create_satellite(ps)
    cfg = None
    if detector != "none":
        cfg = {
            "start_at_iter": 2, "max_intervention_iters": 20,
            "blend_len": 30, "goal_switch_buffer": 15, "min_consecutive": 7,
            "exit_fudge": 2.0, "min_prior_decrease_knots": 5, "min_spike_ratio": 2.0,
            "kp_q": 0.3, "kd_w": 2.0, "rw_scale": 0.0, "omega_max": 0.30, "verbose": False,
        }
        if detector == "winding":
            cfg["winding_detector"] = True
            cfg["winding_excess_threshold"] = threshold if threshold is not None else np.pi

    t0 = time.time()
    try:
        result = trajOpt(ps, sat, x0, r0, v0, jtime, qgoal, bs, debug=True, spike_removal_cfg=cfg)
        X, U, stop, snaps = result[0], result[1], result[2], result[3]
    except Exception as e:
        return {"name": name, "error": str(e)[:80]}
    elapsed = time.time() - t0

    # Extract constraint violation from stop reason (format: "... violation <num> <= ...")
    ctol = None
    if "violation" in stop:
        try:
            ctol = float(stop.split("violation")[1].split("<=")[0].strip())
        except:
            pass

    pe = pe_profile(X)
    trav, direct = excess_full(X)
    return {
        "name": name,
        "stop": stop.split(":")[0] if ":" in stop else stop,
        "iters": len(snaps),
        "elapsed_s": elapsed,
        "pe_mean_deg": float(pe.mean()),
        "pe_max_deg": float(pe.max()),
        "pe_final_deg": float(pe[-1]),
        "excess_deg": float(np.degrees(trav - direct)),
        "traveled_deg": float(np.degrees(trav)),
        "ctol": ctol,
    }


def format_row(r):
    if "error" in r:
        return f"{r['name']:<30} ERROR: {r['error']}"
    ctol_s = f"{r['ctol']:.2e}" if r['ctol'] is not None else "   n/a  "
    return (f"{r['name']:<30} "
            f"{r['stop']:<30} "
            f"it={r['iters']:>5}  "
            f"t={r['elapsed_s']:>5.1f}s  "
            f"PE_mean={r['pe_mean_deg']:>5.1f}°  "
            f"PE_fin={r['pe_final_deg']:>5.1f}°  "
            f"excess={r['excess_deg']:>6.1f}°  "
            f"ctol={ctol_s}")


# -----------------------------------------------------------------------------
# Main matrix
# -----------------------------------------------------------------------------
print("=" * 120)
print("MAIN MATRIX: 2 angle weights × 3 detectors")
print("=" * 120)
results_matrix = []
for angle_w in [1e4, 1e6]:
    for det in ["none", "three-pass", "winding"]:
        name = f"w={angle_w:.0e}  {det}"
        print(f"  running: {name} ...", flush=True)
        r = run_case(name, angle_w, det)
        print("    " + format_row(r))
        results_matrix.append(r)

print()
print("=" * 120)
print("THRESHOLD SWEEP: 1e4 × winding @ {π/2, 3π/4, π, 5π/4, 3π/2}")
print("=" * 120)
sweep = []
for frac, label in [(0.5, "π/2"), (0.75, "3π/4"), (1.0, "π"), (1.25, "5π/4"), (1.5, "3π/2")]:
    thresh = frac * np.pi
    name = f"w=1e+04  winding@{label} ({np.degrees(thresh):.0f}°)"
    print(f"  running: {name} ...", flush=True)
    r = run_case(name, 1e4, "winding", threshold=thresh)
    print("    " + format_row(r))
    sweep.append(r)

print()
print("=" * 120)
print("SUMMARY TABLE")
print("=" * 120)
print(f"{'case':<40} {'stop':<30} {'iters':>6} {'time':>7} {'PE_mean':>9} {'excess':>9} {'ctol':>11}")
print("-" * 120)
for r in results_matrix + sweep:
    if "error" in r:
        print(f"{r['name']:<40} ERROR")
        continue
    ctol_s = f"{r['ctol']:.2e}" if r['ctol'] is not None else "   n/a  "
    print(f"{r['name']:<40} {r['stop']:<30} {r['iters']:>6} {r['elapsed_s']:>6.1f}s "
          f"{r['pe_mean_deg']:>7.1f}°  {r['excess_deg']:>7.1f}°  {ctol_s:>10}")
