"""Trace regularization trajectory across iterations by wrapping
saltro_py.backward_pass and saltro_py.forward_pass.  Logs each call's
(reg, ok) plus computes Q_uu at the final iteration's failing knot to
inspect eigenvalue structure.

Runs the failing cases: 0+3 RW (spike on), dt=30 3000s (spike on), ω=5× 3000s.
"""
import sys, time, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from trajOpt import trajOpt
from sat_3_1_hybrid import create_satellite as create_3_1
from sat_0_3_rw     import create_satellite as create_0_3


# ---- Wrap backward_pass and forward_pass to capture reg trajectory ----
_orig_bp = saltro_py.backward_pass
_orig_fp = saltro_py.forward_pass

call_log = []   # list of dicts: {phase, reg, ok, iter}
iter_counter = [0]  # mutable, bumped by fp success


def logged_bp(*args, **kwargs):
    reg = args[13] if len(args) > 13 else kwargs.get("reg", None)
    result = _orig_bp(*args, **kwargs)
    ok = bool(result[0])
    call_log.append({"phase": "BP", "reg": float(reg) if reg is not None else None,
                     "ok": ok, "iter": iter_counter[0]})
    return result


def logged_fp(*args, **kwargs):
    result = _orig_fp(*args, **kwargs)
    ok = bool(result[0])
    call_log.append({"phase": "FP", "reg": None, "ok": ok, "iter": iter_counter[0]})
    if ok:
        iter_counter[0] += 1
    return result


saltro_py.backward_pass = logged_bp
saltro_py.forward_pass = logged_fp


def build(dt=10.0, angle=1e4, max_iters=200, max_outer=30):
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = 1
    ps.num_passes = 1
    ps.passes[0].dt = dt
    ps.passes[0].ilqr.cost_tol = 1e-6
    ps.passes[0].ilqr.max_iters = max_iters
    ps.passes[0].ilqr.grad_tol = 0.0
    ps.passes[0].auglag.max_outer_iters = max_outer
    ps.passes[0].auglag.constraint_tol = 1e-3
    c = ps.passes[0].cost
    c.angle = angle; c.ang_vel = angle/100
    c.control_mult = 1.0
    c.mtq_control_weight = 1e-1; c.rw_control_weight = 1.0
    c.ang_cost_func_type = 3; c.use_cost_hess = True
    c.setTerminalEmphasis(100.0)
    ps.passes[0].reg.reg_init = 1e-6; ps.passes[0].reg.reg_max = 1e30
    ps.passes[0].reg.reg_scale = 1.6
    ps.passes[0].linesearch.max_iters = 24
    ps.passes[0].linesearch.beta1 = 1e-10; ps.passes[0].linesearch.beta2 = 5000.0
    return ps


def run_case(name, sat_fn, time_s=1000.0, dt=10.0, omega0=0.01, use_spike=True):
    global call_log
    call_log = []
    iter_counter[0] = 0

    ps = build(dt=dt)
    sat = sat_fn(ps)
    x0 = np.hstack([[omega0, omega0, omega0], [1, 0, 0, 0], np.zeros(sat.numRW)])
    r0 = np.array([7e6, 0, 0]); v0 = np.array([0, 7.5e3, 0])
    jtime = np.array([0.22, 0.22 + time_s/(36525*86400)])
    qg = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
    cfg = None
    if use_spike:
        cfg = {"start_at_iter": 2, "max_intervention_iters": 20, "blend_len": 30,
               "goal_switch_buffer": 15, "min_consecutive": 7, "exit_fudge": 2.0,
               "min_prior_decrease_knots": 5, "min_spike_ratio": 2.0,
               "kp_q": 0.3, "kd_w": 2.0, "rw_scale": 0.0,
               "omega_max": 0.30, "verbose": False}
    t0 = time.time()
    try:
        X, U, stop, snaps, *_ = trajOpt(
            ps, sat, x0, r0, v0, jtime,
            np.tile(qg[:, None], (1, 2)),
            np.array([[1,1],[0,0],[0,0]], dtype=float),
            debug=True, spike_removal_cfg=cfg,
        )
    except Exception as e:
        stop = f"EXCEPTION: {type(e).__name__}"
        snaps = []

    wall = time.time() - t0

    # Summarize reg trajectory
    bp_calls = [c for c in call_log if c["phase"] == "BP"]
    fp_calls = [c for c in call_log if c["phase"] == "FP"]
    bp_fails = [c for c in bp_calls if not c["ok"]]
    fp_fails = [c for c in fp_calls if not c["ok"]]
    regs = [c["reg"] for c in bp_calls if c["reg"] is not None]

    print(f"\n=== {name} ===")
    print(f"  wall={wall:.1f}s  stop={stop.split(':')[0] if ':' in stop else stop}")
    print(f"  total BP calls: {len(bp_calls)}  ({len(bp_fails)} fails)")
    print(f"  total FP calls: {len(fp_calls)}  ({len(fp_fails)} fails)")
    print(f"  reg trajectory: min={min(regs):.2e}  max={max(regs):.2e}  "
          f"median={np.median(regs):.2e}  mean={np.mean(regs):.2e}")

    # Show reg evolution sampled through run (show every 5% of iterations)
    total_iters = iter_counter[0]
    if total_iters > 0:
        print(f"  reg@iter: ", end="")
        checkpoints = [0, total_iters//10, total_iters//4, total_iters//2,
                       3*total_iters//4, 9*total_iters//10, total_iters-1]
        for cp in checkpoints:
            # Find first BP call at this iter
            matches = [c for c in bp_calls if c["iter"] == cp]
            if matches:
                print(f"it{cp}={matches[0]['reg']:.1e}", end=" ")
        print()

    # Look at the LAST several BP calls (just before reg_exceeded / end)
    print(f"  last 10 BP calls:")
    for c in bp_calls[-10:]:
        print(f"    iter{c['iter']:4d}  reg={c['reg']:.2e}  {'OK' if c['ok'] else 'FAIL'}")


# ------------------------- Run selected cases -------------------------
print("="*100)
print("REG TRACE DIAGNOSTIC")
print("="*100)
run_case("3+1 baseline   NO spike", create_3_1)
run_case("3+1 baseline   w/spike",  create_3_1, use_spike=True)
run_case("0+3 RW-only    NO spike", create_0_3)
run_case("0+3 RW-only    w/spike",  create_0_3, use_spike=True)
run_case("3+1 dt=30 3000 w/spike",  create_3_1, time_s=3000.0, dt=30.0, use_spike=True)
run_case("3+1 dt=30 3000 NO spike", create_3_1, time_s=3000.0, dt=30.0, use_spike=False)
run_case("3+1 omega=5x   w/spike",  create_3_1, omega0=0.05, use_spike=True)
run_case("3+1 omega=5x   NO spike", create_3_1, omega0=0.05, use_spike=False)
