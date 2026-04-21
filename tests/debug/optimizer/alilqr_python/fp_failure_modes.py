"""Run failing cases with SALTRO_FP_VERBOSE=1 to categorize FP failure modes."""
import sys, os, subprocess, re, collections
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRIVER = HERE / "_fp_driver.py"

DRIVER_CODE = r"""
import sys, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from trajOpt import trajOpt
from sat_3_1_hybrid import create_satellite as create_3_1
from sat_0_3_rw     import create_satellite as create_0_3

case = sys.argv[1]

ps = saltro_py.PlannerSettings()
ps.init_traj.initcontroller = 1; ps.num_passes = 1
ps.passes[0].ilqr.cost_tol = 1e-6
ps.passes[0].ilqr.grad_tol = 0.0
ps.passes[0].auglag.constraint_tol = 1e-3
c = ps.passes[0].cost
c.control_mult = 1.0
c.mtq_control_weight = 1e-1; c.rw_control_weight = 1.0
c.ang_cost_func_type = 3; c.use_cost_hess = True
ps.passes[0].reg.reg_init = 1e-6; ps.passes[0].reg.reg_max = 1e30
ps.passes[0].reg.reg_scale = 1.6
ps.passes[0].linesearch.max_iters = 24
ps.passes[0].linesearch.beta1 = 1e-10; ps.passes[0].linesearch.beta2 = 5000.0
# Short run — we just want FP-failure samples, not full convergence.
ps.passes[0].ilqr.max_iters = 20
ps.passes[0].auglag.max_outer_iters = 2

if case == "dt30":
    ps.passes[0].dt = 30.0
    sat = create_3_1(ps); time_s = 3000.0
    c.angle = 1e4; c.ang_vel = 1e2; c.setTerminalEmphasis(100.0)
elif case == "rw":
    ps.passes[0].dt = 10.0
    sat = create_0_3(ps); time_s = 1000.0
    c.angle = 1e4; c.ang_vel = 1e2; c.setTerminalEmphasis(100.0)
elif case == "baseline":
    ps.passes[0].dt = 10.0
    sat = create_3_1(ps); time_s = 1000.0
    c.angle = 1e4; c.ang_vel = 1e2; c.setTerminalEmphasis(100.0)
elif case == "omega5x":
    ps.passes[0].dt = 10.0
    sat = create_3_1(ps); time_s = 1000.0
    c.angle = 1e4; c.ang_vel = 1e2; c.setTerminalEmphasis(100.0)

omega0 = 0.05 if case == "omega5x" else 0.01
x0 = np.hstack([[omega0, omega0, omega0], [1, 0, 0, 0], np.zeros(sat.numRW)])
r0 = np.array([7e6, 0, 0]); v0 = np.array([0, 7.5e3, 0])
jtime = np.array([0.22, 0.22 + time_s/(36525*86400)])
qg = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
qgoal = np.tile(qg[:, None], (1, 2))
bs = np.array([[1,1],[0,0],[0,0]], dtype=float)

try:
    X, U, stop, *_ = trajOpt(ps, sat, x0, r0, v0, jtime, qgoal, bs,
                              debug=True, spike_removal_cfg=None)
    print(f"STOP: {stop}", file=sys.stderr)
except Exception as e:
    print(f"EXCEPTION: {type(e).__name__}: {e}", file=sys.stderr)
"""

DRIVER.write_text(DRIVER_CODE)

CASES = ["baseline", "dt30", "rw", "omega5x"]

for case in CASES:
    print(f"\n{'='*100}\nCASE: {case}\n{'='*100}")
    env = dict(os.environ, SALTRO_FP_VERBOSE="1",
               PYTHONPATH=str(Path(__file__).resolve().parents[3] / "build"))
    try:
        res = subprocess.run(
            ["/mnt/c/Users/LV - Patrick McKeen/saltro/venv/bin/python",
             "-u", str(DRIVER), case],
            capture_output=True, text=True, env=env, timeout=60)
    except subprocess.TimeoutExpired:
        print("  TIMEOUT")
        continue

    fp_lines = [l for l in res.stderr.splitlines() if l.startswith("[FP]")]
    # Categorize
    rollout_fails = [l for l in fp_lines if "rollout_fail" in l]
    ls_rejects   = [l for l in fp_lines if "ls_reject" in l]
    all_failed   = [l for l in fp_lines if "ALL TRIALS FAILED" in l]

    rollout_reasons = collections.Counter()
    for l in rollout_fails:
        m = re.search(r"reason=(\S+)", l)
        if m: rollout_reasons[m.group(1)] += 1

    # z-ratio distribution
    zvals = []
    for l in ls_rejects:
        m = re.search(r"z=([-e\d\.inf]+)", l)
        if m:
            try: zvals.append(float(m.group(1)))
            except: pass

    print(f"  total FP ALL-FAILED events : {len(all_failed)}")
    print(f"  total rollout failures     : {len(rollout_fails)}")
    for r, n in rollout_reasons.most_common():
        print(f"     - {r}: {n}")
    print(f"  total linesearch rejects   : {len(ls_rejects)}")
    if zvals:
        import statistics
        z_neg = sum(1 for z in zvals if z < 0)
        z_small = sum(1 for z in zvals if 0 <= z < 1e-10)
        z_big   = sum(1 for z in zvals if z > 5000)
        z_finite = [z for z in zvals if -1e20 < z < 1e20]
        print(f"  z-distribution: N={len(zvals)}  negative={z_neg}  tiny(<1e-10)={z_small}  huge(>5000)={z_big}")
        if z_finite:
            print(f"     finite z range: [{min(z_finite):.2e}, {max(z_finite):.2e}]  median={statistics.median(z_finite):.2e}")

    # Print last 8 FP log lines to see endgame
    print("  last FP trials:")
    for l in fp_lines[-8:]:
        print(f"    {l[:180]}")

    # Print exit reason
    for l in res.stderr.splitlines()[-5:]:
        if "STOP:" in l or "EXCEPTION:" in l:
            print(f"  {l}")
