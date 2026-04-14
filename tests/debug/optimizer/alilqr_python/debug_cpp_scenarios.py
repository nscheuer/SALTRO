"""Multi-scenario C++ optimizer test: baseline vs antispike.

Tests several slew angles and initial conditions to verify:
  1. Baseline (no spike removal) converges or produces reasonable results
  2. Antispike doesn't break anything on easy scenarios
  3. Antispike helps on hard scenarios with spikes
"""
import os
import sys
import time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from sat_3_1_hybrid import create_satellite


def make_settings(spike_enabled, max_spike_knots=30):
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = 1
    ps.num_passes = 1
    ps.passes[0].dt = 10.0
    ps.passes[0].ilqr.cost_tol = 1e-3
    ps.passes[0].ilqr.max_iters = 20
    ps.passes[0].auglag.max_outer_iters = 10
    ps.passes[0].auglag.constraint_tol = 1e-3

    cost = ps.passes[0].cost
    cost.angle = 1e2
    cost.ang_vel = 1e1
    cost.control_mult = 1.0
    cost.mtq_control_weight = 1e-1
    cost.rw_control_weight = 1.0
    cost.angle_N = 1e2
    cost.ang_vel_N = 1e1
    cost.ang_cost_func_type = 3
    cost.use_cost_hess = False

    ps.disturbances.plan_for_aero = False
    ps.disturbances.plan_for_gg = False
    ps.disturbances.plan_for_srp = False
    ps.disturbances.plan_for_prop = False
    ps.disturbances.plan_for_gendist = False
    ps.disturbances.plan_for_resdipole = False

    ps.passes[0].reg.reg_init = 1e-6
    ps.passes[0].reg.reg_max = 1e10
    ps.passes[0].reg.reg_scale = 10.0

    ps.passes[0].linesearch.max_iters = 20
    ps.passes[0].linesearch.beta1 = 1e-10
    ps.passes[0].linesearch.beta2 = 5000.0

    if spike_enabled:
        sr = ps.passes[0].spike_removal
        sr.enabled = True
        sr.start_at_iter = 3
        sr.max_intervention_iters = 5
        sr.blend_len = 20
        sr.goal_switch_buffer = 15
        sr.min_consecutive = 7
        sr.exit_fudge = 2.0
        sr.min_prior_decrease_knots = 10
        sr.min_spike_ratio = 3.0
        sr.max_spike_knots = max_spike_knots
        sr.kp_q = 0.3
        sr.kd_w = 2.0
        sr.rw_scale = 0.0
        sr.omega_max = 0.30
        sr.verbose = True

    return ps


def run_trajopt(label, ps, x0, r0, v0, jtime, qgoal, boresight):
    sat = create_satellite(ps)
    old_stdout = os.dup(1)
    log_path = Path(__file__).parent / f"cpp_{label.replace(' ','_')}_log.txt"
    log_fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(log_fd, 1)
    sys.stdout.flush()
    t0 = time.time()
    ok, X, U, _K = saltro_py.trajOpt(ps, sat, x0, r0, v0, jtime, qgoal, boresight)
    elapsed = time.time() - t0
    os.dup2(old_stdout, 1)
    os.close(log_fd); os.close(old_stdout)
    spike_lines = []
    for line in open(log_path):
        if "SpikeRemoval" in line and "skipping" not in line:
            spike_lines.append(line.strip())
    return ok, X, U, elapsed, spike_lines


def pe_deg(X, qg):
    N = X.shape[1]
    errs = np.zeros(N)
    for k in range(N):
        q = X[3:7, k]
        dot = np.clip(abs(float(np.dot(q, qg))), 0, 1)
        errs[k] = 2.0 * np.degrees(np.arccos(dot))
    return errs


def run_scenario(name, x0, qgoal, jtime, boresight, r0, v0):
    print(f"\n{'='*70}")
    print(f"SCENARIO: {name}")
    print(f"{'='*70}")
    qg = qgoal[:, 0]

    results = {}
    for mode, spike_en in [("baseline", False), ("antispike", True)]:
        label = f"{name}_{mode}"
        ps = make_settings(spike_en)
        ok, X, U, t, spikes = run_trajopt(label, ps, x0, r0, v0, jtime, qgoal, boresight)
        N = X.shape[1]
        pe = pe_deg(X, qg)
        conv = "CONVERGED" if ok else "not converged"
        for s in spikes:
            print(f"  [{mode}] {s}")
        print(f"  {mode:10s}: {conv}  N={N}  max_pe={pe.max():.1f}°  "
              f"final_pe={pe[-1]:.1f}°  time={t:.1f}s")
        results[mode] = {"ok": ok, "pe": pe, "t": t, "X": X}

    # Did antispike hurt?
    base_final = results["baseline"]["pe"][-1]
    anti_final = results["antispike"]["pe"][-1]
    if anti_final > base_final + 10:
        print(f"  ** WARNING: antispike final_pe ({anti_final:.1f}°) worse than baseline ({base_final:.1f}°)")
    elif anti_final < base_final - 5:
        print(f"  ** BETTER: antispike final_pe ({anti_final:.1f}°) < baseline ({base_final:.1f}°)")
    else:
        print(f"  ** SIMILAR: antispike and baseline within 10°")

    return results


def main():
    r0 = np.array([7e6, 0.0, 0.0])
    v0 = np.array([0.0, 7.5e3, 0.0])

    # Scenario 1: Small 30° slew (easy, shouldn't have spikes)
    print("\n" + "#"*70)
    print("# Testing multiple scenarios")
    print("#"*70)

    # 30-deg slew about Z
    ang = np.radians(30) / 2
    qg_30 = np.array([np.cos(ang), 0, 0, np.sin(ang)])
    jtime_short = np.array([0.22, 0.22 + 500.0 / (36525.0 * 86400.0)])
    run_scenario(
        "30deg_slew",
        x0=np.array([0.01, 0.01, 0.01, 1, 0, 0, 0, 0.0]),
        qgoal=np.tile(qg_30[:, None], (1, 2)),
        jtime=jtime_short,
        boresight=np.array([[1,1],[0,0],[0,0]], dtype=float),
        r0=r0, v0=v0,
    )

    # Scenario 2: 90° slew (the standard test case)
    qg_90 = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
    jtime_med = np.array([0.22, 0.22 + 1000.0 / (36525.0 * 86400.0)])
    run_scenario(
        "90deg_slew",
        x0=np.array([0.01, 0.01, 0.01, 1, 0, 0, 0, 0.0]),
        qgoal=np.tile(qg_90[:, None], (1, 2)),
        jtime=jtime_med,
        boresight=np.array([[1,1],[0,0],[0,0]], dtype=float),
        r0=r0, v0=v0,
    )

    # Scenario 3: 90° slew with larger initial omega (more likely to spike)
    run_scenario(
        "90deg_large_omega",
        x0=np.array([0.05, 0.05, 0.05, 1, 0, 0, 0, 0.0]),
        qgoal=np.tile(qg_90[:, None], (1, 2)),
        jtime=jtime_med,
        boresight=np.array([[1,1],[0,0],[0,0]], dtype=float),
        r0=r0, v0=v0,
    )

    print("\n" + "="*70)
    print("ALL SCENARIOS COMPLETE")


if __name__ == "__main__":
    main()
