"""Reproducibility diagnostic: run the same MTQ-only scenario multiple times
and check whether solver results are deterministic.

Scenario chosen: omega_10x (wide_test scenario 13) — one of the cases
showing large PE variation between principled (49°) and regfloor (125°) runs.

If results are identical across runs: solver is deterministic; the difference
between principled and regfloor is a real effect of the config change
(e.g., the reg snap-to-zero discontinuity).

If results vary: solver has run-to-run nondeterminism (likely from
time-seeded RNG in ExcitationController), and a single wide run is not
enough evidence of config effects.
"""
import sys, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from sat_3_0_mtq import create_satellite
from trajOpt import trajOpt

qg = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])   # 90° slew about z
x0 = np.array([0.10, 0.10, 0.10, 1, 0, 0, 0])        # omega_10x: ω0 = 0.10 rad/s per axis
r0 = np.array([7e6, 0, 0]); v0 = np.array([0, 7.5e3, 0])
jtime = np.array([0.22, 0.22 + 1000.0 / (36525.0 * 86400.0)])
qgoal = np.tile(qg[:, None], (1, 2))
bs = np.array([[1, 1], [0, 0], [0, 0]], dtype=float)


def build_ps(reg_init):
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = 1
    ps.num_passes = 1
    ps.passes[0].dt = 10.0
    ps.passes[0].ilqr.cost_tol = 1e-6
    ps.passes[0].ilqr.max_iters = 200
    ps.passes[0].ilqr.grad_tol = 0.0
    ps.passes[0].auglag.max_outer_iters = 30
    ps.passes[0].auglag.constraint_tol = 1e-3
    c = ps.passes[0].cost
    c.angle = 1e4; c.ang_vel = 1e2
    c.control_mult = 1.0; c.mtq_control_weight = 0.1
    c.ang_cost_func_type = 3; c.use_cost_hess = True
    c.setTerminalEmphasis(100.0)
    for a in ["aero", "gg", "srp", "prop", "gendist", "resdipole"]:
        setattr(ps.disturbances, "plan_for_" + a, False)
    ps.passes[0].reg.reg_init = reg_init
    ps.passes[0].reg.reg_max = 1e30
    ps.passes[0].reg.reg_scale = 1.6
    ps.passes[0].linesearch.max_iters = 24
    ps.passes[0].linesearch.beta1 = 1e-10
    ps.passes[0].linesearch.beta2 = 5000.0
    return ps


def pe_profile(X, qg):
    return np.array([2 * np.degrees(np.arccos(min(abs(float(np.dot(X[3:7, k], qg))), 1)))
                     for k in range(X.shape[1])])


spike_cfg = {
    "start_at_iter": 2, "max_intervention_iters": 20,
    "blend_len": 30, "goal_switch_buffer": 15, "min_consecutive": 7,
    "exit_fudge": 2.0, "min_prior_decrease_knots": 5, "min_spike_ratio": 2.0,
    "kp_q": 0.3, "kd_w": 2.0, "rw_scale": -1.0, "omega_max": 0.30, "verbose": False,
}

N_TRIALS = 3
for reg_init in (0.0, 1e-12):
    print(f"\n=== reg_init = {reg_init} ===")
    results = []
    for trial in range(N_TRIALS):
        ps = build_ps(reg_init)
        sat = create_satellite(ps)
        X, U, stop, snaps, trans, dt_val, ctol, elapsed = trajOpt(
            ps, sat, x0.copy(), r0, v0, jtime, qgoal, bs,
            debug=True, spike_removal_cfg=spike_cfg,
        )
        pe = pe_profile(X, qg)
        print(f"  trial {trial}: stop={stop}  it={len(snaps)}  "
              f"PE_fin={pe[-1]:7.3f}°  PE_mean={pe.mean():6.1f}°  t={elapsed:5.1f}s")
        results.append((stop, len(snaps), pe[-1], pe.mean()))

    # Check reproducibility
    unique = set(results)
    if len(unique) == 1:
        print("  → DETERMINISTIC (all trials identical)")
    else:
        print(f"  → NONDETERMINISTIC ({len(unique)} distinct outcomes)")
