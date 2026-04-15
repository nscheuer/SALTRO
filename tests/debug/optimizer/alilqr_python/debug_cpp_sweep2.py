"""Sweep 2: focus on getting antispike to sub-degree."""
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


def make_settings(spike_enabled, angle, ang_vel, angle_N, ang_vel_N,
                  mtq_w, max_iters, max_outer, use_hess=True):
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = 1
    ps.num_passes = 1
    ps.passes[0].dt = 10.0
    ps.passes[0].ilqr.cost_tol = 1e-6  # tighter convergence
    ps.passes[0].ilqr.max_iters = max_iters
    ps.passes[0].auglag.max_outer_iters = max_outer
    ps.passes[0].auglag.constraint_tol = 1e-3

    cost = ps.passes[0].cost
    cost.angle = angle
    cost.ang_vel = ang_vel
    cost.control_mult = 1.0
    cost.mtq_control_weight = mtq_w
    cost.rw_control_weight = 1.0
    cost.angle_N = angle_N
    cost.ang_vel_N = ang_vel_N
    cost.ang_cost_func_type = 3
    cost.use_cost_hess = use_hess

    for a in ['aero', 'gg', 'srp', 'prop', 'gendist', 'resdipole']:
        setattr(ps.disturbances, f'plan_for_{a}', False)

    ps.passes[0].reg.reg_init = 1e-6
    ps.passes[0].reg.reg_max = 1e10
    ps.passes[0].reg.reg_scale = 10.0

    ps.passes[0].linesearch.max_iters = 24
    ps.passes[0].linesearch.beta1 = 1e-10
    ps.passes[0].linesearch.beta2 = 5000.0

    if spike_enabled:
        sr = ps.passes[0].spike_removal
        sr.enabled = True
        sr.start_at_iter = 2
        sr.max_intervention_iters = 20
        sr.blend_len = 30
        sr.goal_switch_buffer = 15
        sr.min_consecutive = 7
        sr.exit_fudge = 2.0
        sr.min_prior_decrease_knots = 10
        sr.min_spike_ratio = 3.0
        sr.max_spike_knots = 55
        sr.kp_q = 0.3
        sr.kd_w = 2.0
        sr.rw_scale = 0.0
        sr.omega_max = 0.30
        sr.verbose = False

    return ps


def run_one(spike_enabled, angle, ang_vel, angle_N, ang_vel_N,
            mtq_w, max_iters, max_outer, use_hess=True):
    ps = make_settings(spike_enabled, angle, ang_vel, angle_N, ang_vel_N,
                       mtq_w, max_iters, max_outer, use_hess)
    sat = create_satellite(ps)

    jtime = np.array([0.22, 0.22 + 1000.0 / (36525.0 * 86400.0)])
    qgoal = np.array([
        [np.sqrt(2)/2, np.sqrt(2)/2],
        [0.0, 0.0],
        [0.0, 0.0],
        [np.sqrt(2)/2, np.sqrt(2)/2],
    ])
    boresight = np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    x0 = np.array([0.01, 0.01, 0.01, 1.0, 0.0, 0.0, 0.0, 0.0])
    r0 = np.array([7e6, 0.0, 0.0])
    v0 = np.array([0.0, 7.5e3, 0.0])
    qg = qgoal[:, 0]

    old_stdout = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    sys.stdout.flush()
    t0 = time.time()
    ok, X, U, _K = saltro_py.trajOpt(ps, sat, x0, r0, v0, jtime, qgoal, boresight)
    elapsed = time.time() - t0
    os.dup2(old_stdout, 1)
    os.close(devnull)
    os.close(old_stdout)

    pe = np.array([2*np.degrees(np.arccos(min(abs(float(np.dot(X[3:7, k], qg))), 1)))
                   for k in range(X.shape[1])])
    return ok, pe.max(), pe[-1], elapsed


def main():
    configs = [
        # Very high terminal weights
        ("H: term1e5",     1e2, 1e1, 1e5, 1e4, 1e-1, 200, 50),
        ("I: term1e6",     1e2, 1e1, 1e6, 1e5, 1e-1, 200, 50),
        # High stage + high terminal
        ("J: all1e4",      1e4, 1e3, 1e5, 1e4, 1e-2, 200, 50),
        # Defaults with more iters
        ("K: def_long",    1e3, 1e4, 1e4, 1e5, 1e-2, 500, 100),
        # Original with way more iters
        ("L: orig_long",   1e2, 1e1, 1e2, 1e1, 1e-1, 500, 100),
        # Moderate boost
        ("M: mid",         1e3, 1e2, 1e3, 1e2, 1e-1, 200, 50),
    ]

    print(f"{'Config':<18s} {'Mode':<6s} {'OK':<4s} {'MaxPE':>7s} {'FinalPE':>8s} {'Time':>6s}")
    print("-" * 55)
    sys.stdout.flush()

    for label, angle, ang_vel, angle_N, ang_vel_N, mtq_w, mi, mo in configs:
        for spike in [False, True]:
            mode = "anti" if spike else "base"
            ok, max_pe, final_pe, t = run_one(
                spike, angle, ang_vel, angle_N, ang_vel_N, mtq_w, mi, mo
            )
            conv = "Y" if ok else "N"
            print(f"{label:<18s} {mode:<6s} {conv:<4s} {max_pe:7.1f} {final_pe:8.2f} {t:6.1f}s")
            sys.stdout.flush()
        print()
        sys.stdout.flush()


if __name__ == "__main__":
    main()
