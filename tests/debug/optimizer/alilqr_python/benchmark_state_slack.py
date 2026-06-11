"""Benchmark: state-constraint slack + polish vs baseline AL on hard cases.

Runs saltro_py.alilqr (the C++ outer loop) on a battery of feasible-but-
binding problems where state constraints (wmax, RW momentum) are stressed,
comparing use_state_slack=False against the two-phase slack -> polish solver.

Usage:
    PYTHONPATH=<repo>/build python benchmark_state_slack.py [--quick]
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
import saltro_py

SEC_PER_CENTURY = 36525.0 * 86400.0


def make_settings(dt, wmax, max_outer=20, ilqr_iters=50):
    s = saltro_py.PlannerSettings()
    s.init_traj.initcontroller = 2
    s.num_passes = 1
    s.passes[0].dt = dt
    s.passes[0].ilqr.cost_tol = 1e-5
    s.passes[0].ilqr.max_iters = ilqr_iters

    s.passes[0].auglag.max_outer_iters = max_outer
    s.passes[0].auglag.constraint_tol = 1e-3

    cost = s.passes[0].cost
    cost.angle = 1.0
    cost.ang_vel = 1e1
    cost.ang_vel_mag = 0.0
    cost.ang_vel_err_dir = 0.0
    cost.control_mult = 1.0
    cost.mtq_control_weight = 1e-2
    cost.rw_control_weight = 1.0
    cost.magic_control_weight = 0.0
    cost.rw_AM_weight = 0.0
    cost.rw_stic_weight = 0.0
    cost.RWh_stiction_mult = 0.0
    cost.RWh_ok_mult = 0.0
    cost.angle_N = 0.0
    cost.ang_vel_N = 0.0
    cost.ang_vel_mag_N = 0.0
    cost.ang_vel_err_dir_N = 0.0
    cost.ang_cost_func_type = 3
    cost.use_cost_hess = True

    s.passes[0].reg.reg_init = 1e-6
    s.passes[0].reg.reg_max = 1e10
    s.passes[0].reg.reg_scale = 10.0
    s.passes[0].reg.use_dynamics_hess = False
    s.passes[0].reg.use_constraint_hess = False

    s.passes[0].linesearch.max_iters = 24
    s.passes[0].linesearch.beta1 = 1e-10
    s.passes[0].linesearch.beta2 = 5000.0

    s.constraints.wmax = wmax
    return s


def make_satellite(settings, n_mtq=0, rw_hmax=0.02, n_rw=3):
    J = np.diag([0.067, 0.071, 0.069])
    sat = saltro_py.Satellite(J, settings)
    axes = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
    for i in range(n_mtq):
        sat.addMTQ(axes[i], 0.2)
    for i in range(n_rw):
        sat.addRW(axes[i], 0.001, 1e-5, 0.0, rw_hmax)
    return sat


def quat_about_z(angle_rad):
    return np.array([math.cos(angle_rad / 2.0), 0.0, 0.0, math.sin(angle_rad / 2.0)])


def quat_angle_deg(q_a, q_b):
    d = abs(float(np.dot(q_a, q_b)))
    d = min(1.0, d)
    return math.degrees(2.0 * math.acos(d))


def run_case(settings, sat, x0, tf, dt, goal_quat):
    N = int(tf / dt) + 1
    jtime = 0.22 + np.arange(N) * dt / SEC_PER_CENTURY

    q_goal = np.tile(goal_quat.reshape(4, 1), (1, N))
    boresight = np.tile(np.array([[1.0], [0.0], [0.0]]), (1, N))

    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7.5e3, 0.0])
    ok, R, V, B, S, rho = saltro_py.generate_orbit(r0, v0, jtime.reshape(1, -1), 1, 2, 0, 0, 0)
    assert ok, "orbit generation failed"
    rho = np.asarray(rho, dtype=float).reshape(1, -1)[:, :N]
    R, V, B, S = (np.asarray(a, dtype=float)[:, :N] for a in (R, V, B, S))

    ok, X, U = saltro_py.warm_start(settings, sat, x0, jtime, q_goal, boresight, R, V, B, S, rho)
    assert ok, "warm start failed"
    X = np.asarray(X, dtype=float)[:, :N]
    U = np.asarray(U, dtype=float)[:, :N]

    t0 = time.perf_counter()
    ok, X_out, U_out, status, max_c = saltro_py.alilqr(
        settings, 0, sat, X, U, R, V, B, S, rho, jtime, boresight, q_goal
    )
    elapsed = time.perf_counter() - t0

    X_out = np.asarray(X_out)
    w_max_traj = float(np.max(np.linalg.norm(X_out[0:3, :], axis=0)))
    h_max_traj = float(np.max(np.abs(X_out[7:, :]))) if X_out.shape[0] > 7 else 0.0
    final_err = quat_angle_deg(X_out[3:7, -1], goal_quat)
    # State-constraint violation EXCLUDING the fixed initial knot (which the
    # optimizer cannot repair when x0 itself violates wmax).
    wmax = settings.constraints.wmax
    w_norms_tail = np.linalg.norm(X_out[0:3, 1:], axis=0)
    w_viol_tail = float(np.max(np.maximum(0.0, (w_norms_tail**2 - wmax**2) / wmax**2)))
    return {
        "ok": bool(ok),
        "status": str(status),
        "max_c": float(max_c),
        "time_s": elapsed,
        "final_err_deg": final_err,
        "w_peak": w_max_traj,
        "h_peak": h_max_traj,
        "w_viol_tail": w_viol_tail,
        "X": X_out,
        "U": np.asarray(U_out),
    }


CASES = [
    # name, n_mtq, rw_hmax, wmax, tf, slew_deg, w0, penalty_scale
    ("slew90  wmax=0.015 (mild)",        0, 0.02,   0.015, 200.0, 90.0,  np.zeros(3), 10.0),
    ("slew90  wmax=0.012 (tight)",       0, 0.02,   0.012, 200.0, 90.0,  np.zeros(3), 10.0),
    ("slew90  wmax=0.009 (razor)",       0, 0.02,   0.009, 200.0, 90.0,  np.zeros(3), 10.0),
    ("slew180 wmax=0.018 (tight)",       0, 0.02,   0.018, 300.0, 180.0, np.zeros(3), 10.0),
    ("slew180 wmax=0.013 (razor)",       0, 0.02,   0.013, 300.0, 180.0, np.zeros(3), 10.0),
    ("slew90  hmax=0.0012 (h binding)",  0, 0.0012, 0.05,  200.0, 90.0,  np.zeros(3), 10.0),
    ("slew90  wmax=0.012 hot-start",     0, 0.02,   0.012, 200.0, 90.0,  np.array([0.0095, -0.003, 0.0]), 10.0),
    ("3+3 slew90 wmax=0.012 (tight)",    3, 0.02,   0.012, 200.0, 90.0,  np.zeros(3), 10.0),
    ("3+3 slew180 wmax=0.013 (razor)",   3, 0.02,   0.013, 300.0, 180.0, np.zeros(3), 10.0),
    # --- escalation: genuinely hard / pathological ---
    # Infeasible margin: 90 deg in 200 s needs ~0.0079 rad/s average.
    ("slew90  wmax=0.007 (infeasible)",  0, 0.02,   0.007, 200.0, 90.0,  np.zeros(3), 10.0),
    # Initial state violates wmax 2x: the k=0 violation is unfixable, so the
    # baseline ramps mu against it forever; judge by w_viol_tail (k >= 1).
    ("detumble |w0|=2*wmax + slew90",    0, 0.02,   0.012, 300.0, 90.0,  np.array([0.014, -0.014, 0.014]), 10.0),
    # Long horizon, tight everything.
    ("3+3 long slew180 wmax=0.013",      3, 0.02,   0.013, 600.0, 180.0, np.zeros(3), 10.0),
    # Aggressive penalty ramp: mu blows up 30x per outer iter.
    ("3+3 slew180 razor ramp=30",        3, 0.02,   0.013, 300.0, 180.0, np.zeros(3), 30.0),
    ("slew90 razor ramp=30",             0, 0.02,   0.009, 200.0, 90.0,  np.zeros(3), 30.0),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="first 3 cases only")
    parser.add_argument("--slack-rho", type=float, default=50.0)
    parser.add_argument("--slack-off-tol", type=float, default=0.02)
    parser.add_argument("--slack-stall-iters", type=int, default=0,
                        help="opt-in stall fallback (0 = disabled, the C++ default)")
    parser.add_argument("--max-outer", type=int, default=20)
    args = parser.parse_args()

    cases = CASES[:3] if args.quick else CASES

    header = (
        f"{'case':36s} {'mode':9s} {'ok':3s} {'status':20s} "
        f"{'max_c':>10s} {'t[s]':>7s} {'err[deg]':>9s} {'w_peak':>8s} {'h_peak':>9s} {'w_viol@k>0':>11s}"
    )
    print(header)
    print("-" * len(header))

    for name, n_mtq, rw_hmax, wmax, tf, slew_deg, w0, penalty_scale in cases:
        for mode in ("baseline", "slack"):
            settings = make_settings(dt=10.0, wmax=wmax, max_outer=args.max_outer)
            settings.passes[0].auglag.penalty_scale = penalty_scale
            if mode == "slack":
                aug = settings.passes[0].auglag
                aug.use_state_slack = True
                aug.slack_rho = args.slack_rho
                aug.slack_off_tol = args.slack_off_tol
                aug.slack_stall_iters = args.slack_stall_iters
            sat = make_satellite(settings, n_mtq=n_mtq, rw_hmax=rw_hmax)

            x0 = np.zeros(sat.stateDim)
            x0[0:3] = w0
            x0[3:7] = np.array([1.0, 0.0, 0.0, 0.0])

            goal = quat_about_z(math.radians(slew_deg))
            try:
                r = run_case(settings, sat, x0, tf, 10.0, goal)
            except Exception as exc:  # noqa: BLE001 - benchmark must keep going
                print(f"{name:36s} {mode:9s} EXC {type(exc).__name__}: {exc}")
                continue

            print(
                f"{name:36s} {mode:9s} {'Y' if r['ok'] else 'N':3s} {r['status']:20s} "
                f"{r['max_c']:10.4g} {r['time_s']:7.2f} {r['final_err_deg']:9.3f} "
                f"{r['w_peak']:8.4f} {r['h_peak']:9.5f} {r['w_viol_tail']:11.4g}"
            )


if __name__ == "__main__":
    main()
