"""A/B benchmark: dense vs square-root backward pass on the P2.2
planner-discovered spin maneuver (generate_fig_spin.py scenario).

Uses the saltro_py module built from the feat/sqrt-bp-prstack worktree and
the unmodified GenADCS scenario builders from generate_fig_spin.py.

afc=4 was removed by PR #53 (it equals afc=1 with the constant 2 absorbed
into the angle weight), so this runs afc=1 with doubled angle weights —
the documented migration, numerically identical cost.
"""
import sys, os, time, json
import importlib.util

BUILD = "/Users/patrickmckeen/ADCS_wt/saltro-sqrt/build"
GENADCS = "/Users/patrickmckeen/Documents/Generalized_ADCS"
sys.path.insert(0, BUILD)            # saltro_py from the sqrt branch
sys.path.insert(0, GENADCS)          # ADCS package

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import saltro_py
assert saltro_py.__file__.startswith(BUILD), saltro_py.__file__

# Import the real scenario file by path (its main() only runs under __main__).
spec = importlib.util.spec_from_file_location(
    "genfigspin", os.path.join(GENADCS, "papers/Planner/generate_fig_spin.py"))
gfs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gfs)

import ADCS
from ADCS.CONOPS.goallist import GoalList
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.helpers.math_helpers import rot_mat, normalize
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.orbits.orbit import Orbit


def plan(ps, real_sat, x_0, os_0, goal_list, tf, dt, use_sqrt):
    """generate_fig_spin.plan_trajectory with a use_sqrt_bp toggle + timing."""
    t_start = float(os_0.J2000)
    t_end = float(t_start + tf * TimeConstants.sec2cent)
    n_steps = max(1, int(np.ceil(tf / dt)))
    jtime = np.ascontiguousarray(np.linspace(t_start, t_end, n_steps + 1, dtype=np.float64))

    sim_orbit = Orbit(os_0, t_end, dt=dt, use_J2=True, fast=True, verbose=False)
    q_goal = np.empty((4, jtime.size), dtype=np.float64)
    boresight = np.empty((3, jtime.size), dtype=np.float64)
    for i, t_k in enumerate(jtime):
        os_at_t = sim_orbit.get_os(float(t_k))
        ag = goal_list.get_active_goal(float(t_k), time_units="centuries")
        tr, _ = ag.to_ref(os_at_t)
        tr = np.asarray(tr, dtype=np.float64).reshape(4)
        q_goal[:, i] = tr if np.isnan(tr[0]) else normalize(tr)
        boresight[:, i] = np.asarray(real_sat.get_boresight(), dtype=np.float64).reshape(3)

    cpp_settings = ps.to_cpp()
    for p in cpp_settings.passes:
        p.reg.use_sqrt_bp = bool(use_sqrt)

    cpp_sat = saltro_py.Satellite()
    cpp_sat.setInertia(np.asarray(real_sat.J_COM, dtype=np.float64))
    for act in real_sat.actuators:
        if isinstance(act, MTQ):
            cpp_sat.addMTQ(np.asarray(act.axis, dtype=np.float64), float(act.u_max))
    for act in real_sat.actuators:
        if isinstance(act, RW):
            cpp_sat.addRW(np.asarray(act.axis, dtype=np.float64),
                          float(act.u_max), float(act.J), float(act.h), float(act.h_max))
    r0 = np.asarray(os_0.R, dtype=np.float64).reshape(3) * 1.0e3
    v0 = np.asarray(os_0.V, dtype=np.float64).reshape(3) * 1.0e3
    x0_clean = np.asarray(x_0, dtype=np.float64).reshape(-1)

    t0 = time.perf_counter()
    ok, Xs, Us, K = saltro_py.trajOpt(
        cpp_settings, cpp_sat, x0_clean, r0, v0,
        np.ascontiguousarray(jtime),
        np.ascontiguousarray(q_goal),
        np.ascontiguousarray(boresight),
    )
    wall = time.perf_counter() - t0
    return bool(ok), np.asarray(Xs), np.asarray(Us), jtime, wall, real_sat


def metrics(ok, X, U, jtime, wall, real_sat, dt):
    t_sec = (jtime - jtime[0]) * 36525.0 * 86400.0
    N = X.shape[1]
    q_hist = X[3:7, :].T
    bz = np.array([0.0, 0.0, 1.0])
    goal_eci = np.array([0.0, 0.0, 1.0])
    pe = np.array([
        np.degrees(np.arccos(np.clip(np.dot(rot_mat(q_hist[i]) @ bz, goal_eci), -1, 1)))
        for i in range(N)])
    m30 = t_sec >= (t_sec[-1] - 30.0)
    omega = X[0:3, :]
    h_rw = X[7, :] if X.shape[0] >= 8 else None

    mtq_axes, mtq_umax, rw_umax, rw_hmax = [], [], None, None
    for act in real_sat.actuators:
        if isinstance(act, MTQ):
            mtq_umax.append(float(act.u_max))
        elif isinstance(act, RW):
            rw_umax, rw_hmax = float(act.u_max), float(act.h_max)
    n_mtq = len(mtq_umax)
    mtq_sat = float(np.max(np.abs(U[:n_mtq, :]) / np.array(mtq_umax)[:, None])) if U.size else float("nan")
    rw_sat = float(np.max(np.abs(U[n_mtq, :]) / rw_umax)) if (rw_umax and U.shape[0] > n_mtq) else float("nan")
    h_sat = float(np.max(np.abs(h_rw)) / rw_hmax) if (h_rw is not None and rw_hmax) else float("nan")

    return dict(
        ok=ok, wall_s=wall,
        pe_final_deg=float(pe[-1]),
        pe_mean_last30_deg=float(np.mean(pe[m30])),
        pe_max_deg=float(np.max(pe)),
        omega_z_final_dps=float(np.degrees(omega[2, -1])),
        omega_z_max_dps=float(np.degrees(np.max(np.abs(omega[2, :])))),
        mtq_sat=mtq_sat, rw_sat=rw_sat, h_rw_sat=h_sat,
        all_finite=bool(np.all(np.isfinite(X)) and np.all(np.isfinite(U))),
    ), pe, t_sec, omega


def run_case(tf, dt, prop, label, out, penalty_max=None, penalty_scale=None,
             max_outer=None):
    real_sat = gfs.build_3p1_with_prop(prop)
    goal = ADCS.goals.ECI_Goal(np.array([0.0, 0.0, 1.0]))
    t0_cent = 0.22
    goal_list = GoalList({t0_cent: goal})
    x_0 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    os0 = ADCS.Orbital_State(
        ephem=ADCS.Ephemeris(), J2000=t0_cent,
        R=7000.0 * np.array([0, np.sqrt(2) / 2, np.sqrt(2) / 2]),
        V=np.array([8.0, 0.0, 0.0]))

    results = {}
    curves = {}
    for use_sqrt in (False, True):
        # afc=4 -> afc=1 with doubled angle weight (PR #53 migration).
        ps = gfs.make_planner_settings(real_sat, dt=dt, gauss_newton=True,
                                       ang_cost_func_type=1)
        ps.passes[0].cost.angle = 2.0e4
        ps.passes[0].cost.angle_N = 2.0e6
        if penalty_max is not None:
            ps.passes[0].aug_lag.penalty_max = penalty_max
        if penalty_scale is not None:
            ps.passes[0].aug_lag.penalty_scale = penalty_scale
        if max_outer is not None:
            ps.passes[0].aug_lag.max_outer_iters = max_outer
        tag = "sqrt" if use_sqrt else "dense"
        print(f"[{label}] running {tag} ...", flush=True)
        t0 = time.perf_counter()
        try:
            ok, X, U, jtime, wall, _ = plan(ps, real_sat, x_0, os0, goal_list, tf, dt, use_sqrt)
        except RuntimeError as exc:
            wall = time.perf_counter() - t0
            results[tag] = dict(ok=False, wall_s=wall, error=str(exc))
            print(f"[{label}] {tag}: NOT CONVERGED after {wall:.1f}s ({exc})", flush=True)
            continue
        m, pe, t_sec, omega = metrics(ok, X, U, jtime, wall, real_sat, dt)
        results[tag] = m
        curves[tag] = (t_sec, pe, omega)
        print(f"[{label}] {tag}: ok={m['ok']} wall={m['wall_s']:.1f}s "
              f"PE_m30={m['pe_mean_last30_deg']:.2f}deg PEf={m['pe_final_deg']:.2f}deg "
              f"wz_f={m['omega_z_final_dps']:.2f}dps mtq_sat={m['mtq_sat']:.2f} "
              f"h_sat={m['h_rw_sat']:.2f}", flush=True)

    if not curves:
        return results
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for tag, color in (("dense", "tab:blue"), ("sqrt", "tab:orange")):
        if tag not in curves:
            continue
        t_sec, pe, omega = curves[tag]
        axes[0].plot(t_sec, pe, color=color, label=f"{tag} (ok={results[tag]['ok']})")
        axes[1].plot(t_sec, np.degrees(omega[2, :]), color=color, label=tag)
    axes[0].set_ylabel("pointing error [deg]"); axes[0].legend(); axes[0].grid(alpha=.3)
    axes[0].set_title(f"{label}: dense vs sqrt backward pass")
    axes[1].set_ylabel("omega_z [deg/s]"); axes[1].set_xlabel("t [s]"); axes[1].grid(alpha=.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"spin_ab_{label}.png"), dpi=130)
    return results


if __name__ == "__main__":
    out = "/tmp/sqrt_spin_ab"
    cases = sys.argv[1:] or ["default", "hard", "highmu"]
    all_res = {}
    if "default" in cases:
        # Committed P2.2 scenario (generate_fig_spin defaults).
        all_res["p22_default"] = run_case(240.0, 1.0, [4.0e-5, 0.0, 0.0], "p22_default", out)
    if "hard" in cases:
        # SPINNING_MANEUVER diagnosis-level disturbance (known AL-stall case).
        all_res["p22_hard"] = run_case(500.0, 1.0, [3.0e-4, 0.0, 0.0], "p22_hard", out)
    if "highmu" in cases:
        # High-penalty stress: the UPDATE_4 "closest to feasible" AL schedule
        # (penalty_max=1e8, scale=4) — the regime where mu*c^T*c outer
        # products cost the dense recursion half its precision.
        all_res["p22_highmu"] = run_case(500.0, 1.0, [3.0e-4, 0.0, 0.0], "p22_highmu", out,
                                         penalty_max=1e8, penalty_scale=4.0, max_outer=60)
    suffix = "_".join(cases)
    with open(os.path.join(out, f"results_{suffix}.json"), "w") as f:
        json.dump(all_res, f, indent=2)
    print(json.dumps(all_res, indent=2))
