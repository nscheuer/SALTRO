"""Paper-2 (P2.2) planner-discovered SPINNING maneuver — self-contained.

From a true cold start (omega = 0, q = identity) under a body-fixed off-axis
propulsion torque, the planner must DISCOVER a spinning trajectory: spinning
the body about its pointing axis turns the body-fixed disturbance into an
inertial vector that averages to zero over a rotation, so mean pointing holds
with near-zero steady-state actuator effort.

This is the end-to-end demonstration of three SALTRO features that must all be
present in the build this runs against:
  - vec-pointing 2-DOF cost + Gauss-Newton PSD Hessian (cost_hess_gauss_newton).
    Without GN the vec Hessian goes indefinite and the backward pass fails.
  - prop-disturbance application in the dynamics. Without it the planner cannot
    see the disturbance and there is nothing to spin to reject.
  - the backward pass linearizing with settings.disturbances (so A_k/B_k agree
    with the disturbed forward rollout).

Fully self-contained: builds the BeaverCube2 3+1 satellite inline, sets every
cost/disturbance knob through saltro_py directly, resolves saltro_py from this
worktree's own build/ relative to this file. No GenADCS import, no absolute
paths.

ang_cost_func_type = 4 ((1-c)^2, f''=2) is the documented default for the
GN-PSD Hessian. On a build where afc=4 has been removed (PR #53), pass
`--afc 1 --angle 2e4 --angle-N 2e6` (the numerically identical migration).

Usage:
    python debug_3_1_spin_dt1.py
    python debug_3_1_spin_dt1.py --afc 1 --angle 2e4 --angle-N 2e6   # post-#53
    python debug_3_1_spin_dt1.py --prop 3e-4 --tf 500                # hard case
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))

import saltro_py  # noqa: E402
from final_viewer import plot_final_trajectory  # noqa: E402

# Hundredths-of-a-Julian-century per second (TimeConstants.sec2cent), inlined
# so the script needs nothing outside saltro_py.
SEC2CENT = 1.0 / (36525.0 * 86400.0)

# BeaverCube2 principal+products inertia (kg m^2), 3+1 actuator set.
J_COM = np.array([
    [3.136490806e-02,  5.883040000e-05, -6.713613570e-03],
    [5.883040000e-05,  3.409127827e-02, -1.233475600e-04],
    [-6.713613570e-03, -1.233475600e-04, 1.004091997e-02],
])


def create_satellite(plannersettings):
    """BeaverCube2 3 MTQ + 1 RW. RW is on +z, the boresight / spin axis, so the
    pointing-direction-locked spin averages out the body +x prop torque."""
    sat = saltro_py.Satellite(J_COM, plannersettings)
    sat.addMTQ(np.array([1.0, 0.0, 0.0]), 0.2)
    sat.addMTQ(np.array([0.0, 1.0, 0.0]), 0.2)
    sat.addMTQ(np.array([0.0, 0.0, 1.0]), 0.2)
    # addRW(axis, u_max, J_wheel, h0, h_max)
    sat.addRW(np.array([0.0, 0.0, 1.0]), 0.0023, 5.7e-6, 0.0, 0.0036)
    return sat


def create_planner_settings(dt, prop, afc, angle, angle_N, gauss_newton):
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = 2  # integrated B-dot warm start

    ps.num_passes = 1
    p = ps.passes[0]
    p.dt = float(dt)
    p.ilqr.max_iters = 100
    p.ilqr.cost_tol = 1e-5
    p.auglag.max_outer_iters = 100
    p.auglag.constraint_tol = 1e-3

    c = p.cost
    c.angle = angle
    c.angle_N = angle_N
    c.ang_vel = 0.0          # NO cost on body-rate magnitude: spinning is free
    c.ang_vel_N = 0.0
    c.ang_vel_mag = 0.0
    c.ang_vel_mag_N = 0.0
    c.ang_vel_err_dir = 0.0  # no Lyapunov crossterm (interferes with the spin)
    c.ang_vel_err_dir_N = 0.0
    c.ang_vel_roll_ratio = 0.1       # down-weight roll-about-pointing-axis rate
    c.ang_vel_err_dir_ratio = 0.0
    c.control_mult = 1.0
    c.mtq_control_weight = 1.0
    c.rw_control_weight = 1.0
    c.rw_AM_weight = 1e4     # curvature along the wheel-momentum axis (GN angle
                             # Hessian only provides curvature along dc/dtheta)
    c.ang_cost_func_type = int(afc)
    c.use_cost_hess = True
    c.cost_hess_gauss_newton = bool(gauss_newton)

    p.reg.reg_init = 1e-3
    p.reg.reg_min = 1e-8

    # Body-fixed off-axis prop torque on +x; the disturbance to be spun out.
    d = ps.disturbances
    d.plan_for_aero = False
    d.plan_for_gg = False
    d.plan_for_srp = False
    d.plan_for_gendist = False
    d.plan_for_resdipole = False
    d.plan_for_prop = True
    d.prop_torque = np.array([prop, 0.0, 0.0])
    return ps


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prop", type=float, default=4.0e-5,
                    help="body +x prop torque (N m). 4e-5 = paper default.")
    ap.add_argument("--tf", type=float, default=240.0)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--afc", type=int, default=4, choices=[0, 1, 2, 3, 4],
                    help="ang_cost_func_type. 4 = paper; use 1 on PR #53 builds.")
    ap.add_argument("--angle", type=float, default=1e4)
    ap.add_argument("--angle-N", type=float, default=1e6)
    ap.add_argument("--no-gauss-newton", action="store_true")
    args = ap.parse_args()
    use_gn = not args.no_gauss_newton

    ps = create_planner_settings(args.dt, args.prop, args.afc,
                                 args.angle, args.angle_N, use_gn)
    sat = create_satellite(ps)

    # Constant ECI vec goal: pin body +z (boresight, RW axis) on ECI +z.
    # vec mode: q_goal row 0 = NaN, rows 1..3 = target ECI direction.
    t0 = 0.22
    n_steps = max(1, int(np.ceil(args.tf / args.dt)))
    jtime = np.ascontiguousarray(
        np.linspace(t0, t0 + args.tf * SEC2CENT, n_steps + 1))
    M = jtime.size
    vec_goal = np.vstack([np.full(M, np.nan), np.zeros(M), np.zeros(M), np.ones(M)])
    boresight = np.vstack([np.zeros(M), np.zeros(M), np.ones(M)])

    # Cold start: omega = 0, q = identity, h_rw = 0.
    x0 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    # Orbit (7000 km, 45-deg plane) — only sets the B-field / GG environment.
    r0 = 7000.0e3 * np.array([0.0, np.sqrt(2) / 2, np.sqrt(2) / 2])
    v0 = 8.0e3 * np.array([1.0, 0.0, 0.0])

    print("SALTRO planner-discovered spin maneuver (P2.2), self-contained")
    print(f"  saltro_py = {saltro_py.__file__}")
    print(f"  prop      = {args.prop:.1e} N m on +x  "
          f"({args.prop*1e6:.1f} uN m), tf={args.tf:.0f}s dt={args.dt:.1f}s")
    print(f"  afc={args.afc}, GN={use_gn}, cold start omega=0 q=identity")

    t = time.perf_counter()
    ok, X, U, _K = saltro_py.trajOpt(ps, sat, x0, r0, v0,
                                     jtime, vec_goal, boresight)
    wall = time.perf_counter() - t
    print(f"  -> trajOpt ok = {ok}  ({wall:.1f}s, X{X.shape} U{U.shape})")

    # ---- diagnostics -------------------------------------------------------
    N = X.shape[1]
    t_sec = (jtime - jtime[0]) / SEC2CENT
    omega_deg = X[0:3, :].T * (180.0 / np.pi)
    q_hist = X[3:7, :].T
    h_rw = X[7, :] if X.shape[0] >= 8 else None

    def rot_zaxis(q):
        # body +z expressed in ECI: third column of R(q), q = [w,x,y,z].
        w, x, y, z = q
        return np.array([2 * (x * z + w * y),
                         2 * (y * z - w * x),
                         1 - 2 * (x * x + y * y)])

    bz_eci = np.array([rot_zaxis(q_hist[i]) for i in range(N)])
    pe_deg = np.degrees(np.arccos(np.clip(bz_eci[:, 2], -1.0, 1.0)))

    last = slice(-max(1, int(30 / max(args.dt, 1e-6))), None)
    om_mean = omega_deg[last, :].mean(axis=0)
    pe_mean = float(pe_deg[last].mean())
    print()
    print(f"  steady-state body rate (last 30 s):      {om_mean.round(2)} deg/s")
    print(f"  steady-state pointing error (last 30 s): {pe_mean:.2f} deg")
    if h_rw is not None:
        print(f"  final stored RW momentum:                {h_rw[-1]*1000:.2f} mN m s")
    if abs(om_mean[2]) > 1.0:
        print(f"  -> DISCOVERED spin about body +z: {om_mean[2]:+.2f} deg/s")
    else:
        print(f"  -> no significant steady-state spin (|omega_z| < 1 deg/s)")
    print(f"  all finite: {bool(np.all(np.isfinite(X)) and np.all(np.isfinite(U)))}")

    if not ok:
        raise RuntimeError("trajOpt did not converge")

    plot_final_trajectory(
        X,
        U,
        ps.passes[0].dt,
        satellite=sat,
        q_goal=vec_goal,
        boresight_body=boresight,
        jtime=jtime,
        title=f"P2.2 spin (ok={ok}, afc={args.afc}, GN={use_gn})",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
