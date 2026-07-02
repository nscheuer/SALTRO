"""Paper-2 (P2.2) autonomous SPINNING maneuver via Python AL-iLQR.

Mirrors the corresponding alilqr_cpp 179-deg spin-acquisition case, but routes
the solve through the Python AL-iLQR debug path and displays the standard
Python viewer output.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import saltro_py
from trajOpt import trajOpt


SEC2CENT = 1.0 / (36525.0 * 86400.0)

J_THESIS = np.array([[0.1, 0.0, 0.00013],
                     [0.0, 0.05, -0.00021],
                     [0.00013, -0.00021, 0.005]])
Q0_THESIS = np.array([-0.232, -0.664, -0.234, -0.671])
Q0_THESIS = Q0_THESIS / np.linalg.norm(Q0_THESIS)
U_MTQ_MAX = np.array([0.19, 0.57, 0.57])
U_RW_MAX = 2e-4
PROP = np.array([3e-4, 0.0, 0.0])
DT = 1.0
TF = 500.0


def create_satellite(plannersettings):
    sat = saltro_py.Satellite(J_THESIS, plannersettings)
    sat.addMTQ(np.array([1.0, 0.0, 0.0]), 0.19)
    sat.addMTQ(np.array([0.0, 1.0, 0.0]), 0.57)
    sat.addMTQ(np.array([0.0, 0.0, 1.0]), 0.57)
    sat.addRW(np.array([0.0, 1.0, 0.0]), U_RW_MAX, 2e-6, 0.0, 2e-3)
    return sat


def create_planner_settings(gauss_newton):
    plannersettings = saltro_py.PlannerSettings()
    plannersettings.init_traj.initcontroller = 3
    plannersettings.constraints.wmax = np.deg2rad(60.0)
    plannersettings.constraints.rw_momentum_limit_scale = 0.85

    plannersettings.num_passes = 1
    p = plannersettings.passes[0]
    p.dt = DT
    p.ilqr.max_iters = 150
    p.auglag.max_outer_iters = 35
    p.auglag.penalty_init = 0.01
    p.auglag.penalty_scale = 3.0
    p.auglag.penalty_max = 1e15

    p.reg.reg_init = 1e-3
    p.reg.use_dynamics_hess = True
    p.reg.psd_clip_quu_ddp = True

    c = p.cost
    c.angle = 3e7
    c.angle_N = 1e7
    c.ang_vel = 1e2
    c.ang_vel_N = 1e2
    c.control_mult = 1.0
    c.mtq_control_weight = 1000.0
    c.rw_control_weight = 2.5e7
    c.ang_cost_func_type = 3
    c.use_cost_hess = True
    c.cost_hess_gauss_newton = bool(gauss_newton)
    c.ang_vel_roll_ratio = 0.0
    c.ang_vel_err_dir_ratio = 0.0
    c.rw_AM_weight = 1e4
    c.RWh_ok_mult = 0.5
    c.rw_stic_weight = 0.0
    c.RWh_stiction_mult = 0.05

    d = plannersettings.disturbances
    d.plan_for_aero = False
    d.plan_for_gg = False
    d.plan_for_srp = False
    d.plan_for_gendist = False
    d.plan_for_resdipole = False
    d.plan_for_prop = True
    d.prop_torque = PROP.copy()
    return plannersettings


def grids(dt=DT):
    r_hat0 = np.array([0.0, np.cos(np.deg2rad(51.5)), np.sin(np.deg2rad(51.5))])
    v_hat0 = np.array([1.0, 0.0, 0.0])
    r0 = 6800.0e3 * r_hat0
    v0 = 7.65e3 * v_hat0
    n_rate = np.linalg.norm(v0) / np.linalg.norm(r0)

    t0 = 0.22
    n = int(TF / dt) + 1
    jtime = np.linspace(t0, t0 + TF * SEC2CENT, n)
    t_sec = (jtime - t0) / SEC2CENT

    qg = np.empty((4, n))
    bs = np.empty((3, n))
    for i, ts in enumerate(t_sec):
        v_hat = -np.sin(n_rate * ts) * r_hat0 + np.cos(n_rate * ts) * v_hat0
        qg[0, i] = np.nan
        qg[1:, i] = -v_hat
        bs[:, i] = np.array([0.0, 0.0, 1.0])
    return jtime, qg, bs, r0, v0


def rot_bz_eci(q):
    w, x, y, z = q
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gauss-newton",
        action="store_true",
        help="ablation: GN Hessian instead of full Newton (over-commands the MTQ near the antipode).",
    )
    args = parser.parse_args()

    plannersettings = create_planner_settings(args.gauss_newton)
    satellite = create_satellite(plannersettings)
    jtime, qgoal, boresight, r0, v0 = grids()
    x0 = np.concatenate([np.zeros(3), Q0_THESIS, [0.0]])

    pe0 = np.degrees(np.arccos(np.clip(
        rot_bz_eci(Q0_THESIS) @ (qgoal[1:, 0] / np.linalg.norm(qgoal[1:, 0])),
        -1.0,
        1.0,
    )))
    print("P2.2 autonomous spin — 179 deg acquisition under overwhelming prop (Python AL-iLQR)")
    print(f"  saltro_py = {saltro_py.__file__}")
    print(f"  PE_0 = {pe0:.1f} deg, prop = {PROP[0]*1e6:.0f} uN m body-x, "
          f"tf={TF:.0f}s dt={DT:.1f}s, full_newton={not args.gauss_newton}")

    X, U, stop_reason, snapshots, transitions, dt, cost_tol, elapsed_time = trajOpt(
        plannersettings,
        satellite,
        x0,
        r0,
        v0,
        np.ascontiguousarray(jtime),
        np.ascontiguousarray(qgoal),
        np.ascontiguousarray(boresight),
        debug=True,
    )

    X = np.asarray(X)
    U = np.asarray(U)
    print(f"  -> trajOpt stop_reason = {stop_reason}  "
          f"({elapsed_time:.1f}s, X{X.shape} U{U.shape})")

    n = X.shape[1]
    t_sec = np.arange(n) * dt
    pe = np.array([float(np.degrees(np.arccos(np.clip(
        rot_bz_eci(X[3:7, k]) @ (qgoal[1:, k] / np.linalg.norm(qgoal[1:, k])),
        -1.0,
        1.0,
    )))) for k in range(n)])
    w_deg = X[0:3, :] * 180.0 / np.pi
    h_mnms = X[7, :] * 1000.0
    mtq = float(np.max(np.abs(U[0:3, :].T) / U_MTQ_MAX))
    rw = float(np.max(np.abs(U[3, :])) / U_RW_MAX)
    below1 = np.nonzero(pe < 1.0)[0]
    t_acq = t_sec[below1[0]] if below1.size else float("nan")

    print(f"  PE_fin={pe[-1]:.2f} deg  PE_m30={float(np.mean(pe[-30:])):.2f} deg  "
          f"t(<1deg)={t_acq:.0f} s")
    print(f"  <wz>_m30={float(np.mean(w_deg[2, -30:])):.2f} deg/s  "
          f"|h|max={float(np.abs(h_mnms).max()):.2f} mN m s (limit 2.0, margin 1.7)")
    print(f"  MTQ peak={mtq:.2f}x u_max  RW peak={rw:.2f}x u_max")
    print(f"  Final cost: {snapshots[-1]['J']:.6e}")

    from ilqr_viewer import launch_viewer
    launch_viewer(snapshots, transitions, stop_reason, dt, cost_tol)


if __name__ == "__main__":
    main()
