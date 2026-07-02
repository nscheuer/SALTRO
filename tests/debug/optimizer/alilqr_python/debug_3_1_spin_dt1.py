"""Paper-2 (P2.2) planner-discovered SPINNING maneuver via Python AL-iLQR.

Matches the corresponding alilqr_cpp debug case, but routes the solve through
the Python AL-iLQR path and displays the standard Python debug viewer.
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

J_COM = np.array([
    [3.136490806e-02, 5.883040000e-05, -6.713613570e-03],
    [5.883040000e-05, 3.409127827e-02, -1.233475600e-04],
    [-6.713613570e-03, -1.233475600e-04, 1.004091997e-02],
])


def create_satellite(plannersettings):
    sat = saltro_py.Satellite(J_COM, plannersettings)
    sat.addMTQ(np.array([1.0, 0.0, 0.0]), 0.2)
    sat.addMTQ(np.array([0.0, 1.0, 0.0]), 0.2)
    sat.addMTQ(np.array([0.0, 0.0, 1.0]), 0.2)
    sat.addRW(np.array([0.0, 0.0, 1.0]), 0.0023, 5.7e-6, 0.0, 0.0036)
    return sat


def create_planner_settings(dt, prop, afc, angle, angle_N, gauss_newton):
    plannersettings = saltro_py.PlannerSettings()

    plannersettings.init_traj.initcontroller = 2
    plannersettings.num_passes = 1

    p = plannersettings.passes[0]
    p.dt = float(dt)
    p.ilqr.max_iters = 100
    p.ilqr.cost_tol = 1e-5
    p.auglag.max_outer_iters = 100
    p.auglag.constraint_tol = 1e-3

    cost = p.cost
    cost.angle = angle
    cost.angle_N = angle_N
    cost.ang_vel = 0.0
    cost.ang_vel_N = 0.0
    cost.ang_vel_mag = 0.0
    cost.ang_vel_mag_N = 0.0
    cost.ang_vel_err_dir = 0.0
    cost.ang_vel_err_dir_N = 0.0
    cost.ang_vel_roll_ratio = 0.1
    cost.ang_vel_err_dir_ratio = 0.0
    cost.control_mult = 1.0
    cost.mtq_control_weight = 1.0
    cost.rw_control_weight = 1.0
    cost.rw_AM_weight = 1e4
    cost.ang_cost_func_type = int(afc)
    cost.use_cost_hess = True
    cost.cost_hess_gauss_newton = bool(gauss_newton)

    p.reg.reg_init = 1e-3
    p.reg.reg_min = 1e-8

    disturbances = plannersettings.disturbances
    disturbances.plan_for_aero = False
    disturbances.plan_for_gg = False
    disturbances.plan_for_srp = False
    disturbances.plan_for_gendist = False
    disturbances.plan_for_resdipole = False
    disturbances.plan_for_prop = True
    disturbances.prop_torque = np.array([prop, 0.0, 0.0])

    return plannersettings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prop", type=float, default=4.0e-5)
    parser.add_argument("--tf", type=float, default=240.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--afc", type=int, default=4, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--angle", type=float, default=1e4)
    parser.add_argument("--angle-N", type=float, default=1e6, dest="angle_N")
    parser.add_argument("--no-gauss-newton", action="store_true")
    args = parser.parse_args()

    plannersettings = create_planner_settings(
        args.dt,
        args.prop,
        args.afc,
        args.angle,
        args.angle_N,
        not args.no_gauss_newton,
    )
    satellite = create_satellite(plannersettings)

    t0 = 0.22
    n_steps = max(1, int(np.ceil(args.tf / args.dt)))
    jtime = np.ascontiguousarray(
        np.linspace(t0, t0 + args.tf * SEC2CENT, n_steps + 1)
    )
    m = jtime.size
    qgoal = np.vstack([np.full(m, np.nan), np.zeros(m), np.zeros(m), np.ones(m)])
    boresight = np.vstack([np.zeros(m), np.zeros(m), np.ones(m)])

    x0 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    r0 = 7000.0e3 * np.array([0.0, np.sqrt(2) / 2, np.sqrt(2) / 2])
    v0 = 8.0e3 * np.array([1.0, 0.0, 0.0])

    X, U, stop_reason, snapshots, transitions, dt, cost_tol, elapsed_time = trajOpt(
        plannersettings,
        satellite,
        x0,
        r0,
        v0,
        jtime,
        qgoal,
        boresight,
        debug=True,
    )

    print(f"AL-iLQR finished: {stop_reason}")
    print(f"Final cost: {snapshots[-1]['J']:.6e}")
    print(f"Elapsed time: {elapsed_time:.3f} seconds")

    from ilqr_viewer import launch_viewer
    launch_viewer(snapshots, transitions, stop_reason, dt, cost_tol)


if __name__ == "__main__":
    main()
