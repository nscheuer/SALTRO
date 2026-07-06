import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from sat_3_3_hybrid import create_satellite
from trajOpt import trajOpt
from ilqr_viewer import launch_viewer


def create_planner_settings():
    plannersettings = saltro_py.PlannerSettings()

    plannersettings.init_traj.initcontroller = 2

    plannersettings.num_passes = 1
    plannersettings.passes[0].dt = 10.0
    plannersettings.passes[0].ilqr.cost_tol = 1e-5
    plannersettings.passes[0].ilqr.max_iters = 20

    plannersettings.passes[0].auglag.max_outer_iters = 10
    plannersettings.passes[0].auglag.constraint_tol = 1e-3

    cost = plannersettings.passes[0].cost
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
    cost.RWh_knee_frac = 0.0
    cost.angle_N = 0.0
    cost.ang_vel_N = 0.0
    cost.ang_vel_mag_N = 0.0
    cost.ang_vel_err_dir_N = 0.0
    cost.ang_cost_func_type = 4
    cost.use_cost_hess = True
    cost.cost_hess_gauss_newton = True

    plannersettings.disturbances.plan_for_aero = False
    plannersettings.disturbances.plan_for_gg = False
    plannersettings.disturbances.plan_for_srp = False
    plannersettings.disturbances.plan_for_prop = False
    plannersettings.disturbances.plan_for_gendist = False
    plannersettings.disturbances.plan_for_resdipole = False

    plannersettings.passes[0].reg.reg_init = 1e-6
    plannersettings.passes[0].reg.reg_max = 1e10
    plannersettings.passes[0].reg.reg_scale = 10.0
    plannersettings.passes[0].reg.use_dynamics_hess = False
    plannersettings.passes[0].reg.use_constraint_hess = False

    plannersettings.passes[0].linesearch.max_iters = 24
    plannersettings.passes[0].linesearch.beta1 = 1e-10
    plannersettings.passes[0].linesearch.beta2 = 5000.0

    return plannersettings


def main():
    plannersettings = create_planner_settings()
    satellite = create_satellite(plannersettings)

    jtime = np.array([0.22, 0.22 + 400 / (36525 * 86400)])
    # Vector pointing version: each goal column is [NaN, x, y, z], where the
    # last three entries are the inertial target direction.
    # This case asks the body +X boresight to point toward inertial -Y.
    # With the identity quaternion, body +X is aligned with inertial +X, so the
    # initial boresight-to-target error is 90 degrees.
    qgoal = np.array([
        [np.nan, np.nan],
        [0.0, 0.0],
        [-1.0, -1.0],
        [0.0, 0.0],
    ])
    boresight = np.array([
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ])

    w0 = np.array([0.01, 0.01, 0.01])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    h0 = np.array([0.0, 0.0, 0.0])
    x0 = np.hstack((w0, q0, h0))

    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7.5e3, 0.0])

    X, U, stop_reason, snapshots, transitions, dt, cost_tol, elapsed_time = trajOpt(
        plannersettings, satellite, x0, r0, v0, jtime, qgoal, boresight, debug=True
    )

    print(f"AL-iLQR finished: {stop_reason}")
    print(f"Final cost: {snapshots[-1]['J']:.6e}")
    print(f"Elapsed time: {elapsed_time:.3f} seconds")

    launch_viewer(snapshots, transitions, stop_reason, dt, cost_tol)


if __name__ == "__main__":
    main()
