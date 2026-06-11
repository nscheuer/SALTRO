import sys
import time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))

import saltro_py
from final_viewer import plot_final_trajectory


def create_planner_settings():
    plannersettings = saltro_py.PlannerSettings()

    plannersettings.init_traj.initcontroller = 0

    plannersettings.num_passes = 1
    plannersettings.passes[0].dt = 10.0
    plannersettings.passes[0].ilqr.cost_tol = 1e-5
    plannersettings.passes[0].ilqr.max_iters = 20

    plannersettings.passes[0].auglag.max_outer_iters = 10
    plannersettings.passes[0].auglag.constraint_tol = 1e-3

    cost = plannersettings.passes[0].cost
    cost.angle = 1e2
    cost.ang_vel = 1e1
    cost.ang_vel_mag = 0.0
    cost.ang_vel_err_dir = 0.0
    cost.control_mult = 1.0
    cost.mtq_control_weight = 0.0
    cost.rw_control_weight = 0.0
    cost.magic_control_weight = 1e-2
    cost.rw_AM_weight = 0.0
    cost.rw_stic_weight = 0.0
    cost.RWh_stiction_mult = 0.0
    cost.RWh_ok_mult = 0.0
    cost.angle_N = 1e2
    cost.ang_vel_N = 1e1
    cost.ang_vel_mag_N = 0.0
    cost.ang_vel_err_dir_N = 0.0
    cost.ang_cost_func_type = 3
    cost.use_cost_hess = True

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


def create_satellite(plannersettings: saltro_py.PlannerSettings) -> saltro_py.Satellite:
    """3-magic-actuator satellite for a direct-torque 90-degree slew demo."""
    J = np.diag([0.067, 0.071, 0.069])

    satellite = saltro_py.Satellite(J, plannersettings)
    satellite.addMagic(np.array([1.0, 0.0, 0.0]), 0.02)
    satellite.addMagic(np.array([0.0, 1.0, 0.0]), 0.02)
    satellite.addMagic(np.array([0.0, 0.0, 1.0]), 0.02)

    return satellite


def main():
    plannersettings = create_planner_settings()
    satellite = create_satellite(plannersettings)

    jtime = np.array([0.22, 0.22 + 1000 / (36525 * 86400)])
    qgoal = np.array([
        [np.sqrt(2) / 2, np.sqrt(2) / 2],
        [0.0, 0.0],
        [0.0, 0.0],
        [np.sqrt(2) / 2, np.sqrt(2) / 2],
    ])
    boresight = np.array([
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ])

    w0 = np.array([0.0, 0.0, 0.0])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    x0 = np.hstack((w0, q0))

    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7.5e3, 0.0])

    start_time = time.time()
    ok, X, U, _K = saltro_py.trajOpt(
        plannersettings,
        satellite,
        x0,
        r0,
        v0,
        jtime,
        qgoal,
        boresight,
    )
    elapsed = time.time() - start_time

    if not ok:
        raise RuntimeError("trajOpt failed")

    print(f"trajOpt completed in {elapsed:.3f} seconds")
    print(f"Trajectory shape: X={X.shape}, U={U.shape}")
    print(f"Peak |u_magic|: {np.max(np.abs(U)):.6e}")

    plot_final_trajectory(
        X,
        U,
        plannersettings.passes[0].dt,
        satellite=satellite,
        q_goal=qgoal,
        title="AL-iLQR C++ Final Trajectory (Magic Actuators)",
    )


if __name__ == "__main__":
    main()
