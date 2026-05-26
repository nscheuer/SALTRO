import sys
import time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from final_viewer import plot_final_trajectory


def create_planner_settings():
    plannersettings = saltro_py.PlannerSettings()

    plannersettings.init_traj.initcontroller = 2

    plannersettings.num_passes = 1
    plannersettings.passes[0].dt = 5.0
    plannersettings.passes[0].ilqr.cost_tol = 1e-5
    plannersettings.passes[0].ilqr.max_iters = 20

    plannersettings.passes[0].auglag.max_outer_iters = 30
    plannersettings.passes[0].auglag.constraint_tol = 1e-3

    cost = plannersettings.passes[0].cost
    cost.angle = 1e2
    cost.ang_vel = 1e1
    cost.ang_vel_mag = 0.0
    cost.ang_vel_err_dir = 0.0
    cost.control_mult = 1.0
    cost.mtq_control_weight = 1e-1
    cost.rw_control_weight = 1.0
    cost.magic_control_weight = 0.0
    cost.rw_AM_weight = 0.0
    cost.rw_stic_weight = 0.0
    cost.RWh_max_mult = 0.0
    cost.RWh_stiction_mult = 0.0
    cost.RWh_ok_mult = 0.0
    cost.angle_N = 1e2
    cost.ang_vel_N = 1e1
    cost.ang_vel_mag_N = 0.0
    cost.ang_vel_err_dir_N = 0.0
    cost.ang_cost_func_type = 0
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


def create_satellite_bc2(plannersettings: saltro_py.PlannerSettings) -> saltro_py.Satellite:
    """3 MTQ + 1 RW satellite matching debug_saltro_3+1_reduced.py."""
    J = np.array(
        [
            [0.03136490806, 5.88304e-05, -0.00671361357],
            [5.88304e-05, 0.03409127827, -0.00012334756],
            [-0.00671361357, -0.00012334756, 0.01004091997],
        ],
        dtype=float,
    )

    satellite = saltro_py.Satellite(J, plannersettings)

    satellite.addMTQ(np.array([1.0, 0.0, 0.0]), 0.2)
    satellite.addMTQ(np.array([0.0, 1.0, 0.0]), 0.2)
    satellite.addMTQ(np.array([0.0, 0.0, 1.0]), 0.2)
    satellite.addRW(np.array([0.0, 0.0, 1.0]), 5.7e-6, 0.0023, 0.0, 0.0036)

    return satellite


def main():
    plannersettings = create_planner_settings()
    satellite = create_satellite_bc2(plannersettings)

    jtime = np.array([0.22, 0.22 + 1000 / (36525 * 86400)])

    # Same attitude goal setup as debug_saltro_3+1_reduced.py:
    # body boresight is +Y and inertial target is -Z.
    vector_goal = np.array(
        [
            [np.nan, np.nan],
            [0.0, 0.0],
            [0.0, 0.0],
            [-1.0, -1.0],
        ],
        dtype=np.float64,
    )
    boresight = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )

    w0 = np.array([0.0, 0.0, 0.0])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    h0 = np.array([0.0])
    x0 = np.hstack((w0, q0, h0))

    # Same initial orbit setup as debug_saltro_3+1_reduced.py, converted to SALTRO units.
    r0 = 7000e3 * np.array([0.0, np.sqrt(2) / 2, np.sqrt(2) / 2])
    v0 = np.array([8.0e3, 0.0, 0.0])

    start_time = time.time()
    ok, X, U, _K = saltro_py.trajOpt(
        plannersettings,
        satellite,
        x0,
        r0,
        v0,
        jtime,
        vector_goal,
        boresight,
    )
    elapsed = time.time() - start_time

    if not ok:
        raise RuntimeError("trajOpt failed")

    print(f"trajOpt completed in {elapsed:.3f} seconds")
    print(f"Trajectory shape: X={X.shape}, U={U.shape}")

    plot_final_trajectory(
        X,
        U,
        plannersettings.passes[0].dt,
        satellite=satellite,
        q_goal=vector_goal,
        boresight_body=boresight,
        jtime=jtime,
    )


if __name__ == "__main__":
    main()
