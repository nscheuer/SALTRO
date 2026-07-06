import sys
import time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from sat_3_1_hybrid import create_satellite
from final_viewer import plot_final_trajectory


def create_planner_settings():
    plannersettings = saltro_py.PlannerSettings()

    plannersettings.init_traj.initcontroller = 2

    plannersettings.num_passes = 1
    plannersettings.passes[0].dt = 10.0
    plannersettings.passes[0].ilqr.cost_tol = 1e-5
    plannersettings.passes[0].ilqr.max_iters = 100

    plannersettings.passes[0].auglag.max_outer_iters = 100
    plannersettings.passes[0].auglag.constraint_tol = 1e-3
    plannersettings.passes[0].auglag.penalty_init = 1e2  # feel the constraints
    # from iter 0 — at the default the first inner solve is ~unconstrained and
    # over-commits a 30x-violating slew the later penalty levels can't unwind.

    cost = plannersettings.passes[0].cost
    cost.angle = 2e3   # type-1 = 0.5(1-c)^2, so 2x weight matches the old type-4 = (1-c)^2 cost exactly
    cost.ang_vel = 1e2  # rate damping: too low leaves residual body spin that
    # nutates the boresight a few degrees at the end (non-diagonal inertia).
    cost.ang_vel_mag = 0.0
    cost.ang_vel_err_dir = 0.0
    cost.control_mult = 1.0
    cost.mtq_control_weight = 1e-1
    cost.rw_control_weight = 1.0
    cost.magic_control_weight = 0.0
    # Non-zero RW-momentum penalty: the vec-mode Gauss-Newton angle Hessian
    # is rank-1 (curvature only along dc/dtheta), so the AL outer loop needs
    # some auxiliary curvature on the h (RW momentum) state to converge a
    # large slew.  The quat-mode debug case can leave this at 0 because its
    # angle Hessian already covers all three q-tangent directions.
    cost.rw_AM_weight = 1e4
    cost.rw_stic_weight = 0.0
    cost.RWh_stiction_mult = 0.0
    cost.RWh_knee_frac = 0.5   # free band below 50% h_max (validated: wheel used ~11%)
    cost.RWh_desat_mult = 0.05  # REQUIRED: flat free band grinds the outer loop (see plannersettings.h)
    cost.angle_N = 2e4   # doubled with the type-4 -> type-1 migration (see cost.angle)
    cost.ang_vel_N = 1e5  # strong terminal rate damping is what actually holds
    # the boresight at the goal — under-weighting it was the residual-nutation
    # "tail-drift". With this + the RWh re-key the slew points sub-degree
    # (min 0.39 deg, last-quarter mean ~1.0 deg) vs strangled (never settles).
    cost.ang_vel_mag_N = 0.0
    cost.ang_vel_err_dir_N = 0.0
    # Bounded-curvature attitude cost. ang_cost_func_type=1 is 0.5*(1-c)^2;
    # with the angle weights doubled above this is identical to the former
    # type-4 = (1-c)^2 (a constant factor of 2 the AL absorbs), and -- unlike
    # acos^2 (type 3) -- it has no antipodal curvature blow-up, so a >90 deg
    # slew stays well-conditioned. (type 4 is removed by the cost-block PR; this
    # keeps the demo valid once that lands.)
    cost.ang_cost_func_type = 1
    cost.use_cost_hess = True
    # Gauss-Newton angle Hessian: rank-1, PSD by construction. Required for the
    # vector-pointing backward pass to stay positive-definite past 90 deg.
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

    jtime = np.array([0.22, 0.22 + 1000 / (36525 * 86400)])
    # Vector pointing version: each goal column is [NaN, x, y, z], where the
    # last three entries are the inertial target direction.
    # This case asks the body +X boresight to point toward inertial -Y.
    # With the identity quaternion, body +X is aligned with inertial +X, so the
    # initial boresight-to-target error is 90 degrees.
    vector_goal = np.array([
        [np.nan, np.nan],
        [0.0, 0.0],  # x component (constant goal)
        [-1.0, -1.0],  # y component (constant goal)
        [0.0, 0.0],  # z component (constant goal)
    ])
    boresight = np.array([
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ])

    w0 = np.array([0.01, 0.01, 0.01])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    h0 = np.array([0.0])
    x0 = np.hstack((w0, q0, h0))

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
        vector_goal,
        boresight,
    )
    elapsed = time.time() - start_time

    if not ok:
        raise RuntimeError("trajOpt failed")

    print(f"trajOpt completed in {elapsed:.3f} seconds")
    print(f"Trajectory shape: X={X.shape}, U={U.shape}")

    plot_final_trajectory(X, U, plannersettings.passes[0].dt, satellite=satellite, q_goal=vector_goal, boresight_body=boresight, jtime=jtime)


if __name__ == "__main__":
    main()
