"""Spike hunt: 3 MTQ + 1 RW, 180deg slew, large initial omega.

Designed to produce mid-trajectory spikes (homotopy artifacts) so we have
a concrete before-image for the antispike work.  Run with:

    python debug_3_1_spike_hunt.py

Use the viewer arrow keys / cost-click to step through iLQR iterations and
look for mid-trajectory bumps in the Pointing Error panel.
"""
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from sat_3_1_hybrid import create_satellite
from trajOpt import trajOpt
from ilqr_viewer import launch_viewer


def create_planner_settings():
    plannersettings = saltro_py.PlannerSettings()

    # IntegratedBdotController warm-start — matches the working slew90 script
    plannersettings.init_traj.initcontroller = 1

    plannersettings.num_passes = 1
    plannersettings.passes[0].dt = 10.0
    plannersettings.passes[0].ilqr.cost_tol = 1e-3
    plannersettings.passes[0].ilqr.max_iters = 20

    plannersettings.passes[0].auglag.max_outer_iters = 10
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
    # Hessian OFF — more likely to find a different local minimum (spike)
    cost.ang_cost_func_type = 3
    cost.use_cost_hess = False

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

    # 90-degree slew about Z (matches debug_3_1_slew90_dt10.py target)
    # use_cost_hess=False below is the key difference that produces local minima
    jtime = np.array([0.22, 0.22 + 1000.0 / (36525.0 * 86400.0)])
    qgoal = np.array([
        [np.sqrt(2)/2,  np.sqrt(2)/2],
        [0.0,           0.0],
        [0.0,           0.0],
        [np.sqrt(2)/2,  np.sqrt(2)/2],
    ])
    boresight = np.array([
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ])

    # Small initial angular velocity — large w0 breaks the warm-start rollout
    # (invalid_next_state at k=2 for any alpha).  Spikes come from the 180-deg
    # homotopy structure + use_cost_hess=False, not from a huge initial omega.
    w0 = np.array([0.01, 0.01, 0.01])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    h0 = np.array([0.0])
    x0 = np.hstack((w0, q0, h0))

    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7.5e3, 0.0])

    X, U, stop_reason, snapshots, transitions, dt, cost_tol, elapsed_time = trajOpt(
        plannersettings, satellite, x0, r0, v0, jtime, qgoal, boresight, debug=True
    )

    print(f"AL-iLQR finished: {stop_reason}")
    print(f"Final cost: {snapshots[-1]['J']:.6e}")
    print(f"Elapsed time: {elapsed_time:.3f} seconds")
    print(f"Snapshots: {len(snapshots)}")

    # Quick spike summary: print max pointing error per snapshot
    def _quat_multiply(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    print("\nSnapshot | Max pointing error (deg) | Final pointing error (deg)")
    for i, snap in enumerate(snapshots):
        q = snap["X"][3:7, :]
        qg = snap["q_goal"]
        errs = []
        for k in range(q.shape[1]):
            qe = _quat_multiply(np.array([qg[0,k], -qg[1,k], -qg[2,k], -qg[3,k]]), q[:, k])
            errs.append(2.0 * np.arctan2(np.linalg.norm(qe[1:]), abs(qe[0])) * 180.0 / np.pi)
        print(f"  {i:3d}    | {max(errs):8.2f}                  | {errs[-1]:8.2f}")

    launch_viewer(snapshots, transitions, stop_reason, dt, cost_tol)


if __name__ == "__main__":
    main()
