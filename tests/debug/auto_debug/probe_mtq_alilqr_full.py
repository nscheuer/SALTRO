"""Probe full AL-iLQR behavior for MTQ-only case (3_0_slew90_dt60)."""
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(ROOT / "tests" / "debug" / "optimizer" / "alilqr_python"))
sys.path.insert(0, str(ROOT / "tests" / "debug" / "optimizer" / "configs"))

import saltro_py
from sat_3_0_mtq import create_satellite
from trajOpt import _resample_zero_order_hold


def create_planner_settings():
    s = saltro_py.PlannerSettings()
    s.init_traj.initcontroller = 2
    s.num_passes = 1
    s.passes[0].dt = 60.0
    s.passes[0].ilqr.cost_tol = 1e-5
    s.passes[0].ilqr.max_iters = 20
    s.passes[0].auglag.max_outer_iters = 10
    s.passes[0].auglag.constraint_tol = 1e-3

    c = s.passes[0].cost
    c.angle = 1.0
    c.ang_vel = 1e1
    c.ang_vel_mag = 0.0
    c.ang_vel_err_dir = 0.0
    c.control_mult = 1.0
    c.mtq_control_weight = 1e-2
    c.rw_control_weight = 1.0
    c.magic_control_weight = 0.0
    c.rw_AM_weight = 0.0
    c.rw_stic_weight = 0.0
    c.RWh_max_mult = 0.0
    c.RWh_stiction_mult = 0.0
    c.RWh_ok_mult = 0.0
    c.angle_N = 0.0
    c.ang_vel_N = 0.0
    c.ang_vel_mag_N = 0.0
    c.ang_vel_err_dir_N = 0.0
    c.ang_cost_func_type = 3
    c.use_cost_hess = True

    s.disturbances.plan_for_aero = False
    s.disturbances.plan_for_gg = False
    s.disturbances.plan_for_srp = False
    s.disturbances.plan_for_prop = False
    s.disturbances.plan_for_gendist = False
    s.disturbances.plan_for_resdipole = False

    s.passes[0].reg.reg_init = 1e-6
    s.passes[0].reg.reg_max = 1e10
    s.passes[0].reg.reg_scale = 10.0
    s.passes[0].reg.use_dynamics_hess = False
    s.passes[0].reg.use_constraint_hess = False

    s.passes[0].linesearch.max_iters = 24
    s.passes[0].linesearch.beta1 = 1e-10
    s.passes[0].linesearch.beta2 = 5000.0
    return s


def main():
    settings = create_planner_settings()
    sat = create_satellite(settings)

    jtime = np.array([0.22, 0.22 + 5400 / (36525 * 86400)])
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

    dt_sec = settings.passes[0].dt
    jtime_flat, q_goal_fine, boresight_fine = _resample_zero_order_hold(jtime, qgoal, boresight, dt_sec)

    ok_orbit, R, V, B, S, rho = saltro_py.generate_orbit(r0, v0, jtime_flat, 0, 0, 0, 0, 0)
    if not ok_orbit:
        raise RuntimeError("generate_orbit failed")

    q_goal_fine = np.asfortranarray(q_goal_fine)
    boresight_fine = np.asfortranarray(boresight_fine)
    R = np.asfortranarray(R)
    V = np.asfortranarray(V)
    B = np.asfortranarray(B)
    S = np.asfortranarray(S)
    rho = rho.reshape(1, -1) if rho.ndim == 1 else rho

    ok_ws, X0, U0 = saltro_py.warm_start(settings, sat, x0, jtime_flat, q_goal_fine, boresight_fine, R, V, B, S, rho)
    print(f"warm_start ok: {ok_ws}, X0={X0.shape}, U0={U0.shape}")
    if not ok_ws:
        return

    ok_al, X1, U1, status, max_c = saltro_py.alilqr(
        settings,
        0,
        sat,
        X0,
        U0,
        R,
        V,
        B,
        S,
        rho,
        jtime_flat,
        boresight_fine,
        q_goal_fine,
    )
    print(f"alilqr ok: {ok_al}, status={status}, max_c={max_c:.6e}")


if __name__ == "__main__":
    main()
