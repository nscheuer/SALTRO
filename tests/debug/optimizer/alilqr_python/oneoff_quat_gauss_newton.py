"""A/B: full Hessian vs Gauss-Newton on QUATERNION-mode pointing.

For 90° z-rotation, q_goal = [cos(45°), 0, 0, sin(45°)] = [0.7071, 0, 0, 0.7071].
Tests both type=3 (½·acos(d)²) and type=4 (1-d²) shapes, GN on/off.
"""
import sys, time, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from sat_3_1_hybrid import create_satellite
from trajOpt import trajOpt


def quat_dist_deg(q1, q2):
    """Quaternion-distance angle in degrees (full 3-DOF)."""
    return np.degrees(2.0 * np.arccos(min(abs(float(np.dot(q1, q2))), 1.0)))


def pe_profile_quat(X, qg):
    out = np.zeros(X.shape[1])
    for k in range(X.shape[1]):
        q = X[3:7, k]; qn = q / max(np.linalg.norm(q), 1e-12)
        out[k] = quat_dist_deg(qn, qg)
    return out


def run(label, ang_cost_type, gauss_newton, ang_vel_err_dir=100.0,
        ang_vel_err_dir_ratio=0.0):
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = 1
    ps.num_passes = 1
    ps.passes[0].dt = 10.0
    ps.passes[0].ilqr.cost_tol = 1e-6
    ps.passes[0].ilqr.max_iters = 3000
    ps.passes[0].ilqr.grad_tol = 0.0
    ps.passes[0].auglag.max_outer_iters = 30
    ps.passes[0].auglag.constraint_tol = 1e-3
    c = ps.passes[0].cost
    c.angle = 1e4; c.ang_vel = 100.0
    c.control_mult = 1.0; c.mtq_control_weight = 0.1; c.rw_control_weight = 1.0
    c.ang_cost_func_type = ang_cost_type
    c.use_cost_hess = True
    c.cost_hess_gauss_newton = gauss_newton
    c.setTerminalEmphasis(100.0)
    c.ang_vel_err_dir = ang_vel_err_dir
    c.ang_vel_err_dir_ratio = ang_vel_err_dir_ratio
    # Roll ratio is irrelevant in quat mode (3-DOF).
    ps.passes[0].reg.reg_init = 1e-12; ps.passes[0].reg.reg_max = 1e30
    ps.passes[0].reg.reg_scale = 1.6
    ps.passes[0].linesearch.max_iters = 24
    ps.passes[0].linesearch.beta1 = 1e-10
    ps.passes[0].linesearch.beta2 = 5000.0

    sat = create_satellite(ps)
    nRW = sat.numRW
    x0 = np.hstack([np.array([0.01, 0.01, 0.01]),
                    np.array([1.0, 0.0, 0.0, 0.0]),
                    np.zeros(nRW)])
    # 90° z-rotation as quaternion target
    qg = np.array([np.cos(np.radians(45.0)), 0.0, 0.0, np.sin(np.radians(45.0))])
    qgoal = np.tile(qg[:, None], (1, 2))
    bs = np.array([[1, 1], [0, 0], [0, 0]], dtype=float)
    r0 = np.array([7e6, 0, 0]); v0 = np.array([0, 7.5e3, 0])
    jtime = np.array([0.22, 0.22 + 1000.0 / (36525.0 * 86400.0)])

    cfg = {
        "start_at_iter": 2, "max_intervention_iters": 10000,
        "blend_len": 30, "goal_switch_buffer": 15, "min_consecutive": 7,
        "exit_fudge": 2.0, "min_prior_decrease_knots": 5, "min_spike_ratio": 2.0,
        "kp_q": 0.3, "kd_w": 2.0, "rw_scale": -1.0, "omega_max": 0.30,
        "verbose": False, "constraint_gate_ratio": 0.0,
    }

    t0 = time.time()
    try:
        X, U, stop, snaps, trans, dt_val, ctol_cfg, elapsed = trajOpt(
            ps, sat, x0, r0, v0, jtime, qgoal, bs, debug=True, spike_removal_cfg=cfg,
        )
    except Exception as e:
        return {"label": label, "error": str(e)[:80]}
    wall = time.time() - t0
    pe = pe_profile_quat(X, qg)
    return {
        "label": label, "type": ang_cost_type, "GN": gauss_newton,
        "pe_fin": float(pe[-1]), "pe_mean": float(pe.mean()),
        "iters": len(snaps), "wall_s": wall,
        "stop": stop.split(":")[0] if ":" in stop else stop[:60],
    }


if __name__ == "__main__":
    configs = [
        ("type3_full_legacy",   3, False, 100.0, 0.0),
        ("type3_GN_legacy",     3, True,  100.0, 0.0),
        ("type4_full_legacy",   4, False, 100.0, 0.0),
        ("type4_GN_legacy",     4, True,  100.0, 0.0),
        ("type3_full_newpath",  3, False,   0.0, 0.3),
        ("type3_GN_newpath",    3, True,    0.0, 0.3),
        ("type4_full_newpath",  4, False,   0.0, 0.3),
        ("type4_GN_newpath",    4, True,    0.0, 0.3),
    ]
    rows = []
    for cfg in configs:
        print(f"Running {cfg[0]} ...")
        rows.append(run(*cfg))

    print()
    print("QUATERNION-mode 90° z-rotation. PE = quaternion-distance angle (3-DOF).")
    print(f"{'cfg':<26} {'type':>4} {'GN':>5} {'PE_fin':>8} {'PE_mean':>8} {'iters':>6} {'wall_s':>8}  stop")
    for r in rows:
        if "error" in r:
            print(f"{r['label']:<26} ERROR: {r['error']}")
            continue
        print(f"{r['label']:<26} {r['type']:>4d} {str(r['GN']):>5} "
              f"{r['pe_fin']:>8.3f} {r['pe_mean']:>8.3f} "
              f"{r['iters']:>6d} {r['wall_s']:>8.2f}  {r['stop']}")
