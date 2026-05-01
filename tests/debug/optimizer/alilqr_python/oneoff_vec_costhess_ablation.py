"""A/B: vec-mode 00_baseline with use_cost_hess=True vs False.

Isolates whether the new vector-mode cost Hessian is the cause of the
PE_fin=10° regression vs synthetic-q's PE_fin=2°. Prints PE_fin, iters,
wall, stop reason for each config — no plots.
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


def _quat_rotmat(q):
    q0, qv = q[0], q[1:]
    qx, qy, qz = qv
    skew = np.array([[0, -qz, qy], [qz, 0, -qx], [-qy, qx, 0]])
    return (q0*q0 - qv.dot(qv)) * np.eye(3) + 2*np.outer(qv, qv) + 2*q0*skew


def pe_profile_vec(X, r_eci, bs=np.array([1.0, 0.0, 0.0])):
    out = np.zeros(X.shape[1])
    for k in range(X.shape[1]):
        q = X[3:7, k]; qn = q / max(np.linalg.norm(q), 1e-12)
        bs_eci = _quat_rotmat(qn) @ bs
        out[k] = np.degrees(np.arccos(min(max(float(bs_eci.dot(r_eci)), -1.0), 1.0)))
    return out


def run(use_cost_hess):
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = 1
    ps.num_passes = 1
    ps.passes[0].dt = 10.0
    ps.passes[0].ilqr.cost_tol = 1e-6
    ps.passes[0].ilqr.max_iters = 200
    ps.passes[0].ilqr.grad_tol = 0.0
    ps.passes[0].auglag.max_outer_iters = 30
    ps.passes[0].auglag.constraint_tol = 1e-3
    c = ps.passes[0].cost
    c.angle = 1e4; c.ang_vel = 1e2
    c.control_mult = 1.0; c.mtq_control_weight = 0.1; c.rw_control_weight = 1.0
    c.ang_cost_func_type = 3
    c.use_cost_hess = use_cost_hess  # <<< the knob
    c.setTerminalEmphasis(100.0)
    c.ang_vel_err_dir = 1.0 * c.ang_vel
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
    ang = np.radians(90.0)
    eci_target = np.array([np.cos(ang), np.sin(ang), 0.0])
    qg = np.array([np.nan, eci_target[0], eci_target[1], eci_target[2]])
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
    X, U, stop, snaps, trans, dt_val, ctol_cfg, elapsed = trajOpt(
        ps, sat, x0, r0, v0, jtime, qgoal, bs, debug=True, spike_removal_cfg=cfg,
    )
    wall = time.time() - t0
    pe = pe_profile_vec(X, eci_target)
    return {
        "use_cost_hess": use_cost_hess,
        "pe_fin": float(pe[-1]),
        "pe_mean": float(pe.mean()),
        "iters": len(snaps),
        "wall_s": wall,
        "stop": stop.split(":")[0] if ":" in stop else stop[:60],
    }


if __name__ == "__main__":
    print("Running 00_baseline with use_cost_hess=True ...")
    rT = run(True)
    print("Running 00_baseline with use_cost_hess=False ...")
    rF = run(False)

    print()
    print(f"{'cfg':<24} {'PE_fin':>8} {'PE_mean':>8} {'iters':>6} {'wall_s':>8}  stop")
    for r in (rT, rF):
        print(f"use_cost_hess={str(r['use_cost_hess']):<10} "
              f"{r['pe_fin']:>8.3f} {r['pe_mean']:>8.3f} "
              f"{r['iters']:>6d} {r['wall_s']:>8.2f}  {r['stop']}")

    print()
    diff = rF["pe_fin"] - rT["pe_fin"]
    if abs(diff) < 0.5:
        verdict = "INCONCLUSIVE — cost-Hessian flip didn't move PE_fin much. Regression is likely (1) missing roll regularizer, not Hessian definiteness."
    elif diff < 0:
        verdict = f"COST HESSIAN HURTS — turning it off improved PE_fin by {-diff:.2f}°. PSD-clip the (q,q) block, or skip cost Hess in vec-mode."
    else:
        verdict = f"COST HESSIAN HELPS (rules out hypothesis 2) — turning it off worsened PE_fin by {diff:.2f}°."
    print(f"Verdict: {verdict}")
