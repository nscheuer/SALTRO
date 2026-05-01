"""One-off: MTQ-only with DDP on/off (use_dynamics_hess flag).

Matched-conditions test of whether adding the DDP dynamics-Hessian terms
(V_x · f_xx, V_x · f_ux, V_x · f_uu) to the BP Q-matrices changes the
MTQ-only convergence picture.

If DDP helps: thesis figures likely ran with dynamics Hessians ON, and our
port of them via rk4_hessians + reduced-state G-projection is worth keeping.
If no effect: iLQR's re-linearization is sufficient for the warm-started
MTQ-only regime, matching the thesis matlab (which is pure iLQR)."""
import sys, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from sat_3_0_mtq import create_satellite
from trajOpt import trajOpt

qg = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
r0 = np.array([7e6, 0, 0]); v0 = np.array([0, 7.5e3, 0])
jtime = np.array([0.22, 0.22 + 1000.0 / (36525.0 * 86400.0)])
qgoal = np.tile(qg[:, None], (1, 2))
bs = np.array([[1, 1], [0, 0], [0, 0]], dtype=float)

# Test two ω0 regimes:
#   - 0: the "easy" MTQ-only baseline (pre-change ~7° PE, post-principled 5.8°)
#   - 0.01 per-axis: the "hard" case matching wide_test scenario 01 (71° PE baseline)
REGIMES = [
    ("w0=0",    np.array([0.0, 0.0, 0.0, 1, 0, 0, 0])),
    ("w0=0.01", np.array([0.01, 0.01, 0.01, 1, 0, 0, 0])),
]


def build_ps(use_dyn_hess: bool):
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
    c.control_mult = 1.0; c.mtq_control_weight = 0.1
    c.ang_cost_func_type = 3; c.use_cost_hess = True
    c.setTerminalEmphasis(100.0)
    for a in ["aero", "gg", "srp", "prop", "gendist", "resdipole"]:
        setattr(ps.disturbances, "plan_for_" + a, False)
    ps.passes[0].reg.reg_init = 0.0
    ps.passes[0].reg.reg_max = 1e30
    ps.passes[0].reg.reg_scale = 1.6
    ps.passes[0].reg.use_dynamics_hess = use_dyn_hess
    ps.passes[0].linesearch.max_iters = 24
    ps.passes[0].linesearch.beta1 = 1e-10; ps.passes[0].linesearch.beta2 = 5000.0
    return ps


def pe_profile(X, qg):
    return np.array([2 * np.degrees(np.arccos(min(abs(float(np.dot(X[3:7, k], qg))), 1)))
                     for k in range(X.shape[1])])


for reg_label, x0 in REGIMES:
    print(f"\n=== regime: {reg_label} ===")
    for flag_label, flag in [("iLQR (DDP off)", False), ("DDP on", True)]:
        ps = build_ps(flag)
        sat = create_satellite(ps)
        try:
            X, U, stop, snaps, trans, dt_val, ctol, elapsed = trajOpt(
                ps, sat, x0.copy(), r0, v0, jtime, qgoal, bs,
                debug=True, spike_removal_cfg=None,
            )
            pe = pe_profile(X, qg)
            omega_norm = np.linalg.norm(X[0:3, :], axis=0)
            print(f"  {flag_label:20s} stop={stop}  it={len(snaps)}  "
                  f"PE_fin={pe[-1]:7.3f}°  PE_mean={pe.mean():6.1f}°  "
                  f"|ω|_max={np.degrees(omega_norm.max()):6.2f}°/s  t={elapsed:5.1f}s")
        except Exception as e:
            print(f"  {flag_label:20s} FAILED: {e}")
