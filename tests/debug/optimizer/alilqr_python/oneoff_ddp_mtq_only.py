"""DDP A/B on the MTQ-only sat (sat_3_0_mtq).

The hybrid 3+1 result showed the indefinite Quu_ddp eigenmode was 100%
concentrated in the RW axis. With NO RW present, that artifact disappears
entirely — DDP should either help or be neutral here, not hurt.

Mirrors `oneoff_ddp_mitigate.py` configs but uses `sat_3_0_mtq`.
"""
import sys, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from sat_3_0_mtq import create_satellite
from trajOpt import trajOpt

QG = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
R0 = np.array([7e6, 0, 0]); V0 = np.array([0, 7.5e3, 0])
JTIME = np.array([0.22, 0.22 + 1000.0 / (36525.0 * 86400.0)])
QGOAL = np.tile(QG[:, None], (1, 2))
BS = np.array([[1, 1], [0, 0], [0, 0]], dtype=float)
# 3+0 (MTQ-only) state: ω(3) + q(4) + 0 RW = 7-vector.
X0 = np.array([0.01, 0.01, 0.01, 1, 0, 0, 0])


def build_ps(use_dyn_hess: bool, use_eigen_mod: bool, terminal_k: float, psd_clip: bool):
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
    c.ang_cost_func_type = 3; c.use_cost_hess = True
    c.setTerminalEmphasis(terminal_k)
    for a in ["aero", "gg", "srp", "prop", "gendist", "resdipole"]:
        setattr(ps.disturbances, "plan_for_" + a, False)
    ps.passes[0].reg.reg_max = 1e30
    ps.passes[0].reg.reg_scale = 1.6
    ps.passes[0].reg.use_dynamics_hess = use_dyn_hess
    ps.passes[0].reg.use_eigen_modification = use_eigen_mod
    ps.passes[0].reg.psd_clip_quu_ddp = psd_clip
    ps.passes[0].reg.eigen_reg_use_abs = True
    ps.passes[0].reg.eigen_reg_add_mode = True
    ps.passes[0].linesearch.max_iters = 24
    ps.passes[0].linesearch.beta1 = 1e-10
    ps.passes[0].linesearch.beta2 = 5000.0
    return ps


def pe_profile(X, qg):
    return np.array([2 * np.degrees(np.arccos(min(abs(float(np.dot(X[3:7, k], qg))), 1)))
                     for k in range(X.shape[1])])


configs = [
    # label, use_ddp, use_eig, term_k, psd_clip
    ("(A) iLQR baseline                ", False, False, 100.0, False),
    ("(B) DDP, uniform reg             ", True,  False, 100.0, False),
    ("(C) DDP, eigen-mod |λ|+ρ         ", True,  True,  100.0, False),
    ("(D) DDP, terminal_k=1.0          ", True,  False, 1.0,   False),
    ("(F) DDP + PSD-clip both          ", True,  False, 100.0, True),
    ("(H) DDP + PSD-clip + term=1      ", True,  False, 1.0,   True),
]


for label, use_ddp, use_eig, term_k, psd_clip in configs:
    ps = build_ps(use_ddp, use_eig, term_k, psd_clip)
    sat = create_satellite(ps)
    X, U, stop, snaps, trans, dt_val, ctol, elapsed = trajOpt(
        ps, sat, X0.copy(), R0, V0, JTIME, QGOAL, BS,
        debug=True, spike_removal_cfg=None,
    )
    pe = pe_profile(X, QG)
    print(f"{label} stop={stop[:50]:50s}  it={len(snaps):4d}  "
          f"PE_fin={pe[-1]:7.3f}°  PE_mean={pe.mean():6.1f}°  t={elapsed:5.1f}s")
