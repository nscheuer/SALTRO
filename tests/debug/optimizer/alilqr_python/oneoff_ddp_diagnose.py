"""DDP diagnostic: per-iteration accepted alpha estimate, reg, and PE.

Wraps `saltro_py.backward_pass` + `saltro_py.forward_pass` directly so we
control the loop and can record state per iteration. Mirrors trajOpt's
single-pass setup just enough to drive the same scenario as
`oneoff_ddp_baseline.py`.

For each accepted iLQR step it logs:
    iter | reg | max ||d|| | accepted alpha (est) | J | dJ | PE_fin (deg)

The accepted-alpha estimate uses the linear part of the line-search step:
    α_eff = ((U_new − U_bar) · d_total) / ||d_total||²
where d_total[k] = d[k] (open-loop component) summed over knots — not exact
when |α| is large and δx feedback K·δx contributes, but a useful proxy
when feedback is small early in the pass.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from sat_3_1_hybrid import create_satellite
from trajOpt import _resample_zero_order_hold

QG = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
R0 = np.array([7e6, 0, 0]); V0 = np.array([0, 7.5e3, 0])
JTIME = np.array([0.22, 0.22 + 1000.0 / (36525.0 * 86400.0)])
QGOAL = np.tile(QG[:, None], (1, 2))
BS = np.array([[1, 1], [0, 0], [0, 0]], dtype=float)
X0 = np.array([0.01, 0.01, 0.01, 1, 0, 0, 0, 0.0])


def build_ps(use_dyn_hess: bool):
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = 1
    ps.num_passes = 1
    ps.passes[0].dt = 10.0
    ps.passes[0].ilqr.cost_tol = 1e-6
    ps.passes[0].ilqr.max_iters = 60  # cap diagnostic
    ps.passes[0].ilqr.grad_tol = 0.0
    ps.passes[0].auglag.max_outer_iters = 1   # single AL pass for diagnostic
    ps.passes[0].auglag.constraint_tol = 1e-3
    c = ps.passes[0].cost
    c.angle = 1e4; c.ang_vel = 1e2
    c.control_mult = 1.0; c.mtq_control_weight = 0.1; c.rw_control_weight = 1.0
    c.ang_cost_func_type = 3; c.use_cost_hess = True
    c.setTerminalEmphasis(100.0)
    for a in ["aero", "gg", "srp", "prop", "gendist", "resdipole"]:
        setattr(ps.disturbances, "plan_for_" + a, False)
    ps.passes[0].reg.reg_max = 1e30
    ps.passes[0].reg.reg_scale = 1.6
    ps.passes[0].reg.use_dynamics_hess = use_dyn_hess
    ps.passes[0].linesearch.max_iters = 24
    ps.passes[0].linesearch.beta1 = 1e-10
    ps.passes[0].linesearch.beta2 = 5000.0
    return ps


def pe_profile(X, qg):
    return np.array([2 * np.degrees(np.arccos(min(abs(float(np.dot(X[3:7, k], qg))), 1)))
                     for k in range(X.shape[1])])


def run_diagnose(label: str, use_ddp: bool):
    ps = build_ps(use_ddp)
    sat = create_satellite(ps)
    pass_settings = ps.passes[0]

    # Densify the time grid to dt-spaced points (mirrors trajOpt setup).
    dt_sec = pass_settings.dt
    jtime_fine, q_goal, boresight = _resample_zero_order_hold(
        JTIME, QGOAL, BS, dt_sec
    )
    ok, R, V, B, S, rho = saltro_py.generate_orbit(R0, V0, jtime_fine, 0, 0, 0, 0, 0)
    assert ok

    # Use saltro's warm_start to build an initial trajectory matching the
    # configured init_traj.initcontroller (excitation in this scenario).
    rho_2d = rho.reshape(1, -1) if rho.ndim == 1 else rho
    ok, X, U = saltro_py.warm_start(ps, sat, X0, jtime_fine,
                                     np.asfortranarray(q_goal),
                                     np.asfortranarray(boresight),
                                     np.asfortranarray(R),
                                     np.asfortranarray(V),
                                     np.asfortranarray(B),
                                     np.asfortranarray(S),
                                     rho_2d)
    assert ok

    rho = rho_2d
    N = R.shape[1]
    if U.shape[1] >= N:
        U = U[:, :N - 1]

    lambda_aug = []
    mu_aug = []
    n_constraints = 4  # placeholder; alilqr.py constructs from sat.constraints sample
    sample = sat.constraints(0, N, X[:, 0], U[:, 0], S[:, 0], ps.constraints)
    n_constraints = np.asarray(sample).size
    for k in range(N):
        lambda_aug.append(np.zeros(n_constraints))
        mu_aug.append(np.full(n_constraints, pass_settings.auglag.penalty_init))

    reg = pass_settings.reg.reg_init
    dreg = 0.0
    print(f"\n=== {label} ===")
    print(f"{'it':>3} {'reg':>10} {'max_d':>10} {'alpha_est':>10} "
          f"{'lmin_qu':>10} {'lmin_ddp':>10} {'ddp_rel':>9} "
          f"{'J':>11} {'dJ':>11} {'PE_fin':>7}")

    cost_cfg = pass_settings.cost
    J_prev = sat.totalCost(X, U, B, boresight, q_goal, cost_cfg)

    for it in range(pass_settings.ilqr.max_iters):
        ok_bp, K, d, deltaV, Q_uu_hist, Quu_ddp_hist = saltro_py.backward_pass(
            sat, X, U, B, R, V, S, rho,
            boresight, q_goal, ps, lambda_aug, mu_aug, reg,
            return_quu=True,
        )
        if not ok_bp:
            # increase_reg
            dreg = max(dreg * pass_settings.reg.reg_scale, pass_settings.reg.reg_scale)
            reg = max(reg * dreg, pass_settings.reg.reg_min)
            print(f"{it:>3} {reg:>10.2e} {'BP fail':>10}")
            continue

        # Decrease reg on BP success.
        dreg = min(dreg / pass_settings.reg.reg_scale, 1.0 / pass_settings.reg.reg_scale)
        reg = max(dreg * reg, pass_settings.reg.reg_min)

        # Convert numpy stacks to lists for forward_pass.
        K_list = [K[k] for k in range(K.shape[0])]
        d_list = [d[:, k] for k in range(d.shape[1])]
        max_d = max(np.linalg.norm(dk) for dk in d_list) if len(d_list) > 0 else 0.0

        U_bar = U.copy()
        ok_fp, X_new, U_new, J_new = saltro_py.forward_pass(
            sat, X, U, K_list, d_list, deltaV, B, R, V, S, rho,
            boresight, q_goal, ps, lambda_aug, mu_aug, JTIME, J_prev,
        )

        if not ok_fp:
            dreg = max(dreg * pass_settings.reg.reg_scale, pass_settings.reg.reg_scale)
            reg = max(reg * dreg, pass_settings.reg.reg_min)
            reg += pass_settings.reg.reg_bump
            dreg = max(dreg * pass_settings.reg.reg_scale, pass_settings.reg.reg_scale)
            reg = max(reg * dreg, pass_settings.reg.reg_min)
            print(f"{it:>3} {reg:>10.2e} {max_d:>10.2e} {'FP fail':>10}")
            continue

        # alpha estimate: project (U_new − U_bar) along the open-loop d direction.
        # d is shape (nu, N-1).
        dU = U_new - U_bar
        denom = float(np.sum(d * d))
        alpha_est = float(np.sum(d * dU) / denom) if denom > 0 else float('nan')

        # Eigenvalue diagnostics on Q_uu and Quu_ddp.
        # Q_uu_hist shape: (N-1, nu, nu).
        N_steps = Q_uu_hist.shape[0]
        lmin_qu_per = np.array([np.linalg.eigvalsh(Q_uu_hist[k]).min() for k in range(N_steps)])
        lmin_ddp_per = np.array([np.linalg.eigvalsh(Quu_ddp_hist[k]).min() for k in range(N_steps)])
        # ddp_rel: ‖Quu_ddp‖_F / ‖Q_uu_ilqr_part‖_F per knot.
        # Q_uu_ilqr_part = Q_uu - Quu_ddp.
        Q_uu_ilqr = Q_uu_hist - Quu_ddp_hist
        ilqr_norms = np.linalg.norm(Q_uu_ilqr.reshape(N_steps, -1), axis=1)
        ddp_norms = np.linalg.norm(Quu_ddp_hist.reshape(N_steps, -1), axis=1)
        ddp_rel = (ddp_norms / np.maximum(ilqr_norms, 1e-30)).max()

        lmin_qu = lmin_qu_per.min()
        lmin_ddp = lmin_ddp_per.min()

        X = X_new
        U = U_new
        dJ = J_prev - J_new
        pe_fin = pe_profile(X, QG)[-1]
        print(f"{it:>3} {reg:>10.2e} {max_d:>10.2e} {alpha_est:>10.3e} "
              f"{lmin_qu:>10.2e} {lmin_ddp:>10.2e} {ddp_rel:>9.2e} "
              f"{J_new:>11.4e} {dJ:>11.3e} {pe_fin:>7.2f}")
        J_prev = J_new

        if abs(dJ) < pass_settings.ilqr.cost_tol:
            print(f"   converged: |dJ| < {pass_settings.ilqr.cost_tol}")
            break


if __name__ == "__main__":
    run_diagnose("iLQR (DDP off)", use_ddp=False)
    run_diagnose("DDP on",          use_ddp=True)
