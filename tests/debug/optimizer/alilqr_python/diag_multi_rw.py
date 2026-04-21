"""Diagnose multi-RW spike-removal failure by monkey-patching substitute_and_blend
to check quaternion norms after each phase and report the first bad knot.
"""
import sys, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from sat_0_3_rw import create_satellite
from trajOpt import trajOpt
import spike_removal as SR

# Wrap substitute_and_blend with phase-by-phase quaternion-norm checks.
_orig_sub_and_blend = SR.substitute_and_blend

def _check_quats(X, t_from, t_to, label):
    bad = []
    for k in range(max(t_from, 0), min(t_to, X.shape[1])):
        qn = np.linalg.norm(X[3:7, k])
        if not np.isfinite(qn) or abs(qn - 1.0) > 1e-3 or qn < 0.5:
            bad.append((k, qn))
    if bad:
        k0, qn0 = bad[0]
        print(f"    BAD at {label}: first bad k={k0}, |q|={qn0:.6f}  "
              f"(total bad = {len(bad)} of {t_to - t_from})", flush=True)
        # Also dump state at that knot
        x = X[:, k0]
        print(f"      x[:3] (ω)   = {x[:3]}", flush=True)
        print(f"      x[3:7] (q)  = {x[3:7]}", flush=True)
        print(f"      x[7:]  (hw) = {x[7:]}", flush=True)
        return True
    return False


# Reimplement substitute_and_blend inline with step-by-step instrumentation.
from spike_removal import (
    _build_pd_control, _rk4_step, _state_error_reduced
)


def instrumented(X, U, X_pd, U_pd, t_enter, t_exit, B_len,
                 X_nominal_pre, U_bar, K_list, satellite, dist_cfg,
                 B, S, R, V, rho, dt, kp_q=2.0, kd_w=5.0, rw_scale=0.0):
    N = X.shape[1]; nu = U.shape[0]
    print(f"\n  substitute_and_blend: window=({t_enter},{t_exit}) n_pd={t_exit-t_enter} "
          f"nRW={satellite.numRW} nMTQ={satellite.numMTQ}", flush=True)

    n_mtq = satellite.numMTQ; n_rw = satellite.numRW
    def _env(k):
        R_k = R[:, min(k, R.shape[1]-1)]
        B_k = B[:, min(k, B.shape[1]-1)]
        S_k = S[:, min(k, S.shape[1]-1)]
        V_k = V[:, min(k, V.shape[1]-1)]
        rho_k = int(np.round(float(rho[0, min(k, rho.shape[1]-1)])))
        return R_k, B_k, S_k, V_k, rho_k

    # --- Substitution region [t_enter, t_exit) ---
    n_pd = t_exit - t_enter
    X[:, t_enter:t_exit] = X_pd[:, :n_pd]
    if U_pd.shape[1] >= n_pd:
        U[:, t_enter:t_exit] = U_pd[:, :n_pd]
    if X_pd.shape[1] > n_pd:
        X[:, t_exit] = X_pd[:, n_pd]

    q_exit_target = X_nominal_pre[3:7, t_exit]
    blend_end = min(t_exit + B_len, N - 1)

    # --- Blend zone [t_exit, blend_end) ---
    for k in range(t_exit, blend_end):
        lam = float(k - t_exit) / float(B_len)
        R_k, B_k, S_k, V_k, rho_k = _env(k)
        u_pd_k = _build_pd_control(X[:, k], q_exit_target, satellite, B_k, kp_q, kd_w, rw_scale=rw_scale)
        u_ilqr_k = U_bar[:, k] if k < U_bar.shape[1] else np.zeros(nu)
        u_blend = (1.0 - lam) * u_pd_k + lam * u_ilqr_k
        for i in range(n_mtq):
            umax = float(satellite.getMTQ(i).u_max); u_blend[i] = float(np.clip(u_blend[i], -umax, umax))
        for i in range(n_rw):
            umax = float(satellite.getRW(i).u_max); u_blend[n_mtq+i] = float(np.clip(u_blend[n_mtq+i], -umax, umax))
        U[:, k] = u_blend
        x_next = _rk4_step(satellite, X[:, k], u_blend, dt, dist_cfg, R_k, B_k, S_k, V_k, rho_k)
        if k + 1 < N:
            X[:, k + 1] = x_next

    # --- Tail re-rollout [blend_end, N-1) ---
    print(f"    TAIL rollout from k={blend_end} with nominal gain correction", flush=True)
    for k in range(blend_end, N - 1):
        R_k, B_k, S_k, V_k, rho_k = _env(k)
        u_bar_k = U_bar[:, k] if k < U_bar.shape[1] else np.zeros(nu)
        K_k = K_list[k] if (K_list is not None and k < len(K_list) and K_list[k] is not None) else None

        x_at_k = X[:, k].copy()

        if K_k is not None:
            dx = _state_error_reduced(X[:, k], X_nominal_pre[:, k], satellite)
            du_raw = K_k @ dx
            du = du_raw.copy()
            for i in range(n_mtq):
                umax = float(satellite.getMTQ(i).u_max); du[i] = float(np.clip(du[i], -umax, umax))
            for i in range(n_rw):
                umax = float(satellite.getRW(i).u_max); du[n_mtq+i] = float(np.clip(du[n_mtq+i], -umax, umax))
            u_k = u_bar_k + du
            for i in range(n_mtq):
                umax = float(satellite.getMTQ(i).u_max); u_k[i] = float(np.clip(u_k[i], -umax, umax))
            for i in range(n_rw):
                umax = float(satellite.getRW(i).u_max); u_k[n_mtq+i] = float(np.clip(u_k[n_mtq+i], -umax, umax))
        else:
            u_k = u_bar_k
            du = None

        U[:, k] = u_k
        x_next = _rk4_step(satellite, X[:, k], u_k, dt, dist_cfg, R_k, B_k, S_k, V_k, rho_k)
        qn_next = np.linalg.norm(x_next[3:7])

        # First few + any with bad q
        if k <= blend_end + 2 or not np.isfinite(qn_next) or abs(qn_next - 1) > 0.01 or qn_next < 0.5:
            print(f"      k={k}: x[:3]ω={x_at_k[:3]}  |q_k|={np.linalg.norm(x_at_k[3:7]):.4f}  "
                  f"hw={x_at_k[7:]}", flush=True)
            if du is not None:
                print(f"         du_raw norm={np.linalg.norm(du_raw):.3e}  du_clamped={du}  u_bar={u_bar_k}", flush=True)
            print(f"         u={u_k}  → x_next: ω={x_next[:3]} |q_next|={qn_next:.4f} hw={x_next[7:]}", flush=True)
            if not np.isfinite(qn_next) or abs(qn_next - 1) > 0.01 or qn_next < 0.5:
                print(f"         *** NaN/bad q reached at k={k+1} ***", flush=True)
                if k + 1 < N:
                    X[:, k + 1] = x_next
                return

        if k + 1 < N:
            X[:, k + 1] = x_next

SR.substitute_and_blend = instrumented

# Run the RW-only case that fails
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
c.control_mult = 1.0
c.mtq_control_weight = 1e-1; c.rw_control_weight = 1.0
c.angle_N = 1e4; c.ang_vel_N = 1e2
c.ang_cost_func_type = 3; c.use_cost_hess = True
ps.passes[0].reg.reg_init = 1e-6; ps.passes[0].reg.reg_max = 1e30
ps.passes[0].reg.reg_scale = 1.6
ps.passes[0].linesearch.max_iters = 24
ps.passes[0].linesearch.beta1 = 1e-10; ps.passes[0].linesearch.beta2 = 5000.0

sat = create_satellite(ps)
x0 = np.hstack([[0.01, 0.01, 0.01], [1, 0, 0, 0], np.zeros(sat.numRW)])
r0 = np.array([7e6, 0, 0]); v0 = np.array([0, 7.5e3, 0])
jtime = np.array([0.22, 0.22 + 1000.0/(36525*86400)])
qg = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
qgoal = np.tile(qg[:, None], (1, 2))
bs = np.array([[1, 1], [0, 0], [0, 0]], dtype=float)

spike_cfg = {
    "start_at_iter": 2, "max_intervention_iters": 20, "blend_len": 30,
    "goal_switch_buffer": 15, "min_consecutive": 7, "exit_fudge": 2.0,
    "min_prior_decrease_knots": 5, "min_spike_ratio": 2.0,
    "kp_q": 0.3, "kd_w": 2.0, "rw_scale": 0.0, "omega_max": 0.30, "verbose": True,
}

print("=" * 100)
print("0+3 RW-only with spike removal, instrumented")
print("=" * 100)
try:
    X, U, stop, snaps, *_ = trajOpt(ps, sat, x0, r0, v0, jtime, qgoal, bs,
                                     debug=True, spike_removal_cfg=spike_cfg)
    print(f"\nFinal stop: {stop}, iters: {len(snaps)}")
except Exception as e:
    print(f"\nFAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
