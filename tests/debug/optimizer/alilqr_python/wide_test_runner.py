"""Single-axis weakness probe for the ALTRO spike-removal stack.

Each scenario changes ONE thing from the baseline and saves:
    wide_results/<name>.gif          - convergence animation
    wide_results/<name>_final.png    - final-iteration 4-panel
    wide_results/<name>_midway.png   - progression of key iterations

Runs three-pass spike removal (current baseline detector) with verbose=False.
Summary table printed at end.

Env vars:
    WIDE_COSTREF=1            — enable new α-from-β crossterm via
                                ang_vel_err_dir_ratio. Output → wide_results_costref/.
    WIDE_COSTREF_BETA=<f>     — β value for ang_vel_err_dir_ratio (default 0.3).
"""
import os, sys, time, numpy as np
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from trajOpt import trajOpt
from sat_3_1_hybrid import create_satellite as create_3_1
from sat_3_0_mtq    import create_satellite as create_3_0
from sat_0_3_rw     import create_satellite as create_0_3
from sat_3_3_hybrid import create_satellite as create_3_3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

COSTREF_ENABLED = os.environ.get("WIDE_COSTREF") == "1"
COSTREF_BETA = float(os.environ.get("WIDE_COSTREF_BETA", "0.3"))
CURRICULUM_ENABLED = os.environ.get("WIDE_CURRICULUM") == "1"
INITCONTROLLER = int(os.environ.get("WIDE_INITCONTROLLER", "1"))
SCENARIO_FILTER = os.environ.get("WIDE_SCENARIO_FILTER")  # comma-separated
OUT_SUFFIX = os.environ.get("WIDE_OUT_SUFFIX", "")
ANG_COST_TYPE = int(os.environ.get("WIDE_ANG_COST_TYPE", "3"))
GAUSS_NEWTON = os.environ.get("WIDE_GAUSS_NEWTON") == "1"
default_out = "wide_results_costref" if COSTREF_ENABLED else "wide_results"
OUT = Path(__file__).parent / (default_out + OUT_SUFFIX)
OUT.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Baseline
# -----------------------------------------------------------------------------
def baseline_params():
    return dict(
        sat_fn=create_3_1,
        angle=1e4, ang_vel=1e2, mtq_cw=1e-1, rw_cw=1.0,
        dt=10.0, time_s=1000.0,
        omega0=np.array([0.01, 0.01, 0.01]),
        goal_angle_deg=90.0,
        disturbances={},
        use_spike=True,
        max_iters=int(os.environ.get("WIDE_MAX_ITERS", "200")),
        max_outer=int(os.environ.get("WIDE_MAX_OUTER", "30")),
    )


def _config_pass(pass_obj, p, *, angle_weight, max_iters, cost_tol):
    pass_obj.dt = p["dt"]
    pass_obj.ilqr.cost_tol = cost_tol
    pass_obj.ilqr.max_iters = max_iters
    pass_obj.ilqr.grad_tol = 0.0
    pass_obj.auglag.max_outer_iters = p["max_outer"]
    pass_obj.auglag.constraint_tol = 1e-3
    c = pass_obj.cost
    c.angle = angle_weight; c.ang_vel = p["ang_vel"]
    c.control_mult = 1.0
    c.mtq_control_weight = p["mtq_cw"]; c.rw_control_weight = p["rw_cw"]
    c.ang_cost_func_type = ANG_COST_TYPE; c.use_cost_hess = True
    c.cost_hess_gauss_newton = GAUSS_NEWTON
    c.setTerminalEmphasis(100.0)
    if COSTREF_ENABLED:
        c.ang_vel_err_dir_ratio = COSTREF_BETA
        c.ang_vel_err_dir = 0.0  # disable legacy when newpath active
    else:
        # Legacy: w_avang = ang_vel (PhD form). Confirms "legacy" path.
        c.ang_vel_err_dir = c.ang_vel
    pass_obj.reg.reg_init = float(os.environ.get("WIDE_REG_INIT", "0.0"))
    pass_obj.reg.reg_max = 1e30
    pass_obj.reg.reg_scale = 1.6
    # Default: eigen modification ON with kappa_cap=1e-9 and relative floor.
    # Empirically (17-scenario sweep 2026-05-16): case 01 PE_fin 60->9 deg,
    # case 12 PE_fin 123->3 deg, no regressions on previously-good cases,
    # +13% wall time vs eigen-off.  kc=1e-10 fails case 05; kc=1e-8 has
    # 60+% wall-time bloat.  Override via WIDE_EIGEN_MOD=0.
    eigen_mod = os.environ.get("WIDE_EIGEN_MOD", "1") == "1"
    eigen_relfloor = os.environ.get("WIDE_EIGEN_RELFLOOR", "1") == "1"
    if eigen_mod:
        pass_obj.reg.use_eigen_modification = True
        if eigen_relfloor:
            pass_obj.reg.eigen_reg_use_relative_floor = True
        kappa_cap = float(os.environ.get("WIDE_EIGEN_KAPPA_CAP", "1e-9"))
        pass_obj.reg.eigen_reg_condition_cap = kappa_cap
    pass_obj.linesearch.max_iters = 24
    pass_obj.linesearch.beta1 = 1e-10; pass_obj.linesearch.beta2 = 5000.0


def build_planner(p):
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = INITCONTROLLER
    if CURRICULUM_ENABLED:
        # Two-phase: pass 0 detumbles (zero attitude weight, half the iters),
        # pass 1 does full attitude tracking. Trajectory is warm-started from
        # pass 0's output. Mirrors thesis-matlab phase scheduling.
        ps.num_passes = 2
        _config_pass(ps.passes[0], p, angle_weight=0.0,
                     max_iters=max(50, p["max_iters"] // 4), cost_tol=1e-3)
        _config_pass(ps.passes[1], p, angle_weight=p["angle"],
                     max_iters=p["max_iters"], cost_tol=1e-6)
    else:
        ps.num_passes = 1
        _config_pass(ps.passes[0], p, angle_weight=p["angle"],
                     max_iters=p["max_iters"], cost_tol=1e-6)
    for a in ["aero", "gg", "srp", "prop", "gendist", "resdipole"]:
        setattr(ps.disturbances, "plan_for_" + a, p["disturbances"].get(a, False))
    return ps


def run_scenario(name, params):
    ps = build_planner(params)
    sat = params["sat_fn"](ps)

    # Build initial state with the right dimension for the satellite
    nRW = sat.numRW
    w0 = params["omega0"]
    if os.environ.get("WIDE_OMEGA0_ZERO") == "1":
        w0 = np.zeros(3)
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    h0 = np.zeros(nRW)
    x0 = np.hstack([w0, q0, h0])

    ang = np.radians(params["goal_angle_deg"])
    qg = np.array([np.cos(ang/2), 0, 0, np.sin(ang/2)])
    qgoal = np.tile(qg[:, None], (1, 2))
    bs = np.array([[1, 1], [0, 0], [0, 0]], dtype=float)
    r0 = np.array([7e6, 0, 0]); v0 = np.array([0, 7.5e3, 0])
    jtime = np.array([0.22, 0.22 + params["time_s"] / (36525.0 * 86400.0)])

    cfg = None
    if params["use_spike"] and os.environ.get("WIDE_NO_SPIKE") != "1":
        cfg = {
            "start_at_iter": int(os.environ.get("WIDE_SPIKE_START_AT", "2")),
            "max_intervention_iters": 10000,
            "blend_len": 30, "goal_switch_buffer": 15,
            "min_consecutive": int(os.environ.get("WIDE_SPIKE_MIN_CONSEC", "7")),
            "exit_fudge": 2.0, "min_prior_decrease_knots": 5,
            "min_spike_ratio": float(os.environ.get("WIDE_SPIKE_MIN_RATIO", "2.0")),
            "entry_error_max_rad": float(os.environ.get("WIDE_SPIKE_ENTRY_MAX", "0.5")),
            "prior_low_max_rad": float(os.environ.get("WIDE_SPIKE_PRIOR_LOW_MAX", "0.15")),
            "post_low_max_rad": float(os.environ.get("WIDE_SPIKE_POST_LOW_MAX", "0.15")),
            "min_post_stable_knots": int(os.environ.get("WIDE_SPIKE_POST_KNOTS", "10")),
            "post_vs_prior_ratio": float(os.environ.get("WIDE_SPIKE_POST_RATIO", "0.5")),
            "force_mtq_only": os.environ.get("WIDE_SPIKE_FORCE_MTQ") == "1",
            "tail_skip_entry_threshold_rad": float(os.environ.get("WIDE_SPIKE_TAIL_ENTRY_MAX", "0.5")),
            "omega_skip_threshold_rad": float(os.environ.get("WIDE_SPIKE_OMEGA_SKIP", "0.0")),
            "omega_physics_floor_rad_s": float(os.environ.get("WIDE_SPIKE_OMEGA_PHYS_FLOOR", "0.1")),
            "omega_alignment_threshold": float(os.environ.get("WIDE_SPIKE_OMEGA_ALIGN", "0.7")),
            "max_trajectory_transitions": int(os.environ.get("WIDE_SPIKE_MAX_TRANS", "10")),
            "omega_skip_mean_rad_s": float(os.environ.get("WIDE_SPIKE_OMEGA_MEAN_SKIP", "0.1")),
            "pe_rate_skip_threshold_rad_s": float(os.environ.get("WIDE_SPIKE_PE_RATE_SKIP", "0.05")),
            "pd_dt_ref": float(os.environ.get("WIDE_SPIKE_PD_DT_REF", "10.0")),
            "kp_q": float(os.environ.get("WIDE_SPIKE_KP_Q", "0.3")),
            "kd_w": float(os.environ.get("WIDE_SPIKE_KD_W", "2.0")),
            "rw_scale": -1.0, "omega_max": 0.30, "verbose": True,
            "constraint_gate_ratio": float(os.environ.get("WIDE_SPIKE_GATE_RATIO", "0.0")),
        }

    t0 = time.time()
    try:
        X, U, stop, snaps, trans, dt_val, ctol_cfg, elapsed = trajOpt(
            ps, sat, x0, r0, v0, jtime, qgoal, bs, debug=True, spike_removal_cfg=cfg,
        )
    except Exception as e:
        print(f"  {name:<30} ERROR: {str(e)[:80]}")
        return None
    wall = time.time() - t0

    # Metrics
    pe = np.array([2*np.degrees(np.arccos(min(abs(float(np.dot(X[3:7,k], qg))), 1)))
                   for k in range(X.shape[1])])
    q = X[3:7, :]; N = q.shape[1]
    step = np.array([2*np.arccos(min(abs(float(np.dot(q[:,k], q[:,k+1]))), 1)) for k in range(N-1)])
    trav = step.sum()
    direct = 2*np.arccos(min(abs(float(np.dot(q[:,0], q[:,-1]))), 1))
    ctol = None
    if "violation" in stop:
        try: ctol = float(stop.split("violation")[1].split("<=")[0].strip())
        except: pass

    result = {
        "name": name, "stop": stop.split(":")[0] if ":" in stop else stop,
        "iters": len(snaps), "wall_s": wall,
        "pe_mean": float(pe.mean()), "pe_fin": float(pe[-1]),
        "excess_deg": float(np.degrees(trav - direct)),
        "ctol": ctol,
    }

    # Use a single timestamp for all three images so they're consistent.
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Save midway snapshot grid
    save_midway(name, snaps, X, U, ps, sat, qg, r0, v0, jtime, ts=run_ts)
    # Save GIF
    save_gif(name, snaps, ps, sat, qg, ts=run_ts)
    # Save final panel
    save_final(name, X, U, ps, sat, qg, stop, ts=run_ts)

    return result


def pe_profile(X, qg):
    return np.array([2*np.degrees(np.arccos(min(abs(float(np.dot(X[3:7,k], qg))), 1)))
                     for k in range(X.shape[1])])


def save_midway(name, snaps, X_final, U_final, ps, sat, qg, r0, v0, jtime, ts=None):
    N = snaps[0]['X'].shape[1]
    t_arr = np.arange(N) * ps.passes[0].dt
    key_iters = sorted(set(i for i in [0, 5, 10, 20, 50, 100, len(snaps)//4, len(snaps)//2, 3*len(snaps)//4, len(snaps)-1]
                            if 0 <= i < len(snaps)))
    fig, axes = plt.subplots(len(key_iters), 3, figsize=(18, 3*len(key_iters)))
    if ts is None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.suptitle(f"{name}  |  {len(snaps)} iters  |  {ts}", fontsize=11)
    if len(key_iters) == 1:
        axes = axes.reshape(1, 3)
    for row, idx in enumerate(key_iters):
        snap = snaps[idx]
        pe = pe_profile(snap['X'], qg)
        X_s = snap['X']
        U_s = snap['U']
        ax_pe = axes[row, 0]
        ax_pe.plot(t_arr, pe, 'b-', linewidth=1.2)
        ax_pe.axhline(1.0, color='g', linestyle='--', alpha=0.4)
        ax_pe.set_ylim(-5, 185); ax_pe.set_ylabel(f"iter {idx}\nPE (deg)")
        ax_pe.set_title(f"PE mean={pe.mean():.1f}° max={pe.max():.1f}°  J={snap['J']:.2e}", fontsize=9)
        ax_pe.grid(True, alpha=0.3)
        ax_q = axes[row, 1]
        for i in range(4):
            ax_q.plot(t_arr, X_s[3+i, :], linewidth=1.0, label=f"q{i}")
        ax_q.set_ylim(-1.1, 1.1); ax_q.set_ylabel("q"); ax_q.grid(True, alpha=0.3)
        ax_q.legend(loc='upper right', fontsize=7); ax_q.set_title("Quaternion", fontsize=9)
        ax_ctrl = axes[row, 2]
        t_u = t_arr[:U_s.shape[1]]
        for i in range(U_s.shape[0]):
            ax_ctrl.plot(t_u, U_s[i, :], linewidth=0.7)
        ax_ctrl.set_ylabel("u"); ax_ctrl.grid(True, alpha=0.3)
        ax_ctrl.set_title(f"max|u|={np.max(np.abs(U_s)) if U_s.size else 0:.3f}", fontsize=9)
    axes[-1, 0].set_xlabel("t (s)"); axes[-1, 1].set_xlabel("t (s)"); axes[-1, 2].set_xlabel("t (s)")
    plt.tight_layout()
    fig.savefig(OUT / f"{name}_midway.png", dpi=100); plt.close(fig)


def save_final(name, X, U, ps, sat, qg, stop, ts=None):
    N = X.shape[1]
    t_arr = np.arange(N) * ps.passes[0].dt
    pe = pe_profile(X, qg)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    if ts is None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.suptitle(f"{name}  |  {stop.split(':')[0]}  |  {ts}", fontsize=11)
    ax = axes[0, 0]
    ax.plot(t_arr, pe, 'b-', linewidth=1.5); ax.axhline(1.0, color='g', linestyle='--', alpha=0.4)
    ax.set_ylabel("PE (deg)"); ax.set_xlabel("t (s)"); ax.set_ylim(-5, 185)
    ax.set_title(f"PE mean={pe.mean():.1f}° final={pe[-1]:.1f}°"); ax.grid(True, alpha=0.3)
    ax = axes[0, 1]
    for i in range(4):
        ax.plot(t_arr, X[3+i, :], linewidth=1.3, label=f"q{i}")
    ax.set_ylabel("q"); ax.set_xlabel("t (s)"); ax.set_ylim(-1.1, 1.1)
    ax.legend(loc='upper right', fontsize=8); ax.grid(True, alpha=0.3); ax.set_title("Quaternion")
    ax = axes[1, 0]
    t_u = t_arr[:U.shape[1]]
    for i in range(U.shape[0]):
        ax.plot(t_u, U[i, :], linewidth=0.8)
    ax.set_ylabel("u"); ax.set_xlabel("t (s)"); ax.grid(True, alpha=0.3)
    ax.set_title(f"Controls (max={np.max(np.abs(U)) if U.size else 0:.3f})")
    ax = axes[1, 1]
    for i in range(3):
        ax.plot(t_arr, np.degrees(X[i, :]), linewidth=1.3, label=f"w{i}")
    wmax = np.degrees(ps.constraints.wmax)
    ax.axhline(wmax, color='r', linestyle='--', alpha=0.4); ax.axhline(-wmax, color='r', linestyle='--', alpha=0.4)
    ax.set_ylabel("ω (deg/s)"); ax.set_xlabel("t (s)")
    ax.legend(loc='upper right', fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_title(f"ω max={np.degrees(np.max(np.abs(X[0:3, :]))):.1f} °/s")
    plt.tight_layout()
    fig.savefig(OUT / f"{name}_final.png", dpi=120); plt.close(fig)


def save_gif(name, snaps, ps, sat, qg, ts=None):
    N = snaps[0]['X'].shape[1]
    t_arr = np.arange(N) * ps.passes[0].dt
    n_mtq = sat.numMTQ; n_rw = sat.numRW
    mtq_umax = [sat.getMTQ(i).u_max for i in range(n_mtq)]
    rw_umax = [sat.getRW(i).u_max for i in range(n_rw)]
    rw_hmax = [sat.getRW(i).momentumMax for i in range(n_rw)]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    if ts is None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.suptitle(f"{name}  |  {ts}", fontsize=11)

    def animate(frame):
        for ax in axes.flat:
            ax.clear()
        snap = snaps[frame]
        pe = pe_profile(snap['X'], qg)
        X_f = snap['X']; U_f = snap['U']
        J = snap['J']; oi = snap.get('outer_iter', '?')
        # [0,0] PE
        ax = axes[0, 0]
        ax.plot(t_arr, pe, 'b-', linewidth=1.5)
        ax.axhline(1.0, color='g', linestyle='--', alpha=0.4)
        ax.set_ylabel("PE (deg)"); ax.set_xlabel("t (s)"); ax.set_ylim(-5, 185)
        ax.set_title(f"iter {frame}/{len(snaps)-1}  outer={oi}  J={J:.3e}", fontsize=9)
        ax.grid(True, alpha=0.3)
        # [0,1] Quaternion
        ax = axes[0, 1]
        for i in range(4):
            ax.plot(t_arr, X_f[3+i, :], linewidth=1.2, label=f"q{i}")
        ax.set_ylabel("q"); ax.set_ylim(-1.1, 1.1); ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3); ax.set_title("q", fontsize=9)
        # [0,2] ω
        ax = axes[0, 2]
        for i in range(3):
            ax.plot(t_arr, np.degrees(X_f[i, :]), linewidth=1.2, label=f"w{i}")
        wmax_deg = np.degrees(ps.constraints.wmax)
        ax.axhline(wmax_deg, color='r', linestyle='--', alpha=0.4)
        ax.axhline(-wmax_deg, color='r', linestyle='--', alpha=0.4)
        ax.set_ylabel("ω (°/s)"); ax.set_xlabel("t (s)"); ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3); ax.set_title("ω", fontsize=9)
        # [1,0] MTQ utilization %
        ax = axes[1, 0]
        if n_mtq > 0 and U_f.shape[1] > 0:
            t_u = t_arr[:U_f.shape[1]]
            for i in range(n_mtq):
                ax.plot(t_u, U_f[i, :] / mtq_umax[i] * 100, linewidth=0.9, label=f"m{i}")
            ax.axhline(75, color='r', linestyle='--', alpha=0.4, label='AL ceiling')
            ax.axhline(-75, color='r', linestyle='--', alpha=0.4)
        ax.set_ylim(-110, 110); ax.set_ylabel("MTQ %"); ax.set_xlabel("t (s)")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_title("MTQ utilization", fontsize=9)
        # [1,1] RW torque utilization %
        ax = axes[1, 1]
        if n_rw > 0 and U_f.shape[1] > 0:
            t_u = t_arr[:U_f.shape[1]]
            for i in range(n_rw):
                ax.plot(t_u, U_f[n_mtq+i, :] / rw_umax[i] * 100, linewidth=0.9, label=f"τ{i}")
            ax.axhline(75, color='r', linestyle='--', alpha=0.4)
            ax.axhline(-75, color='r', linestyle='--', alpha=0.4)
        ax.set_ylim(-110, 110); ax.set_ylabel("RW τ %"); ax.set_xlabel("t (s)")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_title("RW torque utilization", fontsize=9)
        # [1,2] RW momentum utilization %
        ax = axes[1, 2]
        if n_rw > 0:
            for i in range(n_rw):
                h_i = X_f[7 + i, :]
                ax.plot(t_arr, h_i / rw_hmax[i] * 100, linewidth=1.0, label=f"h{i}")
            ax.axhline(100, color='r', linestyle='--', alpha=0.4)
            ax.axhline(-100, color='r', linestyle='--', alpha=0.4)
        ax.set_ylim(-120, 120); ax.set_ylabel("h_rw %"); ax.set_xlabel("t (s)")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_title("RW momentum", fontsize=9)
        plt.tight_layout()
        return axes.flat

    step = max(1, len(snaps) // 80)
    frames = list(range(0, len(snaps), step))
    if frames[-1] != len(snaps) - 1:
        frames.append(len(snaps) - 1)
    # Hold final frame for 10s at fps=5 = 50 extra frames.
    FPS = 5
    frames = frames + [len(snaps) - 1] * (FPS * 10)
    anim = FuncAnimation(fig, animate, frames=frames, repeat=False)
    anim.save(str(OUT / f"{name}.gif"), writer=PillowWriter(fps=FPS))
    plt.close(fig)


# -----------------------------------------------------------------------------
# Scenarios: baseline + single-axis variations
# -----------------------------------------------------------------------------
def merge(base, **kw):
    d = dict(base)
    for k, v in kw.items():
        if k == "disturbances" and isinstance(v, dict):
            d["disturbances"] = dict(d.get("disturbances", {}), **v)
        else:
            d[k] = v
    return d


B = baseline_params()

SCENARIOS = [
    ("00_baseline",         B),

    # Satellite configs
    ("01_sat_3_0_mtq",      merge(B, sat_fn=create_3_0)),
    ("02_sat_0_3_rw",       merge(B, sat_fn=create_0_3)),
    ("03_sat_3_3_hybrid",   merge(B, sat_fn=create_3_3)),

    # Angle weight
    ("04_angle_1e2_low",    merge(B, angle=1e2, ang_vel=1.0)),
    ("05_angle_1e6_high",   merge(B, angle=1e6, ang_vel=1e4)),

    # Angular-velocity weight
    ("06_angvel_1e4_high",  merge(B, ang_vel=1e4)),
    ("07_angvel_1_low",     merge(B, ang_vel=1.0)),

    # Control weight
    ("08_ctrl_10x_heavy",   merge(B, mtq_cw=10.0, rw_cw=100.0)),
    ("09_ctrl_0.01x_light", merge(B, mtq_cw=1e-3, rw_cw=1e-2)),

    # Trajectory length (keep knots≈100 by scaling dt)
    ("10_short_100s_dt1",   merge(B, dt=1.0,  time_s=100.0)),
    ("11_long_3000s_dt30",  merge(B, dt=30.0, time_s=3000.0)),

    # Initial ω
    ("12_omega_5x",         merge(B, omega0=np.array([0.05, 0.05, 0.05]))),
    ("13_omega_10x",        merge(B, omega0=np.array([0.10, 0.10, 0.10]))),

    # Disturbances
    ("14_aero_on",          merge(B, disturbances={"aero": True})),
    ("15_gg_on",            merge(B, disturbances={"gg": True})),
    ("16_all_disturb_on",   merge(B, disturbances={"aero": True, "gg": True, "srp": True, "resdipole": True})),

    # Larger slew
    ("17_slew_180",         merge(B, goal_angle_deg=180.0)),

    # Tumble-direction ablation: 10x ω₀, hybrid (3+1) sat with RW along body-x.
    # 18: ω along body-x = RW axis → RW can damp directly (solver test).
    # 19: ω along body-y ⊥ to RW axis → RW useless on this axis (physics test).
    ("18_omega_10x_rw_aligned", merge(B, omega0=np.array([0.173, 0.0, 0.0]))),
    ("19_omega_10x_rw_perp",    merge(B, omega0=np.array([0.0, 0.173, 0.0]))),

    # MTQ-only at 10x ω₀ — no RW available, MTQ produces zero torque
    # whenever ω becomes parallel to B during the trajectory.  Pure
    # physics-limited reference.
    ("20_omega_10x_mtq_only",   merge(B, sat_fn=create_3_0,
                                      omega0=np.array([0.10, 0.10, 0.10]))),
]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
results = []
if SCENARIO_FILTER:
    keep = set(s.strip() for s in SCENARIO_FILTER.split(","))
    SCENARIOS = [s for s in SCENARIOS if s[0] in keep]

if COSTREF_ENABLED:
    print(f"Cost refactor: ang_vel_err_dir_ratio = {COSTREF_BETA} (α-from-β PSD crossterm)")
if CURRICULUM_ENABLED:
    print("Curriculum: pass 0 detumble (angle=0), pass 1 full tracking")
if INITCONTROLLER != 1:
    name = {0: "ZeroController", 2: "Bdot"}.get(INITCONTROLLER, str(INITCONTROLLER))
    print(f"Init controller: {name}")
print(f"Running {len(SCENARIOS)} scenarios, output → {OUT}\n")
for name, params in SCENARIOS:
    print(f"→ {name}", flush=True)
    t0 = time.time()
    r = run_scenario(name, params)
    if r is not None:
        print(f"  {r['stop']:<22}  it={r['iters']:>4}  "
              f"PE_mean={r['pe_mean']:>5.1f}°  PE_fin={r['pe_fin']:>5.1f}°  "
              f"excess={r['excess_deg']:>6.1f}°  ctol={('%.2e' % r['ctol']) if r['ctol'] else 'n/a':<10}  "
              f"wall={time.time()-t0:>5.1f}s", flush=True)
        results.append(r)

print("\n" + "="*120)
print("SUMMARY")
print("="*120)
print(f"{'scenario':<28} {'stop':<25} {'iters':>5} {'PE_mean':>8} {'PE_fin':>8} {'excess':>8} {'ctol':>10}")
print("-"*120)
for r in results:
    ctol_s = f"{r['ctol']:.2e}" if r['ctol'] else "  n/a   "
    print(f"{r['name']:<28} {r['stop']:<25} {r['iters']:>5} {r['pe_mean']:>6.1f}°  "
          f"{r['pe_fin']:>6.1f}°  {r['excess_deg']:>6.1f}°  {ctol_s:>10}")
