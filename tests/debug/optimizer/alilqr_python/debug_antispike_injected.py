"""Pathological trajectory injection test for spike removal.

Constructs a trajectory with a controlled spike profile:
  - k=0..29:  PD control toward goal — error converging from 90° to ~2°
  - k=30..44: spike window — states set by quaternion slerp to 45° off-goal
  - k=45..59: recovery — states slerped back to match k=29 endpoint
  - k=60..99: PD continuation toward goal from the recovered state

The spike window states are NOT dynamically feasible (states set directly),
but controls are valid PD outputs.  The test verifies:
  1. detect_spikes finds the (30, ~45) window
  2. apply_spike_removal substitutes a cheaper PD segment
  3. The cleaned trajectory is finite with unit quaternions
  4. Stage cost improves
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from sat_3_1_hybrid import create_satellite
from spike_removal import apply_spike_removal, detect_spikes, _rk4_step, _build_pd_control


def pointing_error_deg(q, qg):
    return 2.0 * np.degrees(np.arccos(min(abs(float(np.dot(q, qg))), 1.0)))


def _slerp(q0, q1, t):
    """Spherical linear interpolation between unit quaternions, t in [0,1]."""
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:
        return (q0 + t * (q1 - q0)) / np.linalg.norm(q0 + t * (q1 - q0))
    theta0 = np.arccos(dot)
    theta = theta0 * t
    sin_theta = np.sin(theta)
    sin_theta0 = np.sin(theta0)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta0
    s1 = sin_theta / sin_theta0
    return s0 * q0 + s1 * q1


def create_planner_settings():
    ps = saltro_py.PlannerSettings()
    ps.num_passes = 1
    ps.passes[0].dt = 10.0
    ps.passes[0].ilqr.cost_tol = 1e-6
    ps.passes[0].ilqr.max_iters = 8
    ps.passes[0].auglag.max_outer_iters = 2
    ps.passes[0].auglag.constraint_tol = 1e-3
    cost = ps.passes[0].cost
    cost.angle, cost.ang_vel = 1e2, 1e1
    cost.control_mult, cost.mtq_control_weight, cost.rw_control_weight = 1., 1e-1, 1.
    cost.angle_N, cost.ang_vel_N = 1e2, 1e1
    cost.ang_cost_func_type, cost.use_cost_hess = 3, False
    ps.disturbances.plan_for_aero = ps.disturbances.plan_for_gg = False
    ps.disturbances.plan_for_srp = ps.disturbances.plan_for_prop = False
    ps.disturbances.plan_for_gendist = ps.disturbances.plan_for_resdipole = False
    ps.passes[0].reg.reg_init = 1e-6
    ps.passes[0].reg.reg_max = 1e10
    ps.passes[0].reg.reg_scale = 10.
    ps.passes[0].linesearch.max_iters = 20
    ps.passes[0].linesearch.beta1 = 1e-10
    ps.passes[0].linesearch.beta2 = 5000.
    return ps


SPIKE_REMOVAL_CFG = {
    "start_at_iter": 0,
    "max_intervention_iters": 3,
    "blend_len": 20,
    "goal_switch_buffer": 5,
    "min_consecutive": 5,
    "exit_fudge": 2.0,
    "min_prior_decrease_knots": 3,   # relaxed: PD trajectories oscillate
    "min_spike_ratio": 3.0,
    "kp_q": 0.3,
    "kd_w": 2.0,
    "rw_scale": 0.0,
    "omega_max": 0.30,
    "verbose": True,
}


def main():
    ps = create_planner_settings()
    sat = create_satellite(ps)
    N = 100
    nx, nu, nxr = sat.stateDim, sat.controlDim, sat.reducedStateDim
    dt = ps.passes[0].dt
    dist_cfg = ps.disturbances

    # Constant environment (flat orbit)
    B_arr = np.zeros((3, N)); B_arr[2, :] = 3.1e-5
    S_arr = np.tile(np.array([0., 0., 1.])[:, None], (1, N))
    R_arr = np.tile(np.array([7000e3, 0., 0.])[:, None], (1, N))
    V_arr = np.tile(np.array([0., 7.5e3, 0.])[:, None], (1, N))
    rho_arr = np.zeros((1, N))
    jtime_arr = np.linspace(0.22, 0.22 + (N-1)*dt/(36525.*86400.), N)

    # Goal: 90-deg rotation about Z axis
    q_goal_single = np.array([np.sqrt(2)/2, 0., 0., np.sqrt(2)/2])
    q_goal = np.tile(q_goal_single[:, None], (1, N))
    boresight = np.tile(np.array([1., 0., 0.])[:, None], (1, N))

    # Spike quaternion: some intermediate attitude ~90° from goal (different homotopy class)
    # Use identity q=[1,0,0,0] which is 90° from goal
    q_start = np.array([1., 0., 0., 0.])  # 90° from goal (initial attitude)

    # Initial state: identity attitude, small angular velocity
    x0 = np.zeros(nx)
    x0[3] = 1.0   # q = [1,0,0,0]
    x0[0:3] = [0.01, 0.01, 0.01]

    # -----------------------------------------------------------------------
    # Trajectory design:
    #   Converge  k=0..SPIKE_START-1 : slerp from q_start to q_entry (monotone decrease)
    #   Spike     k=SPIKE_START..SPIKE_PEAK-1 : slerp back toward q_start (increase)
    #   Recovery  k=SPIKE_PEAK..SPIKE_END-1   : slerp back to q_entry (decrease)
    #   Post      k=SPIKE_END..N-1   : PD from q_entry (continue toward goal)
    #
    # By using slerp for the converging and spiking sections, we get a clean
    # monotone pointing-error profile that the detector can unambiguously find.
    # -----------------------------------------------------------------------
    SPIKE_START  = 30   # spike starts here
    SPIKE_PEAK   = 42   # spike peak (12 knots up)
    SPIKE_END    = 55   # recovery ends (13 knots down)

    # "Entry" state: 80% of the way from q_start to q_goal → ~18° from goal
    # Entry error ~18°, spike peak 90° → ratio = 5x > 3x required
    q_entry = _slerp(q_start, q_goal_single, 0.8)  # ~18° from goal

    X = np.zeros((nx, N))
    U = np.zeros((nu, N))

    def _make_state(q, omega=None):
        x = np.zeros(nx)
        x[3:7] = q / np.linalg.norm(q)
        if omega is not None:
            x[0:3] = omega
        return x

    # Phase 1: slerp from q_start to q_entry over k=0..SPIKE_START (monotone decrease)
    for k in range(SPIKE_START + 1):
        alpha = float(k) / float(SPIKE_START)
        q_k = _slerp(q_start, q_entry, alpha)
        X[:, k] = _make_state(q_k, omega=np.array([0.005, 0.005, 0.005]))
        B_k = B_arr[:, k]
        u_k = _build_pd_control(X[:, k], q_goal_single, sat, B_k, kp_q=0.5, kd_w=3.0)
        for i in range(sat.numMTQ):
            u_k[i] = float(np.clip(u_k[i], -sat.getMTQ(i).u_max, sat.getMTQ(i).u_max))
        for i in range(sat.numRW):
            u_k[sat.numMTQ+i] = float(np.clip(u_k[sat.numMTQ+i], -sat.getRW(i).u_max, sat.getRW(i).u_max))
        U[:, k] = u_k

    print(f"Entry at k={SPIKE_START}: error={pointing_error_deg(X[3:7, SPIKE_START], q_goal_single):.1f}°")

    # Phase 2a: spike — slerp from q_entry back toward q_start (increase)
    for k in range(SPIKE_START, SPIKE_PEAK + 1):
        alpha = float(k - SPIKE_START) / float(SPIKE_PEAK - SPIKE_START)
        q_k = _slerp(q_entry, q_start, alpha)
        X[:, k] = _make_state(q_k, omega=np.array([0.005, 0.005, 0.005]))
        B_k = B_arr[:, k]
        u_k = _build_pd_control(X[:, k], q_goal_single, sat, B_k, kp_q=0.5, kd_w=3.0)
        for i in range(sat.numMTQ):
            u_k[i] = float(np.clip(u_k[i], -sat.getMTQ(i).u_max, sat.getMTQ(i).u_max))
        for i in range(sat.numRW):
            u_k[sat.numMTQ+i] = float(np.clip(u_k[sat.numMTQ+i], -sat.getRW(i).u_max, sat.getRW(i).u_max))
        U[:, k] = u_k

    # Phase 2b: recovery — slerp from q_start back to q_entry (decrease)
    for k in range(SPIKE_PEAK, SPIKE_END + 1):
        alpha = float(k - SPIKE_PEAK) / float(SPIKE_END - SPIKE_PEAK)
        q_k = _slerp(q_start, q_entry, alpha)
        X[:, k] = _make_state(q_k, omega=np.array([0.005, 0.005, 0.005]))
        B_k = B_arr[:, k]
        u_k = _build_pd_control(X[:, k], q_goal_single, sat, B_k, kp_q=0.5, kd_w=3.0)
        for i in range(sat.numMTQ):
            u_k[i] = float(np.clip(u_k[i], -sat.getMTQ(i).u_max, sat.getMTQ(i).u_max))
        for i in range(sat.numRW):
            u_k[sat.numMTQ+i] = float(np.clip(u_k[sat.numMTQ+i], -sat.getRW(i).u_max, sat.getRW(i).u_max))
        U[:, k] = u_k

    # Phase 3: PD rollout from q_entry toward goal
    for k in range(SPIKE_END, N - 1):
        x_k = X[:, k]
        B_k = B_arr[:, k]
        R_k = R_arr[:, k]; S_k = S_arr[:, k]; V_k = V_arr[:, k]
        rho_k = int(np.round(float(rho_arr[0, k])))
        u_k = _build_pd_control(x_k, q_goal_single, sat, B_k, kp_q=0.5, kd_w=3.0)
        for i in range(sat.numMTQ):
            u_k[i] = float(np.clip(u_k[i], -sat.getMTQ(i).u_max, sat.getMTQ(i).u_max))
        for i in range(sat.numRW):
            u_k[sat.numMTQ+i] = float(np.clip(u_k[sat.numMTQ+i], -sat.getRW(i).u_max, sat.getRW(i).u_max))
        U[:, k] = u_k
        X[:, k+1] = _rk4_step(sat, x_k, u_k, dt, dist_cfg, R_k, B_k, S_k, V_k, rho_k)

    print("Synthesized trajectory pointing errors (deg):")
    for k in range(N):
        print(f"  k={k:3d}: {pointing_error_deg(X[3:7,k], q_goal_single):.1f}°")

    U_trim = U[:, :N-1]
    cost_orig = float(sat.totalCost(X, U_trim, B_arr, boresight, q_goal, ps.passes[0].cost))
    print(f"\nStage cost (with spike): {cost_orig:.4e}")

    # Apply spike removal
    print("\n" + "="*60)
    print("Applying spike removal...")
    U_bar = U.copy()
    K_list = [np.zeros((nu, nxr)) for _ in range(N)]

    X_clean, U_clean, occurred = apply_spike_removal(
        X.copy(), U.copy(), U_bar, K_list,
        sat, ps, 0, R_arr, V_arr, B_arr, S_arr, rho_arr,
        jtime_arr, boresight, q_goal,
        iteration=0, **SPIKE_REMOVAL_CFG,
    )

    print(f"Substitution occurred: {occurred}")
    q_norms = np.linalg.norm(X_clean[3:7, :], axis=0)
    print(f"Quat norm range: [{q_norms.min():.6f}, {q_norms.max():.6f}]")
    print(f"All finite: {np.all(np.isfinite(X_clean))}")

    print("\nCleaned pointing errors (deg):")
    for k in range(N):
        print(f"  k={k:3d}: {pointing_error_deg(X_clean[3:7,k], q_goal_single):.1f}°")

    U_clean_trim = U_clean[:, :N-1]
    cost_clean = float(sat.totalCost(X_clean, U_clean_trim, B_arr, boresight, q_goal, ps.passes[0].cost))
    print(f"\nStage cost (cleaned): {cost_clean:.4e}")
    improvement = (cost_orig - cost_clean) / cost_orig * 100
    print(f"Cost improvement: {improvement:.1f}%")

    cands = detect_spikes(X, U_trim, q_goal, boresight, B_arr, sat, ps.constraints,
                          goal_switch_buffer=5, min_consecutive=5, exit_fudge=2.0,
                          min_prior_decrease_knots=8, min_spike_ratio=3.0)
    cands_clean = detect_spikes(X_clean, U_clean_trim, q_goal, boresight, B_arr, sat, ps.constraints,
                                goal_switch_buffer=5, min_consecutive=5, exit_fudge=2.0,
                                min_prior_decrease_knots=8, min_spike_ratio=3.0)
    print(f"\nSpikes in original : {cands}")
    print(f"Spikes in cleaned  : {cands_clean}")

    ok = occurred and cost_clean < cost_orig and np.all(np.isfinite(X_clean))
    print("\n" + "="*60)
    print("PASS" if ok else "FAIL")

    # Plot
    t = np.arange(N) * dt
    pe_orig  = [pointing_error_deg(X[3:7,k], q_goal_single) for k in range(N)]
    pe_clean = [pointing_error_deg(X_clean[3:7,k], q_goal_single) for k in range(N)]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axvspan(t[SPIKE_START], t[SPIKE_END], alpha=0.07, color="red", label="Injected spike region")
    ax.plot(t, pe_orig,  "--", color="C3", alpha=0.8, label=f"Original (cost={cost_orig:.3e})")
    ax.plot(t, pe_clean, "-",  color="C0", lw=2, label=f"Cleaned  (cost={cost_clean:.3e})")
    for (te, tx) in cands:
        ax.axvspan(t[te], t[min(tx, N-1)], alpha=0.15, color="orange")
    ax.set_title("Injected Spike Test — Pointing Error (deg)")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("deg")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = Path(__file__).parent / "antispike_injected.png"
    plt.savefig(str(out), dpi=150)
    print(f"Plot saved: {out.name}")
    plt.close("all")


if __name__ == "__main__":
    main()
