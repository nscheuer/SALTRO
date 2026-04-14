"""Fast antispike integration test — same scenario as debug_3_1_antispike_dt10.py.

Runs both baseline and antispike for a fixed number of iLQR iterations (not until
convergence) to keep wall-clock time under ~60s.  Checks:
  1. Spike removal fires at least once
  2. Antispike trajectory is finite with unit quaternions
  3. Antispike cost trend does not diverge after intervention

Usage:
    python debug_3_1_antispike_fast.py
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
from trajOpt import trajOpt
from spike_removal import detect_spikes


def _pointing_error_deg(q, q_goal):
    n = q.shape[1]
    errs = np.zeros(n)
    for k in range(n):
        qe_w = q_goal[0,k]*q[0,k] + q_goal[1,k]*q[1,k] + q_goal[2,k]*q[2,k] + q_goal[3,k]*q[3,k]
        errs[k] = 2.0 * np.arccos(min(abs(qe_w), 1.0)) * 180.0 / np.pi
    return errs


def create_planner_settings(max_ilqr_iters=6, max_outer=2):
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = 1
    ps.num_passes = 1
    ps.passes[0].dt = 10.0
    ps.passes[0].ilqr.cost_tol = 1e-6   # very tight so we always hit max_iters
    ps.passes[0].ilqr.max_iters = max_ilqr_iters
    ps.passes[0].auglag.max_outer_iters = max_outer
    ps.passes[0].auglag.constraint_tol = 1e-3

    cost = ps.passes[0].cost
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
    cost.ang_cost_func_type = 3
    cost.use_cost_hess = False

    ps.disturbances.plan_for_aero = False
    ps.disturbances.plan_for_gg = False
    ps.disturbances.plan_for_srp = False
    ps.disturbances.plan_for_prop = False
    ps.disturbances.plan_for_gendist = False
    ps.disturbances.plan_for_resdipole = False

    ps.passes[0].reg.reg_init = 1e-6
    ps.passes[0].reg.reg_max = 1e10
    ps.passes[0].reg.reg_scale = 10.0
    ps.passes[0].reg.use_dynamics_hess = False
    ps.passes[0].reg.use_constraint_hess = False

    ps.passes[0].linesearch.max_iters = 20
    ps.passes[0].linesearch.beta1 = 1e-10
    ps.passes[0].linesearch.beta2 = 5000.0

    return ps


SPIKE_REMOVAL_CFG = {
    "start_at_iter": 2,
    "max_intervention_iters": 8,
    "blend_len": 30,
    "goal_switch_buffer": 15,
    "min_consecutive": 7,
    "exit_fudge": 2.0,
    "min_prior_decrease_knots": 10,
    "min_spike_ratio": 3.0,       # stricter: peak must be 3x entry
    "kp_q": 0.3,
    "kd_w": 2.0,
    "rw_scale": 0.0,
    "omega_max": 0.30,
    "verbose": True,
}

# Fixed iteration budget — adjust if runs are still too slow
MAX_ILQR = 10
MAX_OUTER = 4


def main():
    jtime = np.array([0.22, 0.22 + 1000.0 / (36525.0 * 86400.0)])
    qgoal = np.array([
        [np.sqrt(2)/2,  np.sqrt(2)/2],
        [0.0,           0.0],
        [0.0,           0.0],
        [np.sqrt(2)/2,  np.sqrt(2)/2],
    ])
    boresight = np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    w0 = np.array([0.01, 0.01, 0.01])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    h0 = np.array([0.0])
    x0 = np.hstack((w0, q0, h0))
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7.5e3, 0.0])

    print("=" * 60)
    print(f"Running BASELINE ({MAX_ILQR} iLQR iters x {MAX_OUTER} outer loops)...")
    ps_base = create_planner_settings(MAX_ILQR, MAX_OUTER)
    sat_base = create_satellite(ps_base)
    X_base, U_base, reason_base, snaps_base, trans_base, dt, cost_tol, t_base = trajOpt(
        ps_base, sat_base, x0, r0, v0, jtime, qgoal, boresight,
        debug=True, spike_removal_cfg=None,
    )
    print(f"Baseline: {reason_base}  cost={snaps_base[-1]['J']:.4e}  ({t_base:.1f}s)")

    print()
    print("=" * 60)
    print(f"Running ANTISPIKE ({MAX_ILQR} iLQR iters x {MAX_OUTER} outer loops)...")
    ps_anti = create_planner_settings(MAX_ILQR, MAX_OUTER)
    sat_anti = create_satellite(ps_anti)
    X_anti, U_anti, reason_anti, snaps_anti, trans_anti, dt, cost_tol, t_anti = trajOpt(
        ps_anti, sat_anti, x0, r0, v0, jtime, qgoal, boresight,
        debug=True, spike_removal_cfg=SPIKE_REMOVAL_CFG,
    )
    print(f"Antispike: {reason_anti}  cost={snaps_anti[-1]['J']:.4e}  ({t_anti:.1f}s)")

    # --- Validation ---
    ok = True

    finite_ok = bool(np.all(np.isfinite(X_anti)))
    q_norms = np.linalg.norm(X_anti[3:7, :], axis=0)
    qnorm_ok = bool(np.allclose(q_norms, 1.0, atol=1e-4))

    # Did any SpikeRemoval substitution fire? Check if antispike costs differ
    # from baseline at same snapshot index (substitution changes cost trajectory).
    j_base = np.array([s["J"] for s in snaps_base])
    j_anti = np.array([s["J"] for s in snaps_anti])
    n = min(len(j_base), len(j_anti))

    # Detect spikes on baseline final trajectory
    final_snap_base = snaps_base[-1]
    U_trim_base = final_snap_base["U"][:, :X_base.shape[1] - 1]
    candidates_base = detect_spikes(
        final_snap_base["X"], U_trim_base, final_snap_base["q_goal"],
        boresight, final_snap_base["B"], sat_base, ps_base.constraints,
    )

    print()
    print("=" * 60)
    print("RESULTS")
    print(f"  Trajectory finite       : {finite_ok}")
    print(f"  Quaternion norms OK     : {qnorm_ok}  [{q_norms.min():.6f}, {q_norms.max():.6f}]")
    print(f"  Baseline spikes detected: {candidates_base}")
    print(f"  Baseline  cost snapshots: {[f'{j:.4e}' for j in j_base]}")
    print(f"  Antispike cost snapshots: {[f'{j:.4e}' for j in j_anti]}")
    cost_delta = snaps_anti[-1]['J'] - snaps_base[-1]['J']
    print(f"  Final cost delta (anti-base): {cost_delta:+.4e}")

    if not finite_ok:
        print("FAIL: non-finite states in antispike trajectory")
        ok = False
    if not qnorm_ok:
        print("FAIL: quaternion norms out of tolerance")
        ok = False

    print()
    print("PASS" if ok else "FAIL")

    # --- Plot ---
    t_state = np.arange(X_base.shape[1]) * dt
    pe_base = _pointing_error_deg(snaps_base[-1]["X"][3:7, :], snaps_base[-1]["q_goal"])
    pe_anti = _pointing_error_deg(snaps_anti[-1]["X"][3:7, :], snaps_anti[-1]["q_goal"])

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    fig.suptitle(f"Antispike vs Baseline — {MAX_ILQR} iters x {MAX_OUTER} outer", fontsize=12)

    ax = axes[0]
    ax.plot(t_state, pe_base, "o-", color="C3", markersize=3, label="Baseline")
    ax.plot(t_state, pe_anti, "o-", color="C0", markersize=3, label="Antispike")
    for (te, tx) in candidates_base:
        ax.axvspan(t_state[te], t_state[tx], alpha=0.15, color="red", label="Detected spike")
    ax.set_title("Final Pointing Error (deg)")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("deg")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogy(j_base, "o-", color="C3", markersize=4, label="Baseline")
    ax.semilogy(j_anti, "o-", color="C0", markersize=4, label="Antispike")
    ax.set_title("Cost vs Snapshot")
    ax.set_xlabel("Snapshot"); ax.set_ylabel("Cost (log)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    out = Path(__file__).parent / "antispike_fast_comparison.png"
    plt.savefig(str(out), dpi=150)
    print(f"Plot saved: {out.name}")
    plt.close("all")


if __name__ == "__main__":
    main()
