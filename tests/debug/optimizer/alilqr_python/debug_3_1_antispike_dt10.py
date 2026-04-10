"""Antispike integration test: 3 MTQ + 1 RW, 180-deg slew, spike removal enabled.

Runs the same scenario as debug_3_1_spike_hunt.py but with spike_removal_cfg
passed to trajOpt.  Produces a side-by-side comparison plot (no viewer) and
then launches the interactive viewer for the antispike run.

Usage:
    python debug_3_1_antispike_dt10.py
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from sat_3_1_hybrid import create_satellite
from trajOpt import trajOpt
from ilqr_viewer import launch_viewer
from spike_removal import detect_spikes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _pointing_error_deg(q, q_goal):
    n = q.shape[1]
    errs = np.zeros(n)
    for k in range(n):
        qe = _quat_multiply(
            np.array([q_goal[0, k], -q_goal[1, k], -q_goal[2, k], -q_goal[3, k]]),
            q[:, k],
        )
        errs[k] = 2.0 * np.arctan2(np.linalg.norm(qe[1:]), abs(qe[0])) * 180.0 / np.pi
    return errs


# ---------------------------------------------------------------------------
# Planner settings (identical for both runs)
# ---------------------------------------------------------------------------

def create_planner_settings():
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = 1
    ps.num_passes = 1
    ps.passes[0].dt = 10.0
    ps.passes[0].ilqr.cost_tol = 1e-3
    ps.passes[0].ilqr.max_iters = 20
    ps.passes[0].auglag.max_outer_iters = 10
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

    ps.passes[0].linesearch.max_iters = 24
    ps.passes[0].linesearch.beta1 = 1e-10
    ps.passes[0].linesearch.beta2 = 5000.0

    return ps


# Spike removal configuration
SPIKE_REMOVAL_CFG = {
    "start_at_iter": 2,
    "max_intervention_iters": 5,
    "blend_len": 30,
    "goal_switch_buffer": 15,
    "min_consecutive": 7,
    "exit_fudge": 2.0,
    "kp_q": 2.0,
    "kd_w": 5.0,
    "verbose": True,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Scenario: 3 MTQ + 1 RW, 90-deg slew, use_cost_hess=False to produce local minima
    jtime = np.array([0.22, 0.22 + 1000.0 / (36525.0 * 86400.0)])
    qgoal = np.array([
        [np.sqrt(2)/2,  np.sqrt(2)/2],
        [0.0,           0.0],
        [0.0,           0.0],
        [np.sqrt(2)/2,  np.sqrt(2)/2],
    ])
    boresight = np.array([
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ])
    w0 = np.array([0.01, 0.01, 0.01])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    h0 = np.array([0.0])
    x0 = np.hstack((w0, q0, h0))
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7.5e3, 0.0])

    # --- Baseline run (no spike removal) ---
    print("=" * 60)
    print("Running BASELINE (no spike removal)...")
    ps_base = create_planner_settings()
    sat_base = create_satellite(ps_base)
    X_base, U_base, reason_base, snaps_base, trans_base, dt, cost_tol, t_base = trajOpt(
        ps_base, sat_base, x0, r0, v0, jtime, qgoal, boresight,
        debug=True, spike_removal_cfg=None,
    )
    print(f"Baseline finished: {reason_base}")
    print(f"Baseline final cost: {snaps_base[-1]['J']:.6e}  ({t_base:.2f}s)")

    # --- Antispike run ---
    print()
    print("=" * 60)
    print("Running ANTISPIKE (spike removal enabled)...")
    ps_anti = create_planner_settings()
    sat_anti = create_satellite(ps_anti)
    X_anti, U_anti, reason_anti, snaps_anti, trans_anti, dt, cost_tol, t_anti = trajOpt(
        ps_anti, sat_anti, x0, r0, v0, jtime, qgoal, boresight,
        debug=True, spike_removal_cfg=SPIKE_REMOVAL_CFG,
    )
    print(f"Antispike finished: {reason_anti}")
    print(f"Antispike final cost: {snaps_anti[-1]['J']:.6e}  ({t_anti:.2f}s)")

    # --- Summary table ---
    print()
    print("=" * 60)
    print("SUMMARY")
    print(f"  Baseline final cost : {snaps_base[-1]['J']:.6e}")
    print(f"  Antispike final cost: {snaps_anti[-1]['J']:.6e}")
    cost_delta = snaps_anti[-1]['J'] - snaps_base[-1]['J']
    print(f"  Delta (anti - base) : {cost_delta:+.6e}  {'BETTER' if cost_delta <= 0 else 'WORSE'}")

    # --- Comparison plot ---
    t_state = np.arange(X_base.shape[1]) * dt

    # Compute pointing error for final snapshot of each run
    def _pe(snaps):
        snap = snaps[-1]
        return _pointing_error_deg(snap["X"][3:7, :], snap["q_goal"])

    pe_base = _pe(snaps_base)
    pe_anti = _pe(snaps_anti)

    # Detect spikes on baseline final trajectory (diagnostic)
    final_snap_base = snaps_base[-1]
    U_trim_base = final_snap_base["U"][:, :X_base.shape[1] - 1]
    candidates_base = detect_spikes(
        final_snap_base["X"],
        U_trim_base,
        final_snap_base["q_goal"],
        boresight,
        final_snap_base["B"],
        sat_base,
        ps_base.constraints,
    )

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle("Antispike vs Baseline: 3MTQ+1RW 180° Slew", fontsize=13, fontweight="bold")

    # Pointing error comparison
    ax = axes[0]
    ax.plot(t_state, pe_base, "o-", color="C3", markersize=3, label="Baseline")
    ax.plot(t_state, pe_anti, "o-", color="C0", markersize=3, label="Antispike")
    for (te, tx) in candidates_base:
        ax.axvspan(t_state[te], t_state[tx], alpha=0.15, color="red", label="_spike" if te > 0 else "Detected spike")
    ax.set_title("Final Pointing Error (deg)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("deg")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Cost convergence
    ax = axes[1]
    j_base = [s["J"] for s in snaps_base]
    j_anti = [s["J"] for s in snaps_anti]
    ax.semilogy(j_base, "o-", color="C3", markersize=4, label="Baseline")
    ax.semilogy(j_anti, "o-", color="C0", markersize=4, label="Antispike")
    ax.set_title("Cost vs Iteration Snapshot")
    ax.set_xlabel("Snapshot index")
    ax.set_ylabel("Total cost (log)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    # Angular velocity
    ax = axes[2]
    ax.plot(t_state, np.linalg.norm(X_base[0:3, :], axis=0), color="C3", label="||ω|| baseline")
    ax.plot(t_state, np.linalg.norm(X_anti[0:3, :], axis=0), color="C0", label="||ω|| antispike")
    ax.set_title("Angular Velocity Magnitude (final trajectory)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("rad/s")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(Path(__file__).parent / "antispike_comparison.png"), dpi=150)
    print("\nComparison plot saved: antispike_comparison.png")
    plt.show(block=False)

    # Launch interactive viewer for antispike run
    print("Launching interactive viewer for antispike run...")
    launch_viewer(snaps_anti, trans_anti, reason_anti, dt, cost_tol)


if __name__ == "__main__":
    main()
