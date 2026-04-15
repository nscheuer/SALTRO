"""Test C++ spike removal via saltro_py.trajOpt.

Runs the same 3MTQ+1RW 90° slew scenario through the C++ optimizer
with and without spike removal enabled.
"""
import os
import sys
import time
import numpy as np
from pathlib import Path

# Suppress C++ debug output ([BP]/[FP] lines) by redirecting stdout temporarily
# during the C++ calls.
import contextlib

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from sat_3_1_hybrid import create_satellite


def make_settings(spike_enabled, initcontroller=1):
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = initcontroller
    ps.num_passes = 1
    ps.passes[0].dt = 10.0
    ps.passes[0].ilqr.cost_tol = 1e-6
    ps.passes[0].ilqr.max_iters = 200
    ps.passes[0].auglag.max_outer_iters = 50
    ps.passes[0].auglag.constraint_tol = 1e-3

    cost = ps.passes[0].cost
    cost.angle = 1e2
    cost.ang_vel = 1e1
    cost.control_mult = 1.0
    cost.mtq_control_weight = 1e-1
    cost.rw_control_weight = 1.0
    cost.angle_N = 1e6
    cost.ang_vel_N = 1e5
    cost.ang_cost_func_type = 3
    cost.use_cost_hess = True

    ps.disturbances.plan_for_aero = False
    ps.disturbances.plan_for_gg = False
    ps.disturbances.plan_for_srp = False
    ps.disturbances.plan_for_prop = False
    ps.disturbances.plan_for_gendist = False
    ps.disturbances.plan_for_resdipole = False

    ps.passes[0].reg.reg_init = 1e-6
    ps.passes[0].reg.reg_max = 1e10
    ps.passes[0].reg.reg_scale = 10.0

    ps.passes[0].linesearch.max_iters = 24
    ps.passes[0].linesearch.beta1 = 1e-10
    ps.passes[0].linesearch.beta2 = 5000.0

    if spike_enabled:
        sr = ps.passes[0].spike_removal
        sr.enabled = True
        sr.start_at_iter = 2
        sr.max_intervention_iters = 20
        sr.blend_len = 30
        sr.goal_switch_buffer = 15
        sr.min_consecutive = 7
        sr.exit_fudge = 2.0
        sr.min_prior_decrease_knots = 10
        sr.min_spike_ratio = 3.0
        sr.max_spike_knots = 55
        sr.kp_q = 0.3
        sr.kd_w = 2.0
        sr.rw_scale = 0.0
        sr.omega_max = 0.30
        sr.verbose = True

    return ps


def main():
    jtime = np.array([0.22, 0.22 + 1000.0 / (36525.0 * 86400.0)])
    qgoal = np.array([
        [np.sqrt(2)/2, np.sqrt(2)/2],
        [0.0, 0.0],
        [0.0, 0.0],
        [np.sqrt(2)/2, np.sqrt(2)/2],
    ])
    boresight = np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    x0 = np.array([0.01, 0.01, 0.01, 1.0, 0.0, 0.0, 0.0, 0.0])
    r0 = np.array([7e6, 0.0, 0.0])
    v0 = np.array([0.0, 7.5e3, 0.0])

    def run_cpp(label, spike_enabled, initcontroller=1):
        ps = make_settings(spike_enabled, initcontroller)
        sat = create_satellite(ps)
        # Redirect C-level stdout to suppress [BP]/[FP] debug lines.
        old_stdout = os.dup(1)
        log_path = Path(__file__).parent / f"cpp_{label.lower()}_log.txt"
        log_fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        os.dup2(log_fd, 1)
        sys.stdout.flush()
        t0 = time.time()
        ok, X, U, _K = saltro_py.trajOpt(ps, sat, x0, r0, v0, jtime, qgoal, boresight)
        elapsed = time.time() - t0
        os.dup2(old_stdout, 1)
        os.close(log_fd); os.close(old_stdout)
        for line in open(log_path):
            if "SpikeRemoval" in line:
                print(line, end="")
        return ok, X, U, elapsed

    def pe_deg(X, qg):
        N = X.shape[1]
        errs = np.zeros(N)
        for k in range(N):
            q = X[3:7, k]
            dot = np.clip(abs(float(np.dot(q, qg))), 0, 1)
            errs[k] = 2.0 * np.degrees(np.arccos(dot))
        return errs

    qg = qgoal[:, 0]

    results = {}
    for ic, ic_label in [(1, "ExcCtrl"), (2, "BdotCtrl")]:
        for spike, sp_label in [(False, "base"), (True, "anti")]:
            label = f"{ic_label}_{sp_label}"
            print("=" * 60)
            print(f"{label} (initcontroller={ic}, spike={spike})...")
            ok, X, U, t = run_cpp(label, spike, initcontroller=ic)
            pe = pe_deg(X, qg)
            results[label] = (ok, X, U, t, pe)
            conv = "CONVERGED" if ok else "not converged"
            print(f"  -> {conv}  max_pe={pe.max():.1f}°  final_pe={pe[-1]:.2f}°  time={t:.2f}s")
            print()

    # For backwards compat with plotting below
    ok_base, X_base, U_base, t_base, _ = results["ExcCtrl_base"]
    ok_anti, X_anti, U_anti, t_anti, _ = results["ExcCtrl_anti"]

    # --- Summary ---
    print()
    print("=" * 60)
    print("SUMMARY")
    ps_tmp = make_settings(False)
    sat_tmp = create_satellite(ps_tmp)
    cost_cfg = ps_tmp.passes[0].cost
    max_N = max(r[1].shape[1] for r in results.values())
    B_dummy = np.zeros((3, max_N))
    for label, (ok, X, U, t, pe) in results.items():
        N = X.shape[1]
        qn = np.linalg.norm(X[3:7, :], axis=0)
        conv = "CONVERGED" if ok else "not converged"
        U_trim = U[:, :N-1]
        B_n = B_dummy[:, :N]
        bs_n = np.tile(boresight[:, 0:1], (1, N))
        qg_n = np.tile(qgoal[:, 0:1], (1, N))
        cost = sat_tmp.totalCost(X, U_trim, B_n, bs_n, qg_n, cost_cfg)
        print(f"  {label:10s}: {conv}  N={N}  cost={cost:.4e}  "
              f"max_pe={pe.max():.1f}°  final_pe={pe[-1]:.1f}°  "
              f"qnorm=[{qn.min():.6f},{qn.max():.6f}]  time={t:.2f}s")

    # --- Plots ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dt = 10.0
    N_base = X_base.shape[1]
    N_anti = X_anti.shape[1]
    t_base_arr = np.arange(N_base) * dt
    t_anti_arr = np.arange(N_anti) * dt
    pe_base = pe_deg(X_base, qg)
    pe_anti = pe_deg(X_anti, qg)

    # 2x2 grid: rows=initcontroller, cols=base/anti
    configs_plot = [
        ("ExcCtrl_base", "ExcCtrl Baseline"),
        ("ExcCtrl_anti", "ExcCtrl Antispike"),
        ("BdotCtrl_base", "BdotCtrl Baseline"),
        ("BdotCtrl_anti", "BdotCtrl Antispike"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Pointing Error — 3MTQ+1RW 90° Slew", fontsize=14)
    colors = ['r', 'b', 'r', 'b']
    for idx, (key, title) in enumerate(configs_plot):
        ax = axes[idx // 2, idx % 2]
        ok, X, U, t, pe = results[key]
        t_arr = np.arange(len(pe)) * dt
        conv = "CONVERGED" if ok else "not converged"
        ax.set_title(f"{title} — final={pe[-1]:.2f}° ({conv})")
        ax.plot(t_arr, pe, colors[idx] + '-', linewidth=1.5)
        ax.axhline(1.0, color='g', linestyle='--', alpha=0.5, label="1°")
        ax.set_ylabel("deg")
        ax.set_xlabel("Time [s]")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(__file__).parent / "antispike_comparison.png"
    fig.savefig(str(out_path), dpi=150)
    print(f"\nPlot saved: {out_path}")


if __name__ == "__main__":
    main()
