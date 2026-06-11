"""Plots for the state-slack benchmark (benchmark_state_slack.py).

Produces four figures comparing baseline AL vs slack+polish:
  1. slack_traj_headline.png   - trajectories on the hardest case
  2. slack_wmax_sweep.png      - final violation / time across the binding region
  3. slack_outer_trace.png     - violation vs outer iteration (two-phase visible)
  4. slack_battery_summary.png - full 14-case battery summary
"""

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import benchmark_state_slack as bm

OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

C_BASE = "tab:red"
C_SLACK = "tab:blue"
CONSTRAINT_TOL = 1e-3
SLACK_OFF_TOL = 0.02
SLACK_RHO = 50.0


def run(mode, wmax, tf=300.0, slew_deg=180.0, n_mtq=3, max_outer=20, w0=None, penalty_scale=10.0):
    s = bm.make_settings(dt=10.0, wmax=wmax, max_outer=max_outer)
    s.passes[0].auglag.penalty_scale = penalty_scale
    if mode == "slack":
        aug = s.passes[0].auglag
        aug.use_state_slack = True
        aug.slack_rho = SLACK_RHO
        aug.slack_off_tol = SLACK_OFF_TOL
    sat = bm.make_satellite(s, n_mtq=n_mtq)
    x0 = np.zeros(sat.stateDim)
    x0[3] = 1.0
    if w0 is not None:
        x0[0:3] = w0
    goal = bm.quat_about_z(math.radians(slew_deg))
    r = bm.run_case(s, sat, x0, tf, 10.0, goal)
    r["goal"] = goal
    return r


# ----------------------------------------------------------------------------
# Fig 1: headline case trajectories (3+3 slew180, wmax = 0.013)
# ----------------------------------------------------------------------------
print("fig 1: headline trajectories ...")
WMAX_HEAD = 0.013
res = {m: run(m, WMAX_HEAD) for m in ("baseline", "slack")}

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(
    "Headline hard case: 3+3 hybrid, 180$^\\circ$ slew, tf=300 s, wmax=0.013 rad/s\n"
    f"baseline: {res['baseline']['status']} (max c = {res['baseline']['max_c']:.2e})   |   "
    f"slack+polish: {res['slack']['status']} (max c = {res['slack']['max_c']:.2e})",
    fontsize=11,
)

for mode, color in (("baseline", C_BASE), ("slack", C_SLACK)):
    X = res[mode]["X"]
    N = X.shape[1]
    t = np.arange(N) * 10.0
    label = f"{mode} ({'converged' if res[mode]['ok'] else 'FAILED'})"

    w_norm = np.linalg.norm(X[0:3, :], axis=0)
    axes[0, 0].plot(t, w_norm, color=color, label=label)

    err = [bm.quat_angle_deg(X[3:7, k], res[mode]["goal"]) for k in range(N)]
    axes[0, 1].plot(t, err, color=color, label=label)

    viol = np.maximum(0.0, (w_norm**2 - WMAX_HEAD**2) / WMAX_HEAD**2)
    axes[1, 0].plot(t, viol, color=color, label=label)

    h = np.abs(X[7:, :]).max(axis=0)
    axes[1, 1].plot(t, h, color=color, label=label)

axes[0, 0].axhline(WMAX_HEAD, color="k", ls="--", lw=1, label="wmax")
axes[0, 0].set_ylabel("|$\\omega$| [rad/s]")
axes[0, 0].set_title("Angular rate vs limit")

axes[0, 1].set_ylabel("pointing error [deg]")
axes[0, 1].set_title("Attitude error to goal")
axes[0, 1].set_yscale("log")

axes[1, 0].axhline(CONSTRAINT_TOL, color="k", ls=":", lw=1, label="constraint_tol")
axes[1, 0].set_ylabel("normalized $\\omega$ violation")
axes[1, 0].set_title("AngularVelocity constraint violation per knot")
axes[1, 0].set_yscale("symlog", linthresh=1e-6)

axes[1, 1].axhline(0.02, color="k", ls="--", lw=1, label="h_max")
axes[1, 1].set_ylabel("max |h| [N·m·s]")
axes[1, 1].set_title("Reaction-wheel momentum")

for ax in axes.flat:
    ax.set_xlabel("time [s]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / "slack_traj_headline.png", dpi=140)
plt.close(fig)

# ----------------------------------------------------------------------------
# Fig 2: wmax sweep across the binding region
# ----------------------------------------------------------------------------
print("fig 2: wmax sweep ...")
wmaxes = [0.0120, 0.0125, 0.0130, 0.0135, 0.0140, 0.0150, 0.0160]
sweep = {m: [run(m, w) for w in wmaxes] for m in ("baseline", "slack")}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("3+3 hybrid 180$^\\circ$ slew, tf=300 s: sweep of the wmax binding region", fontsize=12)

for mode, color in (("baseline", C_BASE), ("slack", C_SLACK)):
    rs = sweep[mode]
    cs = [r["max_c"] for r in rs]
    ok = np.array([r["ok"] for r in rs])
    ax1.plot(wmaxes, cs, color=color, lw=1.5, label=f"{mode}")
    ax1.scatter(np.array(wmaxes)[ok], np.array(cs)[ok], color=color, marker="o", s=60, zorder=3)
    ax1.scatter(np.array(wmaxes)[~ok], np.array(cs)[~ok], color=color, marker="X", s=110, zorder=3)

    ts = [r["time_s"] for r in rs]
    ax2.plot(wmaxes, ts, color=color, marker="o", label=mode)

ax1.axhline(CONSTRAINT_TOL, color="k", ls=":", lw=1, label="constraint_tol")
ax1.set_yscale("log")
ax1.set_xlabel("wmax [rad/s]")
ax1.set_ylabel("final TRUE max violation")
ax1.set_title("Final violation (X = failed to converge)")
ax1.grid(alpha=0.3)
ax1.legend(fontsize=9)

ax2.set_xlabel("wmax [rad/s]")
ax2.set_ylabel("wall time [s]")
ax2.set_title("Solve time")
ax2.grid(alpha=0.3)
ax2.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT_DIR / "slack_wmax_sweep.png", dpi=140)
plt.close(fig)

# ----------------------------------------------------------------------------
# Fig 3: violation vs outer iteration (deterministic re-run trick)
# ----------------------------------------------------------------------------
print("fig 3: outer-iteration traces ...")
MAX_OUTER = 20
traces = {}
for mode in ("baseline", "slack"):
    cs, oks = [], []
    for n in range(1, MAX_OUTER + 1):
        r = run(mode, WMAX_HEAD, max_outer=n)
        cs.append(r["max_c"])
        oks.append(r["ok"])
        if r["ok"]:
            break  # converged at this outer-iteration budget; trace complete
    traces[mode] = (cs, oks)

fig, ax = plt.subplots(figsize=(8, 5))
for mode, color in (("baseline", C_BASE), ("slack", C_SLACK)):
    cs, oks = traces[mode]
    iters = np.arange(1, len(cs) + 1)
    ax.semilogy(iters, cs, color=color, marker="o", label=f"{mode}" + (" (converges)" if oks[-1] else " (never converges)"))
    if oks[-1]:
        ax.scatter([iters[-1]], [cs[-1]], color=color, marker="*", s=260, zorder=4)

ax.axhline(SLACK_OFF_TOL, color=C_SLACK, ls="--", lw=1, alpha=0.7, label="slack_off_tol (drop slacks, polish)")
ax.axhline(CONSTRAINT_TOL, color="k", ls=":", lw=1, label="constraint_tol")
ax.set_xlabel("outer-iteration budget")
ax.set_ylabel("final TRUE max violation")
ax.set_title(
    "Outer-loop progress, headline case (3+3 slew180, wmax=0.013)\n"
    "each point = a deterministic run truncated at that outer-iteration budget"
)
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT_DIR / "slack_outer_trace.png", dpi=140)
plt.close(fig)

# ----------------------------------------------------------------------------
# Fig 4: full battery summary
# ----------------------------------------------------------------------------
print("fig 4: battery summary ...")
results = []
for name, n_mtq, rw_hmax, wmax, tf, slew_deg, w0, penalty_scale in bm.CASES:
    row = {"name": name}
    for mode in ("baseline", "slack"):
        s = bm.make_settings(dt=10.0, wmax=wmax, max_outer=20)
        s.passes[0].auglag.penalty_scale = penalty_scale
        if mode == "slack":
            aug = s.passes[0].auglag
            aug.use_state_slack = True
            aug.slack_rho = SLACK_RHO
            aug.slack_off_tol = SLACK_OFF_TOL
        sat = bm.make_satellite(s, n_mtq=n_mtq, rw_hmax=rw_hmax)
        x0 = np.zeros(sat.stateDim)
        x0[0:3] = w0
        x0[3] = 1.0
        row[mode] = bm.run_case(s, sat, x0, tf, 10.0, bm.quat_about_z(math.radians(slew_deg)))
    results.append(row)

names = [r["name"] for r in results]
y = np.arange(len(names))
H = 0.38
FLOOR = 1e-8  # display floor for log axis

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
fig.suptitle("Full hard-case battery: baseline AL vs state-slack + polish (dt=10 s)", fontsize=12)

for off, mode, color in ((-H / 2, "baseline", C_BASE), (H / 2, "slack", C_SLACK)):
    cs = np.array([max(r[mode]["max_c"], FLOOR) for r in results])
    ok = np.array([r[mode]["ok"] for r in results])
    bars = ax1.barh(y + off, cs, height=H, color=color, alpha=0.85, label=mode)
    for i, (b, o) in enumerate(zip(bars, ok)):
        if not o:
            ax1.text(b.get_width() * 1.3, b.get_y() + b.get_height() / 2, "FAIL",
                     va="center", fontsize=8, color=color, fontweight="bold")

    ts = np.array([r[mode]["time_s"] for r in results])
    ax2.barh(y + off, ts, height=H, color=color, alpha=0.85, label=mode)

ax1.axvline(CONSTRAINT_TOL, color="k", ls=":", lw=1, label="constraint_tol")
ax1.set_xscale("log")
ax1.set_yticks(y)
ax1.set_yticklabels(names, fontsize=8)
ax1.invert_yaxis()
ax1.set_xlabel("final TRUE max violation")
ax1.set_title("Constraint satisfaction")
ax1.grid(alpha=0.3, axis="x")
ax1.legend(fontsize=9, loc="lower right")

ax2.set_xlabel("wall time [s]")
ax2.set_title("Solve time")
ax2.grid(alpha=0.3, axis="x")
ax2.legend(fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(OUT_DIR / "slack_battery_summary.png", dpi=140)
plt.close(fig)

print(f"wrote 4 figures to {OUT_DIR}")
