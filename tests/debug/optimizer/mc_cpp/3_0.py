import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from sat_3_0_mtq import create_satellite


DT_SECONDS = 10.0
HORIZON_SECONDS = 5000.0
NUM_SIMS = 10
MISSION_NAME = "3 MTQ + 0 RW"


def create_planner_settings():
    plannersettings = saltro_py.PlannerSettings()

    plannersettings.init_traj.initcontroller = 2

    plannersettings.num_passes = 1
    plannersettings.passes[0].dt = DT_SECONDS
    plannersettings.passes[0].ilqr.cost_tol = 1e-5
    plannersettings.passes[0].ilqr.max_iters = 20

    plannersettings.passes[0].auglag.max_outer_iters = 10
    plannersettings.passes[0].auglag.constraint_tol = 1e-3

    cost = plannersettings.passes[0].cost
    cost.angle = 1.0
    cost.ang_vel = 1e1
    cost.ang_vel_mag = 0.0
    cost.ang_vel_err_dir = 0.0
    cost.control_mult = 1.0
    cost.mtq_control_weight = 1e-2
    cost.rw_control_weight = 1.0
    cost.magic_control_weight = 0.0
    cost.rw_AM_weight = 0.0
    cost.rw_stic_weight = 0.0
    cost.RWh_max_mult = 0.0
    cost.RWh_stiction_mult = 0.0
    cost.RWh_ok_mult = 0.0
    cost.angle_N = 0.0
    cost.ang_vel_N = 0.0
    cost.ang_vel_mag_N = 0.0
    cost.ang_vel_err_dir_N = 0.0
    cost.ang_cost_func_type = 3
    cost.use_cost_hess = True

    plannersettings.disturbances.plan_for_aero = False
    plannersettings.disturbances.plan_for_gg = False
    plannersettings.disturbances.plan_for_srp = False
    plannersettings.disturbances.plan_for_prop = False
    plannersettings.disturbances.plan_for_gendist = False
    plannersettings.disturbances.plan_for_resdipole = False

    plannersettings.passes[0].reg.reg_init = 1e-6
    plannersettings.passes[0].reg.reg_max = 1e10
    plannersettings.passes[0].reg.reg_scale = 10.0
    plannersettings.passes[0].reg.use_dynamics_hess = False
    plannersettings.passes[0].reg.use_constraint_hess = False

    plannersettings.passes[0].linesearch.max_iters = 24
    plannersettings.passes[0].linesearch.beta1 = 1e-10
    plannersettings.passes[0].linesearch.beta2 = 5000.0

    return plannersettings


def _count(satellite, name):
    value = getattr(satellite, name)
    return int(value() if callable(value) else value)


def _quat_inverse(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _pointing_error_deg(X, q_goal):
    q = X[3:7, :]
    q_goal_final = q_goal[:, -1]
    err = np.zeros(q.shape[1])
    for k in range(q.shape[1]):
        q_err = _quat_multiply(_quat_inverse(q_goal_final), q[:, k])
        err[k] = 2.0 * np.arctan2(np.linalg.norm(q_err[1:]), abs(q_err[0])) * 180.0 / np.pi
    return err


def _canonicalize_quaternions(X):
    X = np.array(X, copy=True)
    flip = X[3, :] < 0.0
    X[3:7, flip] *= -1.0
    return X


def _common_stack(arrays):
    n = min(arr.shape[1] for arr in arrays)
    return np.stack([arr[:, :n] for arr in arrays], axis=0)


def _series(results, num_mtq, num_rw, q_goal):
    Xs = _common_stack([_canonicalize_quaternions(result["X"]) for result in results])
    Us = _common_stack([result["U"] for result in results])
    pointing = np.stack([_pointing_error_deg(Xs[i], q_goal) for i in range(Xs.shape[0])], axis=0)[:, None, :]

    series = [
        ("Quaternion", Xs[:, 3:7, :], ["q0", "q1", "q2", "q3"], "q", DT_SECONDS),
        ("Angular Velocity", Xs[:, 0:3, :], ["wx", "wy", "wz"], "rad/s", DT_SECONDS),
        ("Pointing Error", pointing, ["err"], "deg", DT_SECONDS),
    ]
    if num_rw > 0:
        series.append(("Wheel Momentum", Xs[:, 7:7 + num_rw, :], [f"h_rw{i}" for i in range(num_rw)], "N m s", DT_SECONDS))
    if num_mtq > 0:
        series.append(("MTQ Control", Us[:, :num_mtq, :], [f"m_mtq{i}" for i in range(num_mtq)], "A m^2", DT_SECONDS))
    if num_rw > 0:
        rw_start = num_mtq
        series.append(("RW Control", Us[:, rw_start:rw_start + num_rw, :], [f"tau_rw{i}" for i in range(num_rw)], "N m", DT_SECONDS))
    return series


def _plot_spaghetti(results, satellite, q_goal, title):
    import matplotlib.pyplot as plt

    num_mtq = _count(satellite, "numMTQ")
    num_rw = _count(satellite, "numRW")
    series = _series(results, num_mtq, num_rw, q_goal)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    axes = axes.ravel()

    for ax, (name, values, labels, ylabel, dt) in zip(axes, series):
        t = np.arange(values.shape[2]) * dt
        for run_idx in range(values.shape[0]):
            for comp_idx, label in enumerate(labels):
                ax.plot(
                    t,
                    values[run_idx, comp_idx, :],
                    color=f"C{comp_idx}",
                    alpha=0.35,
                    linewidth=1.0,
                    label=label if run_idx == 0 else None,
                )
        ax.set_title(name)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for ax in axes[len(series):]:
        ax.axis("off")

    fig.suptitle(f"{title} - Monte Carlo Spaghetti", fontsize=14, fontweight="bold")
    return fig


def _plot_mean_sigma(results, satellite, q_goal, title):
    import matplotlib.pyplot as plt

    num_mtq = _count(satellite, "numMTQ")
    num_rw = _count(satellite, "numRW")
    series = _series(results, num_mtq, num_rw, q_goal)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    axes = axes.ravel()

    for ax, (name, values, labels, ylabel, dt) in zip(axes, series):
        t = np.arange(values.shape[2]) * dt
        mean = np.mean(values, axis=0)
        sigma = np.std(values, axis=0, ddof=1) if values.shape[0] > 1 else np.zeros_like(mean)
        for comp_idx, label in enumerate(labels):
            color = f"C{comp_idx}"
            ax.plot(t, mean[comp_idx, :], color=color, linewidth=1.8, label=f"{label} mean")
            ax.fill_between(
                t,
                mean[comp_idx, :] - sigma[comp_idx, :],
                mean[comp_idx, :] + sigma[comp_idx, :],
                color=color,
                alpha=0.18,
                linewidth=0.0,
            )
        ax.set_title(name)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for ax in axes[len(series):]:
        ax.axis("off")

    fig.suptitle(f"{title} - Monte Carlo Mean +/- 1 sigma", fontsize=14, fontweight="bold")
    return fig


def _mission():
    jtime = np.array([0.22, 0.22 + HORIZON_SECONDS / (36525.0 * 86400.0)])
    q_goal = np.array([
        [np.sqrt(2.0) / 2.0, np.sqrt(2.0) / 2.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [np.sqrt(2.0) / 2.0, np.sqrt(2.0) / 2.0],
    ])
    boresight = np.array([
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ])
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7.5e3, 0.0])
    return jtime, q_goal, boresight, r0, v0


def _initial_state(rng, num_rw, rate_std):
    w0 = rng.normal(0.0, rate_std, size=3)
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    h0 = np.zeros(num_rw)
    return np.hstack((w0, q0, h0))


def run_monte_carlo(num_sims=NUM_SIMS, seed=300, rate_std=0.0):
    rng = np.random.default_rng(seed)
    jtime, q_goal, boresight, r0, v0 = _mission()
    results = []
    last_satellite = None
    run_length_steps = int(round(HORIZON_SECONDS / DT_SECONDS))

    print(
        f"{MISSION_NAME}: num_sims={num_sims}, run_length={HORIZON_SECONDS:.0f}s "
        f"({run_length_steps} steps), dt={DT_SECONDS:.2f}s"
    )

    for run_idx in range(num_sims):
        plannersettings = create_planner_settings()
        satellite = create_satellite(plannersettings)
        num_rw = _count(satellite, "numRW")
        x0 = _initial_state(rng, num_rw, rate_std)

        start = time.time()
        ok, X, U, _K = saltro_py.trajOpt(
            plannersettings,
            satellite,
            x0,
            r0,
            v0,
            jtime,
            q_goal,
            boresight,
        )
        elapsed = time.time() - start
        if not ok:
            raise RuntimeError(f"trajOpt failed on run {run_idx + 1}/{num_sims}")

        results.append({"X": np.asarray(X), "U": np.asarray(U), "x0": x0, "elapsed": elapsed})
        last_satellite = satellite
        print(
            f"run {run_idx + 1:02d}/{num_sims}: "
            f"elapsed={elapsed:.3f}s, w0={np.array2string(x0[:3], precision=4)}"
        )

    avg_elapsed = float(np.mean([result["elapsed"] for result in results])) if results else 0.0
    print(
        f"average elapsed={avg_elapsed:.3f}s over {len(results)} runs "
        f"(run_length={HORIZON_SECONDS:.0f}s, dt={DT_SECONDS:.2f}s)"
    )

    return results, last_satellite, q_goal


def parse_args():
    parser = argparse.ArgumentParser(description=f"Monte Carlo AL-iLQR C++ sweep for {MISSION_NAME}.")
    parser.add_argument("--num-sims", type=int, default=NUM_SIMS, help="Number of Monte Carlo simulations.")
    parser.add_argument("--seed", type=int, default=300, help="Random seed for initial angular rates.")
    parser.add_argument("--rate-std", type=float, default=0.002, help="Initial angular-rate 1-sigma perturbation [rad/s].")
    parser.add_argument("--save-dir", type=Path, default=None, help="Optional directory for PNG plots.")
    parser.add_argument("--no-show", action="store_true", help="Do not open matplotlib windows.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")

    results, satellite, q_goal = run_monte_carlo(args.num_sims, args.seed, args.rate_std)
    title = f"{MISSION_NAME}, 90 deg slew, {HORIZON_SECONDS:.0f}s horizon, dt={DT_SECONDS:.0f}s"
    figs = [
        _plot_spaghetti(results, satellite, q_goal, title),
        _plot_mean_sigma(results, satellite, q_goal, title),
    ]

    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        figs[0].savefig(args.save_dir / "3_0_spaghetti.png", dpi=150)
        figs[1].savefig(args.save_dir / "3_0_mean_sigma.png", dpi=150)

    if not args.no_show:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
