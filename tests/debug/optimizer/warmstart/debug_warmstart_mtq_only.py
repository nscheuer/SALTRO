"""Simple warm-start controller test (3 MTQ only)."""

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
import saltro_py


SEC_PER_CENTURY = 36525.0 * 86400.0


def setup_satellite(init_controller: int = 2):
    """Create 3 MTQ only satellite and planner settings."""
    settings = saltro_py.PlannerSettings()
    settings.num_passes = 1
    settings.passes[0].dt = 10.0
    settings.init_traj.initcontroller = init_controller

    settings.disturbances.plan_for_aero = False
    settings.disturbances.plan_for_gg = False
    settings.disturbances.plan_for_srp = False
    settings.disturbances.plan_for_prop = False
    settings.disturbances.plan_for_gendist = False
    settings.disturbances.plan_for_resdipole = False

    J = np.diag([0.067, 0.071, 0.069])
    satellite = saltro_py.Satellite(J, settings)

    # MTQ only - no reaction wheels
    satellite.addMTQ(np.array([1.0, 0.0, 0.0]), 0.2)
    satellite.addMTQ(np.array([0.0, 1.0, 0.0]), 0.2)
    satellite.addMTQ(np.array([0.0, 0.0, 1.0]), 0.2)

    return satellite, settings


def make_time_grid(N: int, dt: float):
    """Create time grid in Julian centuries."""
    jtime = np.zeros(N)
    dt_centuries = dt / SEC_PER_CENTURY
    for k in range(N):
        jtime[k] = 0.25 + k * dt_centuries
    return jtime


def make_targets(N: int):
    """Create identity attitude target trajectory."""
    attitude_target_traj = np.zeros((4, N))
    attitude_target_traj[0, :] = 1.0
    boresight = np.zeros((3, N))
    boresight[0, :] = 1.0
    return attitude_target_traj, boresight


def make_environment(jtime: np.ndarray):
    """Generate orbital environment."""
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7500.0, 0.0])
    ok, R, V, B, S, rho = saltro_py.generate_orbit(r0, v0, jtime, 0, 0, 0, 0, 0)
    if not ok:
        raise RuntimeError("generate_orbit failed")
    return R, V, B, S, rho.reshape(1, -1)


def make_initial_state(satellite):
    """Create initial state with non-zero angular velocity."""
    x0 = np.zeros(satellite.stateDim)
    x0[0:3] = np.array([0.01, -0.01, 0.01])
    x0[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    return x0


def generate_warmstart(satellite, settings, x0, jtime, attitude_target_traj, boresight, R, V, B, S, rho):
    """Run warm-start and return X, U, elapsed seconds."""
    start_time = time.time()
    ok, X, U = saltro_py.warm_start(
        settings,
        satellite,
        x0,
        jtime,
        attitude_target_traj,
        boresight,
        R,
        V,
        B,
        S,
        rho,
    )
    elapsed = time.time() - start_time

    if not ok:
        raise RuntimeError("warm_start failed")

    return X, U, elapsed


def print_summary(X: np.ndarray, U: np.ndarray, ws_time: float):
    """Print compact diagnostics to evaluate warm-start controller behavior."""
    w0 = np.linalg.norm(X[0:3, 0])
    wf = np.linalg.norm(X[0:3, -1])

    mtq_u = U[0:3, :] if U.shape[0] >= 3 else np.zeros((0, U.shape[1]))

    print(f"Warm-start calculation time: {ws_time * 1000:.2f} ms")
    print(f"Trajectory shape: X={X.shape}, U={U.shape}")
    print(f"Initial angular rates [rad/s]: {X[0:3, 0]}")
    print(f"Final angular rates   [rad/s]: {X[0:3, -1]}")
    print(f"|w| reduction: {w0:.6f} -> {wf:.6f} (x{(w0 / max(wf, 1e-12)):.2f})")

    if mtq_u.size > 0:
        print("MTQ usage [A m^2] (min / mean / max abs per axis):")
        for i in range(mtq_u.shape[0]):
            ua = np.abs(mtq_u[i, :])
            print(f"  m{i}: {ua.min():.4f} / {ua.mean():.4f} / {ua.max():.4f}")


def plot_results(X: np.ndarray, U: np.ndarray, dt: float, satellite=None):
    """Plot quaternion, angular rates, and MTQ control."""
    N = X.shape[1]
    t_state = np.arange(N) * dt
    t_control = np.arange(U.shape[1]) * dt

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    ax_q, ax_w, ax_mtq, ax_empty = axes.flatten()

    q = X[3:7, :]
    w = X[0:3, :]

    for i in range(q.shape[0]):
        ax_q.plot(t_state, q[i, :], label=f"q{i}")
    ax_q.set_title("Quaternion")
    ax_q.set_xlabel("Time [s]")
    ax_q.set_ylabel("q")
    ax_q.grid(True, alpha=0.3)
    ax_q.legend(fontsize=8)

    for i in range(w.shape[0]):
        ax_w.plot(t_state, w[i, :], label=f"w{i}")
    ax_w.plot(t_state, np.linalg.norm(w, axis=0), "k--", label="|w|")
    ax_w.set_title("Angular Rate")
    ax_w.set_xlabel("Time [s]")
    ax_w.set_ylabel("rad/s")
    ax_w.grid(True, alpha=0.3)
    ax_w.legend(fontsize=8)

    mtq_u = U[0:3, :] if U.shape[0] >= 3 else np.zeros((0, U.shape[1]))
    if mtq_u.size > 0:
        for i in range(mtq_u.shape[0]):
            ax_mtq.plot(t_control, mtq_u[i, :], label=f"m_mtq{i}")
        
        # Plot MTQ limits as dotted lines if satellite is provided
        if satellite is not None:
            for i in range(min(mtq_u.shape[0], satellite.numMTQ)):
                mtq = satellite.getMTQ(i)
                u_max = abs(mtq.u_max)
                ax_mtq.axhline(y=u_max, color='gray', linestyle=':', linewidth=1, alpha=0.7)
                ax_mtq.axhline(y=-u_max, color='gray', linestyle=':', linewidth=1, alpha=0.7)
        
        ax_mtq.legend(fontsize=8)
    else:
        ax_mtq.text(0.5, 0.5, "No MTQ controls", ha="center", va="center", transform=ax_mtq.transAxes)
    ax_mtq.set_title("MTQ Control")
    ax_mtq.set_xlabel("Time [s]")
    ax_mtq.set_ylabel("A m^2")
    ax_mtq.grid(True, alpha=0.3)

    ax_empty.axis("off")

    fig.suptitle("Warm-start Controller Test (MTQ Only)", fontsize=13)
    plt.show()


def main():
    """Main entry point: run warm-start and print diagnostics only."""
    N = 100
    dt = 10.0
    init_controller = 2

    satellite, settings = setup_satellite(init_controller)
    jtime = make_time_grid(N, dt)
    attitude_target_traj, boresight = make_targets(N)
    R, V, B, S, rho = make_environment(jtime)
    x0 = make_initial_state(satellite)

    X, U, ws_time = generate_warmstart(
        satellite,
        settings,
        x0,
        jtime,
        attitude_target_traj,
        boresight,
        R,
        V,
        B,
        S,
        rho,
    )

    print(f"initcontroller: {init_controller} (0=zero, 1=excitation, 2=IntegratedBdot)")
    print_summary(X, U, ws_time)
    plot_results(X, U, dt, satellite)


if __name__ == "__main__":
    main()
