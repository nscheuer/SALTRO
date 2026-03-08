"""Simple warm-start controller test (3 RW only)."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ilqr"))

import saltro_py
from create_3rw_sat import create_3rw_satellite
from warmstart_viewer import launch_viewer


SEC_PER_CENTURY = 36525.0 * 86400.0


def setup_planner_settings(init_controller: int = 2, dt: float = 10.0):
    """Create planner settings for warm-start."""
    settings = saltro_py.PlannerSettings()
    settings.num_passes = 1
    settings.passes[0].dt = dt
    settings.init_traj.initcontroller = init_controller

    settings.disturbances.plan_for_aero = False
    settings.disturbances.plan_for_gg = False
    settings.disturbances.plan_for_srp = False
    settings.disturbances.plan_for_prop = False
    settings.disturbances.plan_for_gendist = False
    settings.disturbances.plan_for_resdipole = False

    return settings


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
    """Create initial state with non-zero angular velocity and RW momentum states."""
    x0 = np.zeros(satellite.stateDim)
    x0[0:3] = np.array([0.01, -0.01, 0.01])
    x0[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    # RW momentum states start at zero
    if satellite.numRW > 0:
        x0[7 : 7 + satellite.numRW] = 0.0
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

    rw_u = U[0:3, :] if U.shape[0] >= 3 else np.zeros((0, U.shape[1]))

    print(f"Warm-start calculation time: {ws_time * 1000:.2f} ms")
    print(f"Trajectory shape: X={X.shape}, U={U.shape}")
    print(f"Initial angular rates [rad/s]: {X[0:3, 0]}")
    print(f"Final angular rates   [rad/s]: {X[0:3, -1]}")
    print(f"|w| reduction: {w0:.6f} -> {wf:.6f} (x{(w0 / max(wf, 1e-12)):.2f})")

    if rw_u.size > 0:
        print("RW usage [N m] (min / mean / max abs per axis):")
        for i in range(rw_u.shape[0]):
            ua = np.abs(rw_u[i, :])
            print(f"  rw{i}: {ua.min():.6f} / {ua.mean():.6f} / {ua.max():.6f}")


def main():
    """Main entry point: run warm-start and visualize results."""
    N = 90
    dt = 60.0
    init_controller = 2

    settings = setup_planner_settings(init_controller, dt)
    satellite = create_3rw_satellite(settings)

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
    launch_viewer(X, U, dt, satellite, title="Warm-start: 3 RW Detumble")


if __name__ == "__main__":
    main()
