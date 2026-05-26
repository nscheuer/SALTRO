import sys
import os
from pathlib import Path

import numpy as np
import matplotlib
from matplotlib.gridspec import GridSpec


def _configure_backend():
    # Prefer a desktop backend that does not depend on Qt on Linux.
    if os.environ.get("MPLBACKEND"):
        return

    if sys.platform.startswith("linux"):
        try:
            import tkinter  # noqa: F401

            matplotlib.use("TkAgg")
            return
        except Exception:
            # WebAgg gives an interactive browser-based window when Tk is unavailable.
            try:
                matplotlib.use("WebAgg")
                return
            except Exception:
                matplotlib.use("Agg")


_configure_backend()
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "build"))
import saltro_py


def main():
    # Initial state (same as C++ test)
    r0 = np.array([7000e3, 0.0, 0.0], dtype=float)
    v0 = np.array([0.0, 7.5e3, 0.0], dtype=float)

    # Requested debug horizon and time step: 10 s over 5400 s (inclusive endpoint).
    dt = 10.0
    duration = 5400.0
    n_steps = int(duration / dt) + 1

    # generate_orbit expects Julian centuries, not seconds.
    sec_per_julian_century = 36525.0 * 86400.0
    t_seconds = np.arange(n_steps, dtype=float) * dt
    jtime0_century = 0.0
    jtime = jtime0_century + t_seconds / sec_per_julian_century

    # Run with model 0 for everything
    ok, R, V, B, S, rho = saltro_py.generate_orbit(
        r0,
        v0,
        jtime,
        1,  # orbit_model
        1,  # magnetic_model
        0,  # sun_model
        0,  # eclipse_model
        0,  # density_model
    )

    if not ok:
        print("generate_orbit failed")
        return

    # Basic sanity checks for debug visibility.
    if R.shape != (3, n_steps) or V.shape != (3, n_steps):
        raise RuntimeError(f"Unexpected state shapes: R={R.shape}, V={V.shape}")
    if B.shape != (3, n_steps) or S.shape != (3, n_steps):
        raise RuntimeError(f"Unexpected environment shapes: B={B.shape}, S={S.shape}")
    if rho.shape != (n_steps,):
        raise RuntimeError(f"Unexpected density shape: rho={rho.shape}")
    if not np.all(np.isfinite(R)) or not np.all(np.isfinite(V)):
        raise RuntimeError("Non-finite values detected in propagated state")
    if not np.all(np.isfinite(B)) or not np.all(np.isfinite(S)) or not np.all(np.isfinite(rho)):
        raise RuntimeError("Non-finite values detected in environment data")
    if np.any(rho < 0.0):
        raise RuntimeError("Negative density encountered")

    # Derived diagnostic quantities.
    time_s = t_seconds
    time_h = time_s / 3600.0
    r_mag = np.linalg.norm(R, axis=0)
    v_mag = np.linalg.norm(V, axis=0)
    b_mag = np.linalg.norm(B, axis=0)
    s_mag = np.linalg.norm(S, axis=0)

    earth_radius_m = 6371e3
    altitude_m = r_mag - earth_radius_m

    # Sun direction unit vector for directional time-series debugging.
    s_safe = np.where(s_mag > 0.0, s_mag, 1.0)
    s_hat = S / s_safe

    print(f"Propagation complete: n_steps={n_steps}, dt={dt:.1f}s, duration={duration:.1f}s")
    print(f"|R| [km]: start={r_mag[0] / 1e3:.3f}, end={r_mag[-1] / 1e3:.3f}")
    print(f"|V| [m/s]: start={v_mag[0]:.3f}, end={v_mag[-1]:.3f}")
    print(f"|B| [uT]: start={b_mag[0] * 1e6:.3f}, end={b_mag[-1] * 1e6:.3f}")
    print(f"rho [kg/m^3]: start={rho[0]:.3e}, end={rho[-1]:.3e}")

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.30)

    # Orbit shape views.
    ax_orbit_3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_orbit_3d.plot(R[0] / 1e3, R[1] / 1e3, R[2] / 1e3, color="tab:blue", lw=1.2)
    ax_orbit_3d.scatter(r0[0] / 1e3, r0[1] / 1e3, r0[2] / 1e3, color="tab:green", s=40, label="start")
    ax_orbit_3d.scatter(R[0, -1] / 1e3, R[1, -1] / 1e3, R[2, -1] / 1e3, color="tab:red", s=50, label="end")
    ax_orbit_3d.set_title("Orbit 3D")
    ax_orbit_3d.set_xlabel("X [km]")
    ax_orbit_3d.set_ylabel("Y [km]")
    ax_orbit_3d.set_zlabel("Z [km]")
    ax_orbit_3d.legend(loc="upper right", fontsize=8)

    ax_xy = fig.add_subplot(gs[0, 1])
    ax_xy.plot(R[0] / 1e3, R[1] / 1e3, color="tab:blue", lw=1.2)
    ax_xy.scatter(r0[0] / 1e3, r0[1] / 1e3, color="tab:green", s=28)
    ax_xy.scatter(R[0, -1] / 1e3, R[1, -1] / 1e3, color="tab:red", s=32)
    ax_xy.set_title("Orbit XY")
    ax_xy.set_xlabel("X [km]")
    ax_xy.set_ylabel("Y [km]")
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.grid(alpha=0.35)

    ax_xz = fig.add_subplot(gs[0, 2])
    ax_xz.plot(R[0] / 1e3, R[2] / 1e3, color="tab:orange", lw=1.2)
    ax_xz.scatter(r0[0] / 1e3, r0[2] / 1e3, color="tab:green", s=28)
    ax_xz.scatter(R[0, -1] / 1e3, R[2, -1] / 1e3, color="tab:red", s=32)
    ax_xz.set_title("Orbit XZ")
    ax_xz.set_xlabel("X [km]")
    ax_xz.set_ylabel("Z [km]")
    ax_xz.set_aspect("equal", adjustable="box")
    ax_xz.grid(alpha=0.35)

    # Propagation health and environmental diagnostics.
    ax_alt = fig.add_subplot(gs[1, 0])
    ax_alt.plot(time_h, altitude_m / 1e3, color="tab:blue", lw=1.4)
    ax_alt.set_title("Altitude vs Time")
    ax_alt.set_xlabel("Time [h]")
    ax_alt.set_ylabel("Altitude [km]")
    ax_alt.grid(alpha=0.35)

    ax_v = fig.add_subplot(gs[1, 1])
    ax_v.plot(time_h, V[0], label="Vx", color="tab:red", lw=1.0)
    ax_v.plot(time_h, V[1], label="Vy", color="tab:green", lw=1.0)
    ax_v.plot(time_h, V[2], label="Vz", color="tab:blue", lw=1.0)
    ax_v.plot(time_h, v_mag, label="|V|", color="k", lw=1.4, alpha=0.8)
    ax_v.set_title("Velocity vs Time")
    ax_v.set_xlabel("Time [h]")
    ax_v.set_ylabel("Velocity [m/s]")
    ax_v.legend(loc="best", fontsize=8)
    ax_v.grid(alpha=0.35)

    ax_b = fig.add_subplot(gs[1, 2])
    ax_b.plot(time_h, B[0] * 1e6, label="Bx", color="tab:red", lw=1.0)
    ax_b.plot(time_h, B[1] * 1e6, label="By", color="tab:green", lw=1.0)
    ax_b.plot(time_h, B[2] * 1e6, label="Bz", color="tab:blue", lw=1.0)
    ax_b.plot(time_h, b_mag * 1e6, label="|B|", color="k", lw=1.4, alpha=0.8)
    ax_b.set_title("Magnetic Field vs Time")
    ax_b.set_xlabel("Time [h]")
    ax_b.set_ylabel("B [uT]")
    ax_b.legend(loc="best", fontsize=8)
    ax_b.grid(alpha=0.35)

    ax_rho = fig.add_subplot(gs[2, 0])
    ax_rho.semilogy(time_h, rho, color="tab:purple", lw=1.4)
    ax_rho.set_title("Density vs Time")
    ax_rho.set_xlabel("Time [h]")
    ax_rho.set_ylabel("rho [kg/m^3]")
    ax_rho.grid(alpha=0.35, which="both")

    ax_s = fig.add_subplot(gs[2, 1])
    ax_s.plot(time_h, S[0] / 1e9, label="Sx", color="tab:red", lw=1.0)
    ax_s.plot(time_h, S[1] / 1e9, label="Sy", color="tab:green", lw=1.0)
    ax_s.plot(time_h, S[2] / 1e9, label="Sz", color="tab:blue", lw=1.0)
    ax_s.plot(time_h, s_mag / 1e9, label="|S|", color="k", lw=1.4, alpha=0.8)
    ax_s.set_title("Sun Vector vs Time")
    ax_s.set_xlabel("Time [h]")
    ax_s.set_ylabel("S [Gm]")
    ax_s.legend(loc="best", fontsize=8)
    ax_s.grid(alpha=0.35)

    ax_s_hat = fig.add_subplot(gs[2, 2])
    ax_s_hat.plot(time_h, s_hat[0], label="S_hat_x", color="tab:red", lw=1.0)
    ax_s_hat.plot(time_h, s_hat[1], label="S_hat_y", color="tab:green", lw=1.0)
    ax_s_hat.plot(time_h, s_hat[2], label="S_hat_z", color="tab:blue", lw=1.0)
    ax_s_hat.set_title("Sun Direction vs Time")
    ax_s_hat.set_xlabel("Time [h]")
    ax_s_hat.set_ylabel("Unit vector [-]")
    ax_s_hat.set_ylim(-1.1, 1.1)
    ax_s_hat.legend(loc="best", fontsize=8)
    ax_s_hat.grid(alpha=0.35)

    fig.suptitle(
        f"Orbit Debug Dashboard | dt={dt:.1f} s | duration={duration:.0f} s | N={n_steps}",
        fontsize=14,
        fontweight="bold",
    )
    plt.show()


if __name__ == "__main__":
    main()