import os
import sys
from pathlib import Path

import numpy as np
import matplotlib


def _configure_backend() -> None:
    # Prefer an interactive backend on Linux without requiring Qt.
    if os.environ.get("MPLBACKEND"):
        return

    if sys.platform.startswith("linux"):
        try:
            import tkinter  # noqa: F401

            matplotlib.use("TkAgg")
            return
        except Exception:
            try:
                matplotlib.use("WebAgg")
                return
            except Exception:
                matplotlib.use("Agg")


_configure_backend()
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
SALTRO_BUILD = ROOT / "build"
ADCS_ROOT = ROOT.parent / "Generalized_ADCS"

sys.path.insert(0, str(SALTRO_BUILD))
sys.path.insert(0, str(ADCS_ROOT))

try:
    import saltro_py
except Exception as exc:
    raise RuntimeError(
        f"Unable to import saltro_py from {SALTRO_BUILD}. Build SALTRO first."
    ) from exc

try:
    from ADCS.orbits.ephemeris import Ephemeris
    from ADCS.orbits.orbit import Orbit
    from ADCS.orbits.orbital_state import Orbital_State
    from ADCS.orbits.universal_constants import TimeConstants
except Exception as exc:
    raise RuntimeError(
        f"Unable to import ADCS modules from {ADCS_ROOT}. Use the Generalized_ADCS venv."
    ) from exc


def _generate_saltro_data(
    r0_m: np.ndarray,
    v0_mps: np.ndarray,
    jtime0_century: float,
    dt_s: float,
    duration_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_steps = int(duration_s / dt_s) + 1
    sec_per_julian_century = 36525.0 * 86400.0

    time_s = np.arange(n_steps, dtype=float) * dt_s
    jtime = jtime0_century + time_s / sec_per_julian_century

    ok, R_m, _V_mps, B_t, _S, _rho = saltro_py.generate_orbit(
        r0_m,
        v0_mps,
        jtime,
        1,  # orbit_model: J2 RK4
        2,  # magnetic_model: IGRF-13
        0,
        0,
        0,
    )
    if not ok:
        raise RuntimeError("SALTRO generate_orbit failed")

    if R_m.shape != (3, n_steps) or B_t.shape != (3, n_steps):
        raise RuntimeError(f"Unexpected SALTRO shapes: R={R_m.shape}, B={B_t.shape}")

    bmag_uT = np.linalg.norm(B_t, axis=0) * 1e6
    return time_s, R_m, B_t, bmag_uT


def _generate_adcs_data(
    r0_m: np.ndarray,
    v0_mps: np.ndarray,
    jtime0_century: float,
    dt_s: float,
    duration_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ephem = Ephemeris()
    os0 = Orbital_State(
        ephem=ephem,
        J2000=jtime0_century,
        R=np.asarray(r0_m, dtype=float) / 1e3,
        V=np.asarray(v0_mps, dtype=float) / 1e3,
    )

    orbit = Orbit(
        os0,
        end_time=jtime0_century + duration_s * TimeConstants.sec2cent,
        dt=dt_s,
        use_J2=True,
        fast=False,
        verbose=False,
    )

    times = sorted(orbit.states.keys())
    t0 = times[0]
    time_s = (np.asarray(times, dtype=float) - t0) * TimeConstants.cent2sec

    R_km = np.stack([np.asarray(orbit.states[t].R).reshape(3) for t in times], axis=1)
    B_t = np.stack([np.asarray(orbit.states[t].B).reshape(3) for t in times], axis=1)

    R_m = R_km * 1e3
    bmag_uT = np.linalg.norm(B_t, axis=0) * 1e6
    return time_s, R_m, B_t, bmag_uT


def _report_metrics(B_saltro_t: np.ndarray, B_adcs_t: np.ndarray) -> None:
    bdiff = B_saltro_t - B_adcs_t
    diff_norm = np.linalg.norm(bdiff, axis=0)

    bs_mag = np.linalg.norm(B_saltro_t, axis=0)
    ba_mag = np.linalg.norm(B_adcs_t, axis=0)
    mag_diff = np.abs(bs_mag - ba_mag)

    print("Magnetic field comparison metrics")
    print(f"  B-vector RMS error [uT]: {np.sqrt(np.mean(diff_norm**2)) * 1e6:.6f}")
    print(f"  B-vector max error [uT]: {np.max(diff_norm) * 1e6:.6f}")
    print(f"  |B| RMS error [uT]: {np.sqrt(np.mean(mag_diff**2)) * 1e6:.6f}")
    print(f"  |B| max error [uT]: {np.max(mag_diff) * 1e6:.6f}")


def main() -> None:
    r0_m = np.array([7000e3, 0.0, 0.0], dtype=float)
    v0_mps = np.array([0.0, 7.5e3, 0.0], dtype=float)
    jtime0_century = 0.22
    dt_s = 10.0
    duration_s = 5400.0

    t_s_saltro, R_saltro_m, B_saltro_t, bmag_saltro_uT = _generate_saltro_data(
        r0_m, v0_mps, jtime0_century, dt_s, duration_s
    )
    t_s_adcs, R_adcs_m, B_adcs_t, bmag_adcs_uT = _generate_adcs_data(
        r0_m, v0_mps, jtime0_century, dt_s, duration_s
    )

    n = min(R_saltro_m.shape[1], R_adcs_m.shape[1])
    if R_saltro_m.shape[1] != R_adcs_m.shape[1]:
        print(
            "Warning: sample count mismatch "
            f"(SALTRO={R_saltro_m.shape[1]}, ADCS={R_adcs_m.shape[1]}). Trimming to N={n}."
        )

    t_h = t_s_saltro[:n] / 3600.0
    R_saltro_km = R_saltro_m[:, :n] / 1e3
    R_adcs_km = R_adcs_m[:, :n] / 1e3
    B_saltro_t = B_saltro_t[:, :n]
    B_adcs_t = B_adcs_t[:, :n]
    bmag_saltro_uT = bmag_saltro_uT[:n]
    bmag_adcs_uT = bmag_adcs_uT[:n]

    _report_metrics(B_saltro_t, B_adcs_t)

    fig = plt.figure(figsize=(12, 6))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax2d = fig.add_subplot(1, 2, 2)

    ax3d.plot(
        R_saltro_km[0],
        R_saltro_km[1],
        R_saltro_km[2],
        label="SALTRO orbit",
        lw=1.4,
        color="tab:blue",
    )
    ax3d.plot(
        R_adcs_km[0],
        R_adcs_km[1],
        R_adcs_km[2],
        label="Generalized_ADCS orbit",
        lw=1.4,
        color="tab:orange",
        alpha=0.85,
    )

    ax3d.scatter(
        R_saltro_km[0, 0],
        R_saltro_km[1, 0],
        R_saltro_km[2, 0],
        color="tab:green",
        s=35,
        label="start",
    )
    ax3d.scatter(
        R_saltro_km[0, -1],
        R_saltro_km[1, -1],
        R_saltro_km[2, -1],
        color="tab:red",
        s=35,
        label="end",
    )

    ax3d.set_title("Orbit Comparison (3D)")
    ax3d.set_xlabel("X [km]")
    ax3d.set_ylabel("Y [km]")
    ax3d.set_zlabel("Z [km]")
    ax3d.legend(loc="best", fontsize=8)

    ax2d.plot(t_h, bmag_saltro_uT, label="SALTRO |B|", lw=1.4, color="tab:blue")
    ax2d.plot(t_h, bmag_adcs_uT, label="Generalized_ADCS |B|", lw=1.4, color="tab:orange")
    ax2d.set_title("Magnetic Field Magnitude vs Time")
    ax2d.set_xlabel("Time [h]")
    ax2d.set_ylabel("|B| [uT]")
    ax2d.grid(alpha=0.35)
    ax2d.legend(loc="best")

    fig.suptitle(
        "SALTRO (orbit=1, mag=2) vs Generalized_ADCS (J2)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    fig_bcomp, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 8))

    axs[0].plot(t_h, B_saltro_t[0] * 1e6, label="SALTRO Bx", color="tab:blue", lw=1.2)
    axs[0].plot(t_h, B_adcs_t[0] * 1e6, label="Generalized_ADCS Bx", color="tab:orange", lw=1.2)
    axs[0].set_ylabel("Bx [uT]")
    axs[0].set_title("Magnetic Field Components in Inertial Frame")
    axs[0].grid(alpha=0.35)
    axs[0].legend(loc="best", fontsize=8)

    axs[1].plot(t_h, B_saltro_t[1] * 1e6, label="SALTRO By", color="tab:blue", lw=1.2)
    axs[1].plot(t_h, B_adcs_t[1] * 1e6, label="Generalized_ADCS By", color="tab:orange", lw=1.2)
    axs[1].set_ylabel("By [uT]")
    axs[1].grid(alpha=0.35)
    axs[1].legend(loc="best", fontsize=8)

    axs[2].plot(t_h, B_saltro_t[2] * 1e6, label="SALTRO Bz", color="tab:blue", lw=1.2)
    axs[2].plot(t_h, B_adcs_t[2] * 1e6, label="Generalized_ADCS Bz", color="tab:orange", lw=1.2)
    axs[2].set_ylabel("Bz [uT]")
    axs[2].set_xlabel("Time [h]")
    axs[2].grid(alpha=0.35)
    axs[2].legend(loc="best", fontsize=8)

    fig_bcomp.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
