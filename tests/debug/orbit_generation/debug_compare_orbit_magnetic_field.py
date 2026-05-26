import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
import ppigrf


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


def _gmst_rad(jcentury: np.ndarray | float) -> np.ndarray:
    jcentury = np.asarray(jcentury, dtype=float)
    t2 = jcentury * jcentury
    t3 = t2 * jcentury
    gmst_sec = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * jcentury
        + 0.093104 * t2
        - 6.2e-6 * t3
    )
    gmst = (gmst_sec * (2.0 * np.pi)) / 86400.0
    return np.mod(gmst, 2.0 * np.pi)


def _rot_z(angle: np.ndarray | float) -> np.ndarray:
    angle = np.asarray(angle, dtype=float)
    c = np.cos(angle)
    s = np.sin(angle)
    if angle.ndim == 0:
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    out = np.zeros((angle.size, 3, 3), dtype=float)
    out[:, 0, 0] = c
    out[:, 0, 1] = -s
    out[:, 1, 0] = s
    out[:, 1, 1] = c
    out[:, 2, 2] = 1.0
    return out


def _saltro_eci_to_ecef_dcm(jcentury: np.ndarray | float) -> np.ndarray:
    return _rot_z(-_gmst_rad(jcentury))


def _rotation_angle_deg(c1: np.ndarray, c2: np.ndarray) -> float:
    rel = c1 @ c2.T
    trace_val = float(np.trace(rel))
    cos_ang = np.clip((trace_val - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_ang)))


def _generate_saltro_data(
    r0_m: np.ndarray,
    v0_mps: np.ndarray,
    jtime0_century: float,
    dt_s: float,
    duration_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    R_eci2ecef = _saltro_eci_to_ecef_dcm(jtime)
    B_ecef_t = np.einsum("nij,nj->ni", R_eci2ecef, B_t.T).T
    bmag_uT = np.linalg.norm(B_t, axis=0) * 1e6
    return time_s, jtime, R_m, B_t, B_ecef_t, bmag_uT


def _generate_adcs_data(
    r0_m: np.ndarray,
    v0_mps: np.ndarray,
    jtime0_century: float,
    dt_s: float,
    duration_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    time_cent = np.asarray(times, dtype=float)
    time_s = (time_cent - t0) * TimeConstants.cent2sec

    R_km = np.stack([np.asarray(orbit.states[t].R).reshape(3) for t in times], axis=1)
    B_t = np.stack([np.asarray(orbit.states[t].B).reshape(3) for t in times], axis=1)
    ECEF_km = np.stack([np.asarray(orbit.states[t].ECEF).reshape(3) for t in times], axis=1)
    R_eci2ecef = np.stack([np.asarray(orbit.states[t]._R_eci2ecef).reshape(3, 3) for t in times], axis=0)
    geo_hist = np.stack([np.asarray(orbit.states[t].geocentric).reshape(3) for t in times], axis=1)
    dts = [orbit.states[t].datetime for t in times]

    r_ref = geo_hist[0]
    theta_ref = geo_hist[1] * 180.0 / np.pi
    phi_ref = geo_hist[2] * 180.0 / np.pi
    b_r_ref, b_th_ref, b_ph_ref = ppigrf.igrf_gc(r_ref, theta_ref, phi_ref, dts)
    b_r_ref = np.asarray(b_r_ref)
    b_th_ref = np.asarray(b_th_ref)
    b_ph_ref = np.asarray(b_ph_ref)
    if b_r_ref.ndim == 2 and b_r_ref.shape[0] == b_r_ref.shape[1] == len(times):
        b_r_ref = np.diagonal(b_r_ref)
    if b_th_ref.ndim == 2 and b_th_ref.shape[0] == b_th_ref.shape[1] == len(times):
        b_th_ref = np.diagonal(b_th_ref)
    if b_ph_ref.ndim == 2 and b_ph_ref.shape[0] == b_ph_ref.shape[1] == len(times):
        b_ph_ref = np.diagonal(b_ph_ref)
    B_ref_geo_t = np.vstack([
        np.asarray(b_r_ref, dtype=float).reshape(len(times)),
        np.asarray(b_th_ref, dtype=float).reshape(len(times)),
        np.asarray(b_ph_ref, dtype=float).reshape(len(times)),
    ]) * 1e-9

    ECEF_hist = ECEF_km.T * 1e3
    r_ecef = np.linalg.norm(ECEF_hist, axis=1)
    n_ecef = ECEF_hist / r_ecef[:, None]
    zhat = np.array([0.0, 0.0, 1.0], dtype=float).reshape(1, 3)
    svec = np.cross(zhat, n_ecef)
    svec = svec / np.linalg.norm(svec, axis=1)[:, None]
    tvec = np.cross(svec, n_ecef)
    tvec = tvec / np.linalg.norm(tvec, axis=1)[:, None]

    B_ref_ecef_t = (
        B_ref_geo_t[0][:, None] * n_ecef
        + B_ref_geo_t[2][:, None] * svec
        + B_ref_geo_t[1][:, None] * tvec
    ).T
    B_ecef_t = np.einsum("nij,nj->ni", R_eci2ecef, B_t.T).T

    R_m = R_km * 1e3
    bmag_uT = np.linalg.norm(B_t, axis=0) * 1e6
    return time_s, time_cent, R_m, B_t, B_ecef_t, B_ref_ecef_t, B_ref_geo_t, bmag_uT, ECEF_km * 1e3, R_eci2ecef, geo_hist, np.array(dts, dtype=object)


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


def _report_ecef_metrics(B_saltro_ecef: np.ndarray, B_adcs_ecef: np.ndarray) -> None:
    diff = B_saltro_ecef - B_adcs_ecef
    diff_norm = np.linalg.norm(diff, axis=0)
    mag_diff = np.abs(np.linalg.norm(B_saltro_ecef, axis=0) - np.linalg.norm(B_adcs_ecef, axis=0))

    print("ECEF magnetic comparison metrics")
    print(f"  B_ecef RMS error [uT]: {np.sqrt(np.mean(diff_norm**2)) * 1e6:.6f}")
    print(f"  B_ecef max error [uT]: {np.max(diff_norm) * 1e6:.6f}")
    print(f"  |B_ecef| RMS error [uT]: {np.sqrt(np.mean(mag_diff**2)) * 1e6:.6f}")
    print(f"  |B_ecef| max error [uT]: {np.max(mag_diff) * 1e6:.6f}")


def _report_geocentric_metrics(B_saltro_geo: np.ndarray, B_ref_geo: np.ndarray) -> None:
    diff = B_saltro_geo - B_ref_geo
    diff_norm = np.linalg.norm(diff, axis=0)
    mag_diff = np.abs(np.linalg.norm(B_saltro_geo, axis=0) - np.linalg.norm(B_ref_geo, axis=0))

    print("Geocentric magnetic component metrics")
    print(f"  B_geo RMS error [uT]: {np.sqrt(np.mean(diff_norm**2)) * 1e6:.6f}")
    print(f"  B_geo max error [uT]: {np.max(diff_norm) * 1e6:.6f}")
    print(f"  |B_geo| RMS error [uT]: {np.sqrt(np.mean(mag_diff**2)) * 1e6:.6f}")
    print(f"  |B_geo| max error [uT]: {np.max(mag_diff) * 1e6:.6f}")


def _best_lag_samples(reference: np.ndarray, candidate: np.ndarray) -> int:
    ref = np.asarray(reference, dtype=float).reshape(-1)
    cand = np.asarray(candidate, dtype=float).reshape(-1)

    ref = ref - np.mean(ref)
    cand = cand - np.mean(cand)
    corr = np.correlate(cand, ref, mode="full")
    lag = int(np.argmax(corr) - (ref.size - 1))
    return lag


def _report_phase_metrics(
    t_h: np.ndarray,
    B_saltro_t: np.ndarray,
    B_adcs_t: np.ndarray,
    B_saltro_ecef: np.ndarray,
    B_adcs_ecef: np.ndarray,
    B_saltro_geo: np.ndarray,
    B_ref_geo: np.ndarray,
    bmag_saltro_uT: np.ndarray,
    bmag_adcs_uT: np.ndarray,
    dt_s: float,
) -> None:
    print("Phase alignment diagnostics")
    for idx, axis_name in enumerate(("Bx", "By", "Bz")):
        lag = _best_lag_samples(B_adcs_t[idx], B_saltro_t[idx])
        lag_s = lag * dt_s
        print(f"  {axis_name} best lag: {lag:+d} samples ({lag_s:+.2f} s)")

    lag_mag = _best_lag_samples(bmag_adcs_uT, bmag_saltro_uT)
    print(f"  |B| best lag: {lag_mag:+d} samples ({lag_mag * dt_s:+.2f} s)")

    # Show the first and last aligned samples for quick manual inspection.
    print(
        "  first sample [uT]: "
        f"SALTRO=({B_saltro_t[0,0]*1e6:.3f}, {B_saltro_t[1,0]*1e6:.3f}, {B_saltro_t[2,0]*1e6:.3f}) "
        f"ADCS=({B_adcs_t[0,0]*1e6:.3f}, {B_adcs_t[1,0]*1e6:.3f}, {B_adcs_t[2,0]*1e6:.3f})"
    )
    print(
        "  last sample  [uT]: "
        f"SALTRO=({B_saltro_t[0,-1]*1e6:.3f}, {B_saltro_t[1,-1]*1e6:.3f}, {B_saltro_t[2,-1]*1e6:.3f}) "
        f"ADCS=({B_adcs_t[0,-1]*1e6:.3f}, {B_adcs_t[1,-1]*1e6:.3f}, {B_adcs_t[2,-1]*1e6:.3f})"
    )

    for idx, axis_name in enumerate(("Bx_ecef", "By_ecef", "Bz_ecef")):
        lag = _best_lag_samples(B_adcs_ecef[idx], B_saltro_ecef[idx])
        print(f"  {axis_name} best lag: {lag:+d} samples ({lag * dt_s:+.2f} s)")

    for idx, axis_name in enumerate(("Br", "Btheta", "Bphi")):
        lag = _best_lag_samples(B_ref_geo[idx], B_saltro_geo[idx])
        print(f"  {axis_name} best lag: {lag:+d} samples ({lag * dt_s:+.2f} s)")


def _report_time_frame_metrics(
    jtime_saltro: np.ndarray,
    time_cent_adcs: np.ndarray,
    R_saltro_m: np.ndarray,
    R_adcs_m: np.ndarray,
    adcs_ecef_m: np.ndarray,
    adcs_eci2ecef: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    print("Frame and epoch diagnostics")
    time_cent_saltro = np.asarray(jtime_saltro, dtype=float)
    time_cent_adcs = np.asarray(time_cent_adcs, dtype=float)
    time_delta_sec = (time_cent_saltro - time_cent_adcs) * TimeConstants.cent2sec
    print(
        f"  epoch delta [s]: mean={np.mean(time_delta_sec):+.6e}, "
        f"max={np.max(np.abs(time_delta_sec)):.6e}"
    )

    saltro_eci2ecef = _saltro_eci_to_ecef_dcm(time_cent_saltro)
    rot_angle_deg = np.array(
        [
            _rotation_angle_deg(saltro_eci2ecef[k], adcs_eci2ecef[k])
            for k in range(min(len(saltro_eci2ecef), len(adcs_eci2ecef)))
        ],
        dtype=float,
    )
    print(
        f"  ECI->ECEF rotation angle diff [deg]: mean={np.mean(rot_angle_deg):.6e}, "
        f"max={np.max(rot_angle_deg):.6e}"
    )

    saltro_ecef_m = np.einsum("nij,nj->ni", saltro_eci2ecef, R_saltro_m.T).T
    ecef_diff = saltro_ecef_m - adcs_ecef_m
    ecef_diff_norm = np.linalg.norm(ecef_diff, axis=0)
    print(
        f"  ECEF position diff [m]: RMS={np.sqrt(np.mean(ecef_diff_norm**2)):.6e}, "
        f"max={np.max(ecef_diff_norm):.6e}"
    )

    return time_cent_saltro, rot_angle_deg, ecef_diff_norm


def main() -> None:
    r0_m = np.array([7000e3, 0.0, 0.0], dtype=float)
    v0_mps = np.array([0.0, 7.5e3, 0.0], dtype=float)
    jtime0_century = 0.22
    dt_s = 10.0
    duration_s = 5400.0

    t_s_saltro, jtime_saltro, R_saltro_m, B_saltro_t, B_saltro_ecef_t, bmag_saltro_uT = _generate_saltro_data(
        r0_m, v0_mps, jtime0_century, dt_s, duration_s
    )
    (
        t_s_adcs,
        time_cent_adcs,
        R_adcs_m,
        B_adcs_t,
        B_adcs_ecef_t,
        B_ref_ecef_t,
        B_ref_geo_t,
        bmag_adcs_uT,
        ECEF_adcs_m,
        R_eci2ecef_adcs,
        geo_hist_adcs,
        dts_adcs,
    ) = _generate_adcs_data(
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
    B_saltro_ecef_t = B_saltro_ecef_t[:, :n]
    B_adcs_ecef_t = B_adcs_ecef_t[:, :n]
    B_ref_ecef_t = B_ref_ecef_t[:, :n]
    B_ref_geo_t = B_ref_geo_t[:, :n]
    bmag_saltro_uT = bmag_saltro_uT[:n]
    bmag_adcs_uT = bmag_adcs_uT[:n]
    jtime_saltro = jtime_saltro[:n]
    time_cent_adcs = time_cent_adcs[:n]
    ECEF_adcs_m = ECEF_adcs_m[:, :n]
    R_eci2ecef_adcs = R_eci2ecef_adcs[:n]
    geo_hist_adcs = geo_hist_adcs[:, :n]
    dts_adcs = dts_adcs[:n]

    n_ecef = (ECEF_adcs_m / np.linalg.norm(ECEF_adcs_m, axis=0, keepdims=True)).T
    zhat = np.array([0.0, 0.0, 1.0], dtype=float).reshape(1, 3)
    svec = np.cross(zhat, n_ecef)
    svec = svec / np.linalg.norm(svec, axis=1)[:, None]
    tvec = np.cross(svec, n_ecef)
    tvec = tvec / np.linalg.norm(tvec, axis=1)[:, None]

    B_saltro_geo_t = np.vstack([
        np.einsum("ij,ij->i", B_saltro_ecef_t.T, n_ecef),
        np.einsum("ij,ij->i", B_saltro_ecef_t.T, tvec),
        np.einsum("ij,ij->i", B_saltro_ecef_t.T, svec),
    ])

    _report_metrics(B_saltro_t, B_adcs_t)
    _report_ecef_metrics(B_saltro_ecef_t, B_adcs_ecef_t)
    _report_geocentric_metrics(B_saltro_geo_t, B_ref_geo_t)
    _report_phase_metrics(t_h, B_saltro_t, B_adcs_t, B_saltro_ecef_t, B_adcs_ecef_t, B_saltro_geo_t, B_ref_geo_t, bmag_saltro_uT, bmag_adcs_uT, dt_s)
    time_cent_saltro, rot_angle_deg, ecef_diff_norm = _report_time_frame_metrics(
        jtime_saltro, time_cent_adcs, R_saltro_m[:, :n], R_adcs_m[:, :n], ECEF_adcs_m, R_eci2ecef_adcs
    )

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

    fig_ecef, axs_ecef = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    axs_ecef[0].plot(t_h, B_saltro_ecef_t[0] * 1e6, label="SALTRO Bx_ecef", color="tab:blue", lw=1.2)
    axs_ecef[0].plot(t_h, B_adcs_ecef_t[0] * 1e6, label="ADCS Bx_ecef", color="tab:orange", lw=1.2)
    axs_ecef[0].set_ylabel("Bx [uT]")
    axs_ecef[0].set_title("Magnetic Field in ECEF Frame")
    axs_ecef[0].grid(alpha=0.35)
    axs_ecef[0].legend(loc="best", fontsize=8)

    axs_ecef[1].plot(t_h, B_saltro_ecef_t[1] * 1e6, label="SALTRO By_ecef", color="tab:blue", lw=1.2)
    axs_ecef[1].plot(t_h, B_adcs_ecef_t[1] * 1e6, label="ADCS By_ecef", color="tab:orange", lw=1.2)
    axs_ecef[1].set_ylabel("By [uT]")
    axs_ecef[1].grid(alpha=0.35)
    axs_ecef[1].legend(loc="best", fontsize=8)

    axs_ecef[2].plot(t_h, B_saltro_ecef_t[2] * 1e6, label="SALTRO Bz_ecef", color="tab:blue", lw=1.2)
    axs_ecef[2].plot(t_h, B_adcs_ecef_t[2] * 1e6, label="ADCS Bz_ecef", color="tab:orange", lw=1.2)
    axs_ecef[2].set_ylabel("Bz [uT]")
    axs_ecef[2].set_xlabel("Time [h]")
    axs_ecef[2].grid(alpha=0.35)
    axs_ecef[2].legend(loc="best", fontsize=8)

    fig_ecef.tight_layout()

    fig_geo, axs_geo = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    axs_geo[0].plot(t_h, B_saltro_geo_t[0] * 1e6, label="SALTRO Br", color="tab:blue", lw=1.2)
    axs_geo[0].plot(t_h, B_ref_geo_t[0] * 1e6, label="ppigrf Br", color="tab:orange", lw=1.2)
    axs_geo[0].set_ylabel("Br [uT]")
    axs_geo[0].set_title("Magnetic Field in Geocentric Basis")
    axs_geo[0].grid(alpha=0.35)
    axs_geo[0].legend(loc="best", fontsize=8)

    axs_geo[1].plot(t_h, B_saltro_geo_t[1] * 1e6, label="SALTRO Btheta", color="tab:blue", lw=1.2)
    axs_geo[1].plot(t_h, B_ref_geo_t[1] * 1e6, label="ppigrf Btheta", color="tab:orange", lw=1.2)
    axs_geo[1].set_ylabel("Btheta [uT]")
    axs_geo[1].grid(alpha=0.35)
    axs_geo[1].legend(loc="best", fontsize=8)

    axs_geo[2].plot(t_h, B_saltro_geo_t[2] * 1e6, label="SALTRO Bphi", color="tab:blue", lw=1.2)
    axs_geo[2].plot(t_h, B_ref_geo_t[2] * 1e6, label="ppigrf Bphi", color="tab:orange", lw=1.2)
    axs_geo[2].set_ylabel("Bphi [uT]")
    axs_geo[2].set_xlabel("Time [h]")
    axs_geo[2].grid(alpha=0.35)
    axs_geo[2].legend(loc="best", fontsize=8)

    fig_geo.tight_layout()

    fig_frame, axs_frame = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    axs_frame[0].plot(t_h, rot_angle_deg, color="tab:red", lw=1.2)
    axs_frame[0].set_ylabel("Angle diff [deg]")
    axs_frame[0].set_title("SALTRO vs ADCS ECI->ECEF Frame Diagnostics")
    axs_frame[0].grid(alpha=0.35)

    axs_frame[1].semilogy(t_h, ecef_diff_norm, color="tab:purple", lw=1.2)
    axs_frame[1].set_ylabel("|ΔECEF| [m]")
    axs_frame[1].set_xlabel("Time [h]")
    axs_frame[1].grid(alpha=0.35, which="both")

    fig_frame.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
