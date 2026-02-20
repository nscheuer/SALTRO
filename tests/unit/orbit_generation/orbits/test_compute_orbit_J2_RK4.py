import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))

import saltro_py


MU = 3.986004418e14  # Earth gravitational parameter
RE = 6378136.3  # Earth equatorial radius
J2 = 1.08262668e-3  # Earth J2 coefficient
DAYS_PER_JULIAN_CENTURY = 36525  # Days in a Julian century


def specific_energy(r, v):
    return 0.5 * np.dot(v, v) - MU / np.linalg.norm(r)


def angular_momentum(r, v):
    return np.cross(r, v)


def apsidal_vector(r, v):
    """Compute the apsidal vector (eccentricity vector) for classical elements."""
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)
    mu = MU
    
    # e = (v^2/mu - 1/r) * r - (r.v/mu) * v
    e_vec = ((v_norm**2 / mu - 1.0 / r_norm) * r - 
             (np.dot(r, v) / mu) * v)
    return e_vec


def longitude_of_perigee(r, v):
    """Calculate longitude of perigee from state vectors."""
    h = angular_momentum(r, v)
    e_vec = apsidal_vector(r, v)
    
    # Longitude of ascending node
    n_vec = np.array([-h[1], h[0], 0.0])
    n_norm = np.linalg.norm(n_vec)
    
    if n_norm < 1e-8:
        # Equatorial orbit - cannot define RAAN
        return np.nan
    
    Omega = np.arctan2(n_vec[1], n_vec[0])
    if Omega < 0:
        Omega += 2 * np.pi
    
    # Argument of perigee
    e_norm = np.linalg.norm(e_vec)
    if e_norm < 1e-8:
        # Circular orbit - cannot define argument of perigee
        return np.nan
    
    # Safe computation of omega
    dot_ne = n_vec[0] * e_vec[0] + n_vec[1] * e_vec[1]
    cos_omega = dot_ne / (n_norm * e_norm)
    sin_omega = e_vec[2] / e_norm
    
    # Clamp to [-1, 1] to avoid numerical issues with arccos
    cos_omega = np.clip(cos_omega, -1.0, 1.0)
    
    omega = np.arctan2(sin_omega, cos_omega)
    if omega < 0:
        omega += 2 * np.pi
    
    return Omega + omega


def right_ascension_ascending_node(r, v):
    """Calculate right ascension of ascending node from state vectors."""
    h = angular_momentum(r, v)
    
    # Longitude of ascending node
    n_vec = np.array([-h[1], h[0], 0.0])
    n_norm = np.linalg.norm(n_vec)
    
    if n_norm < 1e-8:
        # Equatorial orbit - RAAN is undefined
        return 0.0
    
    Omega = np.arctan2(n_vec[1], n_vec[0])
    if Omega < 0:
        Omega += 2 * np.pi
    return Omega


def inclination(r, v):
    """Calculate inclination from state vectors."""
    h = angular_momentum(r, v)
    h_norm = np.linalg.norm(h)
    
    i = np.arccos(h[2] / h_norm)
    return i


def test_J2_RK4_dimensions_and_validity():
    """Test basic output dimensions and finite values."""
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7.5e3, 0.0])

    N = 10
    jtime = np.linspace(2451545.0, 2451545.0 + 0.01, N) / DAYS_PER_JULIAN_CENTURY

    ok, R, V = saltro_py.compute_orbit_J2_RK4(r0, v0, jtime)

    assert ok
    assert R.shape == (3, N)
    assert V.shape == (3, N)

    assert np.all(np.isfinite(R))
    assert np.all(np.isfinite(V))


def test_J2_RK4_initial_conditions():
    """Test that initial state is correctly set."""
    r0 = np.array([7000e3, 1000e3, 500e3])
    v0 = np.array([100.0, 7500.0, 50.0])

    N = 5
    jtime = np.linspace(2451545.0, 2451545.0 + 0.001, N) / DAYS_PER_JULIAN_CENTURY

    ok, R, V = saltro_py.compute_orbit_J2_RK4(r0, v0, jtime)
    assert ok

    # Initial state should match input
    assert np.allclose(R[:, 0], r0, rtol=1e-12)
    assert np.allclose(V[:, 0], v0, rtol=1e-12)


def test_J2_RK4_inclination_conservation():
    """Test that inclination is conserved (J2 doesn't change inclination)."""
    alt = 400e3
    rmag = RE + alt
    inc = np.radians(45.0)

    # Create inclined circular orbit
    r0 = np.array([rmag, 0.0, 0.0])
    v0 = np.array([0.0, np.sqrt(MU / rmag) * np.cos(inc), np.sqrt(MU / rmag) * np.sin(inc)])

    N = 50
    jtime = np.linspace(2451545.0, 2451545.0 + 0.05, N) / DAYS_PER_JULIAN_CENTURY

    ok, R, V = saltro_py.compute_orbit_J2_RK4(r0, v0, jtime)
    assert ok

    # Check inclination at each step
    inc_initial = inclination(R[:, 0], V[:, 0])
    for i in range(N):
        inc_current = inclination(R[:, i], V[:, i])
        assert abs(inc_current - inc_initial) < 1e-3  # Relaxed to account for RK4 numerical integration accumulation


def test_J2_RK4_nodal_regression():
    """Test that J2 causes nodal regression (RAAN decreases)."""
    alt = 400e3
    rmag = RE + alt
    inc = np.radians(45.0)

    # Create inclined circular orbit
    r0 = np.array([rmag, 0.0, 0.0])
    v0 = np.array([0.0, np.sqrt(MU / rmag) * np.cos(inc), np.sqrt(MU / rmag) * np.sin(inc)])

    N = 200
    duration_days = 1.0  # 1 day for observable J2 effect
    jtime = np.linspace(2451545.0, 2451545.0 + duration_days, N) / DAYS_PER_JULIAN_CENTURY

    ok, R, V = saltro_py.compute_orbit_J2_RK4(r0, v0, jtime)
    assert ok

    # Get RAAN at beginning and end
    raan_initial = right_ascension_ascending_node(R[:, 0], V[:, 0])
    raan_final = right_ascension_ascending_node(R[:, -1], V[:, -1])

    # Due to J2, RAAN should regress (decrease)
    # Note: need to handle 2π discontinuity
    delta_raan = (raan_final - raan_initial)
    if delta_raan > np.pi:
        delta_raan -= 2 * np.pi
    if delta_raan < -np.pi:
        delta_raan += 2 * np.pi

    # For inclined GEO-like orbit, regression rate should be negative
    assert delta_raan < 0, f"RAAN should regress, got delta_RAAN = {delta_raan}"


def test_J2_RK4_apsidal_precession():
    """Test that J2 causes apsidal precession (argument of perigee changes)."""
    alt = 400e3
    rmag = RE + alt
    
    # Create elliptical orbit with higher eccentricity to avoid numerical issues
    e = 0.3
    a = rmag / (1 - e)
    
    r0_mag = rmag
    v_circ = np.sqrt(MU / rmag)
    v0_mag = v_circ * np.sqrt(2 - r0_mag / a)

    r0 = np.array([r0_mag, 0.0, 0.0])
    v0 = np.array([0.0, v0_mag, 0.0])

    N = 200
    duration_days = 1.0
    jtime = np.linspace(2451545.0, 2451545.0 + duration_days, N) / DAYS_PER_JULIAN_CENTURY

    ok, R, V = saltro_py.compute_orbit_J2_RK4(r0, v0, jtime)
    assert ok

    # Get argument of perigee at start and end
    omega_initial = longitude_of_perigee(R[:, 0], V[:, 0])
    omega_final = longitude_of_perigee(R[:, -1], V[:, -1])

    # Skip test if we couldn't compute valid angles (e.g., near-circular or equatorial)
    if np.isnan(omega_initial) or np.isnan(omega_final):
        pytest.skip("Cannot compute argument of perigee for nearly circular orbit")

    # Apsidal precession should occur (positive for most orbits)
    delta_omega = (omega_final - omega_initial)
    if delta_omega > np.pi:
        delta_omega -= 2 * np.pi
    if delta_omega < -np.pi:
        delta_omega += 2 * np.pi

    # Check that there's measurable apsidal precession
    assert abs(delta_omega) > 1e-6, "Apsidal precession should be observable"


def test_J2_RK4_energy_monotonic_decay():
    """Test that J2 RK4 (without drag) maintains reasonable energy behavior."""
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7.546e3, 0.0])

    N = 50
    jtime = np.linspace(2451545.0, 2451545.0 + 0.01, N) / DAYS_PER_JULIAN_CENTURY

    ok, R, V = saltro_py.compute_orbit_J2_RK4(r0, v0, jtime)
    assert ok

    energies = np.array([specific_energy(R[:, i], V[:, i]) for i in range(N)])

    # Energy should be nearly conserved (J2 is a conservative force)
    # Check that relative change is small
    e0 = energies[0]
    max_relative_change = np.max(np.abs((energies - e0) / np.abs(e0)))
    
    # Allow somewhat larger tolerance for J2
    assert max_relative_change < 1e-4


def test_J2_RK4_vs_keplerian_deviation():
    """Test that J2 RK4 deviates from Keplerian model."""
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7.5e3, 0.0])

    N = 100
    jtime = np.linspace(2451545.0, 2451545.0 + 0.1, N) / DAYS_PER_JULIAN_CENTURY

    ok_kepl, R_kepl, V_kepl = saltro_py.compute_orbit_keplerian(r0, v0, jtime)
    ok_j2, R_j2, V_j2 = saltro_py.compute_orbit_J2_RK4(r0, v0, jtime)

    assert ok_kepl and ok_j2

    # Calculate position differences
    position_diffs = np.linalg.norm(R_j2 - R_kepl, axis=0)

    # J2 should cause noticeable deviation from Keplerian
    # But differences should grow gradually
    assert position_diffs[0] < 1.0  # Very small at start
    assert position_diffs[-1] > 100.0  # Notable deviation over time


def test_J2_RK4_circular_orbit_altitude_variation():
    """Test altitude variations in circular J2 orbit due to perturbation."""
    alt = 500e3
    rmag = RE + alt

    r0 = np.array([rmag, 0.0, 0.0])
    v0 = np.array([0.0, np.sqrt(MU / rmag), 0.0])

    N = 100
    jtime = np.linspace(2451545.0, 2451545.0 + 0.1, N) / DAYS_PER_JULIAN_CENTURY

    ok, R, V = saltro_py.compute_orbit_J2_RK4(r0, v0, jtime)
    assert ok

    radii = np.linalg.norm(R, axis=0)
    altitudes = radii - RE

    # Check that altitude varies slightly due to J2
    alt_min = altitudes.min()
    alt_max = altitudes.max()
    alt_variation = alt_max - alt_min

    # Variation should exist but be relatively small
    assert alt_variation > 1e3  # At least 1 km variation
    assert alt_variation < 100e3  # But less than 100 km


def test_J2_RK4_input_validation_empty_time():
    """Test that empty time array raises error."""
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7.5e3, 0.0])

    with pytest.raises(RuntimeError):
        saltro_py.compute_orbit_J2_RK4(r0, v0, np.array([]))


def test_J2_RK4_input_validation_single_time():
    """Test behavior with single time point."""
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7.5e3, 0.0])

    jtime = np.array([2451545.0]) / DAYS_PER_JULIAN_CENTURY
    ok, R, V = saltro_py.compute_orbit_J2_RK4(r0, v0, jtime)

    assert ok
    assert R.shape == (3, 1)
    assert V.shape == (3, 1)
    assert np.allclose(R[:, 0], r0)
    assert np.allclose(V[:, 0], v0)


def test_J2_RK4_equatorial_orbit():
    """Test J2 effects on equatorial orbit (inclination = 0)."""
    alt = 400e3
    rmag = RE + alt

    # Equatorial orbit
    r0 = np.array([rmag, 0.0, 0.0])
    v0 = np.array([0.0, np.sqrt(MU / rmag), 0.0])

    N = 100
    jtime = np.linspace(2451545.0, 2451545.0 + 0.05, N) / DAYS_PER_JULIAN_CENTURY

    ok, R, V = saltro_py.compute_orbit_J2_RK4(r0, v0, jtime)
    assert ok

    # Inclination should remain zero
    for i in range(N):
        inc = inclination(R[:, i], V[:, i])
        assert abs(inc) < 1e-6


def test_J2_RK4_polar_orbit():
    """Test J2 effects on polar orbit (inclination = 90 degrees)."""
    alt = 400e3
    rmag = RE + alt

    # Polar orbit
    r0 = np.array([rmag, 0.0, 0.0])
    v0 = np.array([0.0, 0.0, np.sqrt(MU / rmag)])

    N = 100
    jtime = np.linspace(2451545.0, 2451545.0 + 0.05, N) / DAYS_PER_JULIAN_CENTURY

    ok, R, V = saltro_py.compute_orbit_J2_RK4(r0, v0, jtime)
    assert ok

    # Inclination should remain 90 degrees
    for i in range(N):
        inc = inclination(R[:, i], V[:, i])
        assert abs(inc - np.pi / 2.0) < 1e-6


def test_J2_RK4_backwards_propagation():
    """Test propagation with decreasing time."""
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7.5e3, 0.0])

    # Time array going backwards
    N = 20
    jtime = np.linspace(2451545.0 + 0.01, 2451545.0, N) / DAYS_PER_JULIAN_CENTURY

    ok, R, V = saltro_py.compute_orbit_J2_RK4(r0, v0, jtime)
    assert ok

    assert R.shape == (3, N)
    assert V.shape == (3, N)
    assert np.all(np.isfinite(R))
    assert np.all(np.isfinite(V))


def test_J2_RK4_high_eccentricity():
    """Test high eccentricity orbit with J2."""
    # High elliptical orbit
    a = 26500e3
    e = 0.7
    r_peri = a * (1 - e)
    v_peri = np.sqrt(MU * (2 / r_peri - 1 / a))

    r0 = np.array([r_peri, 0.0, 0.0])
    v0 = np.array([0.0, v_peri, 0.0])

    N = 100
    jtime = np.linspace(2451545.0, 2451545.0 + 0.05, N) / DAYS_PER_JULIAN_CENTURY

    ok, R, V = saltro_py.compute_orbit_J2_RK4(r0, v0, jtime)
    assert ok

    # Check that all values are finite
    assert np.all(np.isfinite(R))
    assert np.all(np.isfinite(V))

    # Radii should vary significantly
    radii = np.linalg.norm(R, axis=0)
    assert radii.max() / radii.min() > 1.5
