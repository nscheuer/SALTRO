import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))

import saltro_py


MU = 3.986004418e14  # Earth gravitational parameter


def specific_energy(r, v):
    return 0.5 * np.dot(v, v) - MU / np.linalg.norm(r)


def test_keplerian_dimensions_and_validity():
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7.5e3, 0.0])

    N = 10
    jtime = np.linspace(2451545.0, 2451545.0 + 0.01, N)

    ok, R, V = saltro_py.compute_orbit_keplerian(r0, v0, jtime)

    assert ok
    assert R.shape == (3, N)
    assert V.shape == (3, N)

    assert np.all(np.isfinite(R))
    assert np.all(np.isfinite(V))


def test_keplerian_energy_conservation():
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7.546e3, 0.0])

    N = 20
    jtime = np.linspace(2451545.0, 2451545.0 + 0.02, N)

    ok, R, V = saltro_py.compute_orbit_keplerian(r0, v0, jtime)
    assert ok

    energies = [specific_energy(R[:, i], V[:, i]) for i in range(N)]

    e0 = energies[0]
    for e in energies:
        assert abs(e - e0) / abs(e0) < 1e-6


def test_keplerian_circular_orbit_radius_constant():
    Re = 6378136.3
    alt = 400e3
    rmag = Re + alt

    r0 = np.array([rmag, 0.0, 0.0])
    v0 = np.array([0.0, np.sqrt(MU / rmag), 0.0])

    N = 60
    jtime = np.linspace(2451545.0, 2451545.0 + 0.02, N)

    ok, R, V = saltro_py.compute_orbit_keplerian(r0, v0, jtime)
    assert ok

    radii = np.linalg.norm(R, axis=0)

    assert np.all(np.isfinite(radii))
    assert radii.max() - radii.min() < 1e3  # ~1 km tolerance


def test_keplerian_periodicity():
    Re = 6378136.3
    alt = 500e3
    rmag = Re + alt

    r0 = np.array([rmag, 0.0, 0.0])
    v0 = np.array([0.0, np.sqrt(MU / rmag), 0.0])

    period = 2 * np.pi * np.sqrt(rmag**3 / MU)

    N = 50
    jtime = np.linspace(2451545.0, 2451545.0 + period / 86400.0, N)

    ok, R, V = saltro_py.compute_orbit_keplerian(r0, v0, jtime)
    assert ok

    r_final = R[:, -1]
    v_final = V[:, -1]

    assert np.linalg.norm(r_final - r0) < 5e3
    assert np.linalg.norm(v_final - v0) < 5.0


def test_keplerian_input_validation():
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7.5e3, 0.0])

    with pytest.raises(RuntimeError):
        saltro_py.compute_orbit_keplerian(r0, v0, np.array([]))