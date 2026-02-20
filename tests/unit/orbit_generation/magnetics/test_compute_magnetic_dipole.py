import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))

import saltro_py


def test_dipole_dimensions_and_validity():
    N = 5
    Re = 6378136.3

    R = np.zeros((3, N))
    jtime = np.linspace(2451545.0, 2451545.0 + 0.01, N)

    for i in range(N):
        alt = 400e3
        r = Re + alt
        R[:, i] = [r, 0.0, 0.0]

    ok, B = saltro_py.compute_magnetic_dipole(R, jtime)

    assert ok
    assert B.shape == (3, N)

    assert np.all(np.isfinite(B))
    assert np.linalg.norm(B) > 0.0


def test_dipole_reasonable_magnitude_leo():
    N = 3
    Re = 6378136.3

    R = np.zeros((3, N))
    jtime = np.full(N, 2451545.0)

    alts = [200e3, 400e3, 700e3]

    for i in range(N):
        r = Re + alts[i]
        R[:, i] = [r, 0.0, 0.0]

    ok, B = saltro_py.compute_magnetic_dipole(R, jtime)
    assert ok

    Bmag = np.linalg.norm(B, axis=0)

    # Rough Earth dipole field magnitude ranges
    assert np.all(Bmag > 1e-7)
    assert np.all(Bmag < 1e-3)

    assert Bmag[0] > Bmag[1] > Bmag[2]


def test_dipole_circular_orbit_variation():
    N = 60
    Re = 6378136.3
    alt = 500e3
    r = Re + alt

    R = np.zeros((3, N))
    jtime = np.linspace(2451545.0, 2451545.0 + 0.01, N)

    for i in range(N):
        theta = 2.0 * np.pi * i / N
        R[:, i] = [
            r * np.cos(theta),
            r * np.sin(theta),
            0.0,
        ]

    ok, B = saltro_py.compute_magnetic_dipole(R, jtime)
    assert ok

    Bmag = np.linalg.norm(B, axis=0)

    assert np.all(np.isfinite(Bmag))
    assert np.all(Bmag > 0.0)

    # dipole variation shouldn't explode
    assert Bmag.max() / Bmag.min() < 50.0


def test_dipole_input_validation_shape():
    R = np.zeros((2, 5))
    jtime = np.ones(5)

    with pytest.raises(RuntimeError):
        saltro_py.compute_magnetic_dipole(R, jtime)


def test_dipole_input_validation_length():
    R = np.zeros((3, 5))
    jtime = np.ones(4)

    with pytest.raises(RuntimeError):
        saltro_py.compute_magnetic_dipole(R, jtime)