import sys
import numpy as np
from pathlib import Path
import pytest

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
import saltro_py

def test_hp_dimensions_and_validity():
    N = 5
    Re = 6378136.3

    R = np.zeros((3, N))
    S = np.zeros((3, N))

    for i in range(N):
        alt = 200e3 + i * 100e3
        r = Re + alt
        R[:, i] = [r, 0.0, 0.0]
        S[:, i] = [1.0, 0.0, 0.0]

    ok, rho = saltro_py.compute_density_harrispriester(R, S)

    assert ok
    assert rho.shape == (N,)

    assert np.all(np.isfinite(rho))
    assert np.all(rho > 0.0)


def test_hp_order_of_magnitude():
    N = 3
    Re = 6378136.3

    alts = [200e3, 400e3, 700e3]

    R = np.zeros((3, N))
    S = np.zeros((3, N))

    for i in range(N):
        r = Re + alts[i]
        R[:, i] = [r, 0.0, 0.0]
        S[:, i] = [1.0, 0.0, 0.0]

    ok, rho = saltro_py.compute_density_harrispriester(R, S)
    assert ok

    assert 1e-12 < rho[0] < 1e-6
    assert 1e-14 < rho[1] < 1e-8
    assert 1e-16 < rho[2] < 1e-9

    assert rho[0] > rho[1] > rho[2]


def test_hp_circular_orbit_smoothness():
    N = 50
    Re = 6378136.3
    alt = 400e3
    r = Re + alt

    R = np.zeros((3, N))
    S = np.zeros((3, N))

    for i in range(N):
        theta = 2.0 * np.pi * i / N
        R[:, i] = [
            r * np.cos(theta),
            r * np.sin(theta),
            0.0,
        ]
        S[:, i] = [1.0, 0.0, 0.0]

    ok, rho = saltro_py.compute_density_harrispriester(R, S)
    assert ok

    assert np.all(np.isfinite(rho))
    assert np.all(rho > 0.0)

    rho_min = rho.min()
    rho_max = rho.max()

    assert rho_max / rho_min < 20.0