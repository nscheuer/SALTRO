import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))

import saltro_py


# API contract: jtime is Julian centuries T (not Julian Date).
# These helpers keep the old "span in days" intent, but convert to centuries.
def make_jcenturies(N, span_days, T0=0.22):
    return T0 + np.linspace(0.0, span_days, N) / 36525.0


def unitize_columns(V):
    """Return unit vectors (3,N) from V (3,N)."""
    norms = np.linalg.norm(V, axis=0)
    # avoid divide-by-zero
    assert np.all(norms > 0.0)
    return V / norms, norms


def test_sun_noaa_dimensions_and_validity():
    N = 5
    Re = 6378136.3

    R = np.zeros((3, N))
    jtime = make_jcenturies(N, span_days=0.01, T0=0.22)

    for i in range(N):
        R[:, i] = [Re + 400e3, 0.0, 0.0]

    ok, S = saltro_py.compute_sun_noaa(R, jtime)

    assert ok
    assert S.shape == (3, N)
    assert np.all(np.isfinite(S))


def test_sun_noaa_unit_vectors():
    N = 10
    Re = 6378136.3

    R = np.zeros((3, N))
    jtime = make_jcenturies(N, span_days=0.02, T0=0.22)

    for i in range(N):
        R[:, i] = [Re + 500e3, 0.0, 0.0]

    ok, S = saltro_py.compute_sun_noaa(R, jtime)
    assert ok

    # Compute direction unit vectors in the test (C++ returns meters)
    S_hat, norms_m = unitize_columns(S)

    # Optional sanity: spacecraft-to-sun distance should be O(1 AU)
    assert np.all(norms_m > 1.0e11)
    assert np.all(norms_m < 2.0e11)

    norms_hat = np.linalg.norm(S_hat, axis=0)
    assert np.all(norms_hat > 0.9)
    assert np.all(norms_hat < 1.1)


def test_sun_noaa_time_variation_smooth():
    N = 50
    Re = 6378136.3

    R = np.zeros((3, N))
    jtime = make_jcenturies(N, span_days=1.0, T0=0.22)  # ~1 day span

    for i in range(N):
        R[:, i] = [Re + 400e3, 0.0, 0.0]

    ok, S = saltro_py.compute_sun_noaa(R, jtime)
    assert ok

    # Check smoothness on direction (unit vectors), not raw meters
    S_hat, _ = unitize_columns(S)
    diffs = np.linalg.norm(np.diff(S_hat, axis=1), axis=0)
    assert np.all(diffs < 0.1)


def test_sun_noaa_circular_orbit_consistency():
    N = 60
    Re = 6378136.3
    alt = 400e3
    r = Re + alt

    R = np.zeros((3, N))
    jtime = make_jcenturies(N, span_days=0.02, T0=0.22)

    for i in range(N):
        theta = 2 * np.pi * i / N
        R[:, i] = [r * np.cos(theta), r * np.sin(theta), 0.0]

    ok, S = saltro_py.compute_sun_noaa(R, jtime)
    assert ok

    S_hat, norms_m = unitize_columns(S)

    # Optional sanity: distance scale
    assert np.all(norms_m > 1.0e11)
    assert np.all(norms_m < 2.0e11)

    norms_hat = np.linalg.norm(S_hat, axis=0)
    assert np.all(norms_hat > 0.9)
    assert np.all(norms_hat < 1.1)