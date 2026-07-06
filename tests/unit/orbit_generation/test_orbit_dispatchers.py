import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "build"))

import saltro_py

# Python twin of tests/unit/orbit_generation/test_orbit_dispatchers.cpp.
#
# The model-dispatcher functions (compute_orbit/sun/magnetic/eclipse/density)
# route an integer model id to a concrete implementation and return false on an
# unknown id (default case). Only the concrete implementations were tested; the
# dispatchers themselves had no coverage, so an out-of-range model id silently
# returning false (rather than e.g. routing to a wrong model) was unverified.
# The saltro_py compute_* bindings call the dispatchers directly, so the same
# routing is exercised here through the bound python surface.

LEN = 4


def make_jtime():
    # JD, ~86 s steps (mirrors the C++ twin's makeJtime()).
    return 2451545.0 + np.arange(LEN, dtype=float) * 1e-3


def initial_conditions():
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7546.0, 0.0])  # ~circular LEO
    return r0, v0


def test_compute_orbit_dispatcher_valid_model_routes_unknown_returns_false():
    r0, v0 = initial_conditions()
    jtime = make_jtime()

    ok, R, V = saltro_py.compute_orbit(r0, v0, jtime, 0)  # KEPLERIAN
    assert ok
    assert np.all(np.isfinite(R))

    ok_unknown, _, _ = saltro_py.compute_orbit(r0, v0, jtime, 99)
    assert not ok_unknown

    ok_negative, _, _ = saltro_py.compute_orbit(r0, v0, jtime, -1)
    assert not ok_negative


def test_compute_sun_dispatcher_valid_model_routes_unknown_returns_false():
    jtime = make_jtime()
    R = np.zeros((3, LEN))

    ok, S = saltro_py.compute_sun(R, jtime, 0)  # NOAA
    assert ok
    assert np.all(np.isfinite(S))

    ok_unknown, _ = saltro_py.compute_sun(R, jtime, 99)
    assert not ok_unknown


def test_compute_magnetic_dispatcher_valid_model_routes_unknown_returns_false():
    r0, v0 = initial_conditions()
    jtime = make_jtime()

    ok_orbit, R, _V = saltro_py.compute_orbit(r0, v0, jtime, 0)
    assert ok_orbit

    ok, B = saltro_py.compute_magnetic(R, jtime, 0)  # DIPOLE
    assert ok
    assert np.all(np.isfinite(B))

    ok_unknown, _ = saltro_py.compute_magnetic(R, jtime, 99)
    assert not ok_unknown


def test_compute_eclipse_dispatcher_unknown_model_returns_false():
    r0, v0 = initial_conditions()
    jtime = make_jtime()

    ok_orbit, R, _V = saltro_py.compute_orbit(r0, v0, jtime, 0)
    assert ok_orbit
    ok_sun, S = saltro_py.compute_sun(R, jtime, 0)
    assert ok_sun

    ok_unknown, _ = saltro_py.compute_eclipse(R, jtime, S, 99)
    assert not ok_unknown


def test_compute_density_dispatcher_unknown_model_returns_false():
    r0, v0 = initial_conditions()
    jtime = make_jtime()

    ok_orbit, R, _V = saltro_py.compute_orbit(r0, v0, jtime, 0)
    assert ok_orbit
    ok_sun, S = saltro_py.compute_sun(R, jtime, 0)
    assert ok_sun

    ok_unknown, _ = saltro_py.compute_density(R, S, 99)
    assert not ok_unknown
