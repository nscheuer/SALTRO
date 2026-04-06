import math

import numpy as np
import pytest

try:
    import saltro_py

    validateOrbitState = lambda r0, v0: saltro_py.validateOrbitState(r0, v0)
except ImportError as e:
    pytest.skip(f"saltro Python bindings not available: {e}", allow_module_level=True)


R_EARTH = 6378136.3
MU_EARTH = 3.986004418e14


def make_circular_leo(altitude_m: float):
    radius = R_EARTH + altitude_m
    speed = math.sqrt(MU_EARTH / radius)
    r0 = np.array([radius, 0.0, 0.0], dtype=float)
    v0 = np.array([0.0, speed, 0.0], dtype=float)
    return r0, v0


def test_valid_circular_leo_state_passes_validation():
    r0, v0 = make_circular_leo(500e3)

    ok, error_msg = validateOrbitState(r0, v0)

    assert ok
    assert error_msg == "" or error_msg is None


def test_r0_likely_in_km_fails_with_unit_hint():
    r0_km = np.array([6878.0, 0.0, 0.0], dtype=float)
    v0_kms = np.array([0.0, 7.6, 0.0], dtype=float)

    ok, error_msg = validateOrbitState(r0_km, v0_kms)

    assert not ok
    assert error_msg == "r0 magnitude too small; expected meters (did you provide kilometers?)"


def test_non_finite_state_fails():
    r0, v0 = make_circular_leo(500e3)
    r0[1] = np.nan

    ok, error_msg = validateOrbitState(r0, v0)

    assert not ok
    assert error_msg == "r0 contains non-finite values"


def test_highly_elliptical_orbit_fails():
    rp = R_EARTH + 400e3
    ra = R_EARTH + 10000e3
    a = 0.5 * (rp + ra)
    vp = math.sqrt(MU_EARTH * (2.0 / rp - 1.0 / a))

    r0 = np.array([rp, 0.0, 0.0], dtype=float)
    v0 = np.array([0.0, vp, 0.0], dtype=float)

    ok, error_msg = validateOrbitState(r0, v0)

    assert not ok
    assert error_msg == "orbit is too elliptical for LEO use (eccentricity > 0.3)"


def test_bound_orbit_above_leo_ceiling_fails():
    r0, v0 = make_circular_leo(3500e3)

    ok, error_msg = validateOrbitState(r0, v0)

    assert not ok
    assert error_msg == "apogee altitude above LEO bounds"


def test_unbound_orbit_fails():
    r0, v0 = make_circular_leo(500e3)
    escape_speed = math.sqrt(2.0 * MU_EARTH / np.linalg.norm(r0))
    v0 = np.array([0.0, 1.1 * escape_speed, 0.0], dtype=float)

    ok, error_msg = validateOrbitState(r0, v0)

    assert not ok
    assert error_msg == "specific orbital energy is non-negative (orbit not bound to Earth)"
