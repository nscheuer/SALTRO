import numpy as np
import pytest

try:
    import saltro_py

    validateJulianTime = lambda jtime: saltro_py.validateJulianTime(jtime)
except ImportError as e:
    pytest.skip(f"saltro Python bindings not available: {e}", allow_module_level=True)


def test_valid_julian_time_vector_passes_validation():
    jtime = np.array([0.20, 0.26, 0.31, 0.40], dtype=float)

    ok, error_msg = validateJulianTime(jtime)

    assert ok
    assert error_msg == "" or error_msg is None


def test_empty_julian_time_vector_fails():
    jtime = np.array([], dtype=float)

    ok, error_msg = validateJulianTime(jtime)

    assert not ok
    assert error_msg == "jtime is empty"


def test_julian_time_vector_with_nan_fails():
    jtime = np.array([0.21, np.nan, 0.24], dtype=float)

    ok, error_msg = validateJulianTime(jtime)

    assert not ok
    assert error_msg == "jtime contains non-finite values"


def test_julian_time_vector_with_infinity_fails():
    jtime = np.array([0.21, np.inf, 0.24], dtype=float)

    ok, error_msg = validateJulianTime(jtime)

    assert not ok
    assert error_msg == "jtime contains non-finite values"


def test_julian_time_vector_with_zero_fails():
    jtime = np.array([0.21, 0.0, 0.24], dtype=float)

    ok, error_msg = validateJulianTime(jtime)

    assert not ok
    assert error_msg == "jtime contains zero values"


def test_julian_time_below_mission_bounds_fails():
    jtime = np.array([0.19, 0.22, 0.24], dtype=float)

    ok, error_msg = validateJulianTime(jtime)

    assert not ok
    assert error_msg == "jtime is outside mission bounds [0.20, 0.40] Julian centuries"


def test_julian_time_above_mission_bounds_fails():
    jtime = np.array([0.22, 0.24, 0.41], dtype=float)

    ok, error_msg = validateJulianTime(jtime)

    assert not ok
    assert error_msg == "jtime is outside mission bounds [0.20, 0.40] Julian centuries"


def test_julian_time_with_repeated_values_fails():
    jtime = np.array([0.22, 0.22, 0.24], dtype=float)

    ok, error_msg = validateJulianTime(jtime)

    assert not ok
    assert error_msg == "jtime must be strictly increasing"


def test_julian_time_with_decreasing_values_fails():
    jtime = np.array([0.24, 0.23, 0.25], dtype=float)

    ok, error_msg = validateJulianTime(jtime)

    assert not ok
    assert error_msg == "jtime must be strictly increasing"
