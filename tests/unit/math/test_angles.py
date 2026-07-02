import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "build"))

try:
    import saltro_py
except ImportError as e:
    pytest.skip(f"saltro Python bindings not available: {e}", allow_module_level=True)


def test_wrap_to_2pi_keeps_in_range_angles():
    assert math.isclose(saltro_py.wrap_to_2pi(0.0), 0.0, abs_tol=1e-12)
    assert math.isclose(saltro_py.wrap_to_2pi(1.0), 1.0, abs_tol=1e-12)
    assert math.isclose(saltro_py.wrap_to_2pi(math.pi), math.pi, abs_tol=1e-12)


def test_wrap_to_2pi_wraps_above_and_below():
    assert math.isclose(saltro_py.wrap_to_2pi(2.0 * math.pi + 0.5), 0.5, abs_tol=1e-12)
    assert math.isclose(saltro_py.wrap_to_2pi(-0.5), 2.0 * math.pi - 0.5, abs_tol=1e-12)
    assert math.isclose(saltro_py.wrap_to_2pi(4.0 * math.pi + 1.0), 1.0, abs_tol=1e-12)


def test_wrap_to_2pi_output_is_in_bounds():
    for angle in np.arange(-20.0, 20.0 + 1e-12, 0.37):
        wrapped = saltro_py.wrap_to_2pi(float(angle))
        assert 0.0 <= wrapped < 2.0 * math.pi


def test_wrap_to_360_known_values():
    assert math.isclose(saltro_py.wrap_to_360(45.0), 45.0, abs_tol=1e-12)
    assert math.isclose(saltro_py.wrap_to_360(360.0), 0.0, abs_tol=1e-12)
    assert math.isclose(saltro_py.wrap_to_360(450.0), 90.0, abs_tol=1e-12)
    assert math.isclose(saltro_py.wrap_to_360(-90.0), 270.0, abs_tol=1e-12)


def test_wrap_to_360_output_is_in_bounds():
    for angle in np.arange(-1000.0, 1000.0 + 1e-12, 17.3):
        wrapped = saltro_py.wrap_to_360(float(angle))
        assert 0.0 <= wrapped < 360.0


def test_deg2rad_known_values():
    assert math.isclose(saltro_py.deg2rad(180.0), math.pi, abs_tol=1e-12)
    assert math.isclose(saltro_py.deg2rad(90.0), math.pi / 2.0, abs_tol=1e-12)
    assert math.isclose(saltro_py.deg2rad(0.0), 0.0, abs_tol=1e-12)


def test_rad2deg_known_values():
    assert math.isclose(saltro_py.rad2deg(math.pi), 180.0, abs_tol=1e-12)
    assert math.isclose(saltro_py.rad2deg(math.pi / 2.0), 90.0, abs_tol=1e-12)


def test_deg2rad_and_rad2deg_are_mutual_inverses():
    for deg in np.arange(-350.0, 350.0 + 1e-12, 13.0):
        assert math.isclose(saltro_py.rad2deg(saltro_py.deg2rad(float(deg))), float(deg), abs_tol=1e-10)
