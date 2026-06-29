"""Python twin of tests/unit/math/test_angles.cpp.

Mirrors the C++ angle-helper tests against the saltro_py bindings so the math
helpers are exercised identically from both languages.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "build"))

import saltro_py as s

TWO_PI = 2.0 * math.pi


# ----------------------------------------------------------------------------
# wrap_to_2pi
# ----------------------------------------------------------------------------
def test_wrap_to_2pi_keeps_in_range_angles():
    assert abs(s.wrap_to_2pi(0.0) - 0.0) < 1e-12
    assert abs(s.wrap_to_2pi(1.0) - 1.0) < 1e-12
    assert abs(s.wrap_to_2pi(math.pi) - math.pi) < 1e-12


def test_wrap_to_2pi_wraps_above_and_below():
    assert abs(s.wrap_to_2pi(TWO_PI + 0.5) - 0.5) < 1e-12
    assert abs(s.wrap_to_2pi(-0.5) - (TWO_PI - 0.5)) < 1e-12
    assert abs(s.wrap_to_2pi(4.0 * math.pi + 1.0) - 1.0) < 1e-12


def test_wrap_to_2pi_output_always_in_range():
    a = -20.0
    while a <= 20.0:
        w = s.wrap_to_2pi(a)
        assert w >= 0.0
        assert w < TWO_PI
        a += 0.37


# ----------------------------------------------------------------------------
# wrap_to_360
# ----------------------------------------------------------------------------
def test_wrap_to_360_known_values():
    assert abs(s.wrap_to_360(45.0) - 45.0) < 1e-12
    assert abs(s.wrap_to_360(360.0) - 0.0) < 1e-12
    assert abs(s.wrap_to_360(450.0) - 90.0) < 1e-12
    assert abs(s.wrap_to_360(-90.0) - 270.0) < 1e-12


def test_wrap_to_360_output_always_in_range():
    a = -1000.0
    while a <= 1000.0:
        w = s.wrap_to_360(a)
        assert w >= 0.0
        assert w < 360.0
        a += 17.3


# ----------------------------------------------------------------------------
# deg2rad / rad2deg
# ----------------------------------------------------------------------------
def test_deg2rad_known_values():
    assert abs(s.deg2rad(180.0) - math.pi) < 1e-12
    assert abs(s.deg2rad(90.0) - math.pi / 2.0) < 1e-12
    assert abs(s.deg2rad(0.0)) < 1e-12


def test_rad2deg_known_values():
    assert abs(s.rad2deg(math.pi) - 180.0) < 1e-12
    assert abs(s.rad2deg(math.pi / 2.0) - 90.0) < 1e-12


def test_deg2rad_and_rad2deg_are_mutual_inverses():
    d = -350.0
    while d <= 350.0:
        assert abs(s.rad2deg(s.deg2rad(d)) - d) < 1e-10
        d += 13.0
