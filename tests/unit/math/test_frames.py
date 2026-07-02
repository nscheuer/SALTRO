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


def test_gmst_rad_output_is_wrapped():
    for t in np.arange(-0.5, 0.5 + 1e-12, 0.05):
        gmst = saltro_py.gmst_rad(float(t))
        assert 0.0 <= gmst < 2.0 * math.pi


def test_gmst_rad_is_deterministic():
    assert math.isclose(saltro_py.gmst_rad(0.1), saltro_py.gmst_rad(0.1), abs_tol=1e-15)


def test_eci_to_ecef_dcm_is_orthonormal_with_positive_determinant():
    for t in (-0.2, 0.0, 0.1, 0.37):
        c = saltro_py.eci_to_ecef_dcm(t)
        assert np.linalg.norm(c @ c.T - np.eye(3)) < 1e-12
        assert abs(np.linalg.det(c) - 1.0) < 1e-12


def test_ecef_to_eci_dcm_is_orthonormal_with_positive_determinant():
    for t in (-0.2, 0.0, 0.1, 0.37):
        c = saltro_py.ecef_to_eci_dcm(t)
        assert np.linalg.norm(c @ c.T - np.eye(3)) < 1e-12
        assert abs(np.linalg.det(c) - 1.0) < 1e-12


def test_eci_and_ecef_dcms_are_mutual_inverses():
    for t in (-0.2, 0.0, 0.1, 0.37):
        c1 = saltro_py.eci_to_ecef_dcm(t)
        c2 = saltro_py.ecef_to_eci_dcm(t)
        assert np.linalg.norm(c1 - c2.T) < 1e-14
        assert np.linalg.norm(c1 @ c2 - np.eye(3)) < 1e-12


def test_eci_to_ecef_is_pure_z_rotation():
    c = saltro_py.eci_to_ecef_dcm(0.123)
    assert abs(c[2, 2] - 1.0) < 1e-12
    assert abs(c[2, 0]) < 1e-12
    assert abs(c[2, 1]) < 1e-12
    assert abs(c[0, 2]) < 1e-12
    assert abs(c[1, 2]) < 1e-12
