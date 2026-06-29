"""Python twin of tests/unit/math/test_frames.cpp."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "build"))

import numpy as np
import saltro_py as s

TWO_PI = 2.0 * math.pi


# ----------------------------------------------------------------------------
# gmst_rad
# ----------------------------------------------------------------------------
def test_gmst_rad_output_is_wrapped():
    T = -0.5
    while T <= 0.5:
        g = s.gmst_rad(T)
        assert g >= 0.0
        assert g < TWO_PI
        T += 0.05


def test_gmst_rad_is_deterministic():
    assert abs(s.gmst_rad(0.1) - s.gmst_rad(0.1)) < 1e-15


# ----------------------------------------------------------------------------
# eci_to_ecef_dcm / ecef_to_eci_dcm
# ----------------------------------------------------------------------------
def test_eci_to_ecef_dcm_is_orthonormal_det_plus_one():
    for T in (-0.2, 0.0, 0.1, 0.37):
        C = np.array(s.eci_to_ecef_dcm(T))
        assert np.linalg.norm(C @ C.T - np.eye(3)) < 1e-12
        assert abs(np.linalg.det(C) - 1.0) < 1e-12


def test_ecef_to_eci_dcm_is_orthonormal_det_plus_one():
    for T in (-0.2, 0.0, 0.1, 0.37):
        C = np.array(s.ecef_to_eci_dcm(T))
        assert np.linalg.norm(C @ C.T - np.eye(3)) < 1e-12
        assert abs(np.linalg.det(C) - 1.0) < 1e-12


def test_eci_ecef_dcms_are_mutual_transposes_inverses():
    for T in (-0.2, 0.0, 0.1, 0.37):
        C1 = np.array(s.eci_to_ecef_dcm(T))
        C2 = np.array(s.ecef_to_eci_dcm(T))
        assert np.linalg.norm(C1 - C2.T) < 1e-14
        assert np.linalg.norm(C1 @ C2 - np.eye(3)) < 1e-12


def test_eci_to_ecef_dcm_is_pure_z_rotation():
    C = np.array(s.eci_to_ecef_dcm(0.123))
    assert abs(C[2, 2] - 1.0) < 1e-12
    assert abs(C[2, 0]) < 1e-12
    assert abs(C[2, 1]) < 1e-12
    assert abs(C[0, 2]) < 1e-12
    assert abs(C[1, 2]) < 1e-12
