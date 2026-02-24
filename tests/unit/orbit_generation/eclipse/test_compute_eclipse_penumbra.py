import sys
from pathlib import Path
import numpy as np
import pytest
import math

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))

import saltro_py

RE = 6378136.3
RSUN = 6.957e8
AU_M = 1.496e11


def expected_penumbra_eclipse(r_sat: np.ndarray, sun_vec: np.ndarray) -> bool:
    eps = 1e-12
    sun_norm2 = float(np.dot(sun_vec, sun_vec))
    r_norm2 = float(np.dot(r_sat, r_sat))

    if sun_norm2 < eps or r_norm2 < eps:
        return False

    sun_norm = math.sqrt(sun_norm2)
    r_norm = math.sqrt(r_norm2)
    inv_sun = 1.0 / sun_norm
    inv_r = 1.0 / r_norm

    cos_psi = float(np.dot(-r_sat, sun_vec)) * inv_r * inv_sun
    if cos_psi <= 0.0:
        return False
    if cos_psi > 1.0:
        cos_psi = 1.0

    sin_alpha_e = RE * inv_r
    sin_alpha_s = RSUN * inv_sun

    if sin_alpha_e > 1.0:
        sin_alpha_e = 1.0
    if sin_alpha_s > 1.0:
        sin_alpha_s = 1.0

    cos_alpha_e = math.sqrt(max(0.0, 1.0 - sin_alpha_e * sin_alpha_e))
    cos_alpha_s = math.sqrt(max(0.0, 1.0 - sin_alpha_s * sin_alpha_s))

    cos_sum = (cos_alpha_e * cos_alpha_s) - (sin_alpha_e * sin_alpha_s)

    return cos_psi >= cos_sum


class TestEclipsePenumbraBasicFunctionality:
    def test_basic_functionality(self):
        N = 5
        R = np.zeros((3, N))
        S = np.zeros((3, N))

        for i in range(N):
            R[:, i] = [RE + 400e3, 0.0, 0.0]
            S[:, i] = [AU_M, 0.0, 0.0]

        ok, S_out = saltro_py.compute_eclipse_penumbra(R, S)
        assert ok

        for i in range(N):
            assert np.linalg.norm(S_out[:, i]) > 1e10

    def test_detects_shadow(self):
        R = np.array([-(RE + 400e3), 0.0, 0.0]).reshape(3, 1)
        S = np.array([AU_M, 0.0, 0.0]).reshape(3, 1)

        ok, S_out = saltro_py.compute_eclipse_penumbra(R, S)
        assert ok
        assert np.linalg.norm(S_out[:, 0]) < 1e-10


class TestEclipsePenumbraGeometry:
    def test_off_axis_inside_cone(self):
        r = RE + 400e3
        alpha_e = math.asin(RE / r)
        alpha_s = math.asin(RSUN / AU_M)
        psi = (alpha_e + alpha_s) - 1e-4

        R = np.array([-r * math.cos(psi), r * math.sin(psi), 0.0]).reshape(3, 1)
        S = np.array([AU_M, 0.0, 0.0]).reshape(3, 1)

        ok, S_out = saltro_py.compute_eclipse_penumbra(R, S)
        assert ok
        assert np.linalg.norm(S_out[:, 0]) < 1e-10

    def test_off_axis_outside_cone(self):
        r = RE + 400e3
        alpha_e = math.asin(RE / r)
        alpha_s = math.asin(RSUN / AU_M)
        psi = (alpha_e + alpha_s) + 1e-4

        R = np.array([-r * math.cos(psi), r * math.sin(psi), 0.0]).reshape(3, 1)
        S_orig = np.array([AU_M, 0.0, 0.0]).reshape(3, 1)

        ok, S_out = saltro_py.compute_eclipse_penumbra(R, S_orig)
        assert ok
        assert np.linalg.norm(S_out[:, 0] - S_orig[:, 0]) < 1e-6


class TestEclipsePenumbraOrbit:
    def test_orbit_expected_pattern(self):
        N = 24
        R = np.zeros((3, N))
        S = np.zeros((3, N))

        r = RE + 400e3

        for i in range(N):
            theta = 2.0 * math.pi * i / N
            R[:, i] = [r * math.cos(theta), r * math.sin(theta), 0.0]
            S[:, i] = [AU_M, 0.0, 0.0]

        ok, S_out = saltro_py.compute_eclipse_penumbra(R, S)
        assert ok

        for i in range(N):
            expected = expected_penumbra_eclipse(R[:, i], np.array([AU_M, 0.0, 0.0]))
            if expected:
                assert np.linalg.norm(S_out[:, i]) < 1e-10
            else:
                assert np.linalg.norm(S_out[:, i]) > 1e10


class TestEclipsePenumbraVectorPreservation:
    def test_arbitrary_sun_direction(self):
        N = 8
        R = np.zeros((3, N))
        S = np.zeros((3, N))

        r = RE + 400e3
        sun_dir = np.array([1.0, 0.5, 0.3])
        sun_dir = sun_dir / np.linalg.norm(sun_dir)
        sun_dist = AU_M

        for i in range(N):
            theta = 2.0 * math.pi * i / N
            R[:, i] = [r * math.cos(theta), r * math.sin(theta), 0.0]
            S[:, i] = sun_dist * sun_dir

        ok, S_out = saltro_py.compute_eclipse_penumbra(R, S)
        assert ok

        for i in range(N):
            expected = expected_penumbra_eclipse(R[:, i], sun_dist * sun_dir)
            if expected:
                assert np.linalg.norm(S_out[:, i]) < 1e-10
            else:
                assert abs(np.linalg.norm(S_out[:, i]) - sun_dist) < 1e4

    def test_zero_sun_vector_handling(self):
        R = np.zeros((3, 2))
        S = np.zeros((3, 2))

        R[:, 0] = [RE + 400e3, 0.0, 0.0]
        S[:, 0] = [AU_M, 0.0, 0.0]

        R[:, 1] = [-(RE + 400e3), 0.0, 0.0]
        S[:, 1] = [0.0, 0.0, 0.0]

        ok, S_out = saltro_py.compute_eclipse_penumbra(R, S)
        assert ok

        assert np.linalg.norm(S_out[:, 0]) > 1e10
        assert np.linalg.norm(S_out[:, 1]) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
