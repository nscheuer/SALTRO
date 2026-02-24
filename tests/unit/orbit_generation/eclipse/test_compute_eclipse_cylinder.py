import sys
from pathlib import Path
import numpy as np
import pytest
import math

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))

import saltro_py

# Constants
RE = 6378136.3  # Earth's radius in meters
AU_M = 1.496e11  # AU in meters


class TestEclipseCylinderBasicFunctionality:
    """Test basic eclipse cylinder operation"""
    
    def test_basic_functionality(self):
        """Sunlit positions should remain unchanged"""
        N = 5
        R = np.zeros((3, N))
        S = np.zeros((3, N))
        
        for i in range(N):
            R[:, i] = [RE + 400e3, 0.0, 0.0]
            S[:, i] = [1e11, 0.0, 0.0]
        
        ok, S_out = saltro_py.compute_eclipse_cylinder(R, S)
        assert ok
        
        # All positions should be sunlit
        for i in range(N):
            sun_norm = np.linalg.norm(S_out[:, i])
            assert sun_norm > 1e10
    
    def test_detects_shadow(self):
        """Satellite on night side should be eclipsed"""
        R = np.array([[-RE - 400e3, 0.0, 0.0]])
        S = np.array([[1e11, 0.0, 0.0]])
        
        ok, S_out = saltro_py.compute_eclipse_cylinder(R.T, S.T)
        assert ok
        assert np.linalg.norm(S_out[:, 0]) < 1e-10
    
    def test_sunlit_side_unchanged(self):
        """Sunlit positions should preserve sun vector"""
        R = np.array([RE + 400e3, 0.0, 0.0]).reshape(3, 1)
        S_orig = np.array([1e11, 0.0, 0.0]).reshape(3, 1)
        
        ok, S_out = saltro_py.compute_eclipse_cylinder(R, S_orig)
        assert ok
        assert np.linalg.norm(S_out[:, 0] - S_orig[:, 0]) < 1e-6


class TestEclipseCylinderGeometry:
    """Test geometric edge cases"""
    
    def test_off_axis_night_side(self):
        """Satellite behind Earth but offset should still be eclipsed"""
        R = np.array([-(RE + 400e3), RE * 0.5, 0.0]).reshape(3, 1)
        S = np.array([1e11, 0.0, 0.0]).reshape(3, 1)
        
        ok, S_out = saltro_py.compute_eclipse_cylinder(R, S)
        assert ok
        assert np.linalg.norm(S_out[:, 0]) < 1e-10
    
    def test_beyond_shadow_edge(self):
        """Satellite beyond shadow edge should be sunlit"""
        R = np.array([-(RE + 400e3), RE * 2.0, 0.0]).reshape(3, 1)
        S_orig = np.array([1e11, 0.0, 0.0]).reshape(3, 1)
        
        ok, S_out = saltro_py.compute_eclipse_cylinder(R, S_orig)
        assert ok
        assert np.linalg.norm(S_out[:, 0] - S_orig[:, 0]) < 1e-6


class TestEclipseCylinderOrbit:
    """Test eclipse patterns for different orbits"""
    
    def test_circular_orbit_partial_eclipse(self):
        """Circular orbit should have partial eclipse"""
        N = 24
        R = np.zeros((3, N))
        S = np.zeros((3, N))
        
        alt = 400e3
        r = RE + alt
        
        for i in range(N):
            theta = 2.0 * math.pi * i / N
            R[:, i] = [r * math.cos(theta), r * math.sin(theta), 0.0]
            S[:, i] = [AU_M, 0.0, 0.0]
        
        ok, S_out = saltro_py.compute_eclipse_cylinder(R, S)
        assert ok
        
        eclipsed_count = sum(1 for i in range(N) if np.linalg.norm(S_out[:, i]) < 1e-10)
        sunlit_count = N - eclipsed_count
        
        assert eclipsed_count > 0
        assert sunlit_count > 0
    
    def test_equatorial_orbit(self):
        """Equatorial orbit should have binary eclipse/sunlit"""
        N = 16
        R = np.zeros((3, N))
        S = np.zeros((3, N))
        
        r = RE + 400e3
        
        for i in range(N):
            theta = 2.0 * math.pi * i / N
            R[:, i] = [r * math.cos(theta), r * math.sin(theta), 0.0]
            S[:, i] = [AU_M, 0.0, 0.0]
        
        ok, S_out = saltro_py.compute_eclipse_cylinder(R, S)
        assert ok
        
        for i in range(N):
            sun_mag = np.linalg.norm(S_out[:, i])
            is_sunlit = sun_mag > 1e10
            is_eclipsed = sun_mag < 1e-10
            assert is_sunlit or is_eclipsed
    
    def test_high_altitude_orbit(self):
        """GEO altitude should have minimal eclipse"""
        N = 12
        R = np.zeros((3, N))
        S = np.zeros((3, N))
        
        r = RE + 36000e3  # GEO
        
        for i in range(N):
            theta = 2.0 * math.pi * i / N
            R[:, i] = [r * math.cos(theta), r * math.sin(theta), 0.0]
            S[:, i] = [AU_M, 0.0, 0.0]
        
        ok, S_out = saltro_py.compute_eclipse_cylinder(R, S)
        assert ok
        
        eclipsed_count = sum(1 for i in range(N) if np.linalg.norm(S_out[:, i]) < 1e-10)
        assert eclipsed_count < N // 4  # Less than 25%


class TestEclipseCylinderVectorPreservation:
    """Test that sun vectors are preserved when appropriate"""
    
    def test_preserves_magnitude_sunlit(self):
        """Sunlit positions should preserve vector magnitude"""
        N = 3
        R = np.zeros((3, N))
        S = np.zeros((3, N))
        
        for i in range(N):
            R[:, i] = [RE + 400e3 + i * 100e3, 0.0, 0.0]
            S[:, i] = [AU_M, 1e9, 1e8]
        
        original_mags = np.array([np.linalg.norm(S[:, i]) for i in range(N)])
        
        ok, S_out = saltro_py.compute_eclipse_cylinder(R, S)
        assert ok
        
        for i in range(N):
            new_mag = np.linalg.norm(S_out[:, i])
            assert abs(new_mag - original_mags[i]) < 1e-6
    
    def test_zero_sun_vector_handling(self):
        """Zero sun vectors should be skipped"""
        R = np.zeros((3, 2))
        S = np.zeros((3, 2))
        
        R[:, 0] = [RE + 400e3, 0.0, 0.0]
        S[:, 0] = [1e11, 0.0, 0.0]
        
        R[:, 1] = [-(RE + 400e3), 0.0, 0.0]
        S[:, 1] = [0.0, 0.0, 0.0]
        
        ok, S_out = saltro_py.compute_eclipse_cylinder(R, S)
        assert ok
        
        assert np.linalg.norm(S_out[:, 0]) > 1e10
        assert np.linalg.norm(S_out[:, 1]) < 1e-10


class TestEclipseCylinderMathematical:
    """Test mathematical correctness of eclipse calculations"""
    
    def test_multi_orbit_mathematical_model(self):
        """Multi-orbit test with explicit mathematical calculation"""
        N = 100  # 5 complete orbits of 20 points each
        R = np.zeros((3, N))
        S = np.zeros((3, N))
        
        r = RE + 400e3
        
        for i in range(N):
            theta = 2.0 * math.pi * i / (N / 5)  # 5 orbits
            R[:, i] = [r * math.cos(theta), r * math.sin(theta), 0.0]
            S[:, i] = [AU_M, 0.0, 0.0]
        
        ok, S_out = saltro_py.compute_eclipse_cylinder(R, S)
        assert ok
        
        eclipsed_per_orbit = sum(1 for i in range(N) if np.linalg.norm(S_out[:, i]) < 1e-10)
        
        # Calculate expected eclipse fraction from geometry
        # For circular orbit at altitude h: sin(θ_e) = R_E / (R_E + h)
        # where θ_e is the eclipse half-angle
        alt = 400e3
        eclipse_half_angle = math.asin(RE / (RE + alt))
        total_eclipse_angle = 2.0 * eclipse_half_angle  # in radians
        eclipse_fraction = total_eclipse_angle / (2.0 * math.pi)
        
        # Expected eclipsed count with ±10% tolerance for discretization
        expected_eclipsed = int(eclipse_fraction * N)
        lower_bound = int(expected_eclipsed * 0.9)
        upper_bound = int(expected_eclipsed * 1.1)
        
        assert eclipsed_per_orbit >= lower_bound
        assert eclipsed_per_orbit <= upper_bound
    
    def test_arbitrary_sun_direction(self):
        """Test with non-axis-aligned sun direction"""
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
        
        ok, S_out = saltro_py.compute_eclipse_cylinder(R, S)
        assert ok
        
        # Verify all outputs are either normal or zero
        for i in range(N):
            sun_mag = np.linalg.norm(S_out[:, i])
            is_normal = abs(sun_mag - sun_dist) < 1e4
            is_zero = sun_mag < 1e-10
            assert is_normal or is_zero


class TestEclipseCylinderEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_single_point(self):
        """Single sunlit point"""
        R = np.array([RE + 400e3, 0.0, 0.0]).reshape(3, 1)
        S = np.array([AU_M, 0.0, 0.0]).reshape(3, 1)
        
        ok, S_out = saltro_py.compute_eclipse_cylinder(R, S)
        assert ok
        assert np.linalg.norm(S_out[:, 0]) > 1e10
    
    def test_directly_behind_earth(self):
        """Satellite directly behind Earth should be eclipsed"""
        R = np.array([-(RE + 400e3), 0.0, 0.0]).reshape(3, 1)
        S = np.array([AU_M, 0.0, 0.0]).reshape(3, 1)
        
        ok, S_out = saltro_py.compute_eclipse_cylinder(R, S)
        assert ok
        assert np.linalg.norm(S_out[:, 0]) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
