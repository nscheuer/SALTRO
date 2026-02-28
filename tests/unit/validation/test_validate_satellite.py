"""
Comprehensive pytest test suite for satellite validation.

This module tests the validateSatellite() function and the Satellite API's 
input validation across multiple categories:
- Inertia matrix validation
- Actuator (MTQ/RW) configuration validation
- Geometry configuration validation
- Boundary value tests

NOTE ON TEST DESIGN:
The Satellite class performs its own validation in constructors and setters,
raising exceptions for invalid inputs (e.g., non-invertible inertia).
The validateSatellite() function provides an additional safety layer that:
  1. Confirms valid satellites pass validation
  2. Catches edge cases and internal inconsistencies
  3. Provides detailed error messages for debugging

Many negative test cases cannot reach validateSatellite() because the
Satellite API prevents creation of invalid states. These are tested
via pytest.raises to verify API-level validation works correctly.
"""

import pytest
import numpy as np
import math
from unittest.mock import Mock, patch

# These imports assume Python bindings are available
try:
    import saltro_py
    # Convenience aliases for cleaner code
    validateSatellite = lambda sat: saltro_py.validateSatellite(sat)
    Satellite = saltro_py.Satellite
    PlannerSettings = saltro_py.PlannerSettings
    # Get limits from saltro_py module
    MAX_NUM_MTQ = saltro_py.limits.MAX_NUM_MTQ
    MAX_NUM_RW = saltro_py.limits.MAX_NUM_RW
    MAX_NUM_GEOMETRY_FACES = saltro_py.limits.MAX_NUM_GEOMETRY_FACES
    GeometryConfig = saltro_py.GeometryConfig
    GeometryFace = saltro_py.GeometryFace
except ImportError as e:
    pytest.skip(f"saltro Python bindings not available: {e}", allow_module_level=True)


# ============================================================================
# Helper Functions
# ============================================================================

def valid_inertia_matrix():
    """Create a valid 3x3 symmetric positive definite inertia matrix."""
    return np.array([
        [0.067, 0.0, 0.0],
        [0.0, 0.067, 0.0],
        [0.0, 0.0, 0.067]
    ])


# ============================================================================
# Basic Validation Tests
# ============================================================================

class TestBasicValidation:
    """Tests for basic valid satellite configurations."""

    def test_valid_satellite_passes_validation(self):
        """Valid satellite with only inertia should pass validation."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        
        is_valid, error_msg = validateSatellite(sat)
        assert is_valid
        assert error_msg == "" or error_msg is None

    def test_valid_satellite_with_mtq(self):
        """Valid satellite with MTQ should pass validation."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        sat.addMTQ(np.array([1.0, 0.0, 0.0]), 0.2)
        
        is_valid, error_msg = validateSatellite(sat)
        assert is_valid

    def test_valid_satellite_with_rw(self):
        """Valid satellite with RW should pass validation."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        sat.addRW(np.array([1.0, 0.0, 0.0]), 0.01, 0.001, 0.0, 0.1)
        
        is_valid, error_msg = validateSatellite(sat)
        assert is_valid

    def test_valid_satellite_with_geometry(self):
        """Valid satellite with geometry face should pass validation."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        geom = sat.geometryConfig
        
        centroid = np.array([0.0, 0.0, 0.1])
        normal = np.array([0.0, 0.0, 1.0])
        face = GeometryFace(0.01, centroid, normal, 0.1, 0.2, 0.7, 2.2)
        geom.addFace(face)
        
        is_valid, error_msg = validateSatellite(sat)
        assert is_valid


# ============================================================================
# Inertia Matrix Validation Tests
# ============================================================================

class TestInertiaValidation:
    """Tests for inertia matrix validation."""

    def test_invalid_inertia_contains_nan(self):
        """Satellite constructor should reject inertia with NaN."""
        J = valid_inertia_matrix()
        J[0, 0] = np.nan
        settings = PlannerSettings()
        
        with pytest.raises(ValueError):
            Satellite(J, settings)

    def test_invalid_inertia_contains_infinity(self):
        """setInertia should reject inertia with infinity."""
        J = valid_inertia_matrix()
        J[1, 1] = np.inf
        settings = PlannerSettings()
        sat = Satellite(valid_inertia_matrix(), settings)
        
        with pytest.raises(ValueError):
            sat.setInertia(J)

    def test_invalid_inertia_not_symmetric(self):
        """Non-symmetric inertia should be accepted by API but caught by validation."""
        J = valid_inertia_matrix()
        J[0, 1] = 0.01
        J[1, 0] = 0.02  # Different from J[0,1]
        settings = PlannerSettings()
        sat = Satellite(valid_inertia_matrix(), settings)
        
        # setInertia doesn't check symmetry, so it accepts this
        sat.setInertia(J)
        
        is_valid, error_msg = validateSatellite(sat)
        assert not is_valid
        assert error_msg == "inertia matrix is not symmetric"

    def test_invalid_inertia_zero_determinant(self):
        """setInertia should reject inertia with zero determinant."""
        J = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0]
        ])
        settings = PlannerSettings()
        sat = Satellite(valid_inertia_matrix(), settings)
        
        with pytest.raises(ValueError):
            sat.setInertia(J)

    def test_invalid_inertia_negative_determinant(self):
        """Inertia with negative determinant should be accepted by API but caught by validation."""
        J = np.array([
            [-0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.1]
        ])
        settings = PlannerSettings()
        sat = Satellite(valid_inertia_matrix(), settings)
        
        # This matrix has determinant <= 1e-12, so setInertia accepts it
        sat.setInertia(J)
        
        is_valid, error_msg = validateSatellite(sat)
        assert not is_valid
        assert error_msg == "inertia matrix determinant is too small or non-positive"

    def test_invalid_inertia_non_positive_eigenvalue(self):
        """Inertia with non-positive eigenvalue should be caught by validation."""
        J = np.array([
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, -0.05]
        ])
        settings = PlannerSettings()
        sat = Satellite(valid_inertia_matrix(), settings)
        
        sat.setInertia(J)
        
        is_valid, error_msg = validateSatellite(sat)
        assert not is_valid
        assert error_msg == "inertia matrix determinant is too small or non-positive"

    def test_invalid_inertia_magnitude_too_small(self):
        """setInertia should reject inertia that's too small."""
        J = np.array([
            [1e-7, 0.0, 0.0],
            [0.0, 1e-7, 0.0],
            [0.0, 0.0, 1e-7]
        ])
        settings = PlannerSettings()
        sat = Satellite(valid_inertia_matrix(), settings)
        
        with pytest.raises(ValueError):
            sat.setInertia(J)

    def test_invalid_inertia_magnitude_too_large(self):
        """Inertia that's too large should be caught by validation."""
        J = np.array([
            [1e7, 0.0, 0.0],
            [0.0, 1e7, 0.0],
            [0.0, 0.0, 1e7]
        ])
        settings = PlannerSettings()
        sat = Satellite(valid_inertia_matrix(), settings)
        sat.setInertia(J)
        
        is_valid, error_msg = validateSatellite(sat)
        assert not is_valid
        assert error_msg == "inertia matrix magnitude out of reasonable range"


# ============================================================================
# Actuator Count Validation Tests
# ============================================================================

class TestActuatorCountValidation:
    """Tests for actuator count validation."""

    def test_valid_satellite_with_maximum_mtqs(self):
        """Satellite with maximum MTQs should pass validation."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        
        for i in range(MAX_NUM_MTQ):
            axis = np.zeros(3)
            axis[i % 3] = 1.0
            sat.addMTQ(axis, 0.2)
        
        is_valid, error_msg = validateSatellite(sat)
        assert is_valid

    def test_valid_satellite_with_maximum_rws(self):
        """Satellite with maximum RWs should pass validation."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        
        for i in range(MAX_NUM_RW):
            axis = np.zeros(3)
            axis[i % 3] = 1.0
            sat.addRW(axis, 0.01, 0.001, 0.0, 0.1)
        
        is_valid, error_msg = validateSatellite(sat)
        assert is_valid


# ============================================================================
# MTQ Configuration Validation Tests
# ============================================================================

class TestMTQValidation:
    """Tests for MTQ (Magnetorquer) configuration validation."""

    def test_invalid_mtq_axis_contains_nan(self):
        """MTQ with NaN axis should be rejected by API."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        
        axis = np.array([np.nan, 0.0, 0.0])
        with pytest.raises(ValueError):
            sat.addMTQ(axis, 0.2)

    def test_invalid_mtq_max_dipole_is_zero(self):
        """MTQ with zero dipole should be caught by validation."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        
        axis = np.array([1.0, 0.0, 0.0])
        sat.addMTQ(axis, 0.0)
        
        is_valid, error_msg = validateSatellite(sat)
        assert not is_valid
        assert error_msg == "MTQ 0 max dipole invalid"

    def test_invalid_mtq_max_dipole_is_nan(self):
        """MTQ with NaN dipole should be rejected by API."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        
        axis = np.array([1.0, 0.0, 0.0])
        with pytest.raises(ValueError):
            sat.addMTQ(axis, np.nan)

    def test_invalid_mtq_max_dipole_unreasonably_large(self):
        """MTQ with very large dipole should be caught by validation."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        
        axis = np.array([1.0, 0.0, 0.0])
        sat.addMTQ(axis, 2e6)  # > 1e6
        
        is_valid, error_msg = validateSatellite(sat)
        assert not is_valid
        assert error_msg == "MTQ 0 max dipole unreasonably large"

    def test_invalid_second_mtq_has_invalid_dipole(self):
        """Second MTQ with invalid dipole should be caught by validation."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        
        axis = np.array([1.0, 0.0, 0.0])
        sat.addMTQ(axis, 0.2)
        sat.addMTQ(axis, 0.0)  # Zero dipole
        
        is_valid, error_msg = validateSatellite(sat)
        assert not is_valid
        assert error_msg == "MTQ 1 max dipole invalid"


# ============================================================================
# RW Configuration Validation Tests
# ============================================================================

class TestRWValidation:
    """Tests for RW (Reaction Wheel) configuration validation."""

    def test_invalid_rw_axis_contains_nan(self):
        """RW with NaN axis should be rejected by API."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        
        axis = np.array([np.nan, 0.0, 0.0])
        with pytest.raises(ValueError):
            sat.addRW(axis, 0.01, 0.001, 0.0, 0.1)

    def test_invalid_rw_max_torque_is_zero(self):
        """RW with zero torque should be caught by validation."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        
        axis = np.array([1.0, 0.0, 0.0])
        sat.addRW(axis, 0.0, 0.001, 0.0, 0.1)
        
        is_valid, error_msg = validateSatellite(sat)
        assert not is_valid
        assert error_msg == "RW 0 max torque invalid"

    def test_invalid_rw_max_torque_is_nan(self):
        """RW with NaN torque should be rejected by API."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        
        axis = np.array([1.0, 0.0, 0.0])
        with pytest.raises(ValueError):
            sat.addRW(axis, np.nan, 0.001, 0.0, 0.1)

    def test_invalid_rw_max_torque_unreasonably_large(self):
        """RW with very large torque should be caught by validation."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        
        axis = np.array([1.0, 0.0, 0.0])
        sat.addRW(axis, 2e4, 0.001, 0.0, 0.1)  # > 1e4
        
        is_valid, error_msg = validateSatellite(sat)
        assert not is_valid
        assert error_msg == "RW 0 max torque unreasonably large"

    def test_invalid_rw_wheel_inertia_is_zero(self):
        """RW with zero wheel inertia should be rejected by API."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        
        axis = np.array([1.0, 0.0, 0.0])
        with pytest.raises(ValueError):
            sat.addRW(axis, 0.01, 0.0, 0.0, 0.1)

    def test_invalid_rw_wheel_inertia_is_negative(self):
        """RW with negative wheel inertia should be rejected by API."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        
        axis = np.array([1.0, 0.0, 0.0])
        with pytest.raises(ValueError):
            sat.addRW(axis, 0.01, -0.001, 0.0, 0.1)

    def test_invalid_rw_wheel_inertia_is_nan(self):
        """RW with NaN wheel inertia should be rejected by API."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        
        axis = np.array([1.0, 0.0, 0.0])
        with pytest.raises(ValueError):
            sat.addRW(axis, 0.01, np.nan, 0.0, 0.1)

    def test_invalid_rw_wheel_inertia_unreasonably_large(self):
        """RW with very large wheel inertia should be caught by validation."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        
        axis = np.array([1.0, 0.0, 0.0])
        sat.addRW(axis, 0.01, 2e3, 0.0, 0.1)  # > 1e3
        
        is_valid, error_msg = validateSatellite(sat)
        assert not is_valid
        assert error_msg == "RW 0 wheel inertia unreasonably large"

    def test_invalid_rw_initial_momentum_is_nan(self):
        """RW with NaN initial momentum should be caught by validation."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        
        axis = np.array([1.0, 0.0, 0.0])
        sat.addRW(axis, 0.01, 0.001, np.nan, 0.1)
        
        is_valid, error_msg = validateSatellite(sat)
        assert not is_valid
        assert error_msg == "RW 0 momentum values not finite"

    def test_invalid_rw_initial_momentum_is_infinite(self):
        """RW with infinite initial momentum should be caught by validation."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        
        axis = np.array([1.0, 0.0, 0.0])
        sat.addRW(axis, 0.01, 0.001, np.inf, 0.1)
        
        is_valid, error_msg = validateSatellite(sat)
        assert not is_valid
        assert error_msg == "RW 0 momentum values not finite"

    def test_invalid_rw_max_momentum_is_nan(self):
        """RW with NaN max momentum should be rejected by API."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        
        axis = np.array([1.0, 0.0, 0.0])
        with pytest.raises(ValueError):
            sat.addRW(axis, 0.01, 0.001, 0.0, np.nan)

    def test_invalid_rw_max_momentum_is_zero(self):
        """RW with zero max momentum should be caught by validation."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        
        axis = np.array([1.0, 0.0, 0.0])
        sat.addRW(axis, 0.01, 0.001, 0.0, 0.0)
        
        is_valid, error_msg = validateSatellite(sat)
        assert not is_valid
        assert error_msg == "RW 0 max momentum must be positive"

    def test_invalid_rw_max_momentum_is_negative(self):
        """RW with negative max momentum should be rejected by API."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        
        axis = np.array([1.0, 0.0, 0.0])
        with pytest.raises(ValueError):
            sat.addRW(axis, 0.01, 0.001, 0.0, -0.1)

    def test_invalid_rw_initial_momentum_exceeds_max_positive(self):
        """RW where initial momentum exceeds max should be caught by validation."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        
        axis = np.array([1.0, 0.0, 0.0])
        sat.addRW(axis, 0.01, 0.001, 0.15, 0.1)  # 0.15 > 0.1
        
        is_valid, error_msg = validateSatellite(sat)
        assert not is_valid
        assert error_msg == "RW 0 initial momentum exceeds max"

    def test_invalid_rw_initial_momentum_exceeds_max_negative(self):
        """RW where |initial momentum| exceeds max should be caught by validation."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        
        axis = np.array([1.0, 0.0, 0.0])
        sat.addRW(axis, 0.01, 0.001, -0.15, 0.1)  # |-0.15| > 0.1
        
        is_valid, error_msg = validateSatellite(sat)
        assert not is_valid
        assert error_msg == "RW 0 initial momentum exceeds max"

    def test_invalid_rw_max_momentum_unreasonably_large(self):
        """RW with very large max momentum should be caught by validation."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        
        axis = np.array([1.0, 0.0, 0.0])
        sat.addRW(axis, 0.01, 0.001, 0.0, 2e4)  # > 1e4
        
        is_valid, error_msg = validateSatellite(sat)
        assert not is_valid
        assert error_msg == "RW 0 max momentum unreasonably large"

    def test_valid_rw_initial_momentum_at_boundary(self):
        """RW with initial momentum equal to max should pass validation."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        
        axis = np.array([1.0, 0.0, 0.0])
        sat.addRW(axis, 0.01, 0.001, 0.1, 0.1)  # h_init == h_max
        
        is_valid, error_msg = validateSatellite(sat)
        assert is_valid

    def test_valid_rw_initial_momentum_at_negative_boundary(self):
        """RW with initial momentum at negative boundary should pass validation."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        
        axis = np.array([1.0, 0.0, 0.0])
        sat.addRW(axis, 0.01, 0.001, -0.1, 0.1)  # h_init == -h_max
        
        is_valid, error_msg = validateSatellite(sat)
        assert is_valid

    def test_invalid_second_rw_has_invalid_torque(self):
        """Second RW with invalid torque should be caught by validation."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        
        axis1 = np.array([1.0, 0.0, 0.0])
        axis2 = np.array([0.0, 1.0, 0.0])
        sat.addRW(axis1, 0.01, 0.001, 0.0, 0.1)
        sat.addRW(axis2, 0.0, 0.001, 0.0, 0.1)  # Zero torque
        
        is_valid, error_msg = validateSatellite(sat)
        assert not is_valid
        assert error_msg == "RW 1 max torque invalid"


# ============================================================================
# Geometry Configuration Validation Tests
# ============================================================================

class TestGeometryValidation:
    """Tests for geometry configuration validation."""

    def test_valid_satellite_with_multiple_geometry_faces(self):
        """Satellite with multiple geometry faces should pass validation."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        geom = sat.geometryConfig
        
        # Add 6 faces for a cube
        for i in range(6):
            normal = np.zeros(3)
            normal[i % 3] = 1.0 if i < 3 else -1.0
            centroid = normal * 0.05
            
            face = GeometryFace(0.01, centroid, normal, 0.1, 0.2, 0.7, 2.2)
            geom.addFace(face)
        
        is_valid, error_msg = validateSatellite(sat)
        assert is_valid

    def test_invalid_geometry_face_area_is_negative(self):
        """Geometry face with negative area should be caught by validation."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        geom = sat.geometryConfig
        
        centroid = np.array([0.0, 0.0, 0.1])
        normal = np.array([0.0, 0.0, 1.0])
        face = GeometryFace(-0.01, centroid, normal, 0.1, 0.2, 0.7, 2.2)
        geom.addFace(face)
        
        is_valid, error_msg = validateSatellite(sat)
        assert not is_valid
        assert error_msg == "geometry face 0 area invalid"

    def test_invalid_geometry_face_area_is_nan(self):
        """Geometry face with NaN area should be caught by validation."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        geom = sat.geometryConfig
        
        centroid = np.array([0.0, 0.0, 0.1])
        normal = np.array([0.0, 0.0, 1.0])
        face = GeometryFace(np.nan, centroid, normal, 0.1, 0.2, 0.7, 2.2)
        geom.addFace(face)
        
        is_valid, error_msg = validateSatellite(sat)
        assert not is_valid
        assert error_msg == "geometry face 0 area invalid"

    def test_invalid_geometry_centroid_contains_nan(self):
        """Geometry face with NaN centroid should be caught by validation."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        geom = sat.geometryConfig
        
        centroid = np.array([np.nan, 0.0, 0.1])
        normal = np.array([0.0, 0.0, 1.0])
        face = GeometryFace(0.01, centroid, normal, 0.1, 0.2, 0.7, 2.2)
        geom.addFace(face)
        
        is_valid, error_msg = validateSatellite(sat)
        assert not is_valid
        assert error_msg == "geometry face 0 centroid not finite"

    def test_invalid_geometry_centroid_contains_infinity(self):
        """Geometry face with infinite centroid should be caught by validation."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        geom = sat.geometryConfig
        
        centroid = np.array([0.0, np.inf, 0.1])
        normal = np.array([0.0, 0.0, 1.0])
        face = GeometryFace(0.01, centroid, normal, 0.1, 0.2, 0.7, 2.2)
        geom.addFace(face)
        
        is_valid, error_msg = validateSatellite(sat)
        assert not is_valid
        assert error_msg == "geometry face 0 centroid not finite"

    def test_invalid_geometry_normal_not_normalized_too_short(self):
        """Geometry face with short normal should be caught by validation."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        geom = sat.geometryConfig
        
        centroid = np.array([0.0, 0.0, 0.1])
        normal = np.array([0.0, 0.0, 0.5])  # Norm = 0.5 < 0.99
        face = GeometryFace(0.01, centroid, normal, 0.1, 0.2, 0.7, 2.2)
        geom.addFace(face)
        
        is_valid, error_msg = validateSatellite(sat)
        assert not is_valid
        assert error_msg == "geometry face 0 normal not normalized"

    def test_invalid_geometry_normal_not_normalized_too_long(self):
        """Geometry face with long normal should be caught by validation."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        geom = sat.geometryConfig
        
        centroid = np.array([0.0, 0.0, 0.1])
        normal = np.array([0.0, 0.0, 1.5])  # Norm = 1.5 > 1.01
        face = GeometryFace(0.01, centroid, normal, 0.1, 0.2, 0.7, 2.2)
        geom.addFace(face)
        
        is_valid, error_msg = validateSatellite(sat)
        assert not is_valid
        assert error_msg == "geometry face 0 normal not normalized"

    def test_invalid_geometry_normal_contains_nan(self):
        """Geometry face with NaN normal should be caught by validation."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        geom = sat.geometryConfig
        
        centroid = np.array([0.0, 0.0, 0.1])
        normal = np.array([0.0, np.nan, 1.0])
        face = GeometryFace(0.01, centroid, normal, 0.1, 0.2, 0.7, 2.2)
        geom.addFace(face)
        
        is_valid, error_msg = validateSatellite(sat)
        assert not is_valid
        assert error_msg == "geometry face 0 normal not normalized"

    def test_invalid_geometry_specular_coefficient_negative(self):
        """Geometry face with negative specular coefficient should be caught."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        geom = sat.geometryConfig
        
        centroid = np.array([0.0, 0.0, 0.1])
        normal = np.array([0.0, 0.0, 1.0])
        face = GeometryFace(0.01, centroid, normal, -0.1, 0.2, 0.7, 2.2)
        geom.addFace(face)
        
        is_valid, error_msg = validateSatellite(sat)
        assert not is_valid
        assert error_msg == "geometry face 0 specular coefficient invalid"

    def test_invalid_geometry_specular_coefficient_too_large(self):
        """Geometry face with too large specular coefficient should be caught."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        geom = sat.geometryConfig
        
        centroid = np.array([0.0, 0.0, 0.1])
        normal = np.array([0.0, 0.0, 1.0])
        face = GeometryFace(0.01, centroid, normal, 1.5, 0.2, 0.7, 2.2)
        geom.addFace(face)
        
        is_valid, error_msg = validateSatellite(sat)
        assert not is_valid
        assert error_msg == "geometry face 0 specular coefficient invalid"

    def test_invalid_geometry_specular_coefficient_is_nan(self):
        """Geometry face with NaN specular coefficient should be caught."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        geom = sat.geometryConfig
        
        centroid = np.array([0.0, 0.0, 0.1])
        normal = np.array([0.0, 0.0, 1.0])
        face = GeometryFace(0.01, centroid, normal, np.nan, 0.2, 0.7, 2.2)
        geom.addFace(face)
        
        is_valid, error_msg = validateSatellite(sat)
        assert not is_valid
        assert error_msg == "geometry face 0 specular coefficient invalid"

    def test_invalid_geometry_diffuse_coefficient_negative(self):
        """Geometry face with negative diffuse coefficient should be caught."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        geom = sat.geometryConfig
        
        centroid = np.array([0.0, 0.0, 0.1])
        normal = np.array([0.0, 0.0, 1.0])
        face = GeometryFace(0.01, centroid, normal, 0.1, -0.2, 0.7, 2.2)
        geom.addFace(face)
        
        is_valid, error_msg = validateSatellite(sat)
        assert not is_valid
        assert error_msg == "geometry face 0 diffuse coefficient invalid"

    def test_invalid_geometry_diffuse_coefficient_too_large(self):
        """Geometry face with too large diffuse coefficient should be caught."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        geom = sat.geometryConfig
        
        centroid = np.array([0.0, 0.0, 0.1])
        normal = np.array([0.0, 0.0, 1.0])
        face = GeometryFace(0.01, centroid, normal, 0.1, 1.2, 0.7, 2.2)
        geom.addFace(face)
        
        is_valid, error_msg = validateSatellite(sat)
        assert not is_valid
        assert error_msg == "geometry face 0 diffuse coefficient invalid"

    def test_invalid_geometry_diffuse_coefficient_is_nan(self):
        """Geometry face with NaN diffuse coefficient should be caught."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        geom = sat.geometryConfig
        
        centroid = np.array([0.0, 0.0, 0.1])
        normal = np.array([0.0, 0.0, 1.0])
        face = GeometryFace(0.01, centroid, normal, 0.1, np.nan, 0.7, 2.2)
        geom.addFace(face)
        
        is_valid, error_msg = validateSatellite(sat)
        assert not is_valid
        assert error_msg == "geometry face 0 diffuse coefficient invalid"

    def test_invalid_geometry_absorptivity_negative(self):
        """Geometry face with negative absorptivity should be caught."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        geom = sat.geometryConfig
        
        centroid = np.array([0.0, 0.0, 0.1])
        normal = np.array([0.0, 0.0, 1.0])
        face = GeometryFace(0.01, centroid, normal, 0.1, 0.2, -0.7, 2.2)
        geom.addFace(face)
        
        is_valid, error_msg = validateSatellite(sat)
        assert not is_valid
        assert error_msg == "geometry face 0 absorptivity invalid"

    def test_invalid_geometry_absorptivity_too_large(self):
        """Geometry face with too large absorptivity should be caught."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        geom = sat.geometryConfig
        
        centroid = np.array([0.0, 0.0, 0.1])
        normal = np.array([0.0, 0.0, 1.0])
        face = GeometryFace(0.01, centroid, normal, 0.1, 0.2, 1.7, 2.2)
        geom.addFace(face)
        
        is_valid, error_msg = validateSatellite(sat)
        assert not is_valid
        assert error_msg == "geometry face 0 absorptivity invalid"

    def test_invalid_geometry_absorptivity_is_nan(self):
        """Geometry face with NaN absorptivity should be caught."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        geom = sat.geometryConfig
        
        centroid = np.array([0.0, 0.0, 0.1])
        normal = np.array([0.0, 0.0, 1.0])
        face = GeometryFace(0.01, centroid, normal, 0.1, 0.2, np.nan, 2.2)
        geom.addFace(face)
        
        is_valid, error_msg = validateSatellite(sat)
        assert not is_valid
        assert error_msg == "geometry face 0 absorptivity invalid"

    def test_invalid_geometry_optical_coefficients_sum_too_small(self):
        """Geometry face with small optical coefficient sum should be caught."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        geom = sat.geometryConfig
        
        centroid = np.array([0.0, 0.0, 0.1])
        normal = np.array([0.0, 0.0, 1.0])
        # Sum = 0.1 + 0.2 + 0.5 = 0.8 < 0.99
        face = GeometryFace(0.01, centroid, normal, 0.1, 0.2, 0.5, 2.2)
        geom.addFace(face)
        
        is_valid, error_msg = validateSatellite(sat)
        assert not is_valid
        assert error_msg == "geometry face 0 optical coefficients don't sum to 1"

    def test_invalid_geometry_optical_coefficients_sum_too_large(self):
        """Geometry face with large optical coefficient sum should be caught."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        geom = sat.geometryConfig
        
        centroid = np.array([0.0, 0.0, 0.1])
        normal = np.array([0.0, 0.0, 1.0])
        # Sum = 0.4 + 0.4 + 0.3 = 1.1 > 1.01
        face = GeometryFace(0.01, centroid, normal, 0.4, 0.4, 0.3, 2.2)
        geom.addFace(face)
        
        is_valid, error_msg = validateSatellite(sat)
        assert not is_valid
        assert error_msg == "geometry face 0 optical coefficients don't sum to 1"

    def test_valid_geometry_optical_coefficients_sum_to_one(self):
        """Geometry face with coefficients summing to 1.0 should pass."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        geom = sat.geometryConfig
        
        centroid = np.array([0.0, 0.0, 0.1])
        normal = np.array([0.0, 0.0, 1.0])
        # Sum = 0.3 + 0.3 + 0.4 = 1.0
        face = GeometryFace(0.01, centroid, normal, 0.3, 0.3, 0.4, 2.2)
        geom.addFace(face)
        
        is_valid, error_msg = validateSatellite(sat)
        assert is_valid

    def test_valid_geometry_optical_coefficients_sum_within_tolerance(self):
        """Geometry face with coefficients within tolerance should pass."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        geom = sat.geometryConfig
        
        centroid = np.array([0.0, 0.0, 0.1])
        normal = np.array([0.0, 0.0, 1.0])
        # Sum = 0.33 + 0.33 + 0.335 = 0.995 (within tolerance)
        face = GeometryFace(0.01, centroid, normal, 0.33, 0.33, 0.335, 2.2)
        geom.addFace(face)
        
        is_valid, error_msg = validateSatellite(sat)
        assert is_valid

    def test_invalid_geometry_drag_coefficient_negative(self):
        """Geometry face with negative drag coefficient should be caught."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        geom = sat.geometryConfig
        
        centroid = np.array([0.0, 0.0, 0.1])
        normal = np.array([0.0, 0.0, 1.0])
        face = GeometryFace(0.01, centroid, normal, 0.1, 0.2, 0.7, -2.2)
        geom.addFace(face)
        
        is_valid, error_msg = validateSatellite(sat)
        assert not is_valid
        assert error_msg == "geometry face 0 drag coefficient invalid"

    def test_invalid_geometry_drag_coefficient_is_nan(self):
        """Geometry face with NaN drag coefficient should be caught."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        geom = sat.geometryConfig
        
        centroid = np.array([0.0, 0.0, 0.1])
        normal = np.array([0.0, 0.0, 1.0])
        face = GeometryFace(0.01, centroid, normal, 0.1, 0.2, 0.7, np.nan)
        geom.addFace(face)
        
        is_valid, error_msg = validateSatellite(sat)
        assert not is_valid
        assert error_msg == "geometry face 0 drag coefficient invalid"

    def test_invalid_geometry_drag_coefficient_unreasonably_large(self):
        """Geometry face with very large drag coefficient should be caught."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        geom = sat.geometryConfig
        
        centroid = np.array([0.0, 0.0, 0.1])
        normal = np.array([0.0, 0.0, 1.0])
        face = GeometryFace(0.01, centroid, normal, 0.1, 0.2, 0.7, 3.5)  # > 3.0
        geom.addFace(face)
        
        is_valid, error_msg = validateSatellite(sat)
        assert not is_valid
        assert error_msg == "geometry face 0 drag coefficient unreasonably large"

    def test_valid_geometry_zero_drag_coefficient(self):
        """Geometry face with zero drag coefficient should pass."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        geom = sat.geometryConfig
        
        centroid = np.array([0.0, 0.0, 0.1])
        normal = np.array([0.0, 0.0, 1.0])
        face = GeometryFace(0.01, centroid, normal, 0.1, 0.2, 0.7, 0.0)
        geom.addFace(face)
        
        is_valid, error_msg = validateSatellite(sat)
        assert is_valid

    def test_invalid_geometry_second_face_has_invalid_properties(self):
        """Second geometry face with invalid properties should be caught."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        geom = sat.geometryConfig
        
        centroid1 = np.array([0.0, 0.0, 0.1])
        normal1 = np.array([0.0, 0.0, 1.0])
        face1 = GeometryFace(0.01, centroid1, normal1, 0.1, 0.2, 0.7, 2.2)
        geom.addFace(face1)
        
        centroid2 = np.array([0.0, 0.0, -0.1])
        normal2 = np.array([0.0, 0.0, -1.0])
        face2 = GeometryFace(-0.01, centroid2, normal2, 0.1, 0.2, 0.7, 2.2)  # Negative area
        geom.addFace(face2)
        
        is_valid, error_msg = validateSatellite(sat)
        assert not is_valid
        assert error_msg == "geometry face 1 area invalid"


# ============================================================================
# Comprehensive and Boundary Tests
# ============================================================================

class TestComprehensiveAndBoundaryValues:
    """Tests for comprehensive configurations and boundary values."""

    def test_valid_satellite_with_all_actuators_and_geometry(self):
        """Satellite with all actuators and geometry should pass validation."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        
        # Add MTQs
        sat.addMTQ(np.array([1.0, 0.0, 0.0]), 0.2)
        sat.addMTQ(np.array([0.0, 1.0, 0.0]), 0.2)
        sat.addMTQ(np.array([0.0, 0.0, 1.0]), 0.2)
        
        # Add RWs
        sat.addRW(np.array([1.0, 0.0, 0.0]), 0.01, 0.001, 0.05, 0.1)
        sat.addRW(np.array([0.0, 1.0, 0.0]), 0.01, 0.001, -0.03, 0.1)
        
        # Add geometry faces
        geom = sat.geometryConfig
        for i in range(6):
            normal = np.zeros(3)
            normal[i % 3] = 1.0 if i < 3 else -1.0
            centroid = normal * 0.05
            face = GeometryFace(0.01, centroid, normal, 0.1, 0.2, 0.7, 2.2)
            geom.addFace(face)
        
        is_valid, error_msg = validateSatellite(sat)
        assert is_valid

    def test_boundary_minimum_valid_inertia_magnitude(self):
        """Satellite with small valid inertia should pass validation."""
        J = np.array([
            [1e-3, 0.0, 0.0],
            [0.0, 1e-3, 0.0],
            [0.0, 0.0, 1e-3]
        ])
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        
        is_valid, error_msg = validateSatellite(sat)
        assert is_valid

    def test_boundary_maximum_valid_inertia_magnitude(self):
        """Satellite with large valid inertia should pass validation."""
        J = np.array([
            [1e5, 0.0, 0.0],
            [0.0, 1e5, 0.0],
            [0.0, 0.0, 1e5]
        ])
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        
        is_valid, error_msg = validateSatellite(sat)
        assert is_valid

    def test_boundary_mtq_max_dipole_at_upper_limit(self):
        """MTQ with max dipole at limit should pass validation."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        
        axis = np.array([1.0, 0.0, 0.0])
        sat.addMTQ(axis, 1e6)
        
        is_valid, error_msg = validateSatellite(sat)
        assert is_valid

    def test_boundary_rw_max_torque_at_upper_limit(self):
        """RW with max torque at limit should pass validation."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        
        axis = np.array([1.0, 0.0, 0.0])
        sat.addRW(axis, 1e4, 0.001, 0.0, 0.1)
        
        is_valid, error_msg = validateSatellite(sat)
        assert is_valid

    def test_boundary_rw_wheel_inertia_at_upper_limit(self):
        """RW with wheel inertia at limit should pass validation."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        
        axis = np.array([1.0, 0.0, 0.0])
        sat.addRW(axis, 0.01, 1e3, 0.0, 0.1)
        
        is_valid, error_msg = validateSatellite(sat)
        assert is_valid

    def test_boundary_rw_max_momentum_at_upper_limit(self):
        """RW with max momentum at limit should pass validation."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        
        axis = np.array([1.0, 0.0, 0.0])
        sat.addRW(axis, 0.01, 0.001, 0.0, 1e4)
        
        is_valid, error_msg = validateSatellite(sat)
        assert is_valid

    def test_boundary_drag_coefficient_at_upper_limit(self):
        """Geometry face with drag coefficient at limit should pass validation."""
        J = valid_inertia_matrix()
        settings = PlannerSettings()
        sat = Satellite(J, settings)
        geom = sat.geometryConfig
        
        centroid = np.array([0.0, 0.0, 0.1])
        normal = np.array([0.0, 0.0, 1.0])
        face = GeometryFace(0.01, centroid, normal, 0.1, 0.2, 0.7, 3.0)
        geom.addFace(face)
        
        is_valid, error_msg = validateSatellite(sat)
        assert is_valid
