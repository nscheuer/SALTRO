"""
Comprehensive pytest test suite for initial state validation.

This module tests the validateInitialState() function which validates
the initial state vector for trajectory optimization:
- State vector minimum size (7 elements)
- Angular velocity validation (finite, magnitude < 10 rad/s)
- Quaternion validation (finite, normalized: |norm - 1.0| <= 1e-6)
- Reaction wheel momenta validation (finite, if present)

NOTE ON TEST DESIGN:
The validateInitialState function checks:
  1. State vector has minimum size 7 (3 angular velocity + 4 quaternion)
  2. Angular velocity components are finite and magnitude < 10 rad/s
  3. Quaternion components are finite and normalized (within 1e-6 tolerance)
  4. Reaction wheel momenta (if present) are finite

The function requires the quaternion to be normalized.
"""

import pytest
import numpy as np
import math

# These imports assume Python bindings are available
try:
    import saltro_py
    validateInitialState = lambda x0: saltro_py.validateInitialState(x0)
except ImportError as e:
    pytest.skip(f"saltro Python bindings not available: {e}", allow_module_level=True)


# ============================================================================
# Helper Functions
# ============================================================================

def valid_initial_state():
    """Create a valid initial state vector (7 elements)."""
    return np.array([
        0.1, 0.05, -0.02,  # angular velocity
        1.0, 0.0, 0.0, 0.0  # quaternion (normalized)
    ])


def valid_initial_state_with_rw():
    """Create a valid initial state vector with RW momenta (9 elements)."""
    return np.array([
        0.1, 0.05, -0.02,  # angular velocity
        1.0, 0.0, 0.0, 0.0,  # quaternion
        0.05, -0.03  # RW momenta
    ])


# ============================================================================
# Basic Validation Tests
# ============================================================================

def test_valid_initial_state_passes_validation():
    """Valid initial state should pass validation."""
    x0 = valid_initial_state()
    
    is_valid, error_msg = validateInitialState(x0)
    
    assert is_valid
    assert error_msg == "" or error_msg is None


def test_valid_initial_state_with_rw_passes_validation():
    """Valid initial state with RW momenta should pass validation."""
    x0 = valid_initial_state_with_rw()
    
    is_valid, error_msg = validateInitialState(x0)
    
    assert is_valid
    assert error_msg == "" or error_msg is None


def test_valid_state_with_zero_angular_velocity():
    """State with zero angular velocity should be valid."""
    x0 = np.array([
        0.0, 0.0, 0.0,  # zero angular velocity (valid)
        1.0, 0.0, 0.0, 0.0  # quaternion
    ])
    
    is_valid, error_msg = validateInitialState(x0)
    
    assert is_valid


def test_valid_state_with_normalized_quaternion():
    """State with normalized quaternion should pass validation."""
    q_unnorm = np.array([2.0, 1.0, 0.5, -0.3])
    q_norm = q_unnorm / np.linalg.norm(q_unnorm)
    x0 = np.array([
        0.1, 0.05, -0.02,  # angular velocity
        q_norm[0], q_norm[1], q_norm[2], q_norm[3]  # normalized quaternion
    ])
    
    is_valid, error_msg = validateInitialState(x0)
    
    assert is_valid


def test_invalid_state_with_unnormalized_quaternion():
    """State with unnormalized quaternion should fail validation."""
    x0 = np.array([
        0.1, 0.05, -0.02,  # angular velocity
        2.0, 1.0, 0.5, -0.3  # unnormalized quaternion (should fail)
    ])
    
    is_valid, error_msg = validateInitialState(x0)
    
    assert not is_valid
    assert error_msg == "quaternion not normalized"


# ============================================================================
# State Dimension Validation Tests
# ============================================================================

def test_state_too_small_fails_validation():
    """State vector smaller than 7 elements should fail."""
    x0 = np.array([
        0.1, 0.05, -0.02,  # angular velocity
        1.0, 0.0, 0.0  # incomplete quaternion (only 3 components)
    ])
    
    is_valid, error_msg = validateInitialState(x0)
    
    assert not is_valid
    assert error_msg == "initial state x0 is too small"


def test_empty_state_fails_validation():
    """Empty state vector should fail."""
    x0 = np.array([])
    
    is_valid, error_msg = validateInitialState(x0)
    
    assert not is_valid
    assert error_msg == "initial state x0 is too small"


# ============================================================================
# Angular Velocity Validation Tests
# ============================================================================

def test_angular_velocity_with_nan_in_first_component():
    """Angular velocity with NaN in first component should fail."""
    x0 = valid_initial_state()
    x0[0] = np.nan
    
    is_valid, error_msg = validateInitialState(x0)
    
    assert not is_valid
    assert error_msg == "angular velocity component 0 is not finite"


def test_angular_velocity_with_nan_in_second_component():
    """Angular velocity with NaN in second component should fail."""
    x0 = valid_initial_state()
    x0[1] = np.nan
    
    is_valid, error_msg = validateInitialState(x0)
    
    assert not is_valid
    assert error_msg == "angular velocity component 1 is not finite"


def test_angular_velocity_with_nan_in_third_component():
    """Angular velocity with NaN in third component should fail."""
    x0 = valid_initial_state()
    x0[2] = np.nan
    
    is_valid, error_msg = validateInitialState(x0)
    
    assert not is_valid
    assert error_msg == "angular velocity component 2 is not finite"


def test_angular_velocity_with_positive_infinity():
    """Angular velocity with positive infinity should fail."""
    x0 = valid_initial_state()
    x0[0] = np.inf
    
    is_valid, error_msg = validateInitialState(x0)
    
    assert not is_valid
    assert error_msg == "angular velocity component 0 is not finite"


def test_angular_velocity_with_negative_infinity():
    """Angular velocity with negative infinity should fail."""
    x0 = valid_initial_state()
    x0[1] = -np.inf
    
    is_valid, error_msg = validateInitialState(x0)
    
    assert not is_valid
    assert error_msg == "angular velocity component 1 is not finite"


def test_angular_velocity_magnitude_unreasonably_large():
    """Angular velocity with magnitude > 10 rad/s should fail."""
    x0 = np.array([
        8.0, 7.0, 6.0,  # magnitude = sqrt(149) > 10
        1.0, 0.0, 0.0, 0.0
    ])
    
    is_valid, error_msg = validateInitialState(x0)
    
    assert not is_valid
    assert error_msg == "angular velocity magnitude unreasonably large"


def test_angular_velocity_magnitude_exactly_at_limit():
    """Angular velocity magnitude exactly 10.0 should fail."""
    x0 = np.array([
        10.0, 0.0, 0.0,  # magnitude = 10.0 (should fail: > not >=)
        1.0, 0.0, 0.0, 0.0
    ])
    
    is_valid, error_msg = validateInitialState(x0)
    
    assert not is_valid
    assert error_msg == "angular velocity magnitude unreasonably large"


def test_angular_velocity_magnitude_just_below_limit():
    """Angular velocity magnitude just below 10.0 should pass."""
    x0 = np.array([
        9.99, 0.0, 0.0,  # magnitude = 9.99 < 10.0
        1.0, 0.0, 0.0, 0.0
    ])
    
    is_valid, error_msg = validateInitialState(x0)
    
    assert is_valid


def test_large_but_reasonable_angular_velocity():
    """Large but reasonable angular velocity should pass."""
    x0 = np.array([
        5.0, 5.0, 5.0,  # magnitude = sqrt(75) ≈ 8.66 < 10
        1.0, 0.0, 0.0, 0.0
    ])
    
    is_valid, error_msg = validateInitialState(x0)
    
    assert is_valid


# ============================================================================
# Quaternion Validation Tests
# ============================================================================

def test_quaternion_with_nan_in_first_component():
    """Quaternion with NaN in first component should fail."""
    x0 = valid_initial_state()
    x0[3] = np.nan
    
    is_valid, error_msg = validateInitialState(x0)
    
    assert not is_valid
    assert error_msg == "quaternion component 0 is not finite"


def test_quaternion_with_nan_in_second_component():
    """Quaternion with NaN in second component should fail."""
    x0 = valid_initial_state()
    x0[4] = np.nan
    
    is_valid, error_msg = validateInitialState(x0)
    
    assert not is_valid
    assert error_msg == "quaternion component 1 is not finite"


def test_quaternion_with_nan_in_third_component():
    """Quaternion with NaN in third component should fail."""
    x0 = valid_initial_state()
    x0[5] = np.nan
    
    is_valid, error_msg = validateInitialState(x0)
    
    assert not is_valid
    assert error_msg == "quaternion component 2 is not finite"


def test_quaternion_with_nan_in_fourth_component():
    """Quaternion with NaN in fourth component should fail."""
    x0 = valid_initial_state()
    x0[6] = np.nan
    
    is_valid, error_msg = validateInitialState(x0)
    
    assert not is_valid
    assert error_msg == "quaternion component 3 is not finite"


def test_quaternion_with_positive_infinity():
    """Quaternion with positive infinity should fail."""
    x0 = valid_initial_state()
    x0[3] = np.inf
    
    is_valid, error_msg = validateInitialState(x0)
    
    assert not is_valid
    assert error_msg == "quaternion component 0 is not finite"


def test_quaternion_with_negative_infinity():
    """Quaternion with negative infinity should fail."""
    x0 = valid_initial_state()
    x0[5] = -np.inf
    
    is_valid, error_msg = validateInitialState(x0)
    
    assert not is_valid
    assert error_msg == "quaternion component 2 is not finite"


def test_quaternion_all_zeros():
    """Quaternion with all zeros should fail."""
    x0 = np.array([
        0.1, 0.05, -0.02,  # angular velocity
        0.0, 0.0, 0.0, 0.0  # all zero quaternion
    ])
    
    is_valid, error_msg = validateInitialState(x0)
    
    assert not is_valid
    assert error_msg == "quaternion not normalized"


def test_quaternion_normalized_within_positive_tolerance():
    """Quaternion normalized within positive tolerance should pass."""
    x0 = np.array([
        0.1, 0.05, -0.02,  # angular velocity
        1.0 + 9e-7, 0.0, 0.0, 0.0  # norm = 1.0 + 9e-7 (within 1e-6 tolerance)
    ])
    
    is_valid, error_msg = validateInitialState(x0)
    
    assert is_valid


def test_quaternion_normalized_within_negative_tolerance():
    """Quaternion normalized within negative tolerance should pass."""
    x0 = np.array([
        0.1, 0.05, -0.02,  # angular velocity
        1.0 - 9e-7, 0.0, 0.0, 0.0  # norm = 1.0 - 9e-7 (within 1e-6 tolerance)
    ])
    
    is_valid, error_msg = validateInitialState(x0)
    
    assert is_valid


def test_quaternion_norm_slightly_above_tolerance():
    """Quaternion norm slightly above tolerance should fail."""
    x0 = np.array([
        0.1, 0.05, -0.02,  # angular velocity
        1.0 + 2e-6, 0.0, 0.0, 0.0  # norm = 1.0 + 2e-6 (outside 1e-6 tolerance)
    ])
    
    is_valid, error_msg = validateInitialState(x0)
    
    assert not is_valid
    assert error_msg == "quaternion not normalized"


def test_quaternion_norm_slightly_below_tolerance():
    """Quaternion norm slightly below tolerance should fail."""
    x0 = np.array([
        0.1, 0.05, -0.02,  # angular velocity
        1.0 - 2e-6, 0.0, 0.0, 0.0  # norm = 1.0 - 2e-6 (outside 1e-6 tolerance)
    ])
    
    is_valid, error_msg = validateInitialState(x0)
    
    assert not is_valid
    assert error_msg == "quaternion not normalized"


def test_quaternion_with_very_small_norm():
    """Quaternion with very small norm should fail."""
    x0 = np.array([
        0.1, 0.05, -0.02,  # angular velocity
        1e-100, 0.0, 0.0, 0.0  # very small quaternion (should fail)
    ])
    
    is_valid, error_msg = validateInitialState(x0)
    
    assert not is_valid
    assert error_msg == "quaternion not normalized"


def test_quaternion_with_large_norm():
    """Quaternion with large norm should fail."""
    x0 = np.array([
        0.1, 0.05, -0.02,  # angular velocity
        100.0, 50.0, -30.0, 20.0  # large unnormalized quaternion (should fail)
    ])
    
    is_valid, error_msg = validateInitialState(x0)
    
    assert not is_valid
    assert error_msg == "quaternion not normalized"


# ============================================================================
# Reaction Wheel Momentum Validation Tests
# ============================================================================

def test_rw_momentum_with_nan_in_first_wheel():
    """RW momentum with NaN in first wheel should fail."""
    x0 = valid_initial_state_with_rw()
    x0[7] = np.nan
    
    is_valid, error_msg = validateInitialState(x0)
    
    assert not is_valid
    assert error_msg == "reaction wheel momentum 0 is not finite"


def test_rw_momentum_with_nan_in_second_wheel():
    """RW momentum with NaN in second wheel should fail."""
    x0 = valid_initial_state_with_rw()
    x0[8] = np.nan
    
    is_valid, error_msg = validateInitialState(x0)
    
    assert not is_valid
    assert error_msg == "reaction wheel momentum 1 is not finite"


def test_rw_momentum_with_positive_infinity():
    """RW momentum with positive infinity should fail."""
    x0 = valid_initial_state_with_rw()
    x0[7] = np.inf
    
    is_valid, error_msg = validateInitialState(x0)
    
    assert not is_valid
    assert error_msg == "reaction wheel momentum 0 is not finite"


def test_rw_momentum_with_negative_infinity():
    """RW momentum with negative infinity should fail."""
    x0 = valid_initial_state_with_rw()
    x0[8] = -np.inf
    
    is_valid, error_msg = validateInitialState(x0)
    
    assert not is_valid
    assert error_msg == "reaction wheel momentum 1 is not finite"


def test_valid_state_with_multiple_rws():
    """State with multiple RW momenta should pass validation."""
    x0 = np.array([
        0.1, 0.05, -0.02,  # angular velocity
        1.0, 0.0, 0.0, 0.0,  # quaternion
        0.05, -0.03, 0.02, -0.01  # 4 RW momenta
    ])
    
    is_valid, error_msg = validateInitialState(x0)
    
    assert is_valid


def test_valid_state_with_zero_rw_momentum():
    """State with zero RW momentum should be valid."""
    x0 = np.array([
        0.1, 0.05, -0.02,  # angular velocity
        1.0, 0.0, 0.0, 0.0,  # quaternion
        0.0, 0.0  # zero RW momenta (valid)
    ])
    
    is_valid, error_msg = validateInitialState(x0)
    
    assert is_valid


def test_valid_state_with_large_rw_momentum():
    """State with large RW momentum should be valid (no upper limit)."""
    x0 = np.array([
        0.1, 0.05, -0.02,  # angular velocity
        1.0, 0.0, 0.0, 0.0,  # quaternion
        100.0, -50.0  # large RW momenta (no upper limit in validation)
    ])
    
    is_valid, error_msg = validateInitialState(x0)
    
    assert is_valid


# ============================================================================
# Comprehensive Edge Cases
# ============================================================================

def test_all_components_at_extreme_valid_values():
    """All components at extreme but valid values should pass."""
    q_unnorm = np.array([1e6, -1e6, 1e6, -1e6])
    q_norm = q_unnorm / np.linalg.norm(q_unnorm)
    x0 = np.array([
        9.9, 0.0, 0.0,  # angular velocity near limit
        q_norm[0], q_norm[1], q_norm[2], q_norm[3],  # normalized quaternion
        1000.0, -1000.0, 500.0, -500.0  # large RW momenta
    ])
    
    is_valid, error_msg = validateInitialState(x0)
    
    assert is_valid


def test_minimum_valid_state_size():
    """State with exactly 7 elements should be valid."""
    x0 = np.array([
        0.0, 0.0, 0.0,  # zero angular velocity
        1.0, 0.0, 0.0, 0.0  # identity quaternion
    ])
    
    is_valid, error_msg = validateInitialState(x0)
    
    assert is_valid


def test_very_large_state_vector():
    """Very large state vector should be valid if all components are valid."""
    x0 = np.zeros(100)  # Unnecessarily large
    x0[0:3] = [0.1, 0.05, -0.02]  # angular velocity
    x0[3:7] = [1.0, 0.0, 0.0, 0.0]  # quaternion
    # All RW momenta are zero (valid)
    
    is_valid, error_msg = validateInitialState(x0)
    
    assert is_valid


def test_mixed_valid_and_invalid_nan_in_rw():
    """Valid angular velocity and quaternion but NaN in RW should fail."""
    x0 = np.array([
        0.1, 0.05, -0.02,  # valid angular velocity
        1.0, 0.0, 0.0, 0.0,  # valid quaternion
        0.05, np.nan  # NaN in RW
    ])
    
    is_valid, error_msg = validateInitialState(x0)
    
    assert not is_valid
    assert error_msg == "reaction wheel momentum 1 is not finite"


def test_negative_angular_velocities_are_valid():
    """Negative angular velocities should be valid."""
    x0 = np.array([
        -5.0, -3.0, -2.0,  # negative angular velocities (valid)
        1.0, 0.0, 0.0, 0.0
    ])
    
    is_valid, error_msg = validateInitialState(x0)
    
    assert is_valid


def test_negative_quaternion_components_are_valid():
    """Negative quaternion components should be valid."""
    q_unnorm = np.array([-1.0, -0.5, -0.3, -0.2])
    q_norm = q_unnorm / np.linalg.norm(q_unnorm)
    x0 = np.array([
        0.1, 0.05, -0.02,
        q_norm[0], q_norm[1], q_norm[2], q_norm[3]  # negative quaternion normalized (valid)
    ])
    
    is_valid, error_msg = validateInitialState(x0)
    
    assert is_valid
