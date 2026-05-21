import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))

import saltro_py


# ============================================================================
# Helper Functions
# ============================================================================

def valid_inertia_matrix():
    """Create a valid inertia matrix (roughly cube-shaped satellite)."""
    J = np.array([
        [0.067, 0.0, 0.0],
        [0.0, 0.067, 0.0],
        [0.0, 0.0, 0.067]
    ])
    return J


def singular_inertia_matrix():
    """Create a singular (non-invertible) inertia matrix."""
    J = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [3.0, 6.0, 9.0]
    ])
    return J


def infinite_inertia_matrix():
    """Create an inertia matrix with infinite entries."""
    J = valid_inertia_matrix()
    J[0, 0] = np.inf
    return J


# ============================================================================
# Constructor Tests
# ============================================================================

def test_satellite_default_constructor():
    """Test default constructor creates satellite with no actuators."""
    sat = saltro_py.Satellite()
    
    assert sat.numMTQ == 0
    assert sat.numRW == 0
    assert sat.stateDim == 7
    assert sat.reducedStateDim == 6
    assert sat.controlDim == 0
    
    # Inertia should be identity
    assert np.allclose(sat.inertia, np.eye(3))
    assert np.allclose(sat.invInertia, np.eye(3))


def test_satellite_constructor_with_valid_inertia():
    """Test constructor with valid inertia matrix."""
    J = valid_inertia_matrix()
    settings = saltro_py.PlannerSettings()
    
    sat = saltro_py.Satellite(J, settings)
    
    assert np.allclose(sat.inertia, J)
    assert np.allclose(sat.invInertia, np.linalg.inv(J))
    assert sat.numMTQ == 0
    assert sat.numRW == 0


def test_satellite_constructor_with_singular_inertia():
    """Test constructor rejects singular inertia matrix."""
    J = singular_inertia_matrix()
    settings = saltro_py.PlannerSettings()
    
    with pytest.raises(Exception):
        saltro_py.Satellite(J, settings)


def test_satellite_constructor_with_infinite_inertia():
    """Test constructor rejects infinite inertia values."""
    J = infinite_inertia_matrix()
    settings = saltro_py.PlannerSettings()
    
    with pytest.raises(Exception):
        saltro_py.Satellite(J, settings)


# ============================================================================
# Inertia Management Tests
# ============================================================================

def test_satellite_set_inertia_with_valid_matrix():
    """Test setting inertia with a valid matrix."""
    sat = saltro_py.Satellite()
    J = valid_inertia_matrix()
    
    sat.setInertia(J)
    
    assert np.allclose(sat.inertia, J)
    assert np.allclose(sat.invInertia, np.linalg.inv(J))


def test_satellite_set_inertia_with_singular_matrix():
    """Test setInertia rejects singular matrix."""
    sat = saltro_py.Satellite()
    J = singular_inertia_matrix()
    
    with pytest.raises(Exception):
        sat.setInertia(J)


def test_satellite_set_inertia_with_non_finite_entries():
    """Test setInertia rejects non-finite entries."""
    sat = saltro_py.Satellite()
    J = infinite_inertia_matrix()
    
    with pytest.raises(Exception):
        sat.setInertia(J)


def test_satellite_inertia_no_rw_matches_inertia_when_no_rws():
    """Test inertiaNoRW equals inertia when no RWs present."""
    J = valid_inertia_matrix()
    settings = saltro_py.PlannerSettings()
    sat = saltro_py.Satellite(J, settings)
    
    assert np.allclose(sat.inertiaNoRW, J)
    assert np.allclose(sat.invInertiaNoRW, np.linalg.inv(J))


def test_satellite_inertia_no_rw_updates_when_rw_is_added():
    """Test inertiaNoRW is correctly updated when RW is added."""
    J = valid_inertia_matrix()
    settings = saltro_py.PlannerSettings()
    sat = saltro_py.Satellite(J, settings)
    
    axis = np.array([1.0, 0.0, 0.0])
    max_torque = 0.001
    J_rw = 1e-5
    h0 = 0.0
    h_max = 0.01
    
    sat.addRW(axis, max_torque, J_rw, h0, h_max)
    
    # J_noRW should be J - J_rw * axis * axis^T
    expected_J_noRW = J - J_rw * np.outer(axis, axis)
    
    assert np.allclose(sat.inertiaNoRW, expected_J_noRW)
    assert np.allclose(sat.invInertiaNoRW, np.linalg.inv(expected_J_noRW))


# ============================================================================
# Adding Actuators Tests
# ============================================================================

def test_add_single_mtq():
    """Test adding a single MTQ."""
    sat = saltro_py.Satellite()
    axis = np.array([1.0, 0.0, 0.0])
    max_dipole = 0.2
    
    sat.addMTQ(axis, max_dipole)
    
    assert sat.numMTQ == 1
    assert sat.controlDim == 1


def test_add_multiple_mtqs():
    """Test adding multiple MTQs."""
    sat = saltro_py.Satellite()
    
    sat.addMTQ(np.array([1.0, 0.0, 0.0]), 0.2)
    assert sat.numMTQ == 1
    
    sat.addMTQ(np.array([0.0, 1.0, 0.0]), 0.2)
    assert sat.numMTQ == 2
    
    sat.addMTQ(np.array([0.0, 0.0, 1.0]), 0.2)
    assert sat.numMTQ == 3
    
    assert sat.controlDim == 3


def test_add_mtq_up_to_maximum_limit():
    """Test adding MTQs up to the maximum limit."""
    sat = saltro_py.Satellite()
    
    # Add MAX_NUM_MTQ (4) MTQs
    for i in range(saltro_py.limits.MAX_NUM_MTQ):
        axis = np.zeros(3)
        axis[i % 3] = 1.0
        sat.addMTQ(axis, 0.2)
    
    assert sat.numMTQ == saltro_py.limits.MAX_NUM_MTQ


def test_add_mtq_beyond_maximum_limit():
    """Test that adding MTQ beyond maximum limit raises error."""
    sat = saltro_py.Satellite()
    
    # Add MAX_NUM_MTQ (4) MTQs
    for i in range(saltro_py.limits.MAX_NUM_MTQ):
        axis = np.zeros(3)
        axis[i % 3] = 1.0
        sat.addMTQ(axis, 0.2)
    
    # Try to add one more - should raise
    with pytest.raises(Exception):
        sat.addMTQ(np.array([1.0, 0.0, 0.0]), 0.2)


def test_add_single_rw():
    """Test adding a single RW."""
    sat = saltro_py.Satellite()
    axis = np.array([1.0, 0.0, 0.0])
    max_torque = 0.001
    J_rw = 1e-5
    h0 = 0.0
    h_max = 0.01
    
    sat.addRW(axis, max_torque, J_rw, h0, h_max)
    
    assert sat.numRW == 1
    assert sat.stateDim == 8  # 7 + 1 RW
    assert sat.reducedStateDim == 7  # 6 + 1 RW
    assert sat.controlDim == 1


def test_add_multiple_rws():
    """Test adding multiple RWs."""
    sat = saltro_py.Satellite()
    max_torque = 0.001
    J_rw = 1e-5
    h0 = 0.0
    h_max = 0.01
    
    sat.addRW(np.array([1.0, 0.0, 0.0]), max_torque, J_rw, h0, h_max)
    assert sat.numRW == 1
    assert sat.stateDim == 8
    
    sat.addRW(np.array([0.0, 1.0, 0.0]), max_torque, J_rw, h0, h_max)
    assert sat.numRW == 2
    assert sat.stateDim == 9
    
    sat.addRW(np.array([0.0, 0.0, 1.0]), max_torque, J_rw, h0, h_max)
    assert sat.numRW == 3
    assert sat.stateDim == 10


def test_add_rw_up_to_maximum_limit():
    """Test adding RWs up to the maximum limit."""
    sat = saltro_py.Satellite()
    max_torque = 0.001
    J_rw = 1e-5
    h0 = 0.0
    h_max = 0.01
    
    # Add MAX_NUM_RW (4) RWs
    for i in range(saltro_py.limits.MAX_NUM_RW):
        axis = np.zeros(3)
        axis[i % 3] = 1.0
        sat.addRW(axis, max_torque, J_rw, h0, h_max)
    
    assert sat.numRW == saltro_py.limits.MAX_NUM_RW
    assert sat.stateDim == 7 + saltro_py.limits.MAX_NUM_RW


def test_add_rw_beyond_maximum_limit():
    """Test that adding RW beyond maximum limit raises error."""
    sat = saltro_py.Satellite()
    max_torque = 0.001
    J_rw = 1e-5
    h0 = 0.0
    h_max = 0.01
    
    # Add MAX_NUM_RW (4) RWs
    for i in range(saltro_py.limits.MAX_NUM_RW):
        axis = np.zeros(3)
        axis[i % 3] = 1.0
        sat.addRW(axis, max_torque, J_rw, h0, h_max)
    
    # Try to add one more - should raise
    with pytest.raises(Exception):
        sat.addRW(np.array([1.0, 0.0, 0.0]), max_torque, J_rw, h0, h_max)


def test_add_both_mtqs_and_rws():
    """Test adding both MTQs and RWs."""
    sat = saltro_py.Satellite()
    
    sat.addMTQ(np.array([1.0, 0.0, 0.0]), 0.2)
    sat.addMTQ(np.array([0.0, 1.0, 0.0]), 0.2)
    sat.addRW(np.array([0.0, 0.0, 1.0]), 0.001, 1e-5, 0.0, 0.01)
    
    assert sat.numMTQ == 2
    assert sat.numRW == 1
    assert sat.controlDim == 3  # 2 MTQs + 1 RW
    assert sat.stateDim == 8    # 7 + 1 RW


# ============================================================================
# Getting Actuators Tests
# ============================================================================

def test_get_mtq_with_valid_index():
    """Test getting an MTQ with a valid index."""
    sat = saltro_py.Satellite()
    axis = np.array([1.0, 0.0, 0.0])
    max_dipole = 0.2
    
    sat.addMTQ(axis, max_dipole)
    
    mtq = sat.getMTQ(0)
    assert np.allclose(mtq.axis, axis / np.linalg.norm(axis))
    assert mtq.u_max == max_dipole


def test_get_mtq_with_negative_index():
    """Test that getting MTQ with negative index raises error."""
    sat = saltro_py.Satellite()
    sat.addMTQ(np.array([1.0, 0.0, 0.0]), 0.2)
    
    with pytest.raises(Exception):
        sat.getMTQ(-1)


def test_get_mtq_with_index_too_large():
    """Test that getting MTQ with too large index raises error."""
    sat = saltro_py.Satellite()
    sat.addMTQ(np.array([1.0, 0.0, 0.0]), 0.2)
    
    with pytest.raises(Exception):
        sat.getMTQ(1)
    
    with pytest.raises(Exception):
        sat.getMTQ(5)


def test_get_mtq_when_no_mtqs_added():
    """Test that getting MTQ when none exist raises error."""
    sat = saltro_py.Satellite()
    
    with pytest.raises(Exception):
        sat.getMTQ(0)


def test_get_multiple_mtqs_with_correct_indices():
    """Test getting multiple MTQs with correct indices."""
    sat = saltro_py.Satellite()
    axis1 = np.array([1.0, 0.0, 0.0])
    axis2 = np.array([0.0, 1.0, 0.0])
    axis3 = np.array([0.0, 0.0, 1.0])
    max_dipole = 0.2
    
    sat.addMTQ(axis1, max_dipole)
    sat.addMTQ(axis2, max_dipole)
    sat.addMTQ(axis3, max_dipole)
    
    assert np.allclose(sat.getMTQ(0).axis, axis1 / np.linalg.norm(axis1))
    assert np.allclose(sat.getMTQ(1).axis, axis2 / np.linalg.norm(axis2))
    assert np.allclose(sat.getMTQ(2).axis, axis3 / np.linalg.norm(axis3))


def test_get_rw_with_valid_index():
    """Test getting an RW with a valid index."""
    sat = saltro_py.Satellite()
    axis = np.array([1.0, 0.0, 0.0])
    max_torque = 0.001
    J_rw = 1e-5
    h0 = 0.0
    h_max = 0.01
    
    sat.addRW(axis, max_torque, J_rw, h0, h_max)
    
    rw = sat.getRW(0)
    assert np.allclose(rw.axis, axis / np.linalg.norm(axis))
    assert rw.u_max == max_torque
    assert rw.wheelInertia == J_rw
    assert rw.momentum == h0
    assert rw.momentumMax == h_max


def test_get_rw_with_negative_index():
    """Test that getting RW with negative index raises error."""
    sat = saltro_py.Satellite()
    sat.addRW(np.array([1.0, 0.0, 0.0]), 0.001, 1e-5, 0.0, 0.01)
    
    with pytest.raises(Exception):
        sat.getRW(-1)


def test_get_rw_with_index_too_large():
    """Test that getting RW with too large index raises error."""
    sat = saltro_py.Satellite()
    sat.addRW(np.array([1.0, 0.0, 0.0]), 0.001, 1e-5, 0.0, 0.01)
    
    with pytest.raises(Exception):
        sat.getRW(1)
    
    with pytest.raises(Exception):
        sat.getRW(5)


def test_get_rw_when_no_rws_added():
    """Test that getting RW when none exist raises error."""
    sat = saltro_py.Satellite()
    
    with pytest.raises(Exception):
        sat.getRW(0)


def test_get_multiple_rws_with_correct_indices():
    """Test getting multiple RWs with correct indices."""
    sat = saltro_py.Satellite()
    axis1 = np.array([1.0, 0.0, 0.0])
    axis2 = np.array([0.0, 1.0, 0.0])
    axis3 = np.array([0.0, 0.0, 1.0])
    max_torque = 0.001
    J_rw = 1e-5
    h0 = 0.0
    h_max = 0.01
    
    sat.addRW(axis1, max_torque, J_rw, h0, h_max)
    sat.addRW(axis2, max_torque, J_rw, h0, h_max)
    sat.addRW(axis3, max_torque, J_rw, h0, h_max)
    
    assert np.allclose(sat.getRW(0).axis, axis1 / np.linalg.norm(axis1))
    assert np.allclose(sat.getRW(1).axis, axis2 / np.linalg.norm(axis2))
    assert np.allclose(sat.getRW(2).axis, axis3 / np.linalg.norm(axis3))


def test_modify_rw_through_reference():
    """Test modifying RW momentum through reference."""
    sat = saltro_py.Satellite()
    sat.addRW(np.array([1.0, 0.0, 0.0]), 0.001, 1e-5, 0.0, 0.01)
    
    rw = sat.getRW(0)
    assert rw.momentum == 0.0
    
    rw.momentum = 0.005
    assert rw.momentum == 0.005
    assert sat.getRW(0).momentum == 0.005


# ============================================================================
# Dimension Calculations Tests
# ============================================================================

def test_state_dimension_with_no_actuators():
    """Test dimensions with no actuators."""
    sat = saltro_py.Satellite()
    
    assert sat.stateDim == 7
    assert sat.reducedStateDim == 6
    assert sat.controlDim == 0


def test_state_dimension_with_only_mtqs():
    """Test dimensions with only MTQs."""
    sat = saltro_py.Satellite()
    sat.addMTQ(np.array([1.0, 0.0, 0.0]), 0.2)
    sat.addMTQ(np.array([0.0, 1.0, 0.0]), 0.2)
    
    assert sat.stateDim == 7  # MTQs don't add state
    assert sat.reducedStateDim == 6
    assert sat.controlDim == 2


def test_state_dimension_with_only_rws():
    """Test dimensions with only RWs."""
    sat = saltro_py.Satellite()
    sat.addRW(np.array([1.0, 0.0, 0.0]), 0.001, 1e-5, 0.0, 0.01)
    sat.addRW(np.array([0.0, 1.0, 0.0]), 0.001, 1e-5, 0.0, 0.01)
    
    assert sat.stateDim == 9  # 7 + 2 RWs
    assert sat.reducedStateDim == 8  # 6 + 2 RWs
    assert sat.controlDim == 2


def test_state_dimension_with_mixed_actuators():
    """Test dimensions with mixed actuators."""
    sat = saltro_py.Satellite()
    sat.addMTQ(np.array([1.0, 0.0, 0.0]), 0.2)
    sat.addMTQ(np.array([0.0, 1.0, 0.0]), 0.2)
    sat.addMTQ(np.array([0.0, 0.0, 1.0]), 0.2)
    sat.addRW(np.array([1.0, 0.0, 0.0]), 0.001, 1e-5, 0.0, 0.01)
    sat.addRW(np.array([0.0, 1.0, 0.0]), 0.001, 1e-5, 0.0, 0.01)
    
    assert sat.stateDim == 9  # 7 + 2 RWs
    assert sat.reducedStateDim == 8  # 6 + 2 RWs
    assert sat.controlDim == 5  # 3 MTQs + 2 RWs


def test_state_dimension_at_maximum_actuators():
    """Test dimensions at maximum actuators."""
    sat = saltro_py.Satellite()
    
    # Add max MTQs
    for i in range(saltro_py.limits.MAX_NUM_MTQ):
        axis = np.zeros(3)
        axis[i % 3] = 1.0
        sat.addMTQ(axis, 0.2)
    
    # Add max RWs
    for i in range(saltro_py.limits.MAX_NUM_RW):
        axis = np.zeros(3)
        axis[i % 3] = 1.0
        sat.addRW(axis, 0.001, 1e-5, 0.0, 0.01)
    
    assert sat.stateDim == 7 + saltro_py.limits.MAX_NUM_RW
    assert sat.reducedStateDim == 6 + saltro_py.limits.MAX_NUM_RW
    assert sat.controlDim == saltro_py.limits.MAX_NUM_MTQ + saltro_py.limits.MAX_NUM_RW


# ============================================================================
# Settings and Geometry Config Tests
# ============================================================================

def test_set_and_get_settings():
    """Test setting and getting planner settings."""
    sat = saltro_py.Satellite()
    settings = saltro_py.PlannerSettings()
    
    sat.setSettings(settings)
    retrieved_settings = sat.settings
    # Just verify we can get settings back
    assert retrieved_settings is not None


def test_set_and_get_geometry_config():
    """Test setting and getting geometry config."""
    sat = saltro_py.Satellite()
    config = saltro_py.GeometryConfig()
    
    sat.setGeometryConfig(config)
    retrieved_config = sat.geometryConfig
    # Just verify we can get config back
    assert retrieved_config is not None


# ============================================================================
# State Indices Tests
# ============================================================================

def test_state_index_constants():
    """Test state index constants."""
    assert saltro_py.Satellite.AV_INDEX == 0
    assert saltro_py.Satellite.QUAT_INDEX == 3
    assert saltro_py.Satellite.RW_MOMENTUM_INDEX == 7


# ============================================================================
# Edge Cases and Complex Scenarios
# ============================================================================

def test_add_actuators_with_non_normalized_axes():
    """Test adding actuators with non-normalized axes."""
    sat = saltro_py.Satellite()
    axis_unnormalized = np.array([2.0, 0.0, 0.0])  # Not unit length
    
    # Should work - actuator should normalize internally
    sat.addMTQ(axis_unnormalized, 0.2)
    sat.addRW(axis_unnormalized, 0.001, 1e-5, 0.0, 0.01)
    
    # Verify axes are normalized
    assert np.isclose(np.linalg.norm(sat.getMTQ(0).axis), 1.0, rtol=1e-10)
    assert np.isclose(np.linalg.norm(sat.getRW(0).axis), 1.0, rtol=1e-10)


def test_inertia_update_with_multiple_rws():
    """Test inertia update with multiple RWs."""
    J = valid_inertia_matrix()
    sat = saltro_py.Satellite(J, saltro_py.PlannerSettings())
    
    J_rw = 1e-5
    sat.addRW(np.array([1.0, 0.0, 0.0]), 0.001, J_rw, 0.0, 0.01)
    sat.addRW(np.array([0.0, 1.0, 0.0]), 0.001, J_rw, 0.0, 0.01)
    sat.addRW(np.array([0.0, 0.0, 1.0]), 0.001, J_rw, 0.0, 0.01)
    
    # J_noRW should be J - sum of J_rw * axis * axis^T for all RWs
    expected_J_noRW = J.copy()
    expected_J_noRW[0, 0] -= J_rw  # RW along x-axis
    expected_J_noRW[1, 1] -= J_rw  # RW along y-axis
    expected_J_noRW[2, 2] -= J_rw  # RW along z-axis
    
    assert np.allclose(sat.inertiaNoRW, expected_J_noRW, atol=1e-10)


def test_different_actuator_configurations():
    """Test various actuator configurations."""
    sat = saltro_py.Satellite()
    
    # Configuration: 3 MTQs, 1 RW
    sat.addMTQ(np.array([1.0, 0.0, 0.0]), 0.2)
    sat.addMTQ(np.array([0.0, 1.0, 0.0]), 0.2)
    sat.addMTQ(np.array([0.0, 0.0, 1.0]), 0.2)
    sat.addRW(np.array([1.0, 0.0, 0.0]), 0.001, 1e-5, 0.0, 0.01)
    
    assert sat.numMTQ == 3
    assert sat.numRW == 1
    assert sat.controlDim == 4
    assert sat.stateDim == 8


def test_zero_maximum_dipole_mtq():
    """Test MTQ with zero maximum dipole."""
    sat = saltro_py.Satellite()
    
    # Should be allowed - actuator decides validity
    sat.addMTQ(np.array([1.0, 0.0, 0.0]), 0.0)
    assert sat.getMTQ(0).u_max == 0.0


def test_zero_maximum_torque_rw():
    """Test RW with zero maximum torque."""
    sat = saltro_py.Satellite()
    
    # Should be allowed - actuator decides validity
    sat.addRW(np.array([1.0, 0.0, 0.0]), 0.0, 1e-5, 0.0, 0.01)
    assert sat.getRW(0).u_max == 0.0


def test_rw_with_different_initial_momentum_values():
    """Test RWs with different initial momentum values."""
    sat = saltro_py.Satellite()
    axis = np.array([1.0, 0.0, 0.0])
    max_torque = 0.001
    J_rw = 1e-5
    h_max = 0.01
    
    sat.addRW(axis, max_torque, J_rw, 0.0, h_max)
    sat.addRW(axis, max_torque, J_rw, 0.005, h_max)
    sat.addRW(axis, max_torque, J_rw, -0.003, h_max)
    
    assert sat.getRW(0).momentum == 0.0
    assert sat.getRW(1).momentum == 0.005
    assert sat.getRW(2).momentum == -0.003


# ============================================================================
# actuatorTorque — sum identity
# ============================================================================

def test_actuator_torque_matches_sum_of_per_actuator_contributions():
    """Satellite.actuatorTorque(x,u,B) must equal Σ MTQ_i.torque + Σ RW_i.torque.
    Previously untested — the only public API that aggregates body torque from
    every actuator. Catches indexing slip-ups (MTQ vs RW ordering in u),
    sign errors in the cross product, and forgotten actuators."""
    J = valid_inertia_matrix()
    sat = saltro_py.Satellite(J, saltro_py.PlannerSettings())

    sat.addMTQ(np.array([1.0, 0.0, 0.0]), 0.2)
    sat.addMTQ(np.array([0.0, 1.0, 0.0]), 0.2)
    sat.addMTQ(np.array([0.0, 0.0, 1.0]), 0.2)
    sat.addRW(np.array([1.0, 0.0, 0.0]), 0.001, 1e-5, 0.0, 0.01)
    sat.addRW(np.array([0.0, 1.0, 0.0]), 0.001, 1e-5, 0.0, 0.01)

    rng = np.random.default_rng(20260515)

    for trial in range(8):
        x = np.zeros(sat.stateDim)
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = q

        u = np.concatenate([
            rng.uniform(-0.2, 0.2, sat.numMTQ),
            rng.uniform(-0.001, 0.001, sat.numRW),
        ])
        B_eci = rng.normal(size=3) * 1e-5

        tau_total = sat.actuatorTorque(x, u, B_eci)

        # Reproduce: actuatorTorque accepts B in ECI but each MTQ sees B in body.
        # The Satellite class rotates B internally; we mirror that here using
        # the quaternion stored in x.
        from numpy import dot
        q0, q1, q2, q3 = q
        # Rotation matrix from body to ECI (the convention used by saltro)
        R_eci_body = np.array([
            [1 - 2*(q2*q2 + q3*q3), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
            [2*(q1*q2 + q0*q3), 1 - 2*(q1*q1 + q3*q3), 2*(q2*q3 - q0*q1)],
            [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1*q1 + q2*q2)],
        ])
        B_body = R_eci_body.T @ B_eci

        sum_torque = np.zeros(3)
        x_base = x[:7]
        for i in range(sat.numMTQ):
            sum_torque += sat.getMTQ(i).torque(u[i], x_base, B_body)
        for i in range(sat.numRW):
            sum_torque += sat.getRW(i).torque(u[sat.numMTQ + i], x_base)

        assert np.allclose(tau_total, sum_torque, atol=1e-14), (
            f"trial {trial}: tau_total={tau_total}, sum={sum_torque}"
        )
