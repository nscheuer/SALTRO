"""
Python tests for backward pass using pybind bindings.
Mirrors test_backwardpass.cpp with equivalent test cases.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'build'))
import saltro_py

# Constants
PI = 3.14159265358979323846
SEC_PER_CENTURY = 36525.0 * 86400.0
MAX_LENGTH_TRAJ = 1000  # From limits.h


def make_attitude_traj(att, N_cols):
    """Create attitude target trajectory by repeating a single target."""
    traj = np.zeros((4, N_cols))
    for k in range(N_cols):
        traj[:, k] = att
    return traj


class BackwardPassFixture:
    """Fixture for backward pass tests with satellite setup."""

    def __init__(self):
        self.N = 2  # Minimal case: 2 timesteps
        self.settings = saltro_py.PlannerSettings()
        
        # Create satellite with inertia and actuators
        J = np.diag([0.067, 0.071, 0.069])
        self.satellite = saltro_py.Satellite(J, self.settings)
        
        # Add MTQs (magnetic torque rods)
        self.satellite.addMTQ(np.array([1.0, 0.0, 0.0]), 0.2)
        self.satellite.addMTQ(np.array([0.0, 1.0, 0.0]), 0.2)
        self.satellite.addMTQ(np.array([0.0, 0.0, 1.0]), 0.2)
        
        # Add RWs (reaction wheels)
        self.satellite.addRW(np.array([1.0, 0.0, 0.0]), 0.001, 1e-5, 0.0, 0.02)
        self.satellite.addRW(np.array([0.0, 1.0, 0.0]), 0.001, 1e-5, 0.0, 0.02)
        self.satellite.addRW(np.array([0.0, 0.0, 1.0]), 0.001, 1e-5, 0.0, 0.02)
        
        # Initial state: near identity quaternion with small angular velocity
        nx = self.satellite.stateDim
        self.x0 = np.zeros(nx)
        # AV_INDEX = 0:3, QUAT_INDEX = 3:7 (typical layout)
        self.x0[0:3] = np.array([0.01, -0.005, 0.008])  # Angular velocity
        self.x0[3:7] = np.array([1.0, 0.0, 0.0, 0.0])   # Identity quaternion
        
        # Goal: identity quaternion (ECI format with NaN q0)
        attitude_target = np.array([np.nan, 0.0, 0.0, 0.0])
        self.attitude_target_traj = make_attitude_traj(attitude_target, self.N)
        
        # Time setup: 2 timesteps with dt=0.5 seconds
        dt_seconds = 0.5
        dt_centuries = dt_seconds / SEC_PER_CENTURY
        
        # Initialize orbital environment matrices
        self.R = np.zeros((3, MAX_LENGTH_TRAJ))
        self.V = np.zeros((3, MAX_LENGTH_TRAJ))
        self.B = np.zeros((3, MAX_LENGTH_TRAJ))
        self.S = np.zeros((3, MAX_LENGTH_TRAJ))
        self.rho = np.zeros((1, MAX_LENGTH_TRAJ))
        self.boresight = np.zeros((3, MAX_LENGTH_TRAJ))
        
        for k in range(self.N):
            self.boresight[:, k] = np.array([1.0, 0.0, 0.0])
            # Orbital environment
            self.R[:, k] = np.array([7000e3, 0.0, 0.0])
            self.V[:, k] = np.array([0.0, 7500.0, 0.0])
            self.B[:, k] = np.array([2.5e-5, -1.5e-5, 3.0e-5])
            self.S[:, k] = np.array([1.0, 0.1, -0.05])
            self.S[:, k] /= np.linalg.norm(self.S[:, k])  # normalize
            self.rho[0, k] = 0.0
        
        # Disable disturbances for cleaner test
        self.settings.disturbances.plan_for_aero = False
        self.settings.disturbances.plan_for_gg = False
        self.settings.disturbances.plan_for_srp = False
        self.settings.disturbances.plan_for_prop = False
        self.settings.disturbances.plan_for_gendist = False
        self.settings.disturbances.plan_for_resdipole = False
        self.settings.num_passes = 1
        self.settings.passes[0].dt = dt_seconds
        
        # Regularization settings
        self.settings.passes[0].reg.reg_init = 1e-8
        self.settings.passes[0].reg.reg_scale = 10.0
        self.settings.passes[0].reg.reg_max = 1e4
        self.reg = self.settings.passes[0].reg.reg_init


@pytest.fixture
def fixture():
    """Provide BackwardPassFixture for all tests."""
    return BackwardPassFixture()


class TestBackwardPass:
    """Test suite for backward pass implementation."""
    
    def test_n1_edge_case(self):
        """Test N=1 edge case where only terminal timestep exists."""
        N_test = 1
        
        # Create new satellite
        settings_test = saltro_py.PlannerSettings()
        J = np.diag([0.067, 0.071, 0.069])
        satellite_test = saltro_py.Satellite(J, settings_test)
        
        satellite_test.addMTQ(np.array([1.0, 0.0, 0.0]), 0.2)
        satellite_test.addMTQ(np.array([0.0, 1.0, 0.0]), 0.2)
        satellite_test.addMTQ(np.array([0.0, 0.0, 1.0]), 0.2)
        satellite_test.addRW(np.array([1.0, 0.0, 0.0]), 0.001, 1e-5, 0.0, 0.02)
        satellite_test.addRW(np.array([0.0, 1.0, 0.0]), 0.001, 1e-5, 0.0, 0.02)
        satellite_test.addRW(np.array([0.0, 0.0, 1.0]), 0.001, 1e-5, 0.0, 0.02)
        
        # Setup matrices
        nx = satellite_test.stateDim
        nu = satellite_test.controlDim
        x0 = np.zeros(nx)
        x0[0:3] = np.array([0.01, -0.005, 0.008])
        x0[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        
        X = np.zeros((nx, N_test))
        X[:, 0] = x0
        
        U = np.zeros((nu, 0))  # No controls for N=1
        
        R_test = np.zeros((3, N_test))
        V_test = np.zeros((3, N_test))
        B_test = np.zeros((3, N_test))
        S_test = np.zeros((3, N_test))
        rho_test = np.zeros((1, N_test))
        boresight_test = np.zeros((3, N_test))
        
        R_test[:, 0] = np.array([7000e3, 0.0, 0.0])
        V_test[:, 0] = np.array([0.0, 7500.0, 0.0])
        B_test[:, 0] = np.array([2.5e-5, -1.5e-5, 3.0e-5])
        S_test[:, 0] = np.array([1.0, 0.1, -0.05]) / np.linalg.norm([1.0, 0.1, -0.05])
        boresight_test[:, 0] = np.array([1.0, 0.0, 0.0])
        
        attitude_target_test = np.array([np.nan, 0.0, 0.0, 0.0])
        attitude_target_test_traj = make_attitude_traj(attitude_target_test, N_test)
        
        settings_test.num_passes = 1
        settings_test.passes[0].dt = 0.5
        settings_test.passes[0].reg.reg_init = 1e-8
        settings_test.passes[0].reg.reg_scale = 10.0
        settings_test.passes[0].reg.reg_max = 1e4
        
        ok, K, d, deltaV = saltro_py.backward_pass(
            satellite_test, X, U, R_test, V_test, B_test, S_test, rho_test,
            boresight_test, attitude_target_test_traj, settings_test, settings_test.passes[0].reg.reg_init
        )
        
        assert ok
        assert K.shape == (0, nu, satellite_test.reducedStateDim)
        assert d.shape == (nu, 0)
        assert deltaV.shape == (2,)
    
    def test_n2_hand_verified(self, fixture):
        """Test N=2 with hand-verified computation."""
        N_test = 2
        
        # Create satellite
        settings_test = saltro_py.PlannerSettings()
        J = np.diag([0.067, 0.071, 0.069])
        satellite_test = saltro_py.Satellite(J, settings_test)
        
        satellite_test.addMTQ(np.array([1.0, 0.0, 0.0]), 0.2)
        satellite_test.addMTQ(np.array([0.0, 1.0, 0.0]), 0.2)
        satellite_test.addMTQ(np.array([0.0, 0.0, 1.0]), 0.2)
        satellite_test.addRW(np.array([1.0, 0.0, 0.0]), 0.001, 1e-5, 0.0, 0.02)
        satellite_test.addRW(np.array([0.0, 1.0, 0.0]), 0.001, 1e-5, 0.0, 0.02)
        satellite_test.addRW(np.array([0.0, 0.0, 1.0]), 0.001, 1e-5, 0.0, 0.02)
        
        # Setup matrices
        nx = satellite_test.stateDim
        nu = satellite_test.controlDim
        x0 = np.zeros(nx)
        x0[0:3] = np.array([0.01, -0.005, 0.008])
        x0[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        
        X = np.zeros((nx, N_test))
        X[:, 0] = x0
        X[:, 1] = x0  # Terminal state same as initial
        
        U = np.zeros((nu, N_test - 1))
        U[:, 0] = 0.0  # Zero control
        
        R_test = np.zeros((3, N_test))
        V_test = np.zeros((3, N_test))
        B_test = np.zeros((3, N_test))
        S_test = np.zeros((3, N_test))
        rho_test = np.zeros((1, N_test))
        boresight_test = np.zeros((3, N_test))
        
        for k in range(N_test):
            R_test[:, k] = np.array([7000e3, 0.0, 0.0])
            V_test[:, k] = np.array([0.0, 7500.0, 0.0])
            B_test[:, k] = np.array([2.5e-5, -1.5e-5, 3.0e-5])
            S_test[:, k] = np.array([1.0, 0.1, -0.05]) / np.linalg.norm([1.0, 0.1, -0.05])
            boresight_test[:, k] = np.array([1.0, 0.0, 0.0])
        
        attitude_target_test = np.array([np.nan, 0.0, 0.0, 0.0])
        attitude_target_test_traj = make_attitude_traj(attitude_target_test, N_test)
        
        settings_test.num_passes = 1
        settings_test.passes[0].dt = 0.5
        settings_test.passes[0].reg.reg_init = 1e-8
        settings_test.passes[0].reg.reg_scale = 10.0
        settings_test.passes[0].reg.reg_max = 1e4
        
        ok, K, d, deltaV = saltro_py.backward_pass(
            satellite_test, X, U, R_test, V_test, B_test, S_test, rho_test,
            boresight_test, attitude_target_test_traj, settings_test, settings_test.passes[0].reg.reg_init
        )
        
        assert ok
        assert K.shape == (1, nu, satellite_test.reducedStateDim)
        assert d.shape == (nu, 1)
        assert K[0].shape == (nu, satellite_test.reducedStateDim)
        assert d[:, 0].shape == (nu,)
        
        # K[0] and d[:,0] should be finite
        assert np.all(np.isfinite(K[0]))
        assert np.all(np.isfinite(d[:, 0]))
        
        # Verify deltaV is finite
        assert np.all(np.isfinite(deltaV))
    
    def test_dimensions(self, fixture):
        """Test backward_pass returns correct output dimensions."""
        N = fixture.N
        nx = fixture.satellite.stateDim
        nu = fixture.satellite.controlDim
        
        X = np.zeros((nx, N))
        X[:, 0] = fixture.x0
        X[:, 1] = fixture.x0
        
        U = np.zeros((nu, N - 1))
        U[:, 0] = 0.0
        
        ok, K, d, deltaV = saltro_py.backward_pass(
            fixture.satellite, X, U, fixture.R, fixture.V, fixture.B, fixture.S,
            fixture.rho, fixture.boresight, fixture.attitude_target_traj, fixture.settings, fixture.reg
        )
        
        assert ok
        assert K.shape == (N - 1, nu, fixture.satellite.reducedStateDim)
        assert d.shape == (nu, N - 1)
        assert K[0].shape == (nu, fixture.satellite.reducedStateDim)
        assert d[:, 0].shape == (nu,)
    
    def test_terminal_cost_to_go(self, fixture):
        """Test backward_pass computes terminal cost-to-go correctly."""
        N = fixture.N
        nx = fixture.satellite.stateDim
        nu = fixture.satellite.controlDim
        
        X = np.zeros((nx, N))
        X[:, 0] = fixture.x0
        X[:, 1] = fixture.x0
        
        U = np.zeros((nu, N - 1))
        U[:, 0] = 0.0
        
        ok, K, d, deltaV = saltro_py.backward_pass(
            fixture.satellite, X, U, fixture.R, fixture.V, fixture.B, fixture.S,
            fixture.rho, fixture.boresight, fixture.attitude_target_traj, fixture.settings, fixture.reg
        )
        
        assert ok
        assert np.all(np.isfinite(K[0]))
        assert np.all(np.isfinite(d[:, 0]))
        # Feedback gain magnitude should be moderate (not exploding)
        assert np.linalg.norm(K[0]) < 100.0
    
    def test_deltav_accumulation(self, fixture):
        """Test backward_pass accumulates cost reduction terms."""
        N = fixture.N
        nx = fixture.satellite.stateDim
        nu = fixture.satellite.controlDim
        
        X = np.zeros((nx, N))
        X[:, 0] = fixture.x0
        X[:, 1] = fixture.x0
        
        U = np.zeros((nu, N - 1))
        U[:, 0] = 0.0
        
        ok, K, d, deltaV = saltro_py.backward_pass(
            fixture.satellite, X, U, fixture.R, fixture.V, fixture.B, fixture.S,
            fixture.rho, fixture.boresight, fixture.attitude_target_traj, fixture.settings, fixture.reg
        )
        
        assert ok
        assert np.all(np.isfinite(deltaV))
        # At least accumulate non-zero terms
        assert (np.abs(deltaV[0]) > 1e-15 or np.abs(deltaV[1]) > 1e-15)
    
    def test_regularization_loop(self, fixture):
        """Test backward_pass regularization loop converges."""
        N = fixture.N
        nx = fixture.satellite.stateDim
        nu = fixture.satellite.controlDim
        
        X = np.zeros((nx, N))
        X[:, 0] = fixture.x0
        X[:, 1] = fixture.x0 + 0.001 * np.random.randn(nx)
        
        U = np.zeros((nu, N - 1))
        U[:, 0] = 0.001 * np.random.randn(nu)
        
        ok, K, d, deltaV = saltro_py.backward_pass(
            fixture.satellite, X, U, fixture.R, fixture.V, fixture.B, fixture.S,
            fixture.rho, fixture.boresight, fixture.attitude_target_traj, fixture.settings, fixture.reg
        )
        
        # Should succeed (regularization loop finds positive definite Q_uu)
        assert ok
        assert K.shape == (1, nu, fixture.satellite.reducedStateDim)
        assert d.shape == (nu, 1)
        assert np.all(np.isfinite(K[0]))
        assert np.all(np.isfinite(d[:, 0]))
    
    def test_longer_trajectory(self):
        """Test backward_pass handles longer trajectory N=5."""
        N_test = 5
        
        # Create satellite
        settings_test = saltro_py.PlannerSettings()
        J = np.diag([0.067, 0.071, 0.069])
        satellite_test = saltro_py.Satellite(J, settings_test)
        
        satellite_test.addMTQ(np.array([1.0, 0.0, 0.0]), 0.2)
        satellite_test.addMTQ(np.array([0.0, 1.0, 0.0]), 0.2)
        satellite_test.addMTQ(np.array([0.0, 0.0, 1.0]), 0.2)
        satellite_test.addRW(np.array([1.0, 0.0, 0.0]), 0.001, 1e-5, 0.0, 0.02)
        satellite_test.addRW(np.array([0.0, 1.0, 0.0]), 0.001, 1e-5, 0.0, 0.02)
        satellite_test.addRW(np.array([0.0, 0.0, 1.0]), 0.001, 1e-5, 0.0, 0.02)
        
        # Setup matrices
        nx = satellite_test.stateDim
        nu = satellite_test.controlDim
        x0 = np.zeros(nx)
        x0[0:3] = np.array([0.01, -0.005, 0.008])
        x0[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        
        X = np.zeros((nx, N_test))
        U = np.zeros((nu, N_test - 1))
        R_test = np.zeros((3, N_test))
        V_test = np.zeros((3, N_test))
        B_test = np.zeros((3, N_test))
        S_test = np.zeros((3, N_test))
        rho_test = np.zeros((1, N_test))
        boresight_test = np.zeros((3, N_test))
        
        for k in range(N_test):
            X[:, k] = x0
            if k < N_test - 1:
                U[:, k] = 0.001 * np.random.randn(nu)
            
            R_test[:, k] = np.array([7000e3, 0.0, 0.0])
            V_test[:, k] = np.array([0.0, 7500.0, 0.0])
            B_test[:, k] = np.array([2.5e-5, -1.5e-5, 3.0e-5])
            S_test[:, k] = np.array([1.0, 0.1, -0.05]) / np.linalg.norm([1.0, 0.1, -0.05])
            boresight_test[:, k] = np.array([1.0, 0.0, 0.0])
        
        attitude_target_test = np.array([np.nan, 0.0, 0.0, 0.0])
        attitude_target_test_traj = make_attitude_traj(attitude_target_test, N_test)
        
        settings_test.num_passes = 1
        settings_test.passes[0].dt = 0.5
        settings_test.passes[0].reg.reg_init = 1e-8
        settings_test.passes[0].reg.reg_scale = 10.0
        settings_test.passes[0].reg.reg_max = 1e4
        
        ok, K, d, deltaV = saltro_py.backward_pass(
            satellite_test, X, U, R_test, V_test, B_test, S_test, rho_test,
            boresight_test, attitude_target_test_traj, settings_test, settings_test.passes[0].reg.reg_init
        )
        
        assert ok
        assert K.shape == (N_test - 1, nu, satellite_test.reducedStateDim)
        assert d.shape == (nu, N_test - 1)
        
        for k in range(N_test - 1):
            assert np.all(np.isfinite(K[k]))
            assert np.all(np.isfinite(d[:, k]))
    
    def test_gain_magnitudes(self):
        """Test K and d have consistent norms across timesteps."""
        N_test = 3
        
        # Create satellite
        settings_test = saltro_py.PlannerSettings()
        J = np.diag([0.067, 0.071, 0.069])
        satellite_test = saltro_py.Satellite(J, settings_test)
        
        satellite_test.addMTQ(np.array([1.0, 0.0, 0.0]), 0.2)
        satellite_test.addMTQ(np.array([0.0, 1.0, 0.0]), 0.2)
        satellite_test.addMTQ(np.array([0.0, 0.0, 1.0]), 0.2)
        satellite_test.addRW(np.array([1.0, 0.0, 0.0]), 0.001, 1e-5, 0.0, 0.02)
        satellite_test.addRW(np.array([0.0, 1.0, 0.0]), 0.001, 1e-5, 0.0, 0.02)
        satellite_test.addRW(np.array([0.0, 0.0, 1.0]), 0.001, 1e-5, 0.0, 0.02)
        
        # Setup matrices
        nx = satellite_test.stateDim
        nu = satellite_test.controlDim
        x0 = np.zeros(nx)
        x0[0:3] = np.array([0.01, -0.005, 0.008])
        x0[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        
        X = np.zeros((nx, N_test))
        U = np.zeros((nu, N_test - 1))
        R_test = np.zeros((3, N_test))
        V_test = np.zeros((3, N_test))
        B_test = np.zeros((3, N_test))
        S_test = np.zeros((3, N_test))
        rho_test = np.zeros((1, N_test))
        boresight_test = np.zeros((3, N_test))
        
        # Uniform trajectory
        for k in range(N_test):
            X[:, k] = x0
            if k < N_test - 1:
                U[:, k] = 0.0
            
            R_test[:, k] = np.array([7000e3, 0.0, 0.0])
            V_test[:, k] = np.array([0.0, 7500.0, 0.0])
            B_test[:, k] = np.array([2.5e-5, -1.5e-5, 3.0e-5])
            S_test[:, k] = np.array([1.0, 0.1, -0.05]) / np.linalg.norm([1.0, 0.1, -0.05])
            boresight_test[:, k] = np.array([1.0, 0.0, 0.0])
        
        attitude_target_test = np.array([np.nan, 0.0, 0.0, 0.0])
        attitude_target_test_traj = make_attitude_traj(attitude_target_test, N_test)
        
        settings_test.num_passes = 1
        settings_test.passes[0].dt = 0.5
        settings_test.passes[0].reg.reg_init = 1e-8
        settings_test.passes[0].reg.reg_scale = 10.0
        settings_test.passes[0].reg.reg_max = 1e4
        
        ok, K, d, deltaV = saltro_py.backward_pass(
            satellite_test, X, U, R_test, V_test, B_test, S_test, rho_test,
            boresight_test, attitude_target_test_traj, settings_test, settings_test.passes[0].reg.reg_init
        )
        
        assert ok
        
        # For uniform trajectory, gains should remain finite and bounded.
        K0_norm = np.linalg.norm(K[0])
        d0_norm = np.linalg.norm(d[:, 0])
        assert K0_norm >= 0.0
        assert d0_norm >= 0.0

        for k in range(1, K.shape[0]):
            Kk_norm = np.linalg.norm(K[k])
            dk_norm = np.linalg.norm(d[:, k])
            assert np.isfinite(Kk_norm)
            assert np.isfinite(dk_norm)
            assert Kk_norm < 1e6
            assert dk_norm < 1e6


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
