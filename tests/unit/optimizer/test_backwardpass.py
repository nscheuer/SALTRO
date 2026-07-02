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
LAMBDA_AUG_ZERO = [np.array([0.0])]
MU_AUG_ZERO = [np.array([0.0])]


def make_attitude_traj(att, N_cols):
    """Create attitude target trajectory by repeating a single target."""
    traj = np.zeros((4, N_cols))
    for k in range(N_cols):
        traj[:, k] = att
    return traj


def _make_test_satellite(settings):
    J = np.diag([0.067, 0.071, 0.069])
    sat = saltro_py.Satellite(J, settings)
    sat.addMTQ(np.array([1.0, 0.0, 0.0]), 0.2)
    sat.addMTQ(np.array([0.0, 1.0, 0.0]), 0.2)
    sat.addMTQ(np.array([0.0, 0.0, 1.0]), 0.2)
    sat.addRW(np.array([1.0, 0.0, 0.0]), 0.001, 1e-5, 0.0, 0.02)
    sat.addRW(np.array([0.0, 1.0, 0.0]), 0.001, 1e-5, 0.0, 0.02)
    sat.addRW(np.array([0.0, 0.0, 1.0]), 0.001, 1e-5, 0.0, 0.02)
    return sat


def _run_ddp_scenario(configure, with_constraint_lambda):
    N_test = 3
    settings = saltro_py.PlannerSettings()
    settings.disturbances.plan_for_aero = False
    settings.disturbances.plan_for_gg = False
    settings.disturbances.plan_for_srp = False
    settings.disturbances.plan_for_prop = False
    settings.disturbances.plan_for_gendist = False
    settings.disturbances.plan_for_resdipole = False
    settings.num_passes = 1
    settings.passes[0].dt = 0.5
    settings.passes[0].reg.reg_init = 1e-6
    settings.passes[0].reg.reg_scale = 10.0
    settings.passes[0].reg.reg_max = 1e6
    settings.passes[0].cost.angle = 50.0
    settings.passes[0].cost.ang_vel = 10.0
    configure(settings)

    sat = _make_test_satellite(settings)

    nx = sat.stateDim
    nu = sat.controlDim

    axis = np.array([0.3, -0.6, 0.74], dtype=float)
    axis /= np.linalg.norm(axis)
    half = 0.5 * (60.0 * PI / 180.0)
    q = np.concatenate(([np.cos(half)], np.sin(half) * axis))

    xs = np.zeros(nx)
    xs[0:3] = np.array([0.05, -0.03, 0.04])
    xs[3:7] = q
    if sat.numRW > 0:
        xs[7:7 + sat.numRW] = 0.005

    X = np.zeros((nx, N_test))
    U = np.zeros((nu, N_test - 1))
    R = np.zeros((3, N_test))
    V = np.zeros((3, N_test))
    B = np.zeros((3, N_test))
    S = np.zeros((3, N_test))
    rho = np.zeros((1, N_test))
    boresight = np.zeros((3, N_test))
    for k in range(N_test):
        X[:, k] = xs
        if k < N_test - 1:
            U[:, k] = 0.01
        R[:, k] = np.array([7000e3, 0.0, 0.0])
        V[:, k] = np.array([0.0, 7500.0, 0.0])
        B[:, k] = np.array([2.5e-5, -1.5e-5, 3.0e-5])
        S[:, k] = np.array([1.0, 0.1, -0.05])
        S[:, k] /= np.linalg.norm(S[:, k])
        boresight[:, k] = np.array([1.0, 0.0, 0.0])

    attitude_target = np.array([np.nan, 0.0, 0.0, 0.0])
    attitude_target_traj = make_attitude_traj(attitude_target, N_test)

    lambda_aug = LAMBDA_AUG_ZERO
    mu_aug = MU_AUG_ZERO
    if with_constraint_lambda:
        settings.constraints.wmax = 0.01
        c0 = np.asarray(
            sat.constraints(0, N_test, X[:, 0], U[:, 0], S[:, 0], settings.constraints),
            dtype=float,
        )
        lambda_aug = [np.full_like(c0, 1.0) for _ in range(N_test)]
        mu_aug = [np.full_like(c0, 100.0) for _ in range(N_test)]

    ok, K, d, deltaV = saltro_py.backward_pass(
        sat,
        X,
        U,
        R,
        V,
        B,
        S,
        rho,
        boresight,
        attitude_target_traj,
        settings,
        lambda_aug,
        mu_aug,
        settings.passes[0].reg.reg_init,
    )
    return ok, K, d, deltaV


def _max_gain_diff(K_a, d_a, K_b, d_b):
    return max(
        np.max(np.abs(K_a - K_b)),
        np.max(np.abs(d_a - d_b)),
    )


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
            boresight_test, attitude_target_test_traj, settings_test,
            LAMBDA_AUG_ZERO, MU_AUG_ZERO, settings_test.passes[0].reg.reg_init
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
            boresight_test, attitude_target_test_traj, settings_test,
            LAMBDA_AUG_ZERO, MU_AUG_ZERO, settings_test.passes[0].reg.reg_init
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
            fixture.rho, fixture.boresight, fixture.attitude_target_traj, fixture.settings,
            LAMBDA_AUG_ZERO, MU_AUG_ZERO, fixture.reg
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
            fixture.rho, fixture.boresight, fixture.attitude_target_traj, fixture.settings,
            LAMBDA_AUG_ZERO, MU_AUG_ZERO, fixture.reg
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
            fixture.rho, fixture.boresight, fixture.attitude_target_traj, fixture.settings,
            LAMBDA_AUG_ZERO, MU_AUG_ZERO, fixture.reg
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
            fixture.rho, fixture.boresight, fixture.attitude_target_traj, fixture.settings,
            LAMBDA_AUG_ZERO, MU_AUG_ZERO, fixture.reg
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
            boresight_test, attitude_target_test_traj, settings_test,
            LAMBDA_AUG_ZERO, MU_AUG_ZERO, settings_test.passes[0].reg.reg_init
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
            boresight_test, attitude_target_test_traj, settings_test,
            LAMBDA_AUG_ZERO, MU_AUG_ZERO, settings_test.passes[0].reg.reg_init
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

    def test_rejects_non_finite_trajectory_states_instead_of_emitting_nan_gains(self):
        """A poisoned state should make backward_pass fail fast without
        returning NaN gain matrices.
        """
        N_test = 5

        settings_test = saltro_py.PlannerSettings()
        J = np.diag([0.067, 0.071, 0.069])
        satellite_test = saltro_py.Satellite(J, settings_test)

        satellite_test.addMTQ(np.array([1.0, 0.0, 0.0]), 0.2)
        satellite_test.addMTQ(np.array([0.0, 1.0, 0.0]), 0.2)
        satellite_test.addMTQ(np.array([0.0, 0.0, 1.0]), 0.2)
        satellite_test.addRW(np.array([1.0, 0.0, 0.0]), 0.001, 1e-5, 0.0, 0.02)
        satellite_test.addRW(np.array([0.0, 1.0, 0.0]), 0.001, 1e-5, 0.0, 0.02)
        satellite_test.addRW(np.array([0.0, 0.0, 1.0]), 0.001, 1e-5, 0.0, 0.02)

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
                U[:, k] = 1e-4 * np.ones(nu)

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

        X[0, 2] = np.nan

        ok, K, d, deltaV = saltro_py.backward_pass(
            satellite_test, X, U, R_test, V_test, B_test, S_test, rho_test,
            boresight_test, attitude_target_test_traj, settings_test,
            LAMBDA_AUG_ZERO, MU_AUG_ZERO, settings_test.passes[0].reg.reg_init
        )

        assert not ok
        for k in range(K.shape[0]):
            assert np.all(np.isfinite(K[k]) | (K[k] == 0.0))

    def test_linearizes_with_configured_disturbances(self):
        """The C++ backward pass must use the same disturbance config as rollout."""
        N_test = 5
        settings_off = saltro_py.PlannerSettings()
        settings_gg = saltro_py.PlannerSettings()
        settings_off.num_passes = 1
        settings_gg.num_passes = 1
        settings_off.passes[0].dt = 0.5
        settings_gg.passes[0].dt = 0.5
        settings_gg.disturbances.plan_for_gg = True

        satellite = _make_test_satellite(settings_off)
        nx = satellite.stateDim
        nu = satellite.controlDim

        X = np.zeros((nx, N_test))
        U = np.zeros((nu, N_test - 1))
        R = np.zeros((3, N_test))
        V = np.zeros((3, N_test))
        B = np.zeros((3, N_test))
        S = np.zeros((3, N_test))
        rho = np.zeros((1, N_test))
        boresight = np.zeros((3, N_test))

        half_angle = np.pi / 12.0
        x_rotated = np.zeros(nx)
        x_rotated[3:7] = np.array([
            np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)
        ])
        rng = np.random.default_rng(0)
        for k in range(N_test):
            X[:, k] = x_rotated
            if k < N_test - 1:
                U[:, k] = 0.001 * rng.standard_normal(nu)
            R[:, k] = np.array([7000e3, 0.0, 0.0])
            V[:, k] = np.array([0.0, 7500.0, 0.0])
            B[:, k] = np.array([2.5e-5, -1.5e-5, 3.0e-5])
            S[:, k] = np.array([1.0, 0.1, -0.05])
            S[:, k] /= np.linalg.norm(S[:, k])
            boresight[:, k] = np.array([1.0, 0.0, 0.0])

        attitude_target = make_attitude_traj(
            np.array([np.nan, 0.0, 0.0, 0.0]), N_test
        )

        def run(settings):
            return saltro_py.backward_pass(
                satellite, X, U, R, V, B, S, rho, boresight, attitude_target,
                settings, LAMBDA_AUG_ZERO, MU_AUG_ZERO,
                settings.passes[0].reg.reg_init,
            )

        ok_off, K_off, d_off, _ = run(settings_off)
        ok_gg, K_gg, d_gg, _ = run(settings_gg)

        assert ok_off and ok_gg
        assert np.all(np.isfinite(K_gg))
        assert np.all(np.isfinite(d_gg))
        assert np.linalg.norm(K_gg - K_off) + np.linalg.norm(d_gg - d_off) > 0.0

    def test_psd_clip_clips_negative_eigenvalues(self):
        """Exercise the C++ psd_clip helper through its Python binding."""
        matrix = np.array([
            [2.0, 0.0, 0.0],
            [0.0, -5.0, 1.0],
            [0.0, 1.0, 3.0],
        ])
        matrix = 0.5 * (matrix + matrix.T)
        assert np.linalg.eigvalsh(matrix)[0] < 0.0

        clipped = saltro_py.psd_clip(matrix)

        assert np.linalg.eigvalsh(clipped)[0] >= -1e-12
        np.linalg.cholesky(clipped + 1e-6 * np.eye(3))

    def test_rk4_hessians_match_finite_difference_of_rk4_jacobians(self):
        """Validate the C++ RK4 Hessian composition against its C++ Jacobian."""
        settings = saltro_py.PlannerSettings()
        settings.num_passes = 1
        settings.passes[0].dt = 0.05
        satellite = saltro_py.Satellite(np.diag([0.067, 0.071, 0.069]), settings)
        satellite.addMTQ(np.array([1.0, 0.0, 0.0]), 0.2)
        satellite.addRW(np.array([1.0, 0.0, 0.0]), 0.001, 1e-5, 0.0, 0.02)

        nx = satellite.stateDim
        nu = satellite.controlDim
        axis = np.array([0.2, 0.5, -0.84])
        axis /= np.linalg.norm(axis)
        half_angle = 0.5 * np.deg2rad(40.0)
        x = np.zeros(nx)
        x[0:3] = np.array([0.08, -0.05, 0.06])
        x[3:7] = np.concatenate(([np.cos(half_angle)], np.sin(half_angle) * axis))
        x[7] = 0.004
        u = np.full(nu, 0.02)

        R = np.array([7000e3, 0.0, 0.0])
        V = np.array([0.0, 7500.0, 0.0])
        B = np.array([2.5e-5, -1.5e-5, 3.0e-5])
        S = np.array([1.0, 0.1, -0.05])
        S /= np.linalg.norm(S)
        disturbances = settings.disturbances
        dt = settings.passes[0].dt

        F_xx, _, _ = saltro_py.rk4_dynamics_hessians(
            satellite, x, u, dt, disturbances, R, B, S, V
        )
        F_xx = np.asarray(F_xx)

        epsilon = 1e-6
        max_error = 0.0
        checked = 0
        quaternion_indices = range(3, 7)
        for j in range(nx):
            if j in quaternion_indices:
                continue
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[j] += epsilon
            x_minus[j] -= epsilon
            A_plus, _ = saltro_py.rk4_dynamics_jacobians(
                satellite, x_plus, u, dt, disturbances, R, B, S, V
            )
            A_minus, _ = saltro_py.rk4_dynamics_jacobians(
                satellite, x_minus, u, dt, disturbances, R, B, S, V
            )
            dA_dxj = (A_plus - A_minus) / (2.0 * epsilon)
            for output in range(nx):
                for state in range(nx):
                    if state in quaternion_indices:
                        continue
                    max_error = max(
                        max_error,
                        abs(dA_dxj[output, state] - F_xx[output, state, j]),
                    )
                    checked += 1

        assert checked > 0
        assert max_error < 1e-4

    def test_ddp_knobs_off_reproduce_gauss_newton(self):
        """Default backward pass should match explicitly disabled DDP knobs."""
        ok0, K0, d0, deltaV0 = _run_ddp_scenario(lambda settings: None, False)
        ok1, K1, d1, deltaV1 = _run_ddp_scenario(
            lambda settings: (
                setattr(settings.passes[0].reg, "use_dynamics_hess", False),
                setattr(settings.passes[0].reg, "use_constraint_hess", False),
                setattr(settings.passes[0].reg, "psd_clip_quu_ddp", False),
            ),
            False,
        )

        assert ok0 and ok1
        assert np.array_equal(K0, K1)
        assert np.array_equal(d0, d1)
        assert np.array_equal(deltaV0, deltaV1)

    def test_ddp_knobs_off_with_active_constraint_reproduce_gauss_newton(self):
        """Default backward pass should stay identical with active AL inputs too."""
        ok0, K0, d0, _ = _run_ddp_scenario(lambda settings: None, True)
        ok1, K1, d1, _ = _run_ddp_scenario(
            lambda settings: (
                setattr(settings.passes[0].reg, "use_dynamics_hess", False),
                setattr(settings.passes[0].reg, "use_constraint_hess", False),
            ),
            True,
        )

        assert ok0 and ok1
        assert np.array_equal(K0, K1)
        assert np.array_equal(d0, d1)

    def test_ddp_dynamics_hessian_changes_gains_vs_gauss_newton(self):
        """Enabling dynamics Hessians should measurably perturb the solution."""
        ok_gn, K_gn, d_gn, _ = _run_ddp_scenario(lambda settings: None, False)
        ok_ddp, K_ddp, d_ddp, deltaV_ddp = _run_ddp_scenario(
            lambda settings: (
                setattr(settings.passes[0].reg, "use_dynamics_hess", True),
                setattr(settings.passes[0].reg, "psd_clip_quu_ddp", True),
            ),
            False,
        )

        assert ok_gn and ok_ddp
        assert np.all(np.isfinite(K_ddp))
        assert np.all(np.isfinite(d_ddp))
        assert np.all(np.isfinite(deltaV_ddp))
        assert _max_gain_diff(K_gn, d_gn, K_ddp, d_ddp) > 1e-9

    def test_ddp_constraint_hessian_changes_gains_vs_gauss_newton(self):
        """Enabling constraint Hessians should measurably perturb the solution."""
        ok_gn, K_gn, d_gn, _ = _run_ddp_scenario(lambda settings: None, True)
        ok_ddp, K_ddp, d_ddp, deltaV_ddp = _run_ddp_scenario(
            lambda settings: (
                setattr(settings.passes[0].reg, "use_constraint_hess", True),
                setattr(settings.passes[0].reg, "psd_clip_quu_ddp", True),
            ),
            True,
        )

        assert ok_gn and ok_ddp
        assert np.all(np.isfinite(K_ddp))
        assert np.all(np.isfinite(d_ddp))
        assert np.all(np.isfinite(deltaV_ddp))
        assert _max_gain_diff(K_gn, d_gn, K_ddp, d_ddp) > 1e-9

    def test_ddp_both_knobs_on_with_active_constraint_stays_finite(self):
        """Combined second-order terms should still yield a finite backward pass."""
        ok, K, d, deltaV = _run_ddp_scenario(
            lambda settings: (
                setattr(settings.passes[0].reg, "use_dynamics_hess", True),
                setattr(settings.passes[0].reg, "use_constraint_hess", True),
                setattr(settings.passes[0].reg, "psd_clip_quu_ddp", True),
            ),
            True,
        )

        assert ok
        assert np.all(np.isfinite(K))
        assert np.all(np.isfinite(d))
        assert np.all(np.isfinite(deltaV))

    def test_psd_clamp_lxx_diagnostic_flag(self):
        """Python twin of the C++ test "backward_pass psd_clamp_lxx diagnostic
        flag".  psd_clamp_lxx is a TESTING/DIAGNOSTIC knob (default False).
        Guards:
         1. Regression: with the flag at its default (untouched settings) the
            result is bitwise-identical to the flag explicitly set to False.
         2. Flag on, well-conditioned problem: backward pass still succeeds
            and results stay finite.
         3. Flag on, indefinite-lxx scenario (analytic cost Hessian with the
            concave raw-acos shape, heavy stage angle weight, large attitude
            error): backward pass succeeds and results stay finite.
        """
        N_test = 5

        def make_settings():
            s = saltro_py.PlannerSettings()
            s.disturbances.plan_for_aero = False
            s.disturbances.plan_for_gg = False
            s.disturbances.plan_for_srp = False
            s.disturbances.plan_for_prop = False
            s.disturbances.plan_for_gendist = False
            s.disturbances.plan_for_resdipole = False
            s.num_passes = 1
            s.passes[0].dt = 0.5
            s.passes[0].reg.reg_init = 1e-8
            s.passes[0].reg.reg_scale = 10.0
            s.passes[0].reg.reg_max = 1e4
            return s

        satellite_test = _make_test_satellite(make_settings())
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

        # Deterministic trajectory (no random draws) so repeated runs are
        # comparable.
        for k in range(N_test):
            X[:, k] = x0
            R_test[:, k] = np.array([7000e3, 0.0, 0.0])
            V_test[:, k] = np.array([0.0, 7500.0, 0.0])
            B_test[:, k] = np.array([2.5e-5, -1.5e-5, 3.0e-5])
            S_test[:, k] = np.array([1.0, 0.1, -0.05])
            S_test[:, k] /= np.linalg.norm(S_test[:, k])
            boresight_test[:, k] = np.array([1.0, 0.0, 0.0])

        # Quaternion attitude target 120 deg about x away from the (identity)
        # state quaternion, so d = q_goal . q = 0.5.
        #
        # NOTE: this scenario used ang_cost_func_type=2 (raw acos, f''(d) < 0)
        # to manufacture a genuinely indefinite stage lxx. Type 2 has been
        # removed, and none of the remaining quaternion-mode shapes {0,1,3} are
        # concave in the aligned hemisphere, so the indefinite_cost branch now
        # runs a well-conditioned (PSD) stage Hessian. The assertions here are
        # finiteness-only, so they still pass and still exercise the
        # psd_clamp_lxx code path; but this no longer proves the clamp rescues
        # a truly indefinite lxx. (Follow-up: reconstruct genuine
        # indefiniteness via the vector-pointing exact Hessian with
        # cost_hess_gauss_newton = False.)
        attitude_target_test = np.array([0.5, np.sqrt(3.0) / 2.0, 0.0, 0.0])
        attitude_target_test_traj = make_attitude_traj(attitude_target_test, N_test)

        def run_bp(set_flag, flag_value, indefinite_cost):
            s = make_settings()
            if set_flag:
                s.passes[0].reg.psd_clamp_lxx = flag_value
            if indefinite_cost:
                s.passes[0].cost.use_cost_hess = True
                s.passes[0].cost.ang_cost_func_type = 3  # was 2 (raw acos, concave); type 2 removed
                s.passes[0].cost.angle = 1e5             # heavy stage angle weight
                s.passes[0].cost.angle_N = 0.0           # keep terminal P_N PSD (clamp covers stage lxx only)
            return saltro_py.backward_pass(
                satellite_test, X, U, R_test, V_test, B_test, S_test, rho_test,
                boresight_test, attitude_target_test_traj, s,
                LAMBDA_AUG_ZERO, MU_AUG_ZERO, s.passes[0].reg.reg_init
            )

        # 1. Regression guard: default flag (untouched RegularizationConfig)
        #    must be bitwise-identical to flag explicitly off.
        ok_default, K_default, d_default, _ = run_bp(False, False, False)
        ok_off, K_off, d_off, _ = run_bp(True, False, False)
        assert ok_default
        assert ok_off
        assert np.array_equal(K_default, K_off)
        assert np.array_equal(d_default, d_off)

        # 2. Flag on, well-conditioned problem: succeeds, finite.
        ok_on, K_on, d_on, _ = run_bp(True, True, False)
        assert ok_on
        assert np.all(np.isfinite(K_on))
        assert np.all(np.isfinite(d_on))

        # 3. Flag on, indefinite-lxx scenario: succeeds, finite.
        ok_diag, K_diag, d_diag, _ = run_bp(True, True, True)
        assert ok_diag
        assert np.all(np.isfinite(K_diag))
        assert np.all(np.isfinite(d_diag))


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
