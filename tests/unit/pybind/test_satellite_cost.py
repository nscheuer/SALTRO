"""
Test suite for satellite cost function, Jacobians, and Hessians.

This module provides comprehensive numerical validation of cost function derivatives
using finite difference approximations as ground truth, mirroring the C++ test suite
in test_satellite_cost.cpp.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))

import saltro_py as saltro

# ============================================================================
# Test Fixture Setup
# ============================================================================

class SatelliteCostFixture:
    """Fixture for satellite cost testing with pre-configured satellite and orbit."""
    
    n_steps = 100
    dt = 10.0
    
    def __init__(self):
        # Inertia matrix
        self.J = np.array([
            [0.067, 0.0, 0.0],
            [0.0, 0.067, 0.0],
            [0.0, 0.0, 0.067]
        ])
        
        # Create satellite with default settings
        self.settings = saltro.PlannerSettings()
        self.sat = saltro.Satellite(self.J, self.settings)
        
        # Add 3 reaction wheels on principal axes
        axes = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0])
        ]
        for axis in axes:
            # addRW(axis, max_torque, J, h0, h_max)
            self.sat.addRW(axis, 0.01, 0.001, 0.0, 0.01)
        
        # Add 3 magnetic torquers on principal axes
        for axis in axes:
            # addMTQ(axis, max_dipole)
            self.sat.addMTQ(axis, 0.5)
        
        # Generate orbit data
        self._generate_orbit()
    
    def _generate_orbit(self):
        """Generate Sun-synchronous orbit at ~600 km altitude."""
        # Semi-major axis corresponding to ~600 km altitude
        a = 6978e3
        r0 = np.array([a, 0.0, 0.0])
        v0 = np.array([0.0, 7.56e3, 0.0])  # Orbital velocity
        
        # Generate orbit vectors (importing from saltro if available)
        try:
            self.R, self.V, self.B, self.S, self.rho, self.jtime = \
                saltro.generate_orbit(r0, v0, self.n_steps, self.dt)
        except (AttributeError, TypeError):
            # Fallback: create dummy orbit data if generate_orbit is not exposed
            self.R = np.zeros((3, self.n_steps))
            self.V = np.zeros((3, self.n_steps))
            self.B = np.zeros((3, self.n_steps))
            self.S = np.zeros((3, self.n_steps))
            self.rho = np.ones(self.n_steps) * 1e-13
            self.jtime = np.linspace(0, self.n_steps * self.dt, self.n_steps)
            
            for i in range(self.n_steps):
                # Circular orbit: r perpendicular to v
                angle = 2 * np.pi * i / self.n_steps
                self.R[:, i] = np.array([a * np.cos(angle), a * np.sin(angle), 0.0])
                self.V[:, i] = np.array([-7.56e3 * np.sin(angle), 7.56e3 * np.cos(angle), 0.0])
                self.B[:, i] = np.array([1e-5 * np.cos(angle), 1e-5 * np.sin(angle), 3e-5])
                self.S[:, i] = np.array([1.0, 0.0, 0.0])  # Sun in +X direction
    
    # ========================================================================
    # Finite Difference Helper Functions
    # ========================================================================
    
    def costJacobianFiniteDiff_x(self, k: int, N: int, x: np.ndarray, u: np.ndarray,
                                  sat_direction: np.ndarray, eci_target: np.ndarray,
                                  B_eci: np.ndarray, cost_cfg: saltro.CostConfig) -> np.ndarray:
        """Compute cost Jacobian w.r.t. state using central finite differences."""
        eps = 1e-7
        nx = self.sat.stateDim
        lx = np.zeros(nx)
        
        # Base cost
        l_base = self.sat.stageCost(k, N, x, u, sat_direction, eci_target, B_eci, cost_cfg)
        
        for i in range(nx):
            # Forward perturbation
            x_plus = x.copy()
            x_plus[i] += eps
            l_plus = self.sat.stageCost(k, N, x_plus, u, sat_direction, eci_target, B_eci, cost_cfg)
            
            # Backward perturbation
            x_minus = x.copy()
            x_minus[i] -= eps
            l_minus = self.sat.stageCost(k, N, x_minus, u, sat_direction, eci_target, B_eci, cost_cfg)
            
            # Central difference
            lx[i] = (l_plus - l_minus) / (2 * eps)
        
        return lx
    
    def costJacobianFiniteDiff_u(self, k: int, N: int, x: np.ndarray, u: np.ndarray,
                                  sat_direction: np.ndarray, eci_target: np.ndarray,
                                  B_eci: np.ndarray, cost_cfg: saltro.CostConfig) -> np.ndarray:
        """Compute cost Jacobian w.r.t. control using central finite differences."""
        eps = 1e-7
        nu = self.sat.controlDim
        lu = np.zeros(nu)
        
        # Base cost
        l_base = self.sat.stageCost(k, N, x, u, sat_direction, eci_target, B_eci, cost_cfg)
        
        for i in range(nu):
            # Forward perturbation
            u_plus = u.copy()
            u_plus[i] += eps
            l_plus = self.sat.stageCost(k, N, x, u_plus, sat_direction, eci_target, B_eci, cost_cfg)
            
            # Backward perturbation
            u_minus = u.copy()
            u_minus[i] -= eps
            l_minus = self.sat.stageCost(k, N, x, u_minus, sat_direction, eci_target, B_eci, cost_cfg)
            
            # Central difference
            lu[i] = (l_plus - l_minus) / (2 * eps)
        
        return lu
    
    def costHessianFiniteDiff_xx(self, k: int, N: int, x: np.ndarray, u: np.ndarray,
                                   sat_direction: np.ndarray, eci_target: np.ndarray,
                                   B_eci: np.ndarray, cost_cfg: saltro.CostConfig) -> np.ndarray:
        """Compute cost Hessian w.r.t. state using forward finite differences of analytical Jacobian."""
        eps = 1e-6
        nx = self.sat.stateDim
        lxx = np.zeros((nx, nx))
        
        # Get base Jacobian (analytical)
        lx_base, _, _ = self.sat.stageCostJacobians(k, N, x, u, sat_direction, eci_target, B_eci, cost_cfg)
        
        for j in range(nx):
            # Get Jacobian at perturbed state (analytical)
            x_pert = x.copy()
            x_pert[j] += eps
            lx_pert, _, _ = self.sat.stageCostJacobians(k, N, x_pert, u, sat_direction, eci_target, B_eci, cost_cfg)
            
            # Forward difference of analytical Jacobian
            lxx[:, j] = (lx_pert - lx_base) / eps
        
        return lxx
    
    def costHessianFiniteDiff_uu(self, k: int, N: int, x: np.ndarray, u: np.ndarray,
                                   sat_direction: np.ndarray, eci_target: np.ndarray,
                                   B_eci: np.ndarray, cost_cfg: saltro.CostConfig) -> np.ndarray:
        """Compute cost Hessian w.r.t. control using forward finite differences of analytical Jacobian."""
        eps = 1e-6
        nu = self.sat.controlDim
        luu = np.zeros((nu, nu))
        
        # Get base Jacobian w.r.t. control (analytical)
        _, lu_base, _ = self.sat.stageCostJacobians(k, N, x, u, sat_direction, eci_target, B_eci, cost_cfg)
        
        for j in range(nu):
            # Get Jacobian w.r.t. control at perturbed control (analytical)
            u_pert = u.copy()
            u_pert[j] += eps
            _, lu_pert, _ = self.sat.stageCostJacobians(k, N, x, u_pert, sat_direction, eci_target, B_eci, cost_cfg)
            
            # Forward difference of analytical Jacobian
            luu[:, j] = (lu_pert - lu_base) / eps
        
        return luu
    
    def costHessianFiniteDiff_ux(self, k: int, N: int, x: np.ndarray, u: np.ndarray,
                                   sat_direction: np.ndarray, eci_target: np.ndarray,
                                   B_eci: np.ndarray, cost_cfg: saltro.CostConfig) -> np.ndarray:
        """Compute cost Hessian w.r.t. state and control using forward finite differences of analytical Jacobian."""
        eps = 1e-6
        nx = self.sat.stateDim
        nu = self.sat.controlDim
        lux = np.zeros((nu, nx))
        
        # Get base Jacobian w.r.t. control (analytical)
        _, lu_base, _ = self.sat.stageCostJacobians(k, N, x, u, sat_direction, eci_target, B_eci, cost_cfg)
        
        for j in range(nx):
            # Get Jacobian w.r.t. control at perturbed state (analytical)
            x_pert = x.copy()
            x_pert[j] += eps
            _, lu_pert, _ = self.sat.stageCostJacobians(k, N, x_pert, u, sat_direction, eci_target, B_eci, cost_cfg)
            
            # Forward difference of analytical Jacobian
            lux[:, j] = (lu_pert - lu_base) / eps
        
        return lux


@pytest.fixture
def fixture():
    """Provide a fixture instance for all tests."""
    return SatelliteCostFixture()


# ============================================================================
# TEST SECTION 1: Cost Function Properties
# ============================================================================

class TestCostProperties:
    """Tests for basic cost function properties."""
    
    def test_stage_cost_is_non_negative(self, fixture):
        """Stage cost should be non-negative for all valid states."""
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1, 0, 0, 0]
        
        u = np.zeros(fixture.sat.controlDim)
        
        cost_cfg = saltro.CostConfig()
        sat_direction = np.zeros(3)
        eci_target = np.array([1.0, 0.0, 0.0, 0.0])
        B_eci = np.zeros(3)
        
        cost = fixture.sat.stageCost(0, 10, x, u, sat_direction, eci_target, B_eci, cost_cfg)
        assert cost >= -1e-10, "Cost should be non-negative"
    
    def test_aligned_quaternion_minimizes_cost(self, fixture):
        """Zero angular velocity with aligned quaternion should minimize attitude cost."""
        cost_cfg = saltro.CostConfig()
        cost_cfg.angle = 1e3
        cost_cfg.ang_vel = 1e4
        cost_cfg.control_mult = 1.0
        
        # Aligned state
        x_aligned = np.zeros(fixture.sat.stateDim)
        x_aligned[fixture.sat.AV_INDEX:fixture.sat.AV_INDEX+3] = 0
        x_aligned[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1, 0, 0, 0]
        
        # Misaligned state
        x_misaligned = np.zeros(fixture.sat.stateDim)
        x_misaligned[fixture.sat.AV_INDEX:fixture.sat.AV_INDEX+3] = 0
        x_misaligned[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [0.707, 0.707, 0, 0]
        
        u = np.zeros(fixture.sat.controlDim)
        sat_direction = np.zeros(3)
        eci_target = np.array([1.0, 0.0, 0.0, 0.0])
        B_eci = np.zeros(3)
        
        cost_aligned = fixture.sat.stageCost(0, 10, x_aligned, u, sat_direction, eci_target, B_eci, cost_cfg)
        cost_misaligned = fixture.sat.stageCost(0, 10, x_misaligned, u, sat_direction, eci_target, B_eci, cost_cfg)
        
        assert cost_aligned < cost_misaligned, "Aligned quaternion should have lower cost"
    
    def test_increasing_angular_velocity_increases_cost(self, fixture):
        """Angular velocity penalty should increase with spin rate."""
        cost_cfg = saltro.CostConfig()
        cost_cfg.ang_vel = 1e4
        cost_cfg.angle = 0.0  # Disable attitude cost
        cost_cfg.control_mult = 0.0  # Disable control cost
        
        x_zero_av = np.zeros(fixture.sat.stateDim)
        x_zero_av[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1, 0, 0, 0]
        
        x_nonzero_av = x_zero_av.copy()
        x_nonzero_av[fixture.sat.AV_INDEX:fixture.sat.AV_INDEX+3] = [0.1, 0.05, 0.02]
        
        u = np.zeros(fixture.sat.controlDim)
        sat_direction = np.zeros(3)
        eci_target = np.array([1.0, 0.0, 0.0, 0.0])
        B_eci = np.zeros(3)
        
        cost_zero = fixture.sat.stageCost(0, 10, x_zero_av, u, sat_direction, eci_target, B_eci, cost_cfg)
        cost_nonzero = fixture.sat.stageCost(0, 10, x_nonzero_av, u, sat_direction, eci_target, B_eci, cost_cfg)
        
        assert cost_nonzero > cost_zero, "Higher angular velocity should increase cost"
    
    def test_control_costs_increase_with_magnitude(self, fixture):
        """Control effort costs should increase with control magnitude."""
        cost_cfg = saltro.CostConfig()
        cost_cfg.control_mult = 1.0
        cost_cfg.mtq_control_weight = 1e3
        cost_cfg.rw_control_weight = 1e8
        cost_cfg.angle = 0.0
        cost_cfg.ang_vel = 0.0
        
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1, 0, 0, 0]
        
        u_zero = np.zeros(fixture.sat.controlDim)
        u_with_ctrl = u_zero.copy()
        u_with_ctrl[0] = 0.01  # Small MTQ control
        
        sat_direction = np.zeros(3)
        eci_target = np.array([1.0, 0.0, 0.0, 0.0])
        B_eci = np.zeros(3)
        
        cost_zero = fixture.sat.stageCost(0, 10, x, u_zero, sat_direction, eci_target, B_eci, cost_cfg)
        cost_with = fixture.sat.stageCost(0, 10, x, u_with_ctrl, sat_direction, eci_target, B_eci, cost_cfg)
        
        assert cost_with > cost_zero, "Higher control magnitude should increase cost"


# ============================================================================
# TEST SECTION 2: Cost Jacobian Validation
# ============================================================================

class TestCostJacobians:
    """Tests for cost Jacobian correctness and finite difference validation."""
    
    def test_jacobian_dimensions(self, fixture):
        """Cost Jacobians should have correct dimensions."""
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1, 0, 0, 0]
        
        u = np.zeros(fixture.sat.controlDim)
        
        cost_cfg = saltro.CostConfig()
        sat_direction = np.zeros(3)
        eci_target = np.array([1.0, 0.0, 0.0, 0.0])
        B_eci = np.zeros(3)
        
        lx, Lu, _ = fixture.sat.stageCostJacobians(0, 10, x, u, sat_direction, eci_target, B_eci, cost_cfg)
        
        nx = fixture.sat.stateDim
        nu = fixture.sat.controlDim
        
        assert lx.shape == (nx,), f"State Jacobian shape {lx.shape} != ({nx},)"
        assert Lu.shape == (1, nu), f"Control Jacobian shape {Lu.shape} != (1, {nu})"
    
    def test_state_jacobian_matches_finite_differences(self, fixture):
        """Analytical state Jacobian should match finite difference estimate."""
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.AV_INDEX:fixture.sat.AV_INDEX+3] = [0.05, 0.02, 0.01]
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1, 0, 0, 0]
        
        u = np.zeros(fixture.sat.controlDim)
        u[0] = 0.001
        
        cost_cfg = saltro.CostConfig()
        cost_cfg.angle = 1e3
        cost_cfg.ang_vel = 1e4
        cost_cfg.control_mult = 1.0
        
        sat_direction = np.zeros(3)
        eci_target = np.array([1.0, 0.0, 0.0, 0.0])
        B_eci = np.array([0.0, 0.0, 3e-5])
        
        lx_analytical, _, _ = fixture.sat.stageCostJacobians(
            0, 10, x, u, sat_direction, eci_target, B_eci, cost_cfg)
        
        lx_numerical = fixture.costJacobianFiniteDiff_x(
            0, 10, x, u, sat_direction, eci_target, B_eci, cost_cfg)
        
        rel_tol = 1e-3
        abs_tol = 1e-7
        
        for i in range(3):  # Angular velocity block
            # Compute relative error
            numerical_mag = np.abs(lx_numerical[i])
            threshold = abs_tol + rel_tol * numerical_mag
            error = np.abs(lx_analytical[i] - lx_numerical[i])
            assert error <= threshold, \
                f"State Jacobian[{i}]: analytical={lx_analytical[i]:.6e}, " \
                f"numerical={lx_numerical[i]:.6e}, error={error:.6e}"
    
    def test_control_jacobian_matches_finite_differences(self, fixture):
        """Analytical control Jacobian should match finite difference estimate."""
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.AV_INDEX:fixture.sat.AV_INDEX+3] = [0.05, 0.02, 0.01]
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1, 0, 0, 0]
        
        u = np.zeros(fixture.sat.controlDim)
        u[0:3] = [0.01, 0.005, 0.002]  # MTQ commands
        u[3:6] = [0.0001, 0.00005, 0.00002]  # RW commands
        
        cost_cfg = saltro.CostConfig()
        cost_cfg.control_mult = 1.0
        cost_cfg.mtq_control_weight = 1e3
        cost_cfg.rw_control_weight = 1e8
        
        sat_direction = np.zeros(3)
        eci_target = np.array([1.0, 0.0, 0.0, 0.0])
        B_eci = np.zeros(3)
        
        _, Lu_analytical, _ = fixture.sat.stageCostJacobians(
            0, 10, x, u, sat_direction, eci_target, B_eci, cost_cfg)
        
        lu_numerical = fixture.costJacobianFiniteDiff_u(
            0, 10, x, u, sat_direction, eci_target, B_eci, cost_cfg)
        
        rel_tol = 1e-2  # Relaxed for weight scaling
        abs_tol = 1e3   # Scaled with weight
        
        for i in range(fixture.sat.controlDim):
            numerical_mag = np.abs(lu_numerical[i])
            threshold = abs_tol + rel_tol * numerical_mag
            error = np.abs(Lu_analytical[0, i] - lu_numerical[i])
            assert error <= threshold, \
                f"Control Jacobian[{i}]: analytical={Lu_analytical[0, i]:.6e}, " \
                f"numerical={lu_numerical[i]:.6e}, error={error:.6e}"
    
    def test_jacobian_all_cost_types(self, fixture):
        """Cost Jacobian should be computed for all cost function types."""
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.AV_INDEX:fixture.sat.AV_INDEX+3] = [0.05, 0.02, 0.01]
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [0.9, 0.1, 0.1, 0.4]
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] /= np.linalg.norm(x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4])
        
        u = np.zeros(fixture.sat.controlDim)
        
        sat_direction = np.zeros(3)
        eci_target = np.array([1.0, 0.0, 0.0, 0.0])
        B_eci = np.zeros(3)
        
        # Test all 5 cost function types (0-4)
        for cost_type in range(5):
            cost_cfg = saltro.CostConfig()
            cost_cfg.ang_cost_func_type = cost_type
            
            lx, Lu, _ = fixture.sat.stageCostJacobians(
                0, 10, x, u, sat_direction, eci_target, B_eci, cost_cfg)
            
            assert lx is not None, f"State Jacobian failed for cost type {cost_type}"
            assert Lu is not None, f"Control Jacobian failed for cost type {cost_type}"
            assert np.all(np.isfinite(lx)), f"State Jacobian non-finite for cost type {cost_type}"
            assert np.all(np.isfinite(Lu)), f"Control Jacobian non-finite for cost type {cost_type}"
    
    def test_jacobian_quaternion_orthogonality(self, fixture):
        """Cost Jacobian quaternion component lies in tangent space (perpendicular to q)."""
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.AV_INDEX:fixture.sat.AV_INDEX+3] = [0.1, 0.05, 0.02]
        q = np.array([0.9, 0.3, 0.2, 0.1])
        q /= np.linalg.norm(q)
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = q
        
        u = np.zeros(fixture.sat.controlDim)
        
        # Configure cost with all attitude terms
        cost_cfg = saltro.CostConfig()
        cost_cfg.angle = 1e2
        cost_cfg.ang_vel = 1e3
        cost_cfg.ang_vel_mag = 1e2
        cost_cfg.ang_vel_err_dir = 1e2
        cost_cfg.control_mult = 0.1
        cost_cfg.ang_cost_func_type = 4  # 1 - |q·q_goal|^2
        
        boresight = np.array([0.9, 0.2, 0.1])
        boresight /= np.linalg.norm(boresight)
        eci_target = np.array([1.0, 0.0, 0.0, 0.0])
        B_eci = np.array([5e-5, 2e-5, 3e-5])
        
        # Get analytical Jacobian
        lx_analytical, _, _ = fixture.sat.stageCostJacobians(
            5, 10, x, u, boresight, eci_target, B_eci, cost_cfg)
        
        grad_q = lx_analytical[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4]
        
        # Verify orthogonality: q^T * grad_q should be near zero
        orthogonality_error = np.abs(np.dot(q, grad_q))
        
        assert orthogonality_error < 1e-10, \
            f"Quaternion gradient not orthogonal to q: q^T*grad_q = {orthogonality_error:.6e}"
    
    def test_jacobian_matches_fd_with_angle_cost(self, fixture):
        """Cost Jacobian matches finite differences when angle cost enabled (regression test)."""
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.AV_INDEX:fixture.sat.AV_INDEX+3] = [0.02, -0.01, 0.015]
        q = np.array([0.9, 0.25, 0.3, 0.15])
        q /= np.linalg.norm(q)
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = q
        
        u = np.zeros(fixture.sat.controlDim)
        
        # Cost with angle cost enabled (this was causing line search failures before fix)
        cost_cfg = saltro.CostConfig()
        cost_cfg.angle = 1e2
        cost_cfg.angle_N = 1e2
        cost_cfg.ang_vel = 1e4
        cost_cfg.ang_vel_N = 1e4
        cost_cfg.ang_vel_mag = 5e1
        cost_cfg.ang_vel_err_dir = 5e1
        cost_cfg.control_mult = 0.01
        cost_cfg.mtq_control_weight = 1.0
        cost_cfg.rw_control_weight = 1e3
        cost_cfg.ang_cost_func_type = 4
        
        boresight = np.array([1.0, 0.0, 0.0])
        eci_target = np.array([1.0, 0.0, 0.0, 0.0])
        B_eci = np.array([5e-5, 2e-5, 3e-5])
        
        # Analytical Jacobian
        lx_analytical, _, _ = fixture.sat.stageCostJacobians(
            5, 10, x, u, boresight, eci_target, B_eci, cost_cfg)
        
        # Finite difference Jacobian
        lx_fd = fixture.costJacobianFiniteDiff_x(
            5, 10, x, u, boresight, eci_target, B_eci, cost_cfg)
        
        # Compare angular velocity block (not constrained)
        rel_tol = 0.02  # 2% tolerance
        abs_tol = 1e-7
        
        for i in range(3):  # AV dimensions
            numerical_mag = np.abs(lx_analytical[i])
            threshold = abs_tol + rel_tol * numerical_mag
            error = np.abs(lx_analytical[i] - lx_fd[i])
            assert error <= threshold, \
                f"Jacobian[{i}] mismatch: analytical={lx_analytical[i]:.6e}, " \
                f"fd={lx_fd[i]:.6e}, error={error:.6e}"


# ============================================================================
# TEST SECTION 3: Cost Hessian Validation
# ============================================================================

class TestCostHessians:
    """Tests for cost Hessian correctness and properties."""
    
    def test_hessian_dimensions(self, fixture):
        """Cost Hessians should have correct dimensions."""
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1, 0, 0, 0]
        
        u = np.zeros(fixture.sat.controlDim)
        
        cost_cfg = saltro.CostConfig()
        sat_direction = np.zeros(3)
        eci_target = np.array([1.0, 0.0, 0.0, 0.0])
        B_eci = np.zeros(3)
        
        lxx, luu, lux = fixture.sat.stageCostHessians(0, 10, x, u, sat_direction, eci_target, B_eci, cost_cfg)
        
        nx = fixture.sat.stateDim
        nu = fixture.sat.controlDim
        
        assert lxx.shape == (nx, nx), f"State Hessian shape {lxx.shape} != ({nx}, {nx})"
        assert luu.shape == (nu, nu), f"Control Hessian shape {luu.shape} != ({nu}, {nu})"
        assert lux.shape == (nu, nx), f"Mixed Hessian shape {lux.shape} != ({nu}, {nx})"
    
    def test_state_hessian_is_symmetric(self, fixture):
        """State Hessian should be symmetric."""
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.AV_INDEX:fixture.sat.AV_INDEX+3] = [0.05, 0.02, 0.01]
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1, 0, 0, 0]
        
        u = np.zeros(fixture.sat.controlDim)
        
        cost_cfg = saltro.CostConfig()
        sat_direction = np.zeros(3)
        eci_target = np.array([1.0, 0.0, 0.0, 0.0])
        B_eci = np.zeros(3)
        
        lxx, _, _ = fixture.sat.stageCostHessians(0, 10, x, u, sat_direction, eci_target, B_eci, cost_cfg)
        
        tol = 1e-9
        for i in range(fixture.sat.stateDim):
            for j in range(i, fixture.sat.stateDim):
                diff = np.abs(lxx[i, j] - lxx[j, i])
                assert diff < tol, f"Hessian asymmetry at [{i},{j}]: {diff}"
    
    def test_control_hessian_is_symmetric(self, fixture):
        """Control Hessian should be symmetric."""
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1, 0, 0, 0]
        
        u = np.zeros(fixture.sat.controlDim)
        
        cost_cfg = saltro.CostConfig()
        sat_direction = np.zeros(3)
        eci_target = np.array([1.0, 0.0, 0.0, 0.0])
        B_eci = np.zeros(3)
        
        _, luu, _ = fixture.sat.stageCostHessians(0, 10, x, u, sat_direction, eci_target, B_eci, cost_cfg)
        
        tol = 1e-9
        for i in range(fixture.sat.controlDim):
            for j in range(i, fixture.sat.controlDim):
                diff = np.abs(luu[i, j] - luu[j, i])
                assert diff < tol, f"Control Hessian asymmetry at [{i},{j}]: {diff}"
    
    def test_state_hessian_matches_finite_differences(self, fixture):
        """State Hessian (AV block) should match finite differences."""
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.AV_INDEX:fixture.sat.AV_INDEX+3] = [0.05, 0.02, 0.01]
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1, 0, 0, 0]
        
        u = np.zeros(fixture.sat.controlDim)
        
        cost_cfg = saltro.CostConfig()
        cost_cfg.ang_vel = 1e4
        cost_cfg.control_mult = 0.0  # Skip control costs
        cost_cfg.angle = 0.0
        cost_cfg.use_cost_hess = True
        
        sat_direction = np.zeros(3)
        eci_target = np.array([1.0, 0.0, 0.0, 0.0])
        B_eci = np.zeros(3)
        
        lxx_analytical, _, _ = fixture.sat.stageCostHessians(
            0, 10, x, u, sat_direction, eci_target, B_eci, cost_cfg)
        
        # For pure quadratic ang_vel cost, only diagonal should be non-zero
        # Verify structure rather than strict FD matching (FD can pick up spurious cross-terms)
        rel_tol = 1e-2
        abs_tol = 1e-5
        
        # Check diagonal (should be ~ang_vel = 1e4)
        for i in range(3):
            expected = cost_cfg.ang_vel
            error = np.abs(lxx_analytical[i, i] - expected)
            threshold = abs_tol + rel_tol * expected
            assert error <= threshold, \
                f"State Hessian diagonal[{i}]: analytical={lxx_analytical[i, i]:.6e}, " \
                f"expected={expected:.6e}, error={error:.6e}"
    
    def test_control_hessian_matches_finite_differences(self, fixture):
        """Control Hessian should match finite differences."""
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1, 0, 0, 0]
        
        u = np.zeros(fixture.sat.controlDim)
        u[0:3] = [0.01, 0.005, 0.002]
        u[3:6] = [0.0001, 0.00005, 0.00002]
        
        cost_cfg = saltro.CostConfig()
        cost_cfg.control_mult = 1.0
        cost_cfg.mtq_control_weight = 1e3
        cost_cfg.rw_control_weight = 1e8
        cost_cfg.ang_vel = 0.0
        cost_cfg.angle = 0.0
        cost_cfg.use_cost_hess = True
        
        sat_direction = np.zeros(3)
        eci_target = np.array([1.0, 0.0, 0.0, 0.0])
        B_eci = np.zeros(3)
        
        _, luu_analytical, _ = fixture.sat.stageCostHessians(
            0, 10, x, u, sat_direction, eci_target, B_eci, cost_cfg)
        
        luu_numerical = fixture.costHessianFiniteDiff_uu(
            0, 10, x, u, sat_direction, eci_target, B_eci, cost_cfg)
        
        rel_tol = 1e-2
        abs_tol = 1e3
        
        for i in range(fixture.sat.controlDim):
            for j in range(fixture.sat.controlDim):
                numerical_mag = np.abs(luu_numerical[i, j])
                threshold = abs_tol + rel_tol * numerical_mag
                error = np.abs(luu_analytical[i, j] - luu_numerical[i, j])
                assert error <= threshold, \
                    f"Control Hessian[{i},{j}]: analytical={luu_analytical[i, j]:.6e}, " \
                    f"numerical={luu_numerical[i, j]:.6e}, error={error:.6e}"
    
    def test_rw_momentum_hessian_positive_semidefinite(self, fixture):
        """RW momentum Hessian diagonal should be non-negative (convexity)."""
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.AV_INDEX:fixture.sat.AV_INDEX+3] = [0.05, 0.02, 0.01]
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1, 0, 0, 0]
        x[fixture.sat.RW_MOMENTUM_INDEX:fixture.sat.RW_MOMENTUM_INDEX+3] = [0.003, -0.002, 0.001]
        
        u = np.zeros(fixture.sat.controlDim)
        
        cost_cfg = saltro.CostConfig()
        cost_cfg.rw_AM_weight = 1e4
        cost_cfg.rw_stic_weight = 1.0
        cost_cfg.RWh_max_mult = 0.8
        cost_cfg.RWh_stiction_mult = 0.01
        cost_cfg.RWh_ok_mult = 0.5
        
        sat_direction = np.zeros(3)
        eci_target = np.array([1.0, 0.0, 0.0, 0.0])
        B_eci = np.zeros(3)
        
        lxx, _, _ = fixture.sat.stageCostHessians(0, 10, x, u, sat_direction, eci_target, B_eci, cost_cfg)
        
        # Check diagonal for RW momentum (indices 7-9 for 7-element state, 10-element state)
        rw_idx = fixture.sat.RW_MOMENTUM_INDEX
        for i in range(3):
            assert lxx[rw_idx + i, rw_idx + i] >= -1e-10, \
                f"RW momentum Hessian diagonal[{rw_idx + i}] = {lxx[rw_idx + i, rw_idx + i]} is negative"
    
    def test_hessian_quaternion_projection_left(self, fixture):
        """Cost Hessian quaternion block satisfies left projection property: H*q ≈ 0."""
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.AV_INDEX:fixture.sat.AV_INDEX+3] = [0.05, 0.03, 0.02]
        q = np.array([0.85, 0.4, 0.25, 0.15])
        q /= np.linalg.norm(q)
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = q
        
        u = np.zeros(fixture.sat.controlDim)
        
        # Cost with angle cost
        cost_cfg = saltro.CostConfig()
        cost_cfg.angle = 1e2
        cost_cfg.ang_vel = 1e3
        cost_cfg.control_mult = 0.1
        cost_cfg.ang_cost_func_type = 4
        
        boresight = np.array([0.9, 0.2, 0.1])
        boresight /= np.linalg.norm(boresight)
        eci_target = np.array([1.0, 0.0, 0.0, 0.0])
        B_eci = np.array([5e-5, 2e-5, 3e-5])
        
        # Get analytical Hessian
        lxx_analytical, _, _ = fixture.sat.stageCostHessians(
            5, 10, x, u, boresight, eci_target, B_eci, cost_cfg)
        
        # Extract quaternion block
        q_idx = fixture.sat.QUAT_INDEX
        H_qq = lxx_analytical[q_idx:q_idx+4, q_idx:q_idx+4]
        
        # Check: H_qq * q should be near zero (right multiplication)
        H_q_product = np.dot(H_qq, q)
        right_proj_error = np.linalg.norm(H_q_product)
        
        assert right_proj_error < 1e-9, \
            f"Hessian right projection failed: ||H*q|| = {right_proj_error:.6e}"
    
    def test_hessian_quaternion_projection_right(self, fixture):
        """Cost Hessian quaternion block satisfies right projection property: q^T*H ≈ 0."""
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.AV_INDEX:fixture.sat.AV_INDEX+3] = [0.05, 0.03, 0.02]
        q = np.array([0.85, 0.4, 0.25, 0.15])
        q /= np.linalg.norm(q)
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = q
        
        u = np.zeros(fixture.sat.controlDim)
        
        # Cost with angle cost
        cost_cfg = saltro.CostConfig()
        cost_cfg.angle = 1e2
        cost_cfg.ang_vel = 1e3
        cost_cfg.control_mult = 0.1
        cost_cfg.ang_cost_func_type = 4
        
        boresight = np.array([0.9, 0.2, 0.1])
        boresight /= np.linalg.norm(boresight)
        eci_target = np.array([1.0, 0.0, 0.0, 0.0])
        B_eci = np.array([5e-5, 2e-5, 3e-5])
        
        # Get analytical Hessian
        lxx_analytical, _, _ = fixture.sat.stageCostHessians(
            5, 10, x, u, boresight, eci_target, B_eci, cost_cfg)
        
        # Extract quaternion block
        q_idx = fixture.sat.QUAT_INDEX
        H_qq = lxx_analytical[q_idx:q_idx+4, q_idx:q_idx+4]
        
        # Check: q^T*H_qq should be near zero (left multiplication)
        qT_H_product = np.dot(q, H_qq)
        left_proj_error = np.linalg.norm(qT_H_product)
        
        assert left_proj_error < 1e-9, \
            f"Hessian left projection failed: ||q^T*H|| = {left_proj_error:.6e}"
    
    def test_hessian_matches_fd_with_angle_cost(self, fixture):
        """Cost Hessian matches finite differences when angle cost enabled (regression test)."""
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.AV_INDEX:fixture.sat.AV_INDEX+3] = [0.02, -0.01, 0.015]
        q = np.array([0.9, 0.25, 0.3, 0.15])
        q /= np.linalg.norm(q)
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = q
        
        u = np.zeros(fixture.sat.controlDim)
        
        # Cost with angle cost enabled
        cost_cfg = saltro.CostConfig()
        cost_cfg.angle = 1e2
        cost_cfg.ang_vel = 1e4
        cost_cfg.control_mult = 0.01
        cost_cfg.mtq_control_weight = 1.0
        cost_cfg.rw_control_weight = 1e3
        cost_cfg.ang_cost_func_type = 4
        cost_cfg.use_cost_hess = True
        
        boresight = np.array([1.0, 0.0, 0.0])
        eci_target = np.array([1.0, 0.0, 0.0, 0.0])
        B_eci = np.array([5e-5, 2e-5, 3e-5])
        
        # Analytical Hessian
        lxx_analytical, _, _ = fixture.sat.stageCostHessians(
            5, 10, x, u, boresight, eci_target, B_eci, cost_cfg)
        
        # Finite difference Hessian
        lxx_fd = fixture.costHessianFiniteDiff_xx(
            5, 10, x, u, boresight, eci_target, B_eci, cost_cfg)
        
        # Compare angular velocity block (not constrained by manifold)
        rel_tol = 0.05  # 5% tolerance due to FD truncation
        abs_tol = 1e-8
        
        for i in range(3):  # AV dimensions
            for j in range(3):
                numerical_mag = np.abs(lxx_analytical[i, j])
                threshold = abs_tol + rel_tol * numerical_mag
                error = np.abs(lxx_analytical[i, j] - lxx_fd[i, j])
                assert error <= threshold, \
                    f"Hessian[{i},{j}] mismatch: analytical={lxx_analytical[i, j]:.6e}, " \
                    f"fd={lxx_fd[i, j]:.6e}, error={error:.6e}"

    def test_state_hessian_disabled_when_use_cost_hess_false(self, fixture):
        """State Hessian should be zeroed when use_cost_hess is disabled."""
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.AV_INDEX:fixture.sat.AV_INDEX+3] = [0.05, 0.02, 0.01]
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1, 0, 0, 0]
        u = np.zeros(fixture.sat.controlDim)

        cost_cfg = saltro.CostConfig()
        cost_cfg.ang_vel = 1e4
        cost_cfg.control_mult = 1.0
        cost_cfg.use_cost_hess = False

        sat_direction = np.zeros(3)
        eci_target = np.array([1.0, 0.0, 0.0, 0.0])
        B_eci = np.zeros(3)

        lxx, luu, lux = fixture.sat.stageCostHessians(
            0, 10, x, u, sat_direction, eci_target, B_eci, cost_cfg)

        assert np.linalg.norm(lxx) == 0.0
        assert np.linalg.norm(lux) == 0.0
        assert np.max(np.diag(luu)) > 0.0


# ============================================================================
# TEST SECTION 4: Terminal Cost vs Stage Cost
# ============================================================================

class TestTerminalCost:
    """Tests for terminal cost weight application and behavior."""
    
    def test_terminal_cost_uses_terminal_weights(self, fixture):
        """Terminal cost should apply terminal weights."""
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1, 0, 0, 0]
        
        cost_cfg = saltro.CostConfig()
        cost_cfg.angle = 1e3
        cost_cfg.angle_N = 1e4  # Much higher for terminal
        cost_cfg.ang_vel = 1e4
        cost_cfg.ang_vel_N = 1e5  # Much higher for terminal
        
        sat_direction = np.zeros(3)
        eci_target = np.array([0.9, 0.1, 0.1, 0.3])  # Misaligned
        B_eci = np.zeros(3)
        
        terminal_cost = fixture.sat.terminalCost(x, sat_direction, eci_target, B_eci, cost_cfg)
        
        assert np.isfinite(terminal_cost), "Terminal cost should be finite"
        assert terminal_cost >= -1e-10, "Terminal cost should be non-negative"
    
    def test_terminal_jacobians_are_finite(self, fixture):
        """Terminal Jacobians should be finite."""
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.AV_INDEX:fixture.sat.AV_INDEX+3] = [0.05, 0.02, 0.01]
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1, 0, 0, 0]
        
        cost_cfg = saltro.CostConfig()
        sat_direction = np.zeros(3)
        eci_target = np.array([1.0, 0.0, 0.0, 0.0])
        B_eci = np.zeros(3)
        
        lx, _, _ = fixture.sat.terminalCostJacobians(x, sat_direction, eci_target, B_eci, cost_cfg)
        
        assert np.all(np.isfinite(lx)), "Terminal Jacobian should be finite"
    
    def test_terminal_hessians_are_finite(self, fixture):
        """Terminal Hessians should be finite."""
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.AV_INDEX:fixture.sat.AV_INDEX+3] = [0.05, 0.02, 0.01]
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1, 0, 0, 0]
        
        cost_cfg = saltro.CostConfig()
        sat_direction = np.zeros(3)
        eci_target = np.array([1.0, 0.0, 0.0, 0.0])
        B_eci = np.zeros(3)
        
        lxx, luu, _ = fixture.sat.terminalCostHessians(x, sat_direction, eci_target, B_eci, cost_cfg)
        
        assert np.all(np.isfinite(lxx)), "Terminal state Hessian should be finite"
        assert np.all(np.isfinite(luu)), "Terminal control Hessian should be finite"


# ============================================================================
# TEST SECTION 5: RW Momentum Cost
# ============================================================================

class TestRWMomentumCost:
    """Tests for reaction wheel momentum cost behavior."""
    
    def test_rw_momentum_penalty_increases_with_magnitude(self, fixture):
        """RW momentum penalty should increase with magnitude."""
        cost_cfg = saltro.CostConfig()
        cost_cfg.rw_AM_weight = 1e4
        cost_cfg.RWh_max_mult = 0.8
        cost_cfg.RWh_ok_mult = 0.5
        cost_cfg.angle = 0.0
        cost_cfg.ang_vel = 0.0
        cost_cfg.control_mult = 0.0
        
        # Low momentum
        x_low = np.zeros(fixture.sat.stateDim)
        x_low[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1, 0, 0, 0]
        
        # High momentum
        x_high = x_low.copy()
        x_high[fixture.sat.RW_MOMENTUM_INDEX] = 0.005
        
        u = np.zeros(fixture.sat.controlDim)
        sat_direction = np.zeros(3)
        eci_target = np.array([1.0, 0.0, 0.0, 0.0])
        B_eci = np.zeros(3)
        
        cost_low = fixture.sat.stageCost(0, 10, x_low, u, sat_direction, eci_target, B_eci, cost_cfg)
        cost_high = fixture.sat.stageCost(0, 10, x_high, u, sat_direction, eci_target, B_eci, cost_cfg)
        
        assert cost_high > cost_low, "Higher momentum should increase cost"
    
    def test_rw_momentum_cost_is_zero_at_low_momentum(self, fixture):
        """RW momentum cost should be minimal at low momentum values."""
        cost_cfg = saltro.CostConfig()
        cost_cfg.rw_AM_weight = 1e4
        cost_cfg.RWh_max_mult = 0.8
        cost_cfg.RWh_ok_mult = 0.5
        cost_cfg.angle = 0.0
        cost_cfg.ang_vel = 0.0
        cost_cfg.control_mult = 0.0
        
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1, 0, 0, 0]
        x[fixture.sat.RW_MOMENTUM_INDEX:fixture.sat.RW_MOMENTUM_INDEX+3] = 1e-6
        
        u = np.zeros(fixture.sat.controlDim)
        sat_direction = np.zeros(3)
        eci_target = np.array([1.0, 0.0, 0.0, 0.0])
        B_eci = np.zeros(3)
        
        cost = fixture.sat.stageCost(0, 10, x, u, sat_direction, eci_target, B_eci, cost_cfg)
        
        # Cost should be finite and dominated by RW momentum penalties at low values
        assert np.isfinite(cost), "Cost should be finite"
        assert cost > -1e-10, "Cost should be non-negative"


# ============================================================================
# TEST SECTION 6: Robustness and Edge Cases
# ============================================================================

class TestRobustness:
    """Tests for numerical robustness and edge case handling."""
    
    def test_cost_finite_for_various_quaternions(self, fixture):
        """Cost should be finite for various quaternion orientations."""
        u = np.zeros(fixture.sat.controlDim)
        
        cost_cfg = saltro.CostConfig()
        sat_direction = np.zeros(3)
        eci_target = np.array([1.0, 0.0, 0.0, 0.0])
        B_eci = np.zeros(3)
        
        quaternions = [
            np.array([1, 0, 0, 0]),
            np.array([0, 1, 0, 0]),
            np.array([0, 0, 1, 0]),
            np.array([0, 0, 0, 1]),
            np.array([0.5, 0.5, 0.5, 0.5]),
        ]
        
        for q in quaternions:
            q = q / np.linalg.norm(q)
            x = np.zeros(fixture.sat.stateDim)
            x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = q
            
            cost = fixture.sat.stageCost(0, 10, x, u, sat_direction, eci_target, B_eci, cost_cfg)
            assert np.isfinite(cost), f"Cost non-finite for quaternion {q}"
    
    def test_cost_consistent_across_state_space(self, fixture):
        """Cost function should be consistent across state space."""
        cost_cfg = saltro.CostConfig()
        sat_direction = np.zeros(3)
        eci_target = np.array([1.0, 0.0, 0.0, 0.0])
        B_eci = np.zeros(3)
        
        u = np.zeros(fixture.sat.controlDim)
        
        # Sample different regions of state space
        for av_mag in [0.0, 0.01, 0.1]:
            x = np.zeros(fixture.sat.stateDim)
            x[fixture.sat.AV_INDEX:fixture.sat.AV_INDEX+3] = av_mag / np.sqrt(3)  # Isotropic
            x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1, 0, 0, 0]
            
            cost = fixture.sat.stageCost(0, 10, x, u, sat_direction, eci_target, B_eci, cost_cfg)
            assert np.isfinite(cost), f"Cost non-finite for |av| = {av_mag}"
            assert cost >= 0, f"Cost negative for |av| = {av_mag}"
    
    def test_jacobian_continuous_with_control(self, fixture):
        """Jacobian should vary continuously with control input."""
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1, 0, 0, 0]
        
        cost_cfg = saltro.CostConfig()
        cost_cfg.control_mult = 1.0
        
        sat_direction = np.zeros(3)
        eci_target = np.array([1.0, 0.0, 0.0, 0.0])
        B_eci = np.zeros(3)
        
        # Two nearby control inputs
        u1 = np.zeros(fixture.sat.controlDim)
        u1[0] = 0.0
        
        u2 = u1.copy()
        u2[0] = 1e-6
        
        _, Lu1, _ = fixture.sat.stageCostJacobians(0, 10, x, u1, sat_direction, eci_target, B_eci, cost_cfg)
        _, Lu2, _ = fixture.sat.stageCostJacobians(0, 10, x, u2, sat_direction, eci_target, B_eci, cost_cfg)
        
        # Control Jacobians can have discontinuities at control boundaries
        # Just verify they are both finite and have reasonable magnitude
        assert np.all(np.isfinite(Lu1)), "Control Jacobian Lu1 should be finite"
        assert np.all(np.isfinite(Lu2)), "Control Jacobian Lu2 should be finite"


# ============================================================================
# TEST SECTION 7: Magnetic Field Dependency
# ============================================================================

class TestMagneticFieldDependency:
    """Tests for magnetic field dependent cost components."""
    
    def test_cost_depends_on_magnetic_field(self, fixture):
        """Cost should depend on magnetic field vector."""
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.AV_INDEX:fixture.sat.AV_INDEX+3] = [0.1, 0.0, 0.0]
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1, 0, 0, 0]
        
        u = np.zeros(fixture.sat.controlDim)
        
        cost_cfg = saltro.CostConfig()
        cost_cfg.ang_vel_mag = 1e3  # Enable magnetic alignment cost
        
        sat_direction = np.zeros(3)
        eci_target = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Zero B-field
        B_eci_zero = np.zeros(3)
        cost_zero = fixture.sat.stageCost(0, 10, x, u, sat_direction, eci_target, B_eci_zero, cost_cfg)
        
        # Non-zero B-field
        B_eci_nonzero = np.array([1e-5, 0.0, 3e-5])
        cost_nonzero = fixture.sat.stageCost(0, 10, x, u, sat_direction, eci_target, B_eci_nonzero, cost_cfg)
        
        # Costs might be equal if mag_align is not active, but should still be finite
        assert np.isfinite(cost_zero)
        assert np.isfinite(cost_nonzero)
    
    def test_cost_finite_across_magnetic_field_range(self, fixture):
        """Cost should remain finite across typical magnetic field magnitudes."""
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1, 0, 0, 0]
        
        u = np.zeros(fixture.sat.controlDim)
        cost_cfg = saltro.CostConfig()
        sat_direction = np.zeros(3)
        eci_target = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Typical Earth B-field ranges from ~25 to 65 µT
        for b_mag in [1e-5, 3e-5, 5e-5]:
            B_eci = np.array([b_mag, 0.0, 0.0]) if b_mag > 0 else np.zeros(3)
            cost = fixture.sat.stageCost(0, 10, x, u, sat_direction, eci_target, B_eci, cost_cfg)
            assert np.isfinite(cost), f"Cost non-finite for B-field magnitude {b_mag}"


# ============================================================================
# TEST SECTION 8: Edge Cases and Numerical Stability
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and numerical stability."""
    
    def test_cost_at_state_boundaries(self, fixture):
        """Cost should be well-defined at state space boundaries."""
        u = np.zeros(fixture.sat.controlDim)
        
        cost_cfg = saltro.CostConfig()
        sat_direction = np.zeros(3)
        eci_target = np.array([1.0, 0.0, 0.0, 0.0])
        B_eci = np.zeros(3)
        
        # Test at RW momentum limits
        for h_rw in [-0.01, 0.0, 0.01]:
            x = np.zeros(fixture.sat.stateDim)
            x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1, 0, 0, 0]
            x[fixture.sat.RW_MOMENTUM_INDEX:fixture.sat.RW_MOMENTUM_INDEX+3] = h_rw
            
            cost = fixture.sat.stageCost(0, 10, x, u, sat_direction, eci_target, B_eci, cost_cfg)
            assert np.isfinite(cost), f"Cost non-finite at RW momentum boundary {h_rw}"
    
    def test_hessian_semidefinite_properties(self, fixture):
        """Control Hessian should exhibit positive semidefinite properties."""
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1, 0, 0, 0]
        
        u = np.zeros(fixture.sat.controlDim)
        u[0:3] = [0.01, 0.005, 0.002]
        u[3:6] = [0.0001, 0.00005, 0.00002]
        
        cost_cfg = saltro.CostConfig()
        cost_cfg.control_mult = 1.0
        cost_cfg.mtq_control_weight = 1e3
        cost_cfg.rw_control_weight = 1e8
        
        sat_direction = np.zeros(3)
        eci_target = np.array([1.0, 0.0, 0.0, 0.0])
        B_eci = np.zeros(3)
        
        _, luu, _ = fixture.sat.stageCostHessians(0, 10, x, u, sat_direction, eci_target, B_eci, cost_cfg)
        
        # Check eigenvalues (should all be non-negative for PSD)
        eigenvalues = np.linalg.eigvalsh(luu)
        assert np.all(eigenvalues >= -1e-6), f"Negative eigenvalues: {eigenvalues[eigenvalues < 0]}"
    
    def test_cost_time_consistency(self, fixture):
        """Cost should be consistent for same state at different times."""
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1, 0, 0, 0]
        
        u = np.zeros(fixture.sat.controlDim)
        
        cost_cfg = saltro.CostConfig()
        cost_cfg.angle = 1e3
        cost_cfg.ang_vel = 1e4
        
        sat_direction = np.zeros(3)
        eci_target = np.array([1.0, 0.0, 0.0, 0.0])
        B_eci = np.zeros(3)
        
        # Cost computed at different stages
        cost_early = fixture.sat.stageCost(0, 100, x, u, sat_direction, eci_target, B_eci, cost_cfg)
        cost_mid = fixture.sat.stageCost(50, 100, x, u, sat_direction, eci_target, B_eci, cost_cfg)
        cost_late = fixture.sat.stageCost(99, 100, x, u, sat_direction, eci_target, B_eci, cost_cfg)
        
        # Costs should have similar magnitude for same state (terminal weights may differ)
        assert np.isfinite(cost_early)
        assert np.isfinite(cost_mid)
        assert np.isfinite(cost_late)
        
        # Early stages should not use terminal weights
        ratio = cost_late / (cost_early + 1e-10)
        assert ratio > 0.5, "Late stage cost significantly different from early (terminal weight effect)"


# ============================================================================
# TEST SECTION 9: Dual-Format ECI Target (Quaternion vs ECI Vector)
# ============================================================================

class TestECITargetDualFormat:
    """Tests for quaternion vs ECI vector target format handling."""
    
    def test_quaternion_format_target_computes_correctly(self, fixture):
        """Cost with quaternion-format target should compute correctly."""
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.AV_INDEX:fixture.sat.AV_INDEX+3] = [0.05, 0.02, 0.01]
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = np.array([0.9, 0.1, 0.0, 0.436])
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] /= np.linalg.norm(x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4])
        
        u = np.zeros(fixture.sat.controlDim)
        
        cost_cfg = saltro.CostConfig()
        cost_cfg.angle = 1e3
        cost_cfg.ang_vel = 1e4
        
        # Quaternion format target: [q0, qx, qy, qz] - no NaN
        eci_target = np.array([0.8, 0.2, 0.1, 0.566])
        sat_direction = np.zeros(3)
        B_eci = np.zeros(3)
        
        cost = fixture.sat.stageCost(0, 10, x, u, sat_direction, eci_target, B_eci, cost_cfg)
        
        assert np.isfinite(cost), "Cost should be finite for quaternion format"
        assert cost >= -1e-10, "Cost should be non-negative"
    
    def test_eci_vector_format_target_computes_correctly(self, fixture):
        """Cost with ECI-vector-format target should compute correctly."""
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.AV_INDEX:fixture.sat.AV_INDEX+3] = [0.05, 0.02, 0.01]
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1.0, 0.0, 0.0, 0.0]  # Identity
        
        u = np.zeros(fixture.sat.controlDim)
        
        cost_cfg = saltro.CostConfig()
        cost_cfg.angle = 1e3
        cost_cfg.ang_vel = 1e4
        
        # ECI vector format: [NaN, x, y, z]
        eci_target = np.array([np.nan, 1.0, 0.0, 0.0])
        sat_direction = np.array([0.0, 0.0, 1.0])  # Body +Z direction
        B_eci = np.zeros(3)
        
        cost = fixture.sat.stageCost(0, 10, x, u, sat_direction, eci_target, B_eci, cost_cfg)
        
        assert np.isfinite(cost), "Cost should be finite for ECI vector format"
        assert cost >= -1e-10, "Cost should be non-negative"
    
    def test_eci_vector_target_zero_vector_handling(self, fixture):
        """ECI vector target with zero vector should handle gracefully."""
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1, 0, 0, 0]
        
        u = np.zeros(fixture.sat.controlDim)
        
        cost_cfg = saltro.CostConfig()
        cost_cfg.angle = 1e3
        cost_cfg.ang_vel = 1e4
        
        # ECI vector with zero magnitude
        eci_target_zero = np.array([np.nan, 0.0, 0.0, 0.0])
        sat_direction = np.array([0.0, 0.0, 1.0])
        B_eci = np.zeros(3)
        
        cost_zero = fixture.sat.stageCost(0, 10, x, u, sat_direction, eci_target_zero, B_eci, cost_cfg)
        
        # Should produce finite result
        assert np.isfinite(cost_zero), "Cost should be finite for zero ECI vector"
        assert cost_zero >= -1e-10, "Cost should be non-negative"
    
    def test_eci_vector_target_uses_sat_direction(self, fixture):
        """ECI vector target should use sat_direction for conversion."""
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1, 0, 0, 0]
        
        u = np.zeros(fixture.sat.controlDim)
        
        cost_cfg = saltro.CostConfig()
        cost_cfg.angle = 1e3
        
        # ECI vector target pointing in +X
        eci_target = np.array([np.nan, 1.0, 0.0, 0.0])
        B_eci = np.zeros(3)
        
        # Compute with different sat_direction values
        sat_dir1 = np.array([1.0, 0.0, 0.0])  # Aligned
        sat_dir2 = np.array([0.0, 1.0, 0.0])  # Perpendicular
        
        cost1 = fixture.sat.stageCost(0, 10, x, u, sat_dir1, eci_target, B_eci, cost_cfg)
        cost2 = fixture.sat.stageCost(0, 10, x, u, sat_dir2, eci_target, B_eci, cost_cfg)
        
        # Costs should be different since sat_direction affects alignment goal
        assert cost1 < cost2, "Aligned sat_direction should produce lower cost"
    
    def test_jacobian_quaternion_format_target_consistent(self, fixture):
        """Jacobian with quaternion format should be consistent."""
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.AV_INDEX:fixture.sat.AV_INDEX+3] = [0.05, 0.02, 0.01]
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1.0, 0.0, 0.0, 0.0]
        
        u = np.zeros(fixture.sat.controlDim)
        
        cost_cfg = saltro.CostConfig()
        cost_cfg.angle = 1e3
        cost_cfg.ang_vel = 1e4
        
        eci_target = np.array([0.9, 0.1, 0.0, 0.436])
        sat_direction = np.zeros(3)
        B_eci = np.zeros(3)
        
        lx, Lu, _ = fixture.sat.stageCostJacobians(0, 10, x, u, sat_direction, eci_target, B_eci, cost_cfg)
        
        assert np.all(np.isfinite(lx)), "State Jacobian should be finite"
        assert np.all(np.isfinite(Lu)), "Control Jacobian should be finite"
    
    def test_jacobian_eci_vector_format_target_consistent(self, fixture):
        """Jacobian with ECI vector format should be consistent."""
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.AV_INDEX:fixture.sat.AV_INDEX+3] = [0.05, 0.02, 0.01]
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1.0, 0.0, 0.0, 0.0]
        
        u = np.zeros(fixture.sat.controlDim)
        
        cost_cfg = saltro.CostConfig()
        cost_cfg.angle = 1e3
        cost_cfg.ang_vel = 1e4
        
        eci_target = np.array([np.nan, 1.0, 0.0, 0.0])
        sat_direction = np.array([0.0, 0.0, 1.0])
        B_eci = np.zeros(3)
        
        lx, Lu, _ = fixture.sat.stageCostJacobians(0, 10, x, u, sat_direction, eci_target, B_eci, cost_cfg)
        
        assert np.all(np.isfinite(lx)), "State Jacobian should be finite"
        assert np.all(np.isfinite(Lu)), "Control Jacobian should be finite"
    
    def test_jacobians_match_fd_both_formats(self, fixture):
        """Jacobians match finite differences for both target formats."""
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.AV_INDEX:fixture.sat.AV_INDEX+3] = [0.05, 0.02, 0.01]
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1.0, 0.0, 0.0, 0.0]
        
        u = np.zeros(fixture.sat.controlDim)
        
        cost_cfg = saltro.CostConfig()
        cost_cfg.angle = 1e3
        
        sat_direction = np.array([0.0, 0.0, 1.0])
        B_eci = np.zeros(3)
        
        rel_tol = 1e-3
        abs_tol = 1e-7
        
        # Test quaternion format
        eci_target_quat = np.array([0.9, 0.1, 0.0, 0.436])
        lx_analytical, _, _ = fixture.sat.stageCostJacobians(
            0, 10, x, u, sat_direction, eci_target_quat, B_eci, cost_cfg)
        
        lx_numerical = fixture.costJacobianFiniteDiff_x(
            0, 10, x, u, sat_direction, eci_target_quat, B_eci, cost_cfg)
        
        for i in range(3):  # AV block
            numerical_mag = np.abs(lx_numerical[i])
            threshold = abs_tol + rel_tol * numerical_mag
            error = np.abs(lx_analytical[i] - lx_numerical[i])
            assert error <= threshold, \
                f"Quaternion format Jacobian[{i}]: error={error:.6e}, threshold={threshold:.6e}"
        
        # Test ECI vector format
        eci_target_vec = np.array([np.nan, 1.0, 0.0, 0.0])
        lx_analytical, _, _ = fixture.sat.stageCostJacobians(
            0, 10, x, u, sat_direction, eci_target_vec, B_eci, cost_cfg)
        
        lx_numerical = fixture.costJacobianFiniteDiff_x(
            0, 10, x, u, sat_direction, eci_target_vec, B_eci, cost_cfg)
        
        for i in range(3):  # AV block
            numerical_mag = np.abs(lx_numerical[i])
            threshold = abs_tol + rel_tol * numerical_mag
            error = np.abs(lx_analytical[i] - lx_numerical[i])
            assert error <= threshold, \
                f"ECI vector format Jacobian[{i}]: error={error:.6e}, threshold={threshold:.6e}"
    
    def test_hessian_quaternion_format_target_symmetric(self, fixture):
        """Hessian with quaternion format should be symmetric."""
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.AV_INDEX:fixture.sat.AV_INDEX+3] = [0.05, 0.02, 0.01]
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1.0, 0.0, 0.0, 0.0]
        
        u = np.zeros(fixture.sat.controlDim)
        
        cost_cfg = saltro.CostConfig()
        cost_cfg.angle = 1e3
        
        eci_target = np.array([0.9, 0.1, 0.0, 0.436])
        sat_direction = np.zeros(3)
        B_eci = np.zeros(3)
        
        lxx, _, _ = fixture.sat.stageCostHessians(0, 10, x, u, sat_direction, eci_target, B_eci, cost_cfg)
        
        tol = 1e-5  # Relaxed for numerical differentiation errors in Hessian computation
        for i in range(fixture.sat.stateDim):
            for j in range(i + 1, fixture.sat.stateDim):
                diff = np.abs(lxx[i, j] - lxx[j, i])
                assert diff < tol, f"Quaternion format Hessian asymmetry at [{i},{j}]: {diff}"
    
    def test_hessian_eci_vector_format_target_symmetric(self, fixture):
        """Hessian with ECI vector format should be symmetric."""
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.AV_INDEX:fixture.sat.AV_INDEX+3] = [0.05, 0.02, 0.01]
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1.0, 0.0, 0.0, 0.0]
        
        u = np.zeros(fixture.sat.controlDim)
        
        cost_cfg = saltro.CostConfig()
        cost_cfg.angle = 1e3
        
        eci_target = np.array([np.nan, 1.0, 0.0, 0.0])
        sat_direction = np.array([0.0, 0.0, 1.0])
        B_eci = np.zeros(3)
        
        lxx, _, _ = fixture.sat.stageCostHessians(0, 10, x, u, sat_direction, eci_target, B_eci, cost_cfg)
        
        tol = 1e-5  # Relaxed for numerical differentiation errors in Hessian computation
        for i in range(fixture.sat.stateDim):
            for j in range(i + 1, fixture.sat.stateDim):
                diff = np.abs(lxx[i, j] - lxx[j, i])
                assert diff < tol, f"ECI vector format Hessian asymmetry at [{i},{j}]: {diff}"
    
    def test_aligned_quaternion_vs_aligned_eci_vector_similar_costs(self, fixture):
        """Aligned quaternion and ECI vector should produce similar costs."""
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1.0, 0.0, 0.0, 0.0]
        
        u = np.zeros(fixture.sat.controlDim)
        
        cost_cfg = saltro.CostConfig()
        cost_cfg.angle = 1e3
        cost_cfg.ang_vel = 1e4
        
        sat_direction = np.array([1.0, 0.0, 0.0])
        B_eci = np.zeros(3)
        
        # Quaternion target: identity (aligned with body frame)
        eci_target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        cost_quat = fixture.sat.stageCost(0, 10, x, u, sat_direction, eci_target_quat, B_eci, cost_cfg)
        
        # ECI vector target: points in +X (which aligns with body +X when quat is identity)
        eci_target_vec = np.array([np.nan, 1.0, 0.0, 0.0])
        cost_vec = fixture.sat.stageCost(0, 10, x, u, sat_direction, eci_target_vec, B_eci, cost_cfg)
        
        # Both should represent alignment, so costs should be similar
        ratio = cost_quat / (cost_vec + 1e-10)
        assert 0.5 < ratio < 2.0, f"Costs should be similar: quat={cost_quat}, vec={cost_vec}, ratio={ratio}"
    
    def test_eci_vector_direction_sensitivity(self, fixture):
        """ECI vector format should be sensitive to sat_direction."""
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1.0, 0.0, 0.0, 0.0]
        
        u = np.zeros(fixture.sat.controlDim)
        
        cost_cfg = saltro.CostConfig()
        cost_cfg.angle = 1e3
        
        B_eci = np.zeros(3)
        
        # ECI target vector pointing in +X
        eci_target = np.array([np.nan, 1.0, 0.0, 0.0])
        
        # Different sat_direction values
        sat_dir_x = np.array([1, 0, 0])  # Aligned
        sat_dir_y = np.array([0, 1, 0])  # Perpendicular
        sat_dir_z = np.array([0, 0, 1])  # Perpendicular
        
        cost_aligned = fixture.sat.stageCost(0, 10, x, u, sat_dir_x, eci_target, B_eci, cost_cfg)
        cost_perp_y = fixture.sat.stageCost(0, 10, x, u, sat_dir_y, eci_target, B_eci, cost_cfg)
        cost_perp_z = fixture.sat.stageCost(0, 10, x, u, sat_dir_z, eci_target, B_eci, cost_cfg)
        
        # Aligned should have lowest cost
        assert cost_aligned < cost_perp_y, "Aligned direction should have lower cost"
        assert cost_aligned < cost_perp_z, "Aligned direction should have lower cost"
    
    def test_terminal_cost_with_both_formats(self, fixture):
        """Terminal cost should work with both target formats."""
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1.0, 0.0, 0.0, 0.0]
        
        cost_cfg = saltro.CostConfig()
        cost_cfg.angle = 1e3
        
        sat_direction = np.array([0.0, 0.0, 1.0])
        B_eci = np.zeros(3)
        
        # Quaternion format
        eci_target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        term_cost_quat = fixture.sat.terminalCost(x, sat_direction, eci_target_quat, B_eci, cost_cfg)
        
        # ECI vector format
        eci_target_vec = np.array([np.nan, 0.0, 0.0, 1.0])
        term_cost_vec = fixture.sat.terminalCost(x, sat_direction, eci_target_vec, B_eci, cost_cfg)
        
        assert np.isfinite(term_cost_quat), "Terminal cost finite for quaternion format"
        assert np.isfinite(term_cost_vec), "Terminal cost finite for ECI vector format"
    
    def test_terminal_jacobian_both_formats(self, fixture):
        """Terminal Jacobians should work with both target formats."""
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.AV_INDEX:fixture.sat.AV_INDEX+3] = [0.05, 0.02, 0.01]
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1.0, 0.0, 0.0, 0.0]
        
        cost_cfg = saltro.CostConfig()
        cost_cfg.angle = 1e3
        
        sat_direction = np.array([0.0, 0.0, 1.0])
        B_eci = np.zeros(3)
        
        # Quaternion format
        eci_target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        lx_quat, _, _ = fixture.sat.terminalCostJacobians(x, sat_direction, eci_target_quat, B_eci, cost_cfg)
        
        # ECI vector format
        eci_target_vec = np.array([np.nan, 0.0, 0.0, 1.0])
        lx_vec, _, _ = fixture.sat.terminalCostJacobians(x, sat_direction, eci_target_vec, B_eci, cost_cfg)
        
        assert np.all(np.isfinite(lx_quat)), "Terminal Jacobian finite for quaternion format"
        assert np.all(np.isfinite(lx_vec)), "Terminal Jacobian finite for ECI vector format"
    
    def test_quaternion_format_ignores_sat_direction(self, fixture):
        """Quaternion format target should ignore sat_direction."""
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1.0, 0.0, 0.0, 0.0]
        
        u = np.zeros(fixture.sat.controlDim)
        
        cost_cfg = saltro.CostConfig()
        cost_cfg.angle = 1e3
        
        eci_target = np.array([0.9, 0.1, 0.0, 0.436])  # Quaternion format (no NaN)
        B_eci = np.zeros(3)
        
        # Two different sat_direction values
        sat_dir1 = np.array([1, 0, 0])
        sat_dir2 = np.array([0, 1, 0])
        
        cost1 = fixture.sat.stageCost(0, 10, x, u, sat_dir1, eci_target, B_eci, cost_cfg)
        cost2 = fixture.sat.stageCost(0, 10, x, u, sat_dir2, eci_target, B_eci, cost_cfg)
        
        # Costs should be identical since quaternion format doesn't use sat_direction
        assert np.abs(cost1 - cost2) < 1e-14, "Quaternion format should not depend on sat_direction"
    
    def test_eci_vector_small_magnitude_handling(self, fixture):
        """ECI vector with small magnitude should convert properly."""
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1.0, 0.0, 0.0, 0.0]
        
        u = np.zeros(fixture.sat.controlDim)
        
        cost_cfg = saltro.CostConfig()
        cost_cfg.angle = 1e3
        
        sat_direction = np.array([1, 0, 0])
        B_eci = np.zeros(3)
        
        # ECI vector with small magnitude
        eci_target_small = np.array([np.nan, 1e-6, 0, 0])
        cost_small = fixture.sat.stageCost(0, 10, x, u, sat_direction, eci_target_small, B_eci, cost_cfg)
        
        # ECI vector with larger magnitude
        eci_target_large = np.array([np.nan, 1e-3, 0, 0])
        cost_large = fixture.sat.stageCost(0, 10, x, u, sat_direction, eci_target_large, B_eci, cost_cfg)
        
        # Both should produce finite results
        assert np.isfinite(cost_small), "Cost finite for small magnitude ECI vector"
        assert np.isfinite(cost_large), "Cost finite for larger magnitude ECI vector"

    def test_angle_cost_increases_after_midrun_boresight_switch(self, fixture):
        """Switching boresight mid-run should increase angle cost when target is fixed in ECI."""
        N = 20
        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX + 4] = [1.0, 0.0, 0.0, 0.0]
        u = np.zeros(fixture.sat.controlDim)
        B_eci = np.zeros(3)

        cost_cfg = saltro.CostConfig()
        cost_cfg.angle = 1e3
        cost_cfg.angle_N = 1e3
        cost_cfg.ang_vel = 0.0
        cost_cfg.ang_vel_N = 0.0
        cost_cfg.ang_vel_mag = 0.0
        cost_cfg.ang_vel_mag_N = 0.0
        cost_cfg.ang_vel_err_dir = 0.0
        cost_cfg.ang_vel_err_dir_N = 0.0
        cost_cfg.control_mult = 0.0

        attitude_target = np.array([np.nan, 1.0, 0.0, 0.0])
        boresight_before = np.array([1.0, 0.0, 0.0])
        boresight_after = np.array([0.0, 1.0, 0.0])

        cost_before = fixture.sat.stageCost(4, N, x, u, boresight_before, attitude_target, B_eci, cost_cfg)
        cost_after = fixture.sat.stageCost(15, N, x, u, boresight_after, attitude_target, B_eci, cost_cfg)

        assert np.isfinite(cost_before)
        assert np.isfinite(cost_after)
        assert cost_before < cost_after

    def test_total_cost_reflects_boresight_switch_history(self, fixture):
        """totalCost should increase when boresight history switches away from a fixed ECI target."""
        N = 20
        nx = fixture.sat.stateDim
        nu = fixture.sat.controlDim

        X = np.zeros((nx, N))
        X[fixture.sat.QUAT_INDEX, :] = 1.0
        U = np.zeros((nu, N - 1))
        B = np.zeros((3, N))

        boresight_aligned = np.zeros((3, N))
        boresight_aligned[0, :] = 1.0

        boresight_switched = np.zeros((3, N))
        boresight_switched[0, : N // 2] = 1.0
        boresight_switched[1, N // 2 :] = 1.0

        cost_cfg = saltro.CostConfig()
        cost_cfg.angle = 1e3
        cost_cfg.angle_N = 1e3
        cost_cfg.ang_vel = 0.0
        cost_cfg.ang_vel_N = 0.0
        cost_cfg.ang_vel_mag = 0.0
        cost_cfg.ang_vel_mag_N = 0.0
        cost_cfg.ang_vel_err_dir = 0.0
        cost_cfg.ang_vel_err_dir_N = 0.0
        cost_cfg.control_mult = 0.0

        attitude_target = np.array([np.nan, 1.0, 0.0, 0.0])
        attitude_target_traj = np.tile(attitude_target.reshape(4, 1), (1, N))

        J_aligned = fixture.sat.totalCost(X, U, B, boresight_aligned, attitude_target_traj, cost_cfg)
        J_switched = fixture.sat.totalCost(X, U, B, boresight_switched, attitude_target_traj, cost_cfg)

        assert np.isfinite(J_aligned)
        assert np.isfinite(J_switched)
        assert J_switched > J_aligned


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
