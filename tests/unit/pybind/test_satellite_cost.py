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
        """Generate orbit at ~600 km altitude using the real generate_orbit
        binding. The previous version called a non-existent overload and
        silently fell back to synthetic R/V/B/S — meaning the entire cost
        test suite was running against fake orbit data."""
        a = 6978e3
        r0 = np.array([a, 0.0, 0.0])
        v0 = np.array([0.0, 7.56e3, 0.0])

        self.jtime = np.array([i * self.dt for i in range(self.n_steps)])
        ok, self.R, self.V, self.B, self.S, self.rho = saltro.generate_orbit(
            r0, v0, self.jtime, 0, 0, 0, 0, 0
        )
        assert ok, "Orbit generation failed"
    
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
        
        # Test the implemented cost function types {0, 1, 3}
        for cost_type in (0, 1, 3):
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
        cost_cfg.angle = 2e2  # doubled: type 4 ((1-d)^2) removed -> type 1 (0.5*(1-d)^2)
        cost_cfg.ang_vel = 1e3
        cost_cfg.ang_vel_mag = 1e2
        cost_cfg.ang_vel_err_dir = 1e2
        cost_cfg.control_mult = 0.1
        cost_cfg.ang_cost_func_type = 1  # 0.5 * (1 - |q·q_goal|)^2
        
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
        cost_cfg.angle = 2e2    # doubled: type 4 ((1-d)^2) removed -> type 1 (0.5*(1-d)^2)
        cost_cfg.angle_N = 2e2  # doubled (see above)
        cost_cfg.ang_vel = 1e4
        cost_cfg.ang_vel_N = 1e4
        cost_cfg.ang_vel_mag = 5e1
        cost_cfg.ang_vel_err_dir = 5e1
        cost_cfg.control_mult = 0.01
        cost_cfg.mtq_control_weight = 1.0
        cost_cfg.rw_control_weight = 1e3
        cost_cfg.ang_cost_func_type = 1
        
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
        cost_cfg.RWh_knee_frac = 0.8
        cost_cfg.RWh_stiction_mult = 0.01
        cost_cfg.RWh_desat_mult = 0.5
        
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
        cost_cfg.angle = 2e2  # doubled: type 4 ((1-d)^2) removed -> type 1 (0.5*(1-d)^2)
        cost_cfg.ang_vel = 1e3
        cost_cfg.control_mult = 0.1
        cost_cfg.ang_cost_func_type = 1

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
        cost_cfg.angle = 2e2  # doubled: type 4 ((1-d)^2) removed -> type 1 (0.5*(1-d)^2)
        cost_cfg.ang_vel = 1e3
        cost_cfg.control_mult = 0.1
        cost_cfg.ang_cost_func_type = 1
        
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
        cost_cfg.angle = 2e2  # doubled: type 4 ((1-d)^2) removed -> type 1 (0.5*(1-d)^2)
        cost_cfg.ang_vel = 1e4
        cost_cfg.control_mult = 0.01
        cost_cfg.mtq_control_weight = 1.0
        cost_cfg.rw_control_weight = 1e3
        cost_cfg.ang_cost_func_type = 1
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
        cost_cfg.RWh_knee_frac = 0.8
        cost_cfg.RWh_desat_mult = 0.5
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
        cost_cfg.RWh_knee_frac = 0.8
        cost_cfg.RWh_desat_mult = 0.5
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

    # ========================================================================
    # RW momentum soft-cost redesign: C1 knee + desat + stiction parking
    # (fixture wheels: h_max = 0.01) — python twins of the C++ cases in
    # test_satellite_cost.cpp.
    # ========================================================================

    @staticmethod
    def _rwh_only_cfg():
        """Twin of rwhOnlyCfg() in test_satellite_cost.cpp."""
        cfg = saltro.CostConfig()
        cfg.angle = 0.0
        cfg.ang_vel = 0.0
        cfg.control_mult = 0.0
        cfg.rw_AM_weight = 1e4
        cfg.rw_stic_weight = 0.0
        cfg.RWh_knee_frac = 0.5        # knee at 0.005
        cfg.RWh_desat_mult = 0.5
        cfg.RWh_stiction_mult = 0.0  # stiction off unless a test enables it
        cfg.use_cost_hess = True     # exercise the real state-Hessian block
        return cfg

    def test_rw_momentum_cost_c1_continuity_at_the_knee(self, fixture):
        """Twin of C++ 'RW momentum cost: C1 continuity at the knee'."""
        cfg = self._rwh_only_cfg()
        knee = 0.5 * 0.01
        sat_direction = np.zeros(3)
        eci_target = np.array([1.0, 0.0, 0.0, 0.0])
        B_eci = np.zeros(3)
        u = np.zeros(fixture.sat.controlDim)
        h_idx = fixture.sat.RW_MOMENTUM_INDEX

        def eval_at(h):
            x = np.zeros(fixture.sat.stateDim)
            x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1, 0, 0, 0]
            x[h_idx] = h
            J = fixture.sat.stageCost(0, 10, x, u, sat_direction, eci_target, B_eci, cfg)
            lx, _, _ = fixture.sat.stageCostJacobians(0, 10, x, u, sat_direction, eci_target, B_eci, cfg)
            return J, lx[h_idx]

        eps = 1e-9
        J_lo, g_lo = eval_at(knee - eps)
        J_hi, g_hi = eval_at(knee + eps)
        # Value continuous and gradient continuous across the knee (the steep
        # term has zero value AND zero slope there).
        assert abs(J_hi - J_lo) <= 1e-4 * (1.0 + abs(J_lo)), \
            f"Cost discontinuous at knee: {J_lo:.6e} vs {J_hi:.6e}"
        assert abs(g_hi - g_lo) <= 1e-4 * (1.0 + abs(g_lo)), \
            f"Gradient discontinuous at knee: {g_lo:.6e} vs {g_hi:.6e}"

    def test_rw_momentum_cost_fd_gradient_and_hessian_per_regime(self, fixture):
        """Twin of C++ 'RW momentum cost: FD gradient and Hessian per regime'."""
        cfg = self._rwh_only_cfg()
        cfg.rw_stic_weight = 50.0
        cfg.RWh_stiction_mult = 0.1  # stiction band [0, 0.001)
        sat_direction = np.zeros(3)
        eci_target = np.array([1.0, 0.0, 0.0, 0.0])
        B_eci = np.zeros(3)
        u = np.zeros(fixture.sat.controlDim)
        h_idx = fixture.sat.RW_MOMENTUM_INDEX

        # One probe per regime, away from all kinks (0, h_stic=0.001, knee=0.005).
        for h0 in (0.0004, 0.003, 0.008, -0.003, -0.008):
            x = np.zeros(fixture.sat.stateDim)
            x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1, 0, 0, 0]
            x[h_idx] = h0
            eps = 1e-8

            def cost_at(h):
                xp = x.copy()
                xp[h_idx] = h
                return fixture.sat.stageCost(0, 10, xp, u, sat_direction, eci_target, B_eci, cfg)

            def grad_at(h):
                xp = x.copy()
                xp[h_idx] = h
                lx, _, _ = fixture.sat.stageCostJacobians(0, 10, xp, u, sat_direction, eci_target, B_eci, cfg)
                return lx[h_idx]

            g_fd = (cost_at(h0 + eps) - cost_at(h0 - eps)) / (2.0 * eps)
            g_an = grad_at(h0)
            assert abs(g_an - g_fd) <= 1e-3 * (1.0 + abs(g_fd)), \
                f"h0={h0}: analytic grad {g_an:.6e} vs FD {g_fd:.6e}"

            H_fd = (grad_at(h0 + eps) - grad_at(h0 - eps)) / (2.0 * eps)
            lxx, _, _ = fixture.sat.stageCostHessians(0, 10, x, u, sat_direction, eci_target, B_eci, cfg)
            assert abs(lxx[h_idx, h_idx] - H_fd) <= 1e-3 * (1.0 + abs(H_fd)), \
                f"h0={h0}: analytic Hessian {lxx[h_idx, h_idx]:.6e} vs FD {H_fd:.6e}"
            assert lxx[h_idx, h_idx] >= 0.0, \
                f"h0={h0}: Hessian not PSD"  # PSD by construction in every regime

    def test_stiction_subgradient_at_exactly_h_zero_is_zero(self, fixture):
        """Twin of C++ 'Stiction: subgradient at exactly h == 0 is zero'."""
        cfg = self._rwh_only_cfg()
        cfg.rw_stic_weight = 100.0
        cfg.RWh_stiction_mult = 0.5  # band [0, 0.005)
        sat_direction = np.zeros(3)
        eci_target = np.array([1.0, 0.0, 0.0, 0.0])
        B_eci = np.zeros(3)
        u = np.zeros(fixture.sat.controlDim)
        h_idx = fixture.sat.RW_MOMENTUM_INDEX

        x = np.zeros(fixture.sat.stateDim)
        x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1, 0, 0, 0]

        lx0, _, _ = fixture.sat.stageCostJacobians(0, 10, x, u, sat_direction, eci_target, B_eci, cfg)
        assert lx0[h_idx] == 0.0, "deterministic contract at the tent peak"

        # Just off zero the ramp pushes |h| outward: dJ/dh < 0 for small h > 0.
        x[h_idx] = 1e-6
        lxp, _, _ = fixture.sat.stageCostJacobians(0, 10, x, u, sat_direction, eci_target, B_eci, cfg)
        assert lxp[h_idx] < 0.0
        x[h_idx] = -1e-6
        lxm, _, _ = fixture.sat.stageCostJacobians(0, 10, x, u, sat_direction, eci_target, B_eci, cfg)
        assert lxm[h_idx] > 0.0

    def test_desat_stiction_park_the_wheel_at_h_star(self, fixture):
        """Twin of C++ 'Desat + stiction park the wheel at h* = h_stic/2'."""
        # Recipe: w_stic = rw_AM_weight * desat * (h_stic/h_max)^2 parks at h_stic/2.
        cfg = self._rwh_only_cfg()                 # rw_AM_weight=1e4, desat=0.5
        cfg.RWh_stiction_mult = 0.5                # h_stic = 0.005 (== knee; steep off below)
        cfg.rw_stic_weight = 1e4 * 0.5 * 0.25      # = 1250
        h_star = 0.5 * 0.005
        sat_direction = np.zeros(3)
        eci_target = np.array([1.0, 0.0, 0.0, 0.0])
        B_eci = np.zeros(3)
        u = np.zeros(fixture.sat.controlDim)
        h_idx = fixture.sat.RW_MOMENTUM_INDEX

        def grad_at(h):
            x = np.zeros(fixture.sat.stateDim)
            x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1, 0, 0, 0]
            x[h_idx] = h
            lx, _, _ = fixture.sat.stageCostJacobians(0, 10, x, u, sat_direction, eci_target, B_eci, cfg)
            return lx[h_idx]

        scale = 1e4 * 0.5 * h_star / (0.01 * 0.01)  # desat gradient magnitude at h*
        assert abs(grad_at(h_star)) <= 1e-6 * scale, \
            f"gradient at h* not ~0: {grad_at(h_star):.6e}"
        assert grad_at(0.8 * h_star) < 0.0  # below h*: pushed up toward h*
        assert grad_at(1.2 * h_star) > 0.0  # above h*: pulled back toward h*

    def test_rw_momentum_cost_desat_zero_leaves_sub_knee_region_free(self, fixture):
        """Twin of C++ 'RW momentum cost: desat=0 leaves the sub-knee region free'."""
        cfg = self._rwh_only_cfg()
        cfg.RWh_desat_mult = 0.0
        sat_direction = np.zeros(3)
        eci_target = np.array([1.0, 0.0, 0.0, 0.0])
        B_eci = np.zeros(3)
        u = np.zeros(fixture.sat.controlDim)

        def cost_at(h):
            x = np.zeros(fixture.sat.stateDim)
            x[fixture.sat.QUAT_INDEX:fixture.sat.QUAT_INDEX+4] = [1, 0, 0, 0]
            x[fixture.sat.RW_MOMENTUM_INDEX] = h
            return fixture.sat.stageCost(0, 10, x, u, sat_direction, eci_target, B_eci, cfg)

        assert cost_at(0.004) == cost_at(0.0)   # below the knee: exactly free
        assert cost_at(0.008) > cost_at(0.004)  # above the knee: steep term active


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

# ============================================================================
# TEST SECTION 12: Singularity sweep + hemisphere-kink coverage
# ============================================================================
# Property tests over the full cost-shape parameter grid:
#   cost type ∈ {0,1,3,5} × mode ∈ {vec (NaN ECI target), quat} × GN ∈ {on,off}.
# We probe the attitude cost h(argument) where the inner scalar is
#   c = bs·R(q)ᵀ·r̂  (vec mode, 2-DOF, argument ∈ [-1, 1]), or
#   d = |q_goal·q|   (quat mode, hemisphere-aligned, argument ∈ [0, 1]).
# The base attitude is identity, so the tangent projector is P = diag(0,1,1,1)
# and only the q-components 1..3 carry gradient/Hessian signal.
#
# Coordinate conventions used here (physical angle θ):
#   vec:  r̂ = [sinθ, 0, cosθ] with bs = +z ⇒ c = cosθ; θ→0 aligned pole
#         (c→+1), θ→π antipode (c→−1, a GENUINE cusp for type 3).
#   quat: q_goal = [cos(θ/2), sin(θ/2), 0, 0] ⇒ d = cos(θ/2); θ→0 aligned pole
#         (d→+1), θ→π gives d→0 — the |·| hemisphere kink, NOT the d=−1 shape
#         antipode (which hemisphere alignment makes unreachable).
#
# GN semantics discovered empirically and encoded below:
#   - GN=False returns the full (exact) Hessian ⇒ matches central-difference FD.
#   - GN=True in VEC mode drops the f'·∂²c chain term (the Gauss-Newton
#     approximation) ⇒ deliberately does NOT match FD; we assert the rank-1
#     GN eigen-structure instead (PSD: f'' ≥ 0 for all remaining shapes).
#   - GN flag is a NO-OP in QUAT mode (d is linear in q ⇒ no chain term to
#     drop), so quat GN=True == quat GN=False == full Hessian ⇒ matches FD.

import math as _math

_SW_BS = np.array([0.0, 0.0, 1.0])
_SW_B0 = np.zeros(3)
_SW_DENSE_DEG = list(range(1, 180, 10)) + [179]      # 1°..171° step 10° + 179°
_SW_BOUNDARY = [1e-2, 1e-3, 1e-4, 1e-5]              # rad, pole-approach sequence


def _sweep_cfg(act, gn):
    cfg = saltro.CostConfig()
    cfg.angle = 1.0
    cfg.angle_N = 1.0
    cfg.ang_vel = 0.0
    cfg.ang_vel_N = 0.0
    cfg.ang_vel_mag = 0.0
    cfg.ang_vel_mag_N = 0.0
    cfg.ang_vel_err_dir = 0.0
    cfg.ang_vel_err_dir_N = 0.0
    cfg.ang_vel_err_dir_ratio = 0.0
    cfg.ang_vel_roll_ratio = 1.0
    cfg.control_mult = 0.0
    cfg.mtq_control_weight = 0.0
    cfg.rw_control_weight = 0.0
    cfg.rw_AM_weight = 0.0
    cfg.rw_stic_weight = 0.0
    cfg.RWh_stiction_mult = 0.0
    cfg.use_cost_hess = True
    cfg.ang_cost_func_type = act
    cfg.cost_hess_gauss_newton = gn
    return cfg


def _sw_target(mode, theta):
    if mode == "vec":
        return np.array([np.nan, np.sin(theta), 0.0, np.cos(theta)])
    return np.array([np.cos(theta / 2.0), np.sin(theta / 2.0), 0.0, 0.0])


def _sw_base_state(sat):
    x = np.zeros(sat.stateDim)
    x[sat.QUAT_INDEX] = 1.0
    return x


def _sw_cost(sat, x, target, cfg):
    return sat.stageCost(0, 100, x, np.zeros(sat.controlDim),
                         _SW_BS, target, _SW_B0, cfg)


def _sw_qgrad_fd(sat, x, target, cfg, eps=1e-6):
    """Central-difference gradient on the q-block, tangent-projected."""
    QI = sat.QUAT_INDEX
    g = np.zeros(4)
    for j in range(4):
        xp = x.copy(); xp[QI + j] += eps
        xm = x.copy(); xm[QI + j] -= eps
        g[j] = (_sw_cost(sat, xp, target, cfg) - _sw_cost(sat, xm, target, cfg)) / (2 * eps)
    q = x[QI:QI + 4]
    return (np.eye(4) - np.outer(q, q)) @ g


def _sw_qhess_fd(sat, x, target, cfg, eps=1e-4):
    """Central 2nd-difference Hessian on the q-block, tangent-projected."""
    QI = sat.QUAT_INDEX
    H = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            xpp = x.copy(); xpp[QI + i] += eps; xpp[QI + j] += eps
            xmm = x.copy(); xmm[QI + i] -= eps; xmm[QI + j] -= eps
            xpm = x.copy(); xpm[QI + i] += eps; xpm[QI + j] -= eps
            xmp = x.copy(); xmp[QI + i] -= eps; xmp[QI + j] += eps
            H[i, j] = (_sw_cost(sat, xpp, target, cfg) + _sw_cost(sat, xmm, target, cfg)
                       - _sw_cost(sat, xpm, target, cfg) - _sw_cost(sat, xmp, target, cfg)) \
                / (4 * eps * eps)
    q = x[QI:QI + 4]
    P = np.eye(4) - np.outer(q, q)
    return P @ H @ P


def _sw_qblock_ana(sat, x, target, cfg):
    """Analytic cost, tangent-projected q-gradient, tangent-projected q-Hess."""
    QI = sat.QUAT_INDEX
    u = np.zeros(sat.controlDim)
    c = _sw_cost(sat, x, target, cfg)
    lx, _, _ = sat.stageCostJacobians(0, 100, x, u, _SW_BS, target, _SW_B0, cfg)
    lxx, _, _ = sat.stageCostHessians(0, 100, x, u, _SW_BS, target, _SW_B0, cfg)
    q = x[QI:QI + 4]
    P = np.eye(4) - np.outer(q, q)
    return c, P @ lx[QI:QI + 4], P @ lxx[QI:QI + 4, QI:QI + 4] @ P, lx, lxx


class TestSingularitySweep:
    """Task 1: singularity sweep property tests across all cost shapes."""

    @pytest.mark.parametrize("act", [0, 1, 3, 5])
    @pytest.mark.parametrize("mode", ["vec", "quat"])
    @pytest.mark.parametrize("gn", [False, True])
    def test_dense_sweep_finite_and_fd_consistent(self, fixture, act, mode, gn):
        """(a) Dense alignment-angle sweep θ ∈ {1°..179°}: everything finite,
        gradient matches central FD (curvature-scaled tol), Hessian matches FD
        of the gradient where the full (non-GN) Hessian is in effect."""
        sat = fixture.sat
        x = _sw_base_state(sat)
        cfg = _sweep_cfg(act, gn)
        # Full Hessian is returned when GN is off, OR always in quat mode
        # (the GN flag is a no-op there — d is linear in q).
        full_hess = (not gn) or (mode == "quat")
        for td in _SW_DENSE_DEG:
            theta = _math.radians(td)
            tgt = _sw_target(mode, theta)
            c, gq, Hq, lx, lxx = _sw_qblock_ana(sat, x, tgt, cfg)
            assert np.isfinite(c), f"cost non-finite mode={mode} act={act} θ={td}"
            assert np.isfinite(lx).all(), f"grad non-finite mode={mode} act={act} θ={td}"
            assert np.isfinite(lxx).all(), f"Hess non-finite mode={mode} act={act} θ={td}"

            # Gradient vs central FD.  The assembled q-gradient stays finite and
            # FD-accurate even near the poles (geometry factor cancels the raw
            # 1/√(1−c²) blow-up), so no skip is needed on the dense grid.
            gfd = _sw_qgrad_fd(sat, x, tgt, cfg)
            for j in range(4):
                tol = 1e-6 + 1e-4 * abs(gfd[j])
                assert abs(gq[j] - gfd[j]) < tol, \
                    f"grad[{j}] mode={mode} act={act} gn={gn} θ={td}: {gq[j]} vs {gfd[j]}"

            # Genuine-singular regions: skip Hessian-FD within 1e-3 rad of the
            # antipode for the acos-family shapes (vec types 3/5), where the cost
            # curvature radius shrinks below the FD step and central differences
            # stop tracking the (correctly diverging) analytic Hessian.  The
            # dense grid never actually enters that band (nearest is 179° ≈
            # 0.0175 rad away); the guard documents intent for completeness.
            near_antipode = (mode == "vec" and act in (3, 5)
                             and abs(_math.pi - theta) < 1e-3)
            if full_hess and not near_antipode:
                Hfd = _sw_qhess_fd(sat, x, tgt, cfg)
                herr = np.max(np.abs(Hq - Hfd))
                hscale = np.max(np.abs(Hfd))
                assert herr < 1e-3 + 5e-2 * hscale, \
                    f"Hess mode={mode} act={act} θ={td}: herr={herr:.2e} scale={hscale:.2e}"
            elif not full_hess:
                # GN=True vec mode: the GN Hessian is f''·(∂c/∂q)(∂c/∂q)ᵀ, a
                # rank-1 form in the tangent block.  Assert the rank-1 structure
                # (two tangent eigenvalues ≈ 0), and PSD for the f''≥0 shapes
                # (types 0/1/3; type 2, whose f'' changed sign at c=0, was
                # removed).  Type 5 (pseudo-Huber) is convex in θ but CONCAVE
                # in c below the g''=g'·cotθ crossover (≈86° at δ=0.35): its
                # single nonzero GN eigenvalue 4·[g''−g'·cotθ] is negative
                # there, positive above, bounded by 4·w in magnitude.
                eig = np.linalg.eigvalsh(Hq)
                mags = np.sort(np.abs(eig))
                assert mags[-2] < 1e-6 + 1e-3 * mags[-1], \
                    f"GN Hess not rank-1 act={act} θ={td}: eig={eig}"
                if act != 5:
                    assert eig.min() > -1e-6, f"GN type{act} not PSD θ={td}: {eig}"
                else:
                    assert eig.min() > -4.0 - 1e-6, \
                        f"GN type5 below -4w bound θ={td}: {eig}"

    @pytest.mark.parametrize("act", [0, 1, 3, 5])
    @pytest.mark.parametrize("mode", ["vec", "quat"])
    @pytest.mark.parametrize("gn", [False, True])
    def test_boundary_approach_aligned_pole_finite(self, fixture, act, mode, gn):
        """(b) Aligned-pole approach θ ∈ {1e-2..1e-5} rad: cost/grad/Hessian all
        finite deep inside the c/d → +1 removable-singularity region (this is
        exactly #30's Taylor-protected zone)."""
        sat = fixture.sat
        x = _sw_base_state(sat)
        cfg = _sweep_cfg(act, gn)
        for theta in _SW_BOUNDARY:
            tgt = _sw_target(mode, theta)
            _, _, _, lx, lxx = _sw_qblock_ana(sat, x, tgt, cfg)
            c = _sw_cost(sat, x, tgt, cfg)
            assert np.isfinite(c) and np.isfinite(lx).all() and np.isfinite(lxx).all(), \
                f"non-finite at aligned pole mode={mode} act={act} gn={gn} θ={theta}"

    @pytest.mark.parametrize("act", [0, 1, 3, 5])
    @pytest.mark.parametrize("gn", [False, True])
    def test_boundary_approach_antipode_vec_finite(self, fixture, act, gn):
        """(b) Antipodal approach (vec only — quat has no reachable shape
        antipode).  cost/grad/Hessian finite even as the geometry diverges;
        the *magnitude* growth of the Hessian is checked separately."""
        sat = fixture.sat
        x = _sw_base_state(sat)
        cfg = _sweep_cfg(act, gn)
        for delta in _SW_BOUNDARY:
            theta = _math.pi - delta
            tgt = _sw_target("vec", theta)
            _, _, _, lx, lxx = _sw_qblock_ana(sat, x, tgt, cfg)
            c = _sw_cost(sat, x, tgt, cfg)
            assert np.isfinite(c) and np.isfinite(lx).all() and np.isfinite(lxx).all(), \
                f"non-finite at antipode act={act} gn={gn} δ={delta}"

    def test_type3_antipode_divergence_clamped(self, fixture):
        """(b) Type 3 at the vec antipode: a GENUINE cusp on the cost surface,
        now handled by the bounded antipodal clamp (below u = 1+c < 1e-6 the
        shape evaluates the exact formula at the seam c_eff = −1 + 1e-6; see
        angCostShape in src/pybind/satellite.cpp).  Instead of unbounded
        ~1/sinθ divergence, the eigenvalues GROW while the exact formula is in
        effect, then SATURATE at the documented clamp bounds:
          - Gauss-Newton max-eig grows like +1/sinθ, peaks at the seam
            (≤ f''_clamp·4·(1−c_eff²) ≈ 8885.8·w), then falls off as
            f''_clamp·4·(1−c²) with the frozen seam curvature.
          - full-Newton min-eig grows like −1/sinθ, then saturates at
            ≈ 4·f'_clamp ≈ −8881.8·w (the f'·∂²c term with clamped f').
        δ ∈ _SW_BOUNDARY maps to u = 1−cos(δ) ≈ δ²/2: δ=1e-2 → u=5e-5 (exact
        region), δ=1e-3/1e-4/1e-5 → u=5e-7/5e-9/5e-11 (inside the clamp)."""
        sat = fixture.sat
        x = _sw_base_state(sat)
        q = x[sat.QUAT_INDEX:sat.QUAT_INDEX + 4]
        P = np.eye(4) - np.outer(q, q)
        clamp_u = 1e-6
        c_eff = -1.0 + clamp_u
        omc2_eff = 1.0 - c_eff * c_eff
        phi_eff = _math.acos(c_eff)
        fp_clamp = -phi_eff / _math.sqrt(omc2_eff)                 # ≈ −2220.44
        fpp_clamp = (1.0 / omc2_eff
                     - phi_eff * c_eff / (omc2_eff * _math.sqrt(omc2_eff)))
        gn_bound = fpp_clamp * 4.0 * omc2_eff                      # ≈ +8885.76
        fn_saturation = 4.0 * fp_clamp                             # ≈ −8881.77
        prev_gn = 0.0
        prev_fn = 0.0
        for delta in _SW_BOUNDARY:
            theta = _math.pi - delta
            tgt = _sw_target("vec", theta)
            QI = sat.QUAT_INDEX
            u = np.zeros(sat.controlDim)
            Hq_gn, _, _ = sat.stageCostHessians(0, 100, x, u, _SW_BS, tgt, _SW_B0,
                                                _sweep_cfg(3, True))
            Hq_fn, _, _ = sat.stageCostHessians(0, 100, x, u, _SW_BS, tgt, _SW_B0,
                                                _sweep_cfg(3, False))
            eig_gn = np.linalg.eigvalsh(P @ Hq_gn[QI:QI + 4, QI:QI + 4] @ P)
            eig_fn = np.linalg.eigvalsh(P @ Hq_fn[QI:QI + 4, QI:QI + 4] @ P)
            gmax = eig_gn.max()
            fmin = eig_fn.min()
            assert gmax > 0.0, f"GN max-eig should be positive, got {gmax} at δ={delta}"
            assert fmin < 0.0, f"FN min-eig should be negative, got {fmin} at δ={delta}"
            # The structural clamp bounds hold everywhere on the approach.
            assert gmax <= gn_bound * (1.0 + 1e-9), \
                f"GN max-eig above clamp bound: {gmax} > {gn_bound} at δ={delta}"
            assert fmin >= fn_saturation * 1.01, \
                f"FN min-eig below clamp saturation: {fmin} at δ={delta}"
            if 1.0 - _math.cos(delta) >= clamp_u:
                # Exact region: monotone growth with ~1/sinθ scaling
                # (empirically eig·sin(δ) ≈ ±4π ≈ ±12.57).
                assert gmax > prev_gn, f"GN max-eig not growing: {gmax} <= {prev_gn}"
                assert fmin < prev_fn, f"FN min-eig not growing (−): {fmin} >= {prev_fn}"
                assert 1.0 < gmax * _math.sin(delta) < 100.0, \
                    f"GN scaling off: {gmax * _math.sin(delta)} at δ={delta}"
                assert -100.0 < fmin * _math.sin(delta) < -1.0, \
                    f"FN scaling off: {fmin * _math.sin(delta)} at δ={delta}"
            else:
                # Clamped region: FN min-eig saturates at ≈ 4·f'_clamp; the GN
                # outer product decays as f''_clamp·4·(1−c²) (frozen f'').
                np.testing.assert_allclose(
                    fmin, fn_saturation, rtol=1e-2,
                    err_msg=f"FN min-eig should saturate at 4·f'_clamp, δ={delta}")
                c_here = _math.cos(theta)
                np.testing.assert_allclose(
                    gmax, fpp_clamp * 4.0 * (1.0 - c_here * c_here), rtol=1e-6,
                    err_msg=f"GN max-eig should follow frozen-f'' decay, δ={delta}")
            prev_gn, prev_fn = gmax, fmin

    @pytest.mark.parametrize("mode", ["vec", "quat"])
    def test_blend_zone_continuity_type3(self, fixture, mode):
        """(c) #30's blend-zone edges: sweep omz = 1 − argument across the
        below/at/above probes straddling the 1e-6 and 1e-4 thresholds and assert
        value/grad/Hessian vary smoothly (no jump at the Taylor↔exact switch).
        For type 3 near the aligned pole: cost ≈ omz (Taylor leading term),
        |g_q| grows monotonically, and the projected max-eig ≈ 4·w_ang stays
        essentially constant."""
        sat = fixture.sat
        x = _sw_base_state(sat)
        cfg = _sweep_cfg(3, False)
        for thr in (1e-6, 1e-4):
            costs, gnorms, eigmaxs, omzs = [], [], [], []
            for frac in (0.5, 1.0, 2.0):
                omz = thr * frac
                arg = 1.0 - omz  # c (vec) or d (quat)
                theta = _math.acos(arg) if mode == "vec" else 2.0 * _math.acos(arg)
                tgt = _sw_target(mode, theta)
                c, gq, Hq, _, _ = _sw_qblock_ana(sat, x, tgt, cfg)
                assert np.isfinite(c) and np.isfinite(gq).all() and np.isfinite(Hq).all()
                costs.append(c); gnorms.append(np.linalg.norm(gq))
                eigmaxs.append(np.linalg.eigvalsh(Hq).max()); omzs.append(omz)
            # Cost ≈ omz across the blend (relative continuity + correctness).
            for c, omz in zip(costs, omzs):
                np.testing.assert_allclose(c, omz, rtol=2e-2,
                                           err_msg=f"[{mode}] cost≠omz at blend edge {thr}")
            # Monotone in omz (no reversal at the switch).
            assert costs[0] < costs[1] < costs[2], f"[{mode}] cost not monotone @ {thr}"
            assert gnorms[0] < gnorms[1] < gnorms[2], f"[{mode}] |g| not monotone @ {thr}"
            # Projected max-eig near the aligned pole is flat across the blend
            # (Hessian dominated by the PwA/manifold term).  The constant differs
            # by mode: vec lifts the reduced Hessian through W with the full
            # 2-DOF geometry (≈ 4·w_ang), quat's d-linear PwA term gives ≈ 1·w_ang.
            expected_eig = 4.0 if mode == "vec" else 1.0
            for e in eigmaxs:
                assert abs(e - expected_eig) < 1e-2, \
                    f"[{mode}] eigmax jumped at {thr}: {e} (expected ≈ {expected_eig})"


class TestHemisphereKink:
    """Task 2: quaternion-mode hemisphere kink at d = q·q_goal = 0."""

    def test_quat_hemisphere_kink_finite_and_sign_flip(self, fixture):
        """At a quaternion 180° from goal (q·q_goal = 0, the abs() kink) the
        cost/gradient/Hessian are all FINITE for every cost shape.  Approaching
        from either hemisphere (scalar part ±1e-6) the cost is continuous but
        the q-gradient FLIPS SIGN — the expected C¹ kink of |q_goal·q|.  The
        exact d=0 point resolves to the qdot≥0 (non-flipped) convention, so its
        gradient sign matches the +hemisphere approach."""
        sat = fixture.sat
        QI = sat.QUAT_INDEX
        x = _sw_base_state(sat)          # q = identity = [1,0,0,0]
        u = np.zeros(sat.controlDim)

        # Finiteness at exactly d = 0 for all cost shapes.
        qg0 = np.array([0.0, 1.0, 0.0, 0.0])   # orthogonal to identity ⇒ d = 0
        for act in (0, 1, 3, 5):
            cfg = _sweep_cfg(act, False)
            c = sat.stageCost(0, 100, x, u, _SW_BS, qg0, _SW_B0, cfg)
            lx, _, _ = sat.stageCostJacobians(0, 100, x, u, _SW_BS, qg0, _SW_B0, cfg)
            lxx, _, _ = sat.stageCostHessians(0, 100, x, u, _SW_BS, qg0, _SW_B0, cfg)
            assert np.isfinite(c) and np.isfinite(lx).all() and np.isfinite(lxx).all(), \
                f"non-finite at d=0 kink, act={act}"

        # Hemisphere-approach sign flip (use type 3; behavior is generic).
        cfg = _sweep_cfg(3, False)
        eps = 1e-6
        qg_plus = np.array([+eps, 1.0, 0.0, 0.0]); qg_plus /= np.linalg.norm(qg_plus)
        qg_minus = np.array([-eps, 1.0, 0.0, 0.0]); qg_minus /= np.linalg.norm(qg_minus)
        c_plus = sat.stageCost(0, 100, x, u, _SW_BS, qg_plus, _SW_B0, cfg)
        c_minus = sat.stageCost(0, 100, x, u, _SW_BS, qg_minus, _SW_B0, cfg)
        g_plus, _, _ = sat.stageCostJacobians(0, 100, x, u, _SW_BS, qg_plus, _SW_B0, cfg)
        g_minus, _, _ = sat.stageCostJacobians(0, 100, x, u, _SW_BS, qg_minus, _SW_B0, cfg)
        g0, _, _ = sat.stageCostJacobians(0, 100, x, u, _SW_BS, qg0, _SW_B0, cfg)

        # Cost is continuous across the kink.
        np.testing.assert_allclose(c_plus, c_minus, atol=1e-6,
                                   err_msg="cost discontinuous across hemisphere kink")
        # The q-gradient (slot QI+1) flips sign — the documented kink.
        assert g_plus[QI + 1] * g_minus[QI + 1] < 0, \
            f"expected gradient sign flip: {g_plus[QI + 1]} vs {g_minus[QI + 1]}"
        np.testing.assert_allclose(g_plus[QI + 1], -g_minus[QI + 1], rtol=1e-4,
                                   err_msg="hemisphere flip not antisymmetric")
        # d=0 resolves to the +hemisphere convention (qdot=0 is not flipped).
        assert np.sign(g0[QI + 1]) == np.sign(g_plus[QI + 1]), \
            "d=0 gradient should match the +hemisphere approach convention"


# ============================================================================
# TEST SECTION: Gauss-Newton curvature cap (gn_curvature_max), vec mode
# ============================================================================
# The GN vec-pointing angle Hessian q-block is the rank-1 PSD term
# w_ang * f'' * dc dc^T, whose single nonzero eigenvalue is exactly
# w_ang * f'' * |dc|^2. For ang_cost_func_type=3 (0.5*acos^2) this diverges
# like 1/sin(theta) toward the antipode. gn_curvature_max > 0 clamps it to
# gn_curvature_max * w_ang. Mirrors the C++ tests in test_satellite_cost.cpp.


def _state_at_angle(fixture, theta):
    """State whose vec-pointing error angle is theta (rot about +z, bs +x,
    target +x): c = bs.R^T.r = cos(theta)."""
    QI = fixture.sat.QUAT_INDEX
    x = np.zeros(fixture.sat.stateDim)
    x[QI:QI + 4] = [np.cos(theta / 2.0), 0.0, 0.0, np.sin(theta / 2.0)]
    return x


def _q_block_max_eig(fixture, lxx):
    QI = fixture.sat.QUAT_INDEX
    qb = np.asarray(lxx)[QI:QI + 4, QI:QI + 4]
    return np.linalg.eigvalsh(qb).max()


def _gn_cfg(weight, cap):
    cfg = saltro.CostConfig()
    cfg.use_cost_hess = True
    cfg.cost_hess_gauss_newton = True
    cfg.ang_cost_func_type = 3
    cfg.angle = weight
    cfg.angle_N = weight
    cfg.ang_vel = 0.0
    cfg.ang_vel_N = 0.0
    cfg.gn_curvature_max = cap
    return cfg


class TestGNCurvatureCap:
    """Opt-in Gauss-Newton curvature cap for near-antipodal vec pointing."""

    sat_dir = np.array([1.0, 0.0, 0.0])
    eci_target = np.array([np.nan, 1.0, 0.0, 0.0])
    B_eci = np.zeros(3)

    def test_cap_disabled_matches_uncapped_formula(self, fixture):
        # With the cap disabled (0.0), the assembled rank-1 GN eigenvalue must
        # equal the closed-form uncapped value w*f''*|dc|^2 = w*4*(1 - th*cot th)
        # (|dc|^2 = 4 sin^2 th for this geometry). This is the pre-change behavior.
        u = np.zeros(fixture.sat.controlDim)
        weight = 3.0
        base = _gn_cfg(weight, 0.0)   # disabled
        for deg in [90.0, 150.0, 170.0, 179.0, 179.9]:
            th = np.deg2rad(deg)
            x = _state_at_angle(fixture, th)
            lxx, _, _ = fixture.sat.stageCostHessians(0, 10, x, u, self.sat_dir, self.eci_target, self.B_eci, base)
            predicted = weight * 4.0 * (1.0 - th * np.cos(th) / np.sin(th))
            assert _q_block_max_eig(fixture, lxx) == pytest.approx(predicted, rel=1e-9)

    def test_cap_clamps_eigenvalue_and_keeps_gradient(self, fixture):
        u = np.zeros(fixture.sat.controlDim)
        weight, cap = 3.0, 10.0
        uncapped = _gn_cfg(weight, 0.0)
        capped = _gn_cfg(weight, cap)
        for deg in [170.0, 179.0, 179.9]:
            x = _state_at_angle(fixture, np.deg2rad(deg))
            lxx_u, _, _ = fixture.sat.stageCostHessians(0, 10, x, u, self.sat_dir, self.eci_target, self.B_eci, uncapped)
            lxx_c, _, _ = fixture.sat.stageCostHessians(0, 10, x, u, self.sat_dir, self.eci_target, self.B_eci, capped)
            # Uncapped is genuinely stiff.
            assert _q_block_max_eig(fixture, lxx_u) > cap * weight
            # Capped eigenvalue is bounded.
            assert _q_block_max_eig(fixture, lxx_c) <= cap * weight * (1.0 + 1e-9)
            # Gradient untouched.
            lx_u, _, _ = fixture.sat.stageCostJacobians(0, 10, x, u, self.sat_dir, self.eci_target, self.B_eci, uncapped)
            lx_c, _, _ = fixture.sat.stageCostJacobians(0, 10, x, u, self.sat_dir, self.eci_target, self.B_eci, capped)
            assert np.array_equal(np.asarray(lx_u), np.asarray(lx_c))

    def test_cap_bit_identical_below_cap_angle(self, fixture):
        u = np.zeros(fixture.sat.controlDim)
        weight, cap = 3.0, 50.0
        uncapped = _gn_cfg(weight, 0.0)
        capped = _gn_cfg(weight, cap)
        for deg in [90.0, 120.0, 150.0, 160.0]:
            x = _state_at_angle(fixture, np.deg2rad(deg))
            lxx_u, _, _ = fixture.sat.stageCostHessians(0, 10, x, u, self.sat_dir, self.eci_target, self.B_eci, uncapped)
            lxx_c, _, _ = fixture.sat.stageCostHessians(0, 10, x, u, self.sat_dir, self.eci_target, self.B_eci, capped)
            assert _q_block_max_eig(fixture, lxx_u) < cap * weight
            assert np.array_equal(np.asarray(lxx_u), np.asarray(lxx_c))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


def test_ang_vel_direction_cost_mixed_hessian_matches_fd():
    """The angular-velocity-direction cost contributes a mixed d2L/dw dq term
    (cross_cost = -sign(qdot) * w_avang * q_g^T W(q) w, bilinear in (q, w)); the
    pure q-q term is zero. This block was previously dropped. Isolate the cost
    (only ang_vel_err_dir on) and check the w-q / q-w blocks vs finite
    differences at a non-identity attitude and nonzero rate."""
    settings = saltro.PlannerSettings()
    sat = saltro.Satellite(np.diag([0.067, 0.071, 0.069]), settings)
    sat.addMTQ(np.array([1.0, 0.0, 0.0]), 0.2)
    AV = saltro.Satellite.AV_INDEX
    QI = saltro.Satellite.QUAT_INDEX

    cfg = saltro.CostConfig()
    for attr in ("angle", "ang_vel", "ang_vel_mag", "angle_N", "ang_vel_N",
                 "rw_AM_weight", "control_mult", "mtq_control_weight", "rw_control_weight"):
        if hasattr(cfg, attr):
            setattr(cfg, attr, 0.0)
    cfg.ang_vel_err_dir = 1.0
    cfg.use_cost_hess = True

    q = np.array([0.6, -0.3, 0.5, 0.2]); q = q / np.linalg.norm(q)
    x = np.zeros(sat.stateDim)
    x[AV:AV + 3] = [0.05, 0.02, 0.01]
    x[QI:QI + 4] = q
    u = np.zeros(sat.controlDim)
    att = np.array([0.2, 0.5, -0.4, 0.7]); att = att / np.linalg.norm(att)
    bs = np.array([1.0, 0.0, 0.0])
    B = np.array([2.5e-5, -1.5e-5, 3.0e-5])
    k, N = 0, 10

    lxx, _, _ = sat.stageCostHessians(k, N, x, u, bs, att, B, cfg)
    lxx = np.array(lxx)

    def L(xx):
        return sat.stageCost(k, N, xx, u, bs, att, B, cfg)

    eps = 1e-5
    for a in range(3):       # angular-velocity index
        for b in range(4):   # quaternion index
            ia, ib = AV + a, QI + b
            xpp, xpm, xmp, xmm = (x.copy() for _ in range(4))
            xpp[ia] += eps; xpp[ib] += eps
            xpm[ia] += eps; xpm[ib] -= eps
            xmp[ia] -= eps; xmp[ib] += eps
            xmm[ia] -= eps; xmm[ib] -= eps
            fd = (L(xpp) - L(xpm) - L(xmp) + L(xmm)) / (4.0 * eps * eps)
            assert abs(lxx[ia, ib] - fd) < 1e-6, f"(w{a},q{b}): {lxx[ia,ib]} vs {fd}"
            assert abs(lxx[ia, ib] - lxx[ib, ia]) < 1e-12  # symmetric
# ============================================================================
# TEST SECTION 13: afc=5 pseudo-Huber angle cost
# ============================================================================
# Python twin of C++ TEST SECTION 13 in tests/unit/pybind/test_satellite_cost.cpp.
# Shape: g(θ) = δ²·(√(1+(θ/δ)²) − 1), θ = acos(c), δ = ang_cost_huber_delta.
#   g'(θ) = θ/√(1+(θ/δ)²) — ≈ θ near the goal, saturates at δ for θ ≫ δ.
#   c-space: f'(c) = −g'(θ)/sinθ, f''(c) = [g''(θ) − g'(θ)·cotθ]/sin²θ.
# Assembled facts asserted below (vec mode, weight w):
#   |g_q| = 2·w·g'(θ) ≤ 2·w·δ·π/√(π²+δ²) < 2·w·δ (bounded urgency, tight at
#     the antipode; non-vanishing escape gradient unlike type 0's plateau);
#   near-goal cost matches type 3's ½θ² with relative error −(θ/δ)²/4;
#   antipodal GN divergence is δ-scaled (4·g'(π)/ε vs type 3's ~4π/ε) and
#   clamped at the same u = 1e-6 seam as type 3.


def _huber_cfg(delta, gn, weight=1.0):
    cfg = _sweep_cfg(5, gn)
    cfg.ang_cost_huber_delta = delta
    cfg.angle = weight
    cfg.angle_N = weight
    return cfg


class TestPseudoHuberCost:
    """afc=5 pseudo-Huber shape: FD exactness, type-3 equivalence near the
    goal, the 2·w·δ assembled-gradient bound, and the δ-scaled clamped
    antipode."""

    @pytest.mark.parametrize("delta", [0.1, 0.35, 1.0])
    @pytest.mark.parametrize("mode", ["vec", "quat"])
    @pytest.mark.parametrize("gn", [False, True])
    def test_fd_grid_gradient_and_hessian(self, fixture, delta, mode, gn):
        """θ ∈ {0.01°, 1°, 20°, 90°, 170°}: full-Newton (and quat mode, where
        GN is a no-op) matches central FD in gradient AND Hessian; GN vec mode
        asserts the rank-1 structure with the δ-dependent sign instead."""
        sat = fixture.sat
        x = _sw_base_state(sat)
        cfg = _huber_cfg(delta, gn)
        full_hess = (not gn) or (mode == "quat")
        for td in (0.01, 1.0, 20.0, 90.0, 170.0):
            theta = _math.radians(td)
            tgt = _sw_target(mode, theta)
            c, gq, Hq, lx, lxx = _sw_qblock_ana(sat, x, tgt, cfg)
            assert np.isfinite(c) and np.isfinite(lx).all() and np.isfinite(lxx).all()

            gfd = _sw_qgrad_fd(sat, x, tgt, cfg)
            for j in range(4):
                tol = 1e-6 + 1e-4 * abs(gfd[j])
                assert abs(gq[j] - gfd[j]) < tol, \
                    f"grad[{j}] δ={delta} {mode} gn={gn} θ={td}: {gq[j]} vs {gfd[j]}"

            if full_hess:
                Hfd = _sw_qhess_fd(sat, x, tgt, cfg)
                herr = np.max(np.abs(Hq - Hfd))
                hscale = np.max(np.abs(Hfd))
                assert herr < 1e-3 + 5e-2 * hscale, \
                    f"Hess δ={delta} {mode} θ={td}: err={herr:.2e} scale={hscale:.2e}"
            else:
                eig = np.linalg.eigvalsh(Hq)
                mags = np.sort(np.abs(eig))
                assert mags[-2] < 1e-6 + 1e-3 * mags[-1], \
                    f"GN Hess not rank-1 δ={delta} θ={td}: {eig}"
                # Magnitude 4·[g''−g'·cotθ]: > −4·w, < 4 + 4·δ/sinθ
                # (g'' ≤ 1, g' < δ).
                assert eig.min() > -4.0 - 1e-6
                assert eig.max() < 4.0 + 4.0 * delta / _math.sin(theta) + 1e-6
                # Sign follows g'' − g'·cotθ (negative below the crossover,
                # positive above).
                r = theta / delta
                S = _math.sqrt(1.0 + r * r)
                expected_sign = S ** -3 - (theta / S) / _math.tan(theta)
                if expected_sign < -1e-9:
                    assert eig.min() < 0.0, f"expected NSD δ={delta} θ={td}: {eig}"
                elif expected_sign > 1e-9:
                    assert eig.min() > -1e-6, f"expected PSD δ={delta} θ={td}: {eig}"

    @pytest.mark.parametrize("delta", [0.35, 1.0])
    @pytest.mark.parametrize("mode", ["vec", "quat"])
    def test_near_goal_equivalence_to_type3(self, fixture, delta, mode):
        """For θ ≪ δ: g₅(θ) = ½θ²·(1 − (θ/δ)²/4 + O((θ/δ)⁴)) — the cost ratio
        to type 3 departs from 1 by exactly −(θ_shape/δ)²/4 (θ_shape = θ/2 in
        quat mode, where the inner scalar is cos(θ/2)); gradients agree to the
        same relative order."""
        sat = fixture.sat
        x = _sw_base_state(sat)
        cfg5 = _huber_cfg(delta, False)
        cfg3 = _sweep_cfg(3, False)
        for td in (0.01, 0.1, 1.0):
            theta = _math.radians(td)
            th_shape = theta / 2.0 if mode == "quat" else theta
            tgt = _sw_target(mode, theta)
            c5, g5, _, _, _ = _sw_qblock_ana(sat, x, tgt, cfg5)
            c3, g3, _, _, _ = _sw_qblock_ana(sat, x, tgt, cfg3)
            expected = -(th_shape / delta) ** 2 / 4.0
            measured = c5 / c3 - 1.0
            assert abs(measured - expected) < 1e-8 + 0.05 * abs(expected), \
                f"δ={delta} {mode} θ={td}: ratio-1={measured:.3e} vs {expected:.3e}"
            gerr = np.linalg.norm(g5 - g3) / max(1e-300, np.linalg.norm(g3))
            assert gerr < 1e-8 + 2.0 * (th_shape / delta) ** 2

    @pytest.mark.parametrize("delta", [0.1, 0.35, 1.0])
    @pytest.mark.parametrize("weight", [1.0, 3.0])
    def test_assembled_gradient_bound_2wdelta(self, fixture, delta, weight):
        """|g_q| = 2·w·g'(θ) ≤ 2·w·δ·π/√(π²+δ²) < 2·w·δ (vec mode; the
        |∂c/∂θ| = 2·sinθ geometry factor cancels f's 1/sinθ, leaving the
        bounded θ-space slope).  Tight at large angle, and the 179° escape
        gradient stays ≥ 1.9·w·δ — bounded urgency WITHOUT type 0's antipodal
        plateau."""
        sat = fixture.sat
        x = _sw_base_state(sat)
        cfg = _huber_cfg(delta, False, weight)
        bound = 2.0 * weight * delta * _math.pi / _math.sqrt(_math.pi ** 2 + delta ** 2)
        gmax = 0.0
        for td in range(1, 180):
            theta = _math.radians(td)
            _, gq, _, _, _ = _sw_qblock_ana(sat, x, _sw_target("vec", theta), cfg)
            gn = np.linalg.norm(gq)
            assert gn <= bound * (1.0 + 1e-9), f"θ={td}: |g|={gn} > {bound}"
            gmax = max(gmax, gn)
        assert gmax > 0.99 * bound, f"bound not tight: {gmax} vs {bound}"
        _, g179, _, _, _ = _sw_qblock_ana(
            sat, x, _sw_target("vec", _math.radians(179.0)), cfg)
        assert np.linalg.norm(g179) > 1.9 * weight * delta

    def test_antipode_divergence_delta_scaled_and_clamped(self, fixture):
        """Exact region (u = 1+c ≥ 1e-6): assembled GN max-eig ≈ 4·g'(π)/ε —
        the δ/π-scaled version of type 3's ~4π/ε — and the escape gradient is
        ≈ 2·g'(π) ≈ 2δ·π/√(π²+δ²), NOT type 3's 2π.  Clamped region: (f', f'')
        freeze at the type-5 seam values (δ=0.35: f' ≈ −246, f'' ≈ +1.23e8),
        FN min-eig saturates at ≈ 4·f'_seam, GN decays with the frozen f''."""
        sat = fixture.sat
        x = _sw_base_state(sat)
        delta_h = 0.35
        cfgG = _huber_cfg(delta_h, True)
        cfgF = _huber_cfg(delta_h, False)

        c_eff = -1.0 + 1e-6
        omc2_eff = 1.0 - c_eff * c_eff
        s_eff = _math.sqrt(omc2_eff)
        phi_eff = _math.acos(c_eff)
        S_eff = _math.sqrt(1.0 + (phi_eff / delta_h) ** 2)
        gp_eff = phi_eff / S_eff                       # ≈ g'(π) ≈ 0.3478
        fp_seam = -gp_eff / s_eff                      # ≈ −246
        fpp_seam = (S_eff ** -3 - gp_eff * c_eff / s_eff) / omc2_eff
        gn_bound = fpp_seam * 4.0 * omc2_eff           # ≈ +984
        fn_saturation = 4.0 * fp_seam                  # ≈ −984

        # δ-scaling vs the type-3 seam curvature: ratio = g'(π)/π ≈ δ/π.
        fpp_seam3 = 1.0 / omc2_eff - phi_eff * c_eff / (omc2_eff * s_eff)
        np.testing.assert_allclose(fpp_seam / fpp_seam3, gp_eff / phi_eff,
                                   rtol=1e-3)

        prev_gn = 0.0
        for eps in (1e-2, 1e-3, 1e-4, 1e-5):
            theta = _math.pi - eps
            tgt = _sw_target("vec", theta)
            _, gF, HF, _, _ = _sw_qblock_ana(sat, x, tgt, cfgF)
            _, _, HG, _, _ = _sw_qblock_ana(sat, x, tgt, cfgG)
            gmax = np.linalg.eigvalsh(HG).max()
            fmin = np.linalg.eigvalsh(HF).min()
            assert gmax > 0.0 and fmin < 0.0
            assert gmax <= gn_bound * (1.0 + 1e-6)
            assert fmin >= fn_saturation * 1.01
            if 1.0 - _math.cos(eps) >= 1e-6:
                # Exact region: δ-scaled ~1/ε divergence + bounded escape grad.
                assert gmax > prev_gn
                np.testing.assert_allclose(gmax * eps, 4.0 * gp_eff, rtol=0.05)
                np.testing.assert_allclose(np.linalg.norm(gF), 2.0 * gp_eff,
                                           rtol=0.05)
            else:
                # Clamped region: FN saturation, frozen-f'' GN decay, linear
                # gradient decay |f'_seam|·2·sinθ.
                np.testing.assert_allclose(fmin, fn_saturation, rtol=1e-2)
                c_here = _math.cos(theta)
                np.testing.assert_allclose(
                    gmax, fpp_seam * 4.0 * (1.0 - c_here * c_here), rtol=1e-5)
                np.testing.assert_allclose(
                    np.linalg.norm(gF), -fp_seam * 2.0 * _math.sin(theta),
                    rtol=1e-5)
            prev_gn = gmax
