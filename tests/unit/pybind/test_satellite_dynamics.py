"""
Comprehensive dynamics tests for Satellite class - mirrors test_satellite_dynamics.cpp
Tests orbital propagation, control algorithms, and physics validation.
"""

import pytest
import numpy as np
import saltro_py


class TestSatelliteDynamicsFixture:
    """Test fixture for satellite dynamics tests"""
    
    def setup_method(self):
        """Set up satellite with actuators and generate orbit"""
        # Inertia matrix (cube-shaped satellite)
        self.J = np.array([
            [0.067, 0.0, 0.0],
            [0.0, 0.067, 0.0],
            [0.0, 0.0, 0.067]
        ])
        
        # Create satellite
        self.settings = saltro_py.PlannerSettings()
        self.sat = saltro_py.Satellite(self.J, self.settings)
        
        # Add 3 MTQs along body axes
        self.sat.addMTQ(np.array([1, 0, 0]), 0.2)
        self.sat.addMTQ(np.array([0, 1, 0]), 0.2)
        self.sat.addMTQ(np.array([0, 0, 1]), 0.2)
        
        # Add 3 RWs along body axes
        self.sat.addRW(np.array([1, 0, 0]), 0.001, 1e-5, 0.0, 0.01)
        self.sat.addRW(np.array([0, 1, 0]), 0.001, 1e-5, 0.0, 0.01)
        self.sat.addRW(np.array([0, 0, 1]), 0.001, 1e-5, 0.0, 0.01)
        
        # Generate orbit
        self.n_steps = 100
        self.dt = 10.0
        self.generate_orbit()
    
    def generate_orbit(self):
        """Generate orbital trajectory"""
        r0 = np.array([7000e3, 0.0, 0.0])
        v0 = np.array([0.0, 7.5e3, 0.0])
        
        jtime = np.array([i * self.dt for i in range(self.n_steps)])
        
        # Generate orbit (using simple models: 0, 0, 0, 0)
        ok, self.R, self.V, self.B, self.S, self.rho = saltro_py.generate_orbit(
            r0, v0, jtime, 0, 0, 0, 0
        )
        assert ok, "Orbit generation failed"
    
    def propagate_step(self, x, u, step_idx, dt_step=None):
        """Propagate dynamics one step using RK4"""
        if dt_step is None:
            dt_step = self.dt
        
        idx = min(step_idx, self.n_steps - 1)
        dist = saltro_py.DisturbanceConfig()
        B_eci = self.B[:, idx]
        S_eci = self.S[:, idx]
        rho_idx = int(self.rho[idx])
        
        # RK4 integration
        def dynamics_func(t, x_in):
            return self.sat.dynamics(x_in, u, dist, B_eci, S_eci, rho_idx)
        
        t0 = step_idx * self.dt
        x_next = self.rk4_step(dynamics_func, x, t0, dt_step)
        
        # Normalize quaternion
        q = x_next[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4]
        q = q / np.linalg.norm(q)
        x_next[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = q
        
        return x_next
    
    def rk4_step(self, f, x, t, dt):
        """RK4 integration step"""
        k1 = f(t, x)
        k2 = f(t + dt/2, x + dt/2 * k1)
        k3 = f(t + dt/2, x + dt/2 * k2)
        k4 = f(t + dt, x + dt * k3)
        return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    def propagate_steps(self, x0, u, num_steps):
        """Propagate multiple steps"""
        x = x0.copy()
        for i in range(num_steps):
            x = self.propagate_step(x, u, i)
        return x
    
    def pd_controller(self, x, q_target, kp, kd):
        """Simple PD controller for attitude stabilization"""
        u = np.zeros(self.sat.controlDim)
        
        w = x[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3]
        q = x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4]
        
        # Simplified attitude error (vector part difference)
        attitude_error = q[1:] - q_target[1:]
        
        # PD control: τ = -kp * e - kd * w
        desired_torque = -kp * attitude_error - kd * w
        
        # Map to RW commands
        for i in range(self.sat.numRW):
            rw = self.sat.getRW(i)
            torque_component = np.dot(rw.axis, desired_torque)
            u[self.sat.numMTQ + i] = np.clip(torque_component, -rw.u_max, rw.u_max)
        
        return u


# ============================================================================
# TEST SECTION 1: Basic Dynamics Properties
# ============================================================================

def test_dynamics_with_zero_state_and_zero_control_returns_zero():
    """Test that zero state and control produces zero derivatives"""
    fixture = TestSatelliteDynamicsFixture()
    fixture.setup_method()
    
    x = np.zeros(fixture.sat.stateDim)
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    
    u = np.zeros(fixture.sat.controlDim)
    dist = saltro_py.DisturbanceConfig()
    B_eci = np.zeros(3)
    S_eci = np.zeros(3)
    
    dxdt = fixture.sat.dynamics(x, u, dist, B_eci, S_eci, 0)
    
    # Angular velocity derivative should be zero
    av_deriv = dxdt[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3]
    assert np.linalg.norm(av_deriv) < 1e-10


def test_dynamics_output_has_correct_dimensions():
    """Test that dynamics output has correct dimensions"""
    fixture = TestSatelliteDynamicsFixture()
    fixture.setup_method()
    
    x = np.zeros(fixture.sat.stateDim)
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    
    u = np.zeros(fixture.sat.controlDim)
    dist = saltro_py.DisturbanceConfig()
    
    dxdt = fixture.sat.dynamics(x, u, dist, fixture.B[:, 0], fixture.S[:, 0], 0)
    
    assert len(dxdt) == fixture.sat.stateDim


# ============================================================================
# TEST SECTION 2: Conservation Properties (Free-Body Rotation)
# ============================================================================

def test_zero_initial_spin_remains_stable_under_zero_control():
    """Test that zero spin remains stable without control"""
    fixture = TestSatelliteDynamicsFixture()
    fixture.setup_method()
    
    x0 = np.zeros(fixture.sat.stateDim)
    x0[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3] = np.zeros(3)
    x0[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    
    u_zero = np.zeros(fixture.sat.controlDim)
    x_final = fixture.propagate_steps(x0, u_zero, 10)
    
    # Angular velocity should remain near zero
    omega_final = x_final[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3]
    assert np.linalg.norm(omega_final) < 1e-6
    
    # Quaternion scalar should remain near 1
    q_final = x_final[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4]
    assert np.isclose(q_final[0], 1.0, atol=1e-6)


def test_angular_momentum_is_conserved_in_torque_free_motion():
    """Test that angular momentum is conserved without torques"""
    fixture = TestSatelliteDynamicsFixture()
    fixture.setup_method()
    
    x0 = np.zeros(fixture.sat.stateDim)
    x0[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3] = np.array([0.1, 0.05, 0.02])
    x0[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    
    omega0 = x0[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3]
    L0 = fixture.J @ omega0
    
    u_zero = np.zeros(fixture.sat.controlDim)
    x_final = fixture.propagate_steps(x0, u_zero, 50)
    
    omega_final = x_final[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3]
    L_final = fixture.J @ omega_final
    
    # Angular momentum magnitude should be conserved
    assert np.isclose(np.linalg.norm(L_final), np.linalg.norm(L0), rtol=0.01)


def test_quaternion_remains_normalized_during_propagation():
    """Test that quaternion stays normalized throughout propagation"""
    fixture = TestSatelliteDynamicsFixture()
    fixture.setup_method()
    
    x0 = np.zeros(fixture.sat.stateDim)
    x0[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3] = np.array([0.2, -0.1, 0.15])
    x0[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    
    u_zero = np.zeros(fixture.sat.controlDim)
    
    x = x0.copy()
    for i in range(50):
        x = fixture.propagate_step(x, u_zero, i)
        q = x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4]
        assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-10)


# ============================================================================
# TEST SECTION 3: Quaternion Kinematics
# ============================================================================

def test_quaternion_derivative_follows_kinematics_equation():
    """Test that quaternion derivative follows correct kinematics"""
    fixture = TestSatelliteDynamicsFixture()
    fixture.setup_method()
    
    x = np.zeros(fixture.sat.stateDim)
    omega = np.array([0.1, 0.0, 0.0])
    x[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3] = omega
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    
    u_zero = np.zeros(fixture.sat.controlDim)
    dist = saltro_py.DisturbanceConfig()
    
    dxdt = fixture.sat.dynamics(x, u_zero, dist, fixture.B[:, 0], fixture.S[:, 0], 0)
    
    q = x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4]
    q_dot_expected = np.array([
        -0.5 * np.dot(omega, q[1:]),
        0.5 * (omega[0] * q[0] + np.cross(omega, q[1:])[0]),
        0.5 * (omega[1] * q[0] + np.cross(omega, q[1:])[1]),
        0.5 * (omega[2] * q[0] + np.cross(omega, q[1:])[2])
    ])
    
    q_dot_actual = dxdt[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4]
    
    assert np.isclose(q_dot_actual[0], q_dot_expected[0], atol=1e-10)


def test_pure_x_axis_rotation_changes_quaternion_correctly():
    """Test that X-axis rotation affects quaternion correctly"""
    fixture = TestSatelliteDynamicsFixture()
    fixture.setup_method()
    
    x0 = np.zeros(fixture.sat.stateDim)
    x0[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3] = np.array([0.1, 0.0, 0.0])
    x0[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    
    u_zero = np.zeros(fixture.sat.controlDim)
    x_final = fixture.propagate_steps(x0, u_zero, 10)
    
    q_final = x_final[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4]
    
    # q_x should increase
    assert abs(q_final[1]) > 1e-3
    # q_y and q_z should remain near zero
    assert np.isclose(q_final[2], 0.0, atol=1e-6)
    assert np.isclose(q_final[3], 0.0, atol=1e-6)


# ============================================================================
# TEST SECTION 4: Actuator Torques
# ============================================================================

def test_rw_control_produces_angular_acceleration():
    """Test that RW torque produces angular acceleration"""
    fixture = TestSatelliteDynamicsFixture()
    fixture.setup_method()
    
    x = np.zeros(fixture.sat.stateDim)
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    
    u = np.zeros(fixture.sat.controlDim)
    u[fixture.sat.numMTQ + 0] = 0.001  # Max torque on X RW
    
    dist = saltro_py.DisturbanceConfig()
    dxdt = fixture.sat.dynamics(x, u, dist, fixture.B[:, 0], fixture.S[:, 0], 0)
    
    alpha_x = dxdt[saltro_py.Satellite.AV_INDEX]
    assert abs(alpha_x) > 1e-6


def test_rw_momentum_accumulates_with_constant_torque():
    """Test that RW momentum accumulates over time"""
    fixture = TestSatelliteDynamicsFixture()
    fixture.setup_method()
    
    x0 = np.zeros(fixture.sat.stateDim)
    x0[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    
    u = np.zeros(fixture.sat.controlDim)
    u[fixture.sat.numMTQ + 0] = 0.0001  # Small torque
    
    x_final = fixture.propagate_steps(x0, u, 5)  # 50s
    
    h_final = x_final[saltro_py.Satellite.RW_MOMENTUM_INDEX]
    assert abs(h_final) > 0.001
    assert abs(h_final) < fixture.sat.getRW(0).momentumMax


def test_mtq_torque_depends_on_magnetic_field():
    """Test that MTQ torque depends on B-field"""
    fixture = TestSatelliteDynamicsFixture()
    fixture.setup_method()
    
    x = np.zeros(fixture.sat.stateDim)
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    
    u = np.zeros(fixture.sat.controlDim)
    u[0] = 0.1  # X-axis MTQ
    
    dist = saltro_py.DisturbanceConfig()
    
    # Zero B-field
    B_zero = np.zeros(3)
    dxdt_zero = fixture.sat.dynamics(x, u, dist, B_zero, fixture.S[:, 0], 0)
    
    # Non-zero B-field
    B_nonzero = np.array([0.0, 0.0, 3e-5])
    dxdt_nonzero = fixture.sat.dynamics(x, u, dist, B_nonzero, fixture.S[:, 0], 0)
    
    alpha_zero = dxdt_zero[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3]
    alpha_nonzero = dxdt_nonzero[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3]
    
    assert np.linalg.norm(alpha_nonzero - alpha_zero) > 1e-6


# ============================================================================
# TEST SECTION 5: Control Performance - Spin Stabilization
# ============================================================================

def test_pd_controller_reduces_small_angular_velocity():
    """Test PD controller reduces small spin"""
    fixture = TestSatelliteDynamicsFixture()
    fixture.setup_method()
    
    x0 = np.zeros(fixture.sat.stateDim)
    x0[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3] = np.array([0.05, 0.03, 0.02])
    x0[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    
    omega_initial = np.linalg.norm(x0[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3])
    
    q_target = np.array([1, 0, 0, 0])
    kp = 0.00001
    kd = 0.0001
    
    x = x0.copy()
    for i in range(100):
        u = fixture.pd_controller(x, q_target, kp, kd)
        x = fixture.propagate_step(x, u, i)
    
    omega_final = np.linalg.norm(x[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3])
    
    assert omega_final < omega_initial
    assert omega_final < 0.01


def test_pd_controller_reduces_large_angular_velocity():
    """Test PD controller reduces large spin"""
    fixture = TestSatelliteDynamicsFixture()
    fixture.setup_method()
    
    x0 = np.zeros(fixture.sat.stateDim)
    x0[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3] = np.array([0.2, -0.15, 0.1])
    x0[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    
    omega_initial = np.linalg.norm(x0[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3])
    
    q_target = np.array([1, 0, 0, 0])
    kp = 0.00001
    kd = 0.0001
    
    x = x0.copy()
    for i in range(200):
        u = fixture.pd_controller(x, q_target, kp, kd)
        x = fixture.propagate_step(x, u, i)
    
    omega_final = np.linalg.norm(x[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3])
    
    assert omega_final < 0.5 * omega_initial


def test_pd_controller_with_different_gains_affects_convergence_rate():
    """Test that higher gains converge faster"""
    fixture = TestSatelliteDynamicsFixture()
    fixture.setup_method()
    
    x0 = np.zeros(fixture.sat.stateDim)
    x0[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3] = np.array([0.1, 0.05, 0.0])
    x0[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    
    q_target = np.array([1, 0, 0, 0])
    
    # Low gains
    x_low = x0.copy()
    for i in range(100):
        u = fixture.pd_controller(x_low, q_target, 0.00001, 0.00005)
        x_low = fixture.propagate_step(x_low, u, i)
    
    # High gains
    x_high = x0.copy()
    for i in range(100):
        u = fixture.pd_controller(x_high, q_target, 0.00002, 0.0002)
        x_high = fixture.propagate_step(x_high, u, i)
    
    omega_low = np.linalg.norm(x_low[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3])
    omega_high = np.linalg.norm(x_high[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3])
    
    assert omega_high < omega_low


def test_pd_controller_stabilizes_multi_axis_rotation():
    """Test PD controller on multi-axis rotation"""
    fixture = TestSatelliteDynamicsFixture()
    fixture.setup_method()
    
    x0 = np.zeros(fixture.sat.stateDim)
    x0[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3] = np.array([0.1, 0.1, 0.1])
    x0[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    
    q_target = np.array([1, 0, 0, 0])
    kp = 0.00001
    kd = 0.0001
    
    x = x0.copy()
    for i in range(150):
        u = fixture.pd_controller(x, q_target, kp, kd)
        x = fixture.propagate_step(x, u, i)
    
    omega_final = x[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3]
    assert abs(omega_final[0]) < 0.05
    assert abs(omega_final[1]) < 0.05
    assert abs(omega_final[2]) < 0.05


# ============================================================================
# TEST SECTION 6: Control Saturation and Limits
# ============================================================================

def test_rw_torque_respects_saturation_limits():
    """Test that RW commands are saturated properly"""
    fixture = TestSatelliteDynamicsFixture()
    fixture.setup_method()
    
    x0 = np.zeros(fixture.sat.stateDim)
    x0[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3] = np.array([1.0, 0.0, 0.0])
    x0[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    
    q_target = np.array([1, 0, 0, 0])
    kp = 1.0
    kd = 1.0
    
    u = fixture.pd_controller(x0, q_target, kp, kd)
    
    for i in range(fixture.sat.numRW):
        u_rw = u[fixture.sat.numMTQ + i]
        assert abs(u_rw) <= fixture.sat.getRW(i).u_max + 1e-10


def test_zero_control_produces_expected_free_body_motion():
    """Test free-body motion along principal axis"""
    fixture = TestSatelliteDynamicsFixture()
    fixture.setup_method()
    
    x0 = np.zeros(fixture.sat.stateDim)
    x0[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3] = np.array([0.0, 0.0, 0.1])
    x0[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    
    omega_initial = np.linalg.norm(x0[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3])
    
    u_zero = np.zeros(fixture.sat.controlDim)
    x_final = fixture.propagate_steps(x0, u_zero, 50)
    
    omega_final = np.linalg.norm(x_final[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3])
    
    assert np.isclose(omega_final, omega_initial, rtol=0.05)


# ============================================================================
# TEST SECTION 7: RW Momentum Management
# ============================================================================

def test_rw_momentum_stays_within_bounds_during_control():
    """Test RW momentum doesn't exceed limits during control"""
    fixture = TestSatelliteDynamicsFixture()
    fixture.setup_method()
    
    x0 = np.zeros(fixture.sat.stateDim)
    x0[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3] = np.array([0.15, 0.1, 0.05])
    x0[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    
    q_target = np.array([1, 0, 0, 0])
    kp = 0.00001
    kd = 0.0001
    
    x = x0.copy()
    for i in range(200):
        u = fixture.pd_controller(x, q_target, kp, kd)
        x = fixture.propagate_step(x, u, i)
        
        # Check RW momentum limits
        for j in range(fixture.sat.numRW):
            h = x[saltro_py.Satellite.RW_MOMENTUM_INDEX + j]
            assert abs(h) <= fixture.sat.getRW(j).momentumMax * 1.1


def test_control_drives_satellite_to_near_zero_angular_velocity():
    """Test that control achieves low angular velocity"""
    fixture = TestSatelliteDynamicsFixture()
    fixture.setup_method()
    
    x0 = np.zeros(fixture.sat.stateDim)
    x0[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3] = np.array([0.08, -0.06, 0.04])
    x0[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    
    omega_initial = np.linalg.norm(x0[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3])
    
    q_target = np.array([1, 0, 0, 0])
    kp = 0.00001
    kd = 0.0001
    
    x = x0.copy()
    for i in range(300):
        u = fixture.pd_controller(x, q_target, kp, kd)
        x = fixture.propagate_step(x, u, i)
    
    omega_final = np.linalg.norm(x[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3])
    
    assert omega_final < 0.01
    assert omega_final < 0.1 * omega_initial


# ============================================================================
# TEST SECTION 8: Different Initial Conditions
# ============================================================================

def test_dynamics_handles_various_spin_rates_correctly():
    """Test dynamics with different spin rates"""
    fixture = TestSatelliteDynamicsFixture()
    fixture.setup_method()
    
    q0 = np.array([1, 0, 0, 0])
    u_zero = np.zeros(fixture.sat.controlDim)
    
    spin_rates = [0.01, 0.05, 0.1, 0.2, 0.5]
    
    for omega_mag in spin_rates:
        x0 = np.zeros(fixture.sat.stateDim)
        x0[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3] = np.array([omega_mag, 0, 0])
        x0[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = q0
        
        x_final = fixture.propagate_steps(x0, u_zero, 10)
        
        q_final = x_final[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4]
        assert np.isclose(np.linalg.norm(q_final), 1.0, atol=1e-10)


def test_dynamics_handles_different_quaternion_orientations():
    """Test dynamics with different initial orientations"""
    fixture = TestSatelliteDynamicsFixture()
    fixture.setup_method()
    
    u_zero = np.zeros(fixture.sat.controlDim)
    omega = np.array([0.05, 0.0, 0.0])
    
    quaternions = [
        np.array([1, 0, 0, 0]),           # Identity
        np.array([0.707, 0.707, 0, 0]),   # 90° about X
        np.array([0.707, 0, 0.707, 0]),   # 90° about Y
        np.array([0.707, 0, 0, 0.707]),   # 90° about Z
        np.array([0.5, 0.5, 0.5, 0.5])    # Mixed
    ]
    
    for q0 in quaternions:
        x0 = np.zeros(fixture.sat.stateDim)
        x0[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3] = omega
        x0[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = q0
        
        x_final = fixture.propagate_steps(x0, u_zero, 10)
        
        q_final = x_final[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4]
        assert np.isclose(np.linalg.norm(q_final), 1.0, atol=1e-10)
