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
        # Inertia matrix — non-diagonal, non-isotropic. An isotropic J causes
        # ω × (J·ω) ≡ 0 which hides bugs in the ∂²ω̇/∂ω² Hessian block
        # (e.g., Bug 4 fixed 2026-04-23 where the Hessian was spuriously
        # w-dependent instead of constant). The diagonal chosen to match the
        # sat_3_3_hybrid baseline weights; the small off-diagonals break the
        # isotropy degeneracy without changing the scenario meaningfully.
        self.J = np.array([
            [0.067, 0.004, -0.002],
            [0.004, 0.071, 0.003],
            [-0.002, 0.003, 0.069]
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
        
        # Generate orbit (using simple models: 0, 0, 0, 0, 0)
        ok, self.R, self.V, self.B, self.S, self.rho = saltro_py.generate_orbit(
            r0, v0, jtime, 0, 0, 0, 0, 0
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
            return self.sat.dynamics(x_in, u, dist, self.R[:, idx], B_eci, S_eci, self.V[:, idx], rho_idx)
        
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
    
    dxdt = fixture.sat.dynamics(x, u, dist, np.zeros(3), B_eci, S_eci, np.zeros(3), 0)
    
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
    
    dxdt = fixture.sat.dynamics(x, u, dist, fixture.R[:, 0], fixture.B[:, 0], fixture.S[:, 0], fixture.V[:, 0], 0)
    
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
    
    dxdt = fixture.sat.dynamics(x, u_zero, dist, fixture.R[:, 0], fixture.B[:, 0], fixture.S[:, 0], fixture.V[:, 0], 0)
    
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
    """Test that X-axis rotation affects quaternion correctly.

    The fixture J is intentionally non-diagonal to break the
    ω × (J·ω) ≡ 0 degeneracy (see fixture docstring). With off-diagonal
    J, pure-x initial spin couples energy into the y/z axes — q_y, q_z
    must NOT be asserted zero. The invariant that survives is: q_x grows
    much faster than q_y, q_z and dominates the rotation.
    """
    fixture = TestSatelliteDynamicsFixture()
    fixture.setup_method()

    x0 = np.zeros(fixture.sat.stateDim)
    x0[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3] = np.array([0.1, 0.0, 0.0])
    x0[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])

    u_zero = np.zeros(fixture.sat.controlDim)
    x_final = fixture.propagate_steps(x0, u_zero, 10)

    q_final = x_final[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4]

    # q_x dominates the rotation
    assert abs(q_final[1]) > 1e-3
    # q_y, q_z grow due to inertial coupling but stay much smaller than q_x
    assert abs(q_final[2]) < 0.5 * abs(q_final[1])
    assert abs(q_final[3]) < 0.5 * abs(q_final[1])


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
    dxdt = fixture.sat.dynamics(x, u, dist, fixture.R[:, 0], fixture.B[:, 0], fixture.S[:, 0], fixture.V[:, 0], 0)
    
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
    dxdt_zero = fixture.sat.dynamics(x, u, dist, fixture.R[:, 0], B_zero, fixture.S[:, 0], fixture.V[:, 0], 0)
    
    # Non-zero B-field
    B_nonzero = np.array([0.0, 0.0, 3e-5])
    dxdt_nonzero = fixture.sat.dynamics(x, u, dist, fixture.R[:, 0], B_nonzero, fixture.S[:, 0], fixture.V[:, 0], 0)
    
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
    """Test RW momentum doesn't drift far past limits during control.

    The dynamics itself does NOT clamp RW momentum (h is a free integrator);
    momentum saturation is a controller-side concern. This test only checks
    that under a sensible PD law and a moderate initial spin, the wheels
    never wind up to runaway magnitudes — a slop band of 1.25 × h_max is
    used because the PD controller is unaware of stored momentum.
    """
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

        for j in range(fixture.sat.numRW):
            h = x[saltro_py.Satellite.RW_MOMENTUM_INDEX + j]
            assert abs(h) <= fixture.sat.getRW(j).momentumMax * 1.25


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


# ============================================================================
# TEST SECTION 9: Dynamics Jacobians - Dimensions and Basic Checks
# ============================================================================

def test_jacobians_have_correct_dimensions():
    """Test that Jacobians have correct dimensions"""
    fixture = TestSatelliteDynamicsFixture()
    fixture.setup_method()
    
    x = np.zeros(fixture.sat.stateDim)
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    
    u = np.zeros(fixture.sat.controlDim)
    dist = saltro_py.DisturbanceConfig()
    
    step = 50
    R_eci = fixture.R[:, step]
    B_eci = fixture.B[:, step]
    S_eci = fixture.S[:, step]
    V_eci = fixture.V[:, step]
    
    jac_x, jac_u, jac_dist = fixture.sat.dynamicsJacobians(
        x, u, dist, R_eci, B_eci, S_eci, V_eci
    )
    
    nx = fixture.sat.stateDim
    nu = fixture.sat.controlDim
    
    # jac_x should be (nx x nx)
    assert jac_x.shape == (nx, nx)
    
    # jac_u should be (nx x nu)
    assert jac_u.shape == (nx, nu)
    
    # jac_dist should be (nx x 3)
    assert jac_dist.shape == (nx, 3)


def test_jacobian_blocks_are_finite_and_not_all_nan():
    """Test that Jacobian blocks are finite and not all NaN"""
    fixture = TestSatelliteDynamicsFixture()
    fixture.setup_method()
    
    step = 50
    
    x = np.zeros(fixture.sat.stateDim)
    x[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3] = np.array([0.05, 0.02, 0.01])
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    
    u = np.zeros(fixture.sat.controlDim)
    u[0] = 0.001
    
    dist = saltro_py.DisturbanceConfig()
    dist.plan_for_gg = True
    dist.plan_for_aero = True
    dist.plan_for_srp = True
    
    R_eci = fixture.R[:, step]
    B_eci = fixture.B[:, step]
    S_eci = fixture.S[:, step]
    V_eci = fixture.V[:, step]
    
    jac_x, jac_u, jac_dist = fixture.sat.dynamicsJacobians(
        x, u, dist, R_eci, B_eci, S_eci, V_eci
    )
    
    # All should be finite
    assert np.all(np.isfinite(jac_x))
    assert np.all(np.isfinite(jac_u))
    assert np.all(np.isfinite(jac_dist))
    
    # At least some non-zero entries
    assert np.linalg.norm(jac_x) > 0
    assert np.linalg.norm(jac_u) > 0


# ============================================================================
# TEST SECTION 10: Dynamics Jacobians - Finite Difference Validation
# ============================================================================

def test_jacobian_wrt_state_matches_finite_differences():
    """Test that analytical Jacobian w.r.t. state matches finite differences"""
    fixture = TestSatelliteDynamicsFixture()
    fixture.setup_method()
    
    step = 50
    
    x = np.zeros(fixture.sat.stateDim)
    x[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3] = np.array([0.05, 0.02, 0.01])
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    
    u = np.zeros(fixture.sat.controlDim)
    dist = saltro_py.DisturbanceConfig()
    dist.plan_for_gg = False
    dist.plan_for_aero = False
    dist.plan_for_srp = False
    
    R_eci = fixture.R[:, step]
    B_eci = fixture.B[:, step]
    S_eci = fixture.S[:, step]
    V_eci = fixture.V[:, step]
    
    jac_x_analytical, _, _ = fixture.sat.dynamicsJacobians(
        x, u, dist, R_eci, B_eci, S_eci, V_eci
    )
    
    # Compute Jacobian via finite differences
    eps = 1e-6
    nx = fixture.sat.stateDim
    jac_x_numerical = np.zeros((nx, nx))
    
    for j in range(nx):
        x_plus = x.copy()
        x_minus = x.copy()
        
        x_plus[j] += eps
        x_minus[j] -= eps
        
        f_plus = fixture.sat.dynamics(x_plus, u, dist, R_eci, B_eci, S_eci, V_eci, 0)
        f_minus = fixture.sat.dynamics(x_minus, u, dist, R_eci, B_eci, S_eci, V_eci, 0)
        
        jac_x_numerical[:, j] = (f_plus - f_minus) / (2.0 * eps)
    
    # Compare analytical vs numerical
    rel_tol = 1e-5
    abs_tol = 1e-9
    
    for i in range(nx):
        for j in range(nx):
            analytical = jac_x_analytical[i, j]
            numerical = jac_x_numerical[i, j]
            
            rel_err = abs(analytical - numerical) / abs(numerical) if abs(numerical) > abs_tol else 0.0
            abs_err = abs(analytical - numerical)
            
            assert (rel_err <= rel_tol or abs_err <= abs_tol), \
                f"Mismatch at ({i},{j}): analytical={analytical}, numerical={numerical}"


def test_jacobian_wrt_control_matches_finite_differences():
    """Test that analytical Jacobian w.r.t. control matches finite differences"""
    fixture = TestSatelliteDynamicsFixture()
    fixture.setup_method()
    
    step = 50
    
    x = np.zeros(fixture.sat.stateDim)
    x[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3] = np.array([0.05, 0.02, 0.01])
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    
    u = np.zeros(fixture.sat.controlDim)
    u[0] = 0.01
    
    dist = saltro_py.DisturbanceConfig()
    dist.plan_for_gg = False
    dist.plan_for_aero = False
    dist.plan_for_srp = False
    
    R_eci = fixture.R[:, step]
    B_eci = fixture.B[:, step]
    S_eci = fixture.S[:, step]
    V_eci = fixture.V[:, step]
    
    _, jac_u_analytical, _ = fixture.sat.dynamicsJacobians(
        x, u, dist, R_eci, B_eci, S_eci, V_eci
    )
    
    # Compute Jacobian via finite differences
    eps = 1e-6
    nx = fixture.sat.stateDim
    nu = fixture.sat.controlDim
    jac_u_numerical = np.zeros((nx, nu))
    
    for j in range(nu):
        u_plus = u.copy()
        u_minus = u.copy()
        
        u_plus[j] += eps
        u_minus[j] -= eps
        
        f_plus = fixture.sat.dynamics(x, u_plus, dist, R_eci, B_eci, S_eci, V_eci, 0)
        f_minus = fixture.sat.dynamics(x, u_minus, dist, R_eci, B_eci, S_eci, V_eci, 0)
        
        jac_u_numerical[:, j] = (f_plus - f_minus) / (2.0 * eps)
    
    # Compare analytical vs numerical
    rel_tol = 1e-5
    abs_tol = 1e-9
    
    for i in range(nx):
        for j in range(nu):
            analytical = jac_u_analytical[i, j]
            numerical = jac_u_numerical[i, j]
            
            rel_err = abs(analytical - numerical) / abs(numerical) if abs(numerical) > abs_tol else 0.0
            abs_err = abs(analytical - numerical)
            
            assert (rel_err <= rel_tol or abs_err <= abs_tol)


def test_jacobian_wrt_state_with_disturbances_enabled():
    """Test that Jacobian w.r.t. state works with disturbances enabled"""
    fixture = TestSatelliteDynamicsFixture()
    fixture.setup_method()
    
    step = 50
    
    x = np.zeros(fixture.sat.stateDim)
    x[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3] = np.array([0.05, 0.02, 0.01])
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    
    u = np.zeros(fixture.sat.controlDim)
    dist = saltro_py.DisturbanceConfig()
    dist.plan_for_gg = True
    dist.plan_for_aero = True
    dist.plan_for_srp = True
    
    R_eci = fixture.R[:, step]
    B_eci = fixture.B[:, step]
    S_eci = fixture.S[:, step]
    V_eci = fixture.V[:, step]
    
    jac_x_analytical, _, _ = fixture.sat.dynamicsJacobians(
        x, u, dist, R_eci, B_eci, S_eci, V_eci
    )
    
    # Compute Jacobian via finite differences
    eps = 1e-6
    nx = fixture.sat.stateDim
    jac_x_numerical = np.zeros((nx, nx))
    
    for j in range(nx):
        x_plus = x.copy()
        x_minus = x.copy()
        
        x_plus[j] += eps
        x_minus[j] -= eps
        
        f_plus = fixture.sat.dynamics(x_plus, u, dist, R_eci, B_eci, S_eci, V_eci, 0)
        f_minus = fixture.sat.dynamics(x_minus, u, dist, R_eci, B_eci, S_eci, V_eci, 0)
        
        jac_x_numerical[:, j] = (f_plus - f_minus) / (2.0 * eps)
    
    # Compare analytical vs numerical
    rel_tol = 1e-5
    abs_tol = 1e-9
    
    for i in range(nx):
        for j in range(nx):
            analytical = jac_x_analytical[i, j]
            numerical = jac_x_numerical[i, j]
            
            rel_err = abs(analytical - numerical) / abs(numerical) if abs(numerical) > abs_tol else 0.0
            abs_err = abs(analytical - numerical)
            
            assert (rel_err <= rel_tol or abs_err <= abs_tol)


def _fd_state_jacobian(fixture, x, u, dist, R_eci, B_eci, S_eci, V_eci, eps=1e-6):
    """Central-difference Jacobian of dynamics() w.r.t. the state."""
    nx = fixture.sat.stateDim
    jac = np.zeros((nx, nx))
    for j in range(nx):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[j] += eps
        x_minus[j] -= eps
        f_plus = fixture.sat.dynamics(x_plus, u, dist, R_eci, B_eci, S_eci, V_eci, 0)
        f_minus = fixture.sat.dynamics(x_minus, u, dist, R_eci, B_eci, S_eci, V_eci, 0)
        jac[:, j] = (f_plus - f_minus) / (2.0 * eps)
    return jac


def _assert_jacobians_match(jac_analytical, jac_numerical, rel_tol=1e-5, abs_tol=1e-9):
    rows, cols = jac_numerical.shape
    for i in range(rows):
        for j in range(cols):
            analytical = jac_analytical[i, j]
            numerical = jac_numerical[i, j]
            rel_err = abs(analytical - numerical) / abs(numerical) if abs(numerical) > abs_tol else 0.0
            abs_err = abs(analytical - numerical)
            assert (rel_err <= rel_tol or abs_err <= abs_tol), \
                f"Mismatch at ({i},{j}): analytical={analytical}, numerical={numerical}"


@pytest.mark.parametrize("q0", [
    np.array([1.0, 0.0, 0.0, 0.0]),
    np.array([0.9, 0.2, -0.3, 0.1]) / np.linalg.norm([0.9, 0.2, -0.3, 0.1]),
])
def test_jacobian_wrt_state_with_resdipole_matches_finite_differences(q0):
    """Residual dipole torque tau = m x (R^T B_eci) depends on attitude, so it
    contributes skew(m) * d(R^T B)/dq to the quaternion Jacobian. Validate the
    analytic Jacobian against finite differences with resdipole enabled."""
    fixture = TestSatelliteDynamicsFixture()
    fixture.setup_method()

    step = 50

    x = np.zeros(fixture.sat.stateDim)
    x[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3] = np.array([0.05, 0.02, 0.01])
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = q0

    u = np.zeros(fixture.sat.controlDim)

    dist = saltro_py.DisturbanceConfig()
    dist.plan_for_resdipole = True
    dist.res_dipole = np.array([0.05, -0.02, 0.03])

    R_eci = fixture.R[:, step]
    B_eci = fixture.B[:, step]
    S_eci = fixture.S[:, step]
    V_eci = fixture.V[:, step]
    assert np.linalg.norm(B_eci) > 0

    jac_x_analytical, _, _ = fixture.sat.dynamicsJacobians(
        x, u, dist, R_eci, B_eci, S_eci, V_eci
    )

    # The resdipole quaternion block must actually be exercised (non-zero):
    # with u = 0 and all other disturbances off, the only dwdot/dq term is
    # skew(m) * d(R^T B)/dq.
    av0 = saltro_py.Satellite.AV_INDEX
    q0_idx = saltro_py.Satellite.QUAT_INDEX
    assert np.linalg.norm(jac_x_analytical[av0:av0 + 3, q0_idx:q0_idx + 4]) > 0

    jac_x_numerical = _fd_state_jacobian(fixture, x, u, dist, R_eci, B_eci, S_eci, V_eci)
    _assert_jacobians_match(jac_x_analytical, jac_x_numerical)


def test_jacobian_wrt_state_with_gendist_matches_finite_differences():
    """gendist_torque is a body-fixed constant: dtau/dx = 0. The analytic
    Jacobian must match FD with gendist enabled AND be identical to the
    Jacobian with gendist disabled (no spurious contribution)."""
    fixture = TestSatelliteDynamicsFixture()
    fixture.setup_method()

    step = 50

    x = np.zeros(fixture.sat.stateDim)
    x[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3] = np.array([0.05, 0.02, 0.01])
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])

    u = np.zeros(fixture.sat.controlDim)

    dist_on = saltro_py.DisturbanceConfig()
    dist_on.plan_for_gendist = True
    dist_on.gendist_torque = np.array([-2.0e-5, 3.0e-5, 1.0e-5])

    dist_off = saltro_py.DisturbanceConfig()

    R_eci = fixture.R[:, step]
    B_eci = fixture.B[:, step]
    S_eci = fixture.S[:, step]
    V_eci = fixture.V[:, step]

    jac_x_on, jac_u_on, _ = fixture.sat.dynamicsJacobians(
        x, u, dist_on, R_eci, B_eci, S_eci, V_eci
    )
    jac_x_off, jac_u_off, _ = fixture.sat.dynamicsJacobians(
        x, u, dist_off, R_eci, B_eci, S_eci, V_eci
    )

    # Constant body-fixed torque: exactly zero Jacobian contribution.
    assert np.array_equal(jac_x_on, jac_x_off)
    assert np.array_equal(jac_u_on, jac_u_off)

    jac_x_numerical = _fd_state_jacobian(fixture, x, u, dist_on, R_eci, B_eci, S_eci, V_eci)
    _assert_jacobians_match(jac_x_on, jac_x_numerical)


# ============================================================================
# TEST SECTION 11: Dynamics Hessians - Dimensions and Basic Checks
# ============================================================================

def test_hessians_have_correct_dimensions():
    """Test that Hessians have correct dimensions"""
    fixture = TestSatelliteDynamicsFixture()
    fixture.setup_method()
    
    x = np.zeros(fixture.sat.stateDim)
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    
    u = np.zeros(fixture.sat.controlDim)
    dist = saltro_py.DisturbanceConfig()
    
    hess_xx, hess_ux, hess_uu = fixture.sat.dynamicsHessians(
        x, u, dist, fixture.R[:, 0], fixture.B[:, 0], fixture.S[:, 0], fixture.V[:, 0]
    )
    
    # Hessians are returned as Tensor3 objects which are numpy arrays in Python
    # Each has shape based on the compile-time MAX dimensions
    # We verify they are not empty and have reasonable structure
    assert hess_xx.shape[0] > 0  # At least one slice
    assert hess_ux.shape[0] > 0
    assert hess_uu.shape[0] > 0


def test_hessian_elements_are_finite():
    """Test that Hessian elements are finite"""
    fixture = TestSatelliteDynamicsFixture()
    fixture.setup_method()
    
    step = 50
    
    x = np.zeros(fixture.sat.stateDim)
    x[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3] = np.array([0.05, 0.02, 0.01])
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    
    u = np.zeros(fixture.sat.controlDim)
    dist = saltro_py.DisturbanceConfig()
    dist.plan_for_gg = True
    dist.plan_for_aero = True
    dist.plan_for_srp = True
    
    hess_xx, hess_ux, hess_uu = fixture.sat.dynamicsHessians(
        x, u, dist, fixture.R[:, step], fixture.B[:, step], fixture.S[:, step], fixture.V[:, step]
    )
    
    nx = fixture.sat.stateDim
    
    # Check all slices are finite (up to actual state dimension)
    for i in range(nx):
        assert np.all(np.isfinite(hess_xx[i]))
        assert np.all(np.isfinite(hess_ux[i]))
        assert np.all(np.isfinite(hess_uu[i]))


def test_hessians_are_symmetric_where_expected():
    """Test that Hessian matrices are symmetric where expected"""
    fixture = TestSatelliteDynamicsFixture()
    fixture.setup_method()

    step = 50
    
    x = np.zeros(fixture.sat.stateDim)
    x[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3] = np.array([0.05, 0.02, 0.01])
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    
    u = np.zeros(fixture.sat.controlDim)
    dist = saltro_py.DisturbanceConfig()
    dist.plan_for_gg = True
    dist.plan_for_aero = True
    dist.plan_for_srp = True
    
    hess_xx, hess_ux, hess_uu = fixture.sat.dynamicsHessians(
        x, u, dist, fixture.R[:, step], fixture.B[:, step], fixture.S[:, step], fixture.V[:, step]
    )
    
    nx = fixture.sat.stateDim
    tol = 1e-6
    
    # For smooth dynamics, hess_xx should be symmetric
    for i in range(nx):
        diff = np.linalg.norm(hess_xx[i] - hess_xx[i].T)
        assert diff < tol


# ============================================================================  
# TEST SECTION 12: Dynamics Hessians - Finite Difference Validation
# ============================================================================

# ω outputs (0, 1, 2) and RW-momentum outputs (7, 8, 9) both have analytic
# Hessians that directly match FD on the raw state. Quaternion outputs (3-6)
# differ: `dynamics()` internally normalizes q before computing q̇, but
# `dynamicsHessians()` differentiates the closed-form formula w.r.t. raw q.
# FD-on-raw-q therefore picks up the normalization's projection while analytic
# does not. To test quaternion outputs properly we'd need tangent-space
# perturbations (or a non-normalizing dynamics variant); leaving those out
# here. The covered set still catches Bug 4 (which lives in ω outputs).
@pytest.mark.parametrize("out_idx", [0, 1, 2, 7, 8, 9])
def test_hessian_wrt_state_matches_finite_differences(out_idx):
    """Test that Hessian w.r.t. state matches finite differences for each
    ω / RW-momentum output component. Previously only out_idx=0 was tested,
    which hid Bug 4 (∂²ω̇/∂ω∂ω was w-dependent instead of constant) because
    an isotropic J made ω × (J·ω) identically zero. Fixture now uses
    non-diagonal J.
    """
    fixture = TestSatelliteDynamicsFixture()
    fixture.setup_method()

    step = 50

    x = np.zeros(fixture.sat.stateDim)
    x[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3] = np.array([0.05, 0.02, 0.01])
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])

    u = np.zeros(fixture.sat.controlDim)
    dist = saltro_py.DisturbanceConfig()
    dist.plan_for_gg = True
    dist.plan_for_aero = True
    dist.plan_for_srp = True

    R_eci = fixture.R[:, step]
    B_eci = fixture.B[:, step]
    S_eci = fixture.S[:, step]
    V_eci = fixture.V[:, step]

    hess_xx, _, _ = fixture.sat.dynamicsHessians(
        x, u, dist, R_eci, B_eci, S_eci, V_eci
    )

    eps = 1e-5
    nx = fixture.sat.stateDim

    hess_numerical = np.zeros((nx, nx))

    # Four-point stencil for second derivative
    for j1 in range(nx):
        for j2 in range(j1, nx):
            x_pp = x.copy()
            x_pm = x.copy()
            x_mp = x.copy()
            x_mm = x.copy()

            x_pp[j1] += eps
            x_pp[j2] += eps
            x_pm[j1] += eps
            x_pm[j2] -= eps
            x_mp[j1] -= eps
            x_mp[j2] += eps
            x_mm[j1] -= eps
            x_mm[j2] -= eps

            f_pp = fixture.sat.dynamics(x_pp, u, dist, R_eci, B_eci, S_eci, V_eci, 0)[out_idx]
            f_pm = fixture.sat.dynamics(x_pm, u, dist, R_eci, B_eci, S_eci, V_eci, 0)[out_idx]
            f_mp = fixture.sat.dynamics(x_mp, u, dist, R_eci, B_eci, S_eci, V_eci, 0)[out_idx]
            f_mm = fixture.sat.dynamics(x_mm, u, dist, R_eci, B_eci, S_eci, V_eci, 0)[out_idx]

            second_deriv = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps * eps)

            hess_numerical[j1, j2] = second_deriv
            hess_numerical[j2, j1] = second_deriv

    # Compare
    rel_tol = 5e-3
    abs_tol = 1e-6

    for j1 in range(nx):
        for j2 in range(nx):
            analytical = hess_xx[out_idx][j1, j2]
            numerical = hess_numerical[j1, j2]

            rel_err = abs(analytical - numerical) / abs(numerical) if abs(numerical) > abs_tol else 0.0
            abs_err = abs(analytical - numerical)

            assert (rel_err <= rel_tol or abs_err <= abs_tol), (
                f"out_idx={out_idx}, j1={j1}, j2={j2}: "
                f"ana={analytical:.6e}, fd={numerical:.6e}, "
                f"rel_err={rel_err:.3e}, abs_err={abs_err:.3e}"
            )


@pytest.mark.parametrize("out_idx", [0, 1, 2, 7, 8, 9])
def test_hessian_wrt_state_matches_fd_at_non_identity_attitude(out_idx):
    """Hessian vs finite differences at a NON-IDENTITY quaternion.

    The identity-quaternion tests above are blind to the quaternion-coupled
    (q0) terms of the attitude second derivative: at q=[1,0,0,0] the
    normalization projector is diag(0,1,1,1) and the retraction-curvature
    off-diagonals vanish. The disturbance Hessian assembly + the normalization
    chain rule are exercised in full only off identity. The q-q disturbance
    entries are O(1e-6), so a loose abs tolerance hides errors; we use a
    converged step (eps=3e-4) and a magnitude-relative tolerance.
    """
    fixture = TestSatelliteDynamicsFixture()
    fixture.setup_method()
    step = 50

    q = np.array([0.6, -0.3, 0.5, 0.2])
    q = q / np.linalg.norm(q)
    x = np.zeros(fixture.sat.stateDim)
    x[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3] = np.array([0.05, 0.02, 0.01])
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = q

    u = np.zeros(fixture.sat.controlDim)
    dist = saltro_py.DisturbanceConfig()
    dist.plan_for_gg = True
    R_eci, B_eci, S_eci, V_eci = (fixture.R[:, step], fixture.B[:, step],
                                  fixture.S[:, step], fixture.V[:, step])

    hess_xx, _, _ = fixture.sat.dynamicsHessians(x, u, dist, R_eci, B_eci, S_eci, V_eci)
    H = np.array(hess_xx[out_idx])

    eps = 3e-4
    nx = fixture.sat.stateDim
    QI = saltro_py.Satellite.QUAT_INDEX

    def f(xx):
        return fixture.sat.dynamics(xx, u, dist, R_eci, B_eci, S_eci, V_eci, 0)[out_idx]

    # Only the quaternion block carries the manifold subtlety; check it tightly.
    fd = np.zeros((4, 4))
    for a in range(4):
        for b in range(a, 4):
            xpp, xpm, xmp, xmm = (x.copy() for _ in range(4))
            xpp[QI + a] += eps; xpp[QI + b] += eps
            xpm[QI + a] += eps; xpm[QI + b] -= eps
            xmp[QI + a] -= eps; xmp[QI + b] += eps
            xmm[QI + a] -= eps; xmm[QI + b] -= eps
            v = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4.0 * eps * eps)
            fd[a, b] = v
            fd[b, a] = v

    Hqq = H[QI:QI + 4, QI:QI + 4]
    scale = np.abs(fd).max() + 1e-12
    err = np.abs(Hqq - fd).max()
    # Converged FD is good to ~1e-11 here; the pre-fix bug was ~5.6e-7 on an
    # O(1e-7) entry, so a 1e-3 relative tolerance both passes the fix and would
    # have caught the bug.
    assert err <= 1e-3 * scale + 1e-9, (
        f"out_idx={out_idx}: max|analytic-FD| over q-q block = {err:.3e} "
        f"(scale {scale:.3e})\nanalytic=\n{Hqq}\nfd=\n{fd}"
    )


@pytest.mark.parametrize("out_idx", [3, 4, 5, 6])
def test_qdot_hessian_matches_fd_at_non_identity_attitude(out_idx):
    """q-dot (quaternion-output) Hessians at a non-identity quaternion.

    q-dot = 0.5 W(q) w with q normalized internally. W is linear in q, so the
    raw q-q Hessian is zero, but the normalization retraction term is not -- it
    was previously left at zero (and these outputs were excluded from the FD
    tests). Checks both the q-q and the q-w mixed blocks vs finite differences of
    the renormalizing dynamics. Pure kinematics, so no disturbances are needed.
    """
    fixture = TestSatelliteDynamicsFixture()
    fixture.setup_method()
    q = np.array([0.6, -0.3, 0.5, 0.2])
    q = q / np.linalg.norm(q)
    AV = saltro_py.Satellite.AV_INDEX
    QI = saltro_py.Satellite.QUAT_INDEX
    x = np.zeros(fixture.sat.stateDim)
    x[AV:AV + 3] = np.array([0.05, 0.02, 0.01])
    x[QI:QI + 4] = q
    u = np.zeros(fixture.sat.controlDim)
    dist = saltro_py.DisturbanceConfig()
    R_eci, B_eci, S_eci, V_eci = (fixture.R[:, 50], fixture.B[:, 50],
                                  fixture.S[:, 50], fixture.V[:, 50])

    hess_xx, _, _ = fixture.sat.dynamicsHessians(x, u, dist, R_eci, B_eci, S_eci, V_eci)
    H = np.array(hess_xx[out_idx])

    eps = 1e-4
    idxs = list(range(AV, AV + 3)) + list(range(QI, QI + 4))

    def f(xx):
        return fixture.sat.dynamics(xx, u, dist, R_eci, B_eci, S_eci, V_eci, 0)[out_idx]

    scale = 0.0
    err = 0.0
    for a in idxs:
        for b in idxs:
            xpp, xpm, xmp, xmm = (x.copy() for _ in range(4))
            xpp[a] += eps; xpp[b] += eps
            xpm[a] += eps; xpm[b] -= eps
            xmp[a] -= eps; xmp[b] += eps
            xmm[a] -= eps; xmm[b] -= eps
            fd = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4.0 * eps * eps)
            scale = max(scale, abs(fd))
            err = max(err, abs(H[a, b] - fd))
    assert err <= 1e-4 * (scale + 1.0), f"out_idx={out_idx}: max|analytic-FD| = {err:.3e} (scale {scale:.3e})"


def test_mtq_hessian_wrt_q_matches_fd_at_nonzero_control():
    """MTQ torque q-q Hessian at nonzero MTQ control, non-identity attitude.

    tau_mtq = magvec x B_body (magvec = sum axis_i u_i), linear in B_body, so the
    q-q Hessian is skew(magvec)*d2(R^T B)/dq2. It is zero only at u=0, so the
    previous code (which computed only the d2/du dq mixed block) left it untested.
    """
    fixture = TestSatelliteDynamicsFixture()
    fixture.setup_method()
    AV = saltro_py.Satellite.AV_INDEX
    QI = saltro_py.Satellite.QUAT_INDEX
    q = np.array([0.6, -0.3, 0.5, 0.2])
    q = q / np.linalg.norm(q)
    x = np.zeros(fixture.sat.stateDim)
    x[AV:AV + 3] = np.array([0.05, 0.02, 0.01])
    x[QI:QI + 4] = q
    u = np.zeros(fixture.sat.controlDim)
    u[0], u[1], u[2] = 0.15, -0.1, 0.08  # MTQ dipoles
    dist = saltro_py.DisturbanceConfig()
    R_eci = np.zeros(3)
    B_eci = np.array([2.5e-5, -1.5e-5, 3.0e-5])
    S_eci = np.zeros(3)
    V_eci = np.zeros(3)

    hess_xx, _, _ = fixture.sat.dynamicsHessians(x, u, dist, R_eci, B_eci, S_eci, V_eci)
    eps = 1e-4

    for o in [0, 1, 2]:
        H = np.array(hess_xx[o])

        def f(xx, o=o):
            return fixture.sat.dynamics(xx, u, dist, R_eci, B_eci, S_eci, V_eci, 0)[o]

        block = H[QI:QI + 4, QI:QI + 4]
        scale = np.abs(block).max() + 1e-30
        for a in range(4):
            for b in range(a, 4):
                xpp, xpm, xmp, xmm = (x.copy() for _ in range(4))
                xpp[QI + a] += eps; xpp[QI + b] += eps
                xpm[QI + a] += eps; xpm[QI + b] -= eps
                xmp[QI + a] -= eps; xmp[QI + b] += eps
                xmm[QI + a] -= eps; xmm[QI + b] -= eps
                fd = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4.0 * eps * eps)
                assert abs(block[a, b] - fd) < 1e-4 * scale + 1e-9, (
                    f"out={o} ({a},{b}): analytic={block[a,b]:.6e} fd={fd:.6e}")


def _rot_matrix(q):
    """Body->ECI rotation (Euler-Rodrigues) for a unit quaternion [w,x,y,z]."""
    w, x, y, z = q
    return np.array([
        [w*w + x*x - y*y - z*z, 2*(x*y - w*z),         2*(x*z + w*y)],
        [2*(x*y + w*z),         w*w - x*x + y*y - z*z,  2*(y*z - w*x)],
        [2*(x*z - w*y),         2*(y*z + w*x),          w*w - x*x - y*y + z*z],
    ])


@pytest.mark.parametrize("which", ["aero", "srp"])
def test_disturbance_hessian_matches_fd_at_non_identity_with_geometry(which):
    """Drag/SRP attitude Hessians vs FD at a non-identity quaternion, with real
    surface geometry. The fixtures elsewhere have no faces, so these torques
    (and their Hessians) are otherwise never exercised. A single asymmetric face
    with strong, stable incidence keeps the active-set gate from flipping under
    the FD perturbation. Both rely on the same normalization-chain-rule machinery
    the gg fix corrected; this pins them off identity."""
    fixture = TestSatelliteDynamicsFixture()
    fixture.setup_method()

    cfg = saltro_py.GeometryConfig()
    # area, centroid (offset lever), normal +x, eta_s, eta_d, eta_a, CD
    cfg.addFace(saltro_py.GeometryFace(
        1.0, np.array([0.3, 0.4, 0.1]), np.array([1.0, 0.0, 0.0]),
        0.3, 0.2, 0.1, 2.2))
    fixture.sat.setGeometryConfig(cfg)

    q = np.array([0.6, -0.3, 0.5, 0.2])
    q = q / np.linalg.norm(q)
    QI = saltro_py.Satellite.QUAT_INDEX
    x = np.zeros(fixture.sat.stateDim)
    x[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3] = np.array([0.05, 0.02, 0.01])
    x[QI:QI + 4] = q
    u = np.zeros(fixture.sat.controlDim)

    dist = saltro_py.DisturbanceConfig()
    R_eci = np.zeros(3)
    B_eci = fixture.B[:, 50]
    Rm = _rot_matrix(q)
    if which == "aero":
        dist.plan_for_aero = True
        # body velocity strongly along +x so the face stays active under perturbation
        V_eci = Rm @ np.array([5000.0, 300.0, 150.0])
        S_eci = np.zeros(3)
    else:
        dist.plan_for_srp = True
        S_eci = Rm @ (np.array([0.9, 0.3, 0.2]) / np.linalg.norm([0.9, 0.3, 0.2]))
        V_eci = np.zeros(3)

    hess_xx, _, _ = fixture.sat.dynamicsHessians(x, u, dist, R_eci, B_eci, S_eci, V_eci)

    eps = 1e-3  # eps^2 convergence verified; rel error ~2e-6 here

    def f(xx, o):
        return fixture.sat.dynamics(xx, u, dist, R_eci, B_eci, S_eci, V_eci, 0)[o]

    for o in [0, 1, 2]:
        H = np.array(hess_xx[o])
        block = H[QI:QI + 4, QI:QI + 4]
        scale = np.abs(block).max() + 1e-30
        for a in range(4):
            for b in range(a, 4):
                xpp, xpm, xmp, xmm = (x.copy() for _ in range(4))
                xpp[QI + a] += eps; xpp[QI + b] += eps
                xpm[QI + a] += eps; xpm[QI + b] -= eps
                xmp[QI + a] -= eps; xmp[QI + b] += eps
                xmm[QI + a] -= eps; xmm[QI + b] -= eps
                fd = (f(xpp, o) - f(xpm, o) - f(xmp, o) + f(xmm, o)) / (4.0 * eps * eps)
                rel = abs(block[a, b] - fd) / scale
                assert rel < 1e-4, (
                    f"{which} out={o} ({a},{b}): analytic={block[a,b]:.6e} "
                    f"fd={fd:.6e} rel={rel:.2e}")


# ============================================================================
def test_qdot_output_hessian_matches_fd_at_non_identity():
    """q-dot outputs (3-6): q-dot = 0.5 W(q) w. W is linear in q so the raw q-q
    Hessian is zero, but the normalization retraction term is not. Verify the
    q-q block vs finite differences at a non-identity attitude (these outputs
    are excluded from the omega/h-dot FD test above)."""
    fixture = TestSatelliteDynamicsFixture()
    fixture.setup_method()
    q = np.array([0.6, -0.3, 0.5, 0.2]); q = q / np.linalg.norm(q)
    QI = saltro_py.Satellite.QUAT_INDEX
    x = np.zeros(fixture.sat.stateDim)
    x[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3] = [0.05, 0.02, 0.01]
    x[QI:QI + 4] = q
    u = np.zeros(fixture.sat.controlDim)
    dist = saltro_py.DisturbanceConfig()
    R, B, S, V = (fixture.R[:, 50], fixture.B[:, 50], fixture.S[:, 50], fixture.V[:, 50])
    hess_xx, _, _ = fixture.sat.dynamicsHessians(x, u, dist, R, B, S, V)

    def f(xx, o):
        return fixture.sat.dynamics(xx, u, dist, R, B, S, V, 0)[o]

    eps = 1e-4
    for o in range(QI, QI + 4):           # q-dot outputs
        H = np.array(hess_xx[o])
        for a in range(4):
            for b in range(a, 4):
                xpp, xpm, xmp, xmm = (x.copy() for _ in range(4))
                xpp[QI + a] += eps; xpp[QI + b] += eps
                xpm[QI + a] += eps; xpm[QI + b] -= eps
                xmp[QI + a] -= eps; xmp[QI + b] += eps
                xmm[QI + a] -= eps; xmm[QI + b] -= eps
                fd = (f(xpp, o) - f(xpm, o) - f(xmp, o) + f(xmm, o)) / (4.0 * eps * eps)
                assert abs(H[QI + a, QI + b] - fd) < 1e-7, f"qdot o={o} ({a},{b})"


def test_mtq_dynamics_hessian_matches_fd_at_non_identity_nonzero_control():
    """MTQ torque depends on q (B_body) and is linear in u, so its omega-dot q-q
    Hessian is nonzero only for u != 0 -- not exercised by the u=0 tests. Verify
    at a non-identity attitude with nonzero MTQ control."""
    fixture = TestSatelliteDynamicsFixture()
    fixture.setup_method()
    q = np.array([0.6, -0.3, 0.5, 0.2]); q = q / np.linalg.norm(q)
    AV = saltro_py.Satellite.AV_INDEX; QI = saltro_py.Satellite.QUAT_INDEX
    x = np.zeros(fixture.sat.stateDim)
    x[AV:AV + 3] = [0.05, 0.02, 0.01]
    x[QI:QI + 4] = q
    u = np.zeros(fixture.sat.controlDim)
    u[:fixture.sat.numMTQ] = np.array([0.15, -0.1, 0.08])[:fixture.sat.numMTQ]
    dist = saltro_py.DisturbanceConfig()   # isolate MTQ torque q-dependence
    R, B, S, V = (fixture.R[:, 50], fixture.B[:, 50], fixture.S[:, 50], fixture.V[:, 50])
    hess_xx, _, _ = fixture.sat.dynamicsHessians(x, u, dist, R, B, S, V)

    def f(xx, o):
        return fixture.sat.dynamics(xx, u, dist, R, B, S, V, 0)[o]

    eps = 1e-4
    for o in range(3):                    # omega-dot outputs
        H = np.array(hess_xx[o])
        for a in range(4):
            for b in range(a, 4):
                xpp, xpm, xmp, xmm = (x.copy() for _ in range(4))
                xpp[QI + a] += eps; xpp[QI + b] += eps
                xpm[QI + a] += eps; xpm[QI + b] -= eps
                xmp[QI + a] -= eps; xmp[QI + b] += eps
                xmm[QI + a] -= eps; xmm[QI + b] -= eps
                fd = (f(xpp, o) - f(xpm, o) - f(xmp, o) + f(xmm, o)) / (4.0 * eps * eps)
                assert abs(H[QI + a, QI + b] - fd) < 1e-7, f"mtq o={o} ({a},{b})"


# TEST SECTION 13: Euler's equation - cross-product gyroscopic terms
# ============================================================================

def _diag_J_fixture():
    """A fresh fixture with strictly diagonal J for clean ω×(J·ω) algebra."""
    fixture = TestSatelliteDynamicsFixture()
    fixture.setup_method()
    # Override with a strictly diagonal but non-isotropic J. Distinct moments
    # are required so that ω×(J·ω) is non-zero for non-principal-axis spin.
    Jd = np.diag([0.05, 0.07, 0.09])
    fixture.J = Jd
    fixture.sat.setInertia(Jd)
    return fixture


def test_euler_cross_product_term_appears_in_omega_dot():
    """For diagonal J and zero torque, J_noRW·ω̇ = -ω × (J·ω + h_rw).

    With h_rw = 0 and ω skewed across axes, the Euler term is non-zero and
    has a known sign. This catches sign / order-of-arguments mistakes in
    the gyroscopic cross product. Note the *full* J (not J_noRW) appears
    inside the cross product — RW reaction enters only via the LHS inertia.
    """
    fixture = _diag_J_fixture()
    Jd = fixture.J
    J_noRW = fixture.sat.inertiaNoRW

    x = np.zeros(fixture.sat.stateDim)
    omega = np.array([0.10, 0.07, 0.0])  # nonzero in 2 axes → non-zero ω×Jω
    x[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3] = omega
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])

    u = np.zeros(fixture.sat.controlDim)
    dist = saltro_py.DisturbanceConfig()  # all off

    dxdt = fixture.sat.dynamics(x, u, dist,
                                fixture.R[:, 0], np.zeros(3), fixture.S[:, 0],
                                fixture.V[:, 0], 0)
    omega_dot_actual = dxdt[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3]

    expected = -np.linalg.solve(J_noRW, np.cross(omega, Jd @ omega))

    assert np.allclose(omega_dot_actual, expected, atol=1e-10)


def test_principal_axis_spin_has_zero_euler_torque():
    """Spin purely along a principal axis of (diagonal) J ⇒ ω×J·ω = 0."""
    fixture = _diag_J_fixture()

    for axis_idx in range(3):
        omega = np.zeros(3)
        omega[axis_idx] = 0.15

        x = np.zeros(fixture.sat.stateDim)
        x[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3] = omega
        x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])

        u = np.zeros(fixture.sat.controlDim)
        dist = saltro_py.DisturbanceConfig()

        dxdt = fixture.sat.dynamics(x, u, dist,
                                    fixture.R[:, 0], np.zeros(3), fixture.S[:, 0],
                                    fixture.V[:, 0], 0)
        omega_dot = dxdt[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3]

        assert np.linalg.norm(omega_dot) < 1e-12, (
            f"axis {axis_idx}: expected zero ω̇, got {omega_dot}"
        )


# ============================================================================
# TEST SECTION 14: RW gyroscopic coupling (h_rw × ω term)
# ============================================================================

def test_rw_momentum_couples_to_body_angular_acceleration():
    """Stored RW momentum should produce a gyroscopic torque on the body
    via the ω × h_rw term: ω̇ acquires a contribution -J⁻¹(ω × h_rw).

    Without this coupling, modifying h alone (control u = 0, ω fixed)
    would leave ω̇ unchanged — a bug we want to catch.
    """
    fixture = _diag_J_fixture()

    x_no_h = np.zeros(fixture.sat.stateDim)
    omega = np.array([0.05, 0.0, 0.0])
    x_no_h[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3] = omega
    x_no_h[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])

    x_with_h = x_no_h.copy()
    # Park momentum on the y-axis wheel so ω × h_rw is non-zero
    x_with_h[saltro_py.Satellite.RW_MOMENTUM_INDEX + 1] = 0.005

    u = np.zeros(fixture.sat.controlDim)
    dist = saltro_py.DisturbanceConfig()

    dxdt_no_h = fixture.sat.dynamics(x_no_h, u, dist,
                                      fixture.R[:, 0], np.zeros(3), fixture.S[:, 0],
                                      fixture.V[:, 0], 0)
    dxdt_with_h = fixture.sat.dynamics(x_with_h, u, dist,
                                        fixture.R[:, 0], np.zeros(3), fixture.S[:, 0],
                                        fixture.V[:, 0], 0)

    delta_omega_dot = (
        dxdt_with_h[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3]
        - dxdt_no_h[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3]
    )

    assert np.linalg.norm(delta_omega_dot) > 1e-8, (
        "ω̇ insensitive to RW momentum → gyroscopic coupling missing"
    )


def test_quaternion_stays_normalized_with_nonzero_rw_momentum():
    """A spinning RW must not break the quaternion-normalization invariant
    across propagation. Regression guard if RW gyroscopic coupling is ever
    routed back through the quaternion-derivative path incorrectly.
    """
    fixture = TestSatelliteDynamicsFixture()
    fixture.setup_method()

    x0 = np.zeros(fixture.sat.stateDim)
    x0[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3] = np.array([0.05, 0.02, 0.01])
    x0[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    x0[saltro_py.Satellite.RW_MOMENTUM_INDEX + 0] = 0.004
    x0[saltro_py.Satellite.RW_MOMENTUM_INDEX + 1] = -0.003
    x0[saltro_py.Satellite.RW_MOMENTUM_INDEX + 2] = 0.002

    u_zero = np.zeros(fixture.sat.controlDim)

    x = x0.copy()
    for i in range(40):
        x = fixture.propagate_step(x, u_zero, i)
        q = x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4]
        assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-10)


# ============================================================================
# TEST SECTION 15: Disturbance-input Jacobian (∂f/∂τ_dist)
# ============================================================================

def test_third_jacobian_is_disturbance_torque_sensitivity():
    """The third Jacobian returned by dynamicsJacobians is ∂f/∂τ_dist: the
    sensitivity of the state derivative to an additive (body-frame)
    disturbance torque. The torque enters only the angular-velocity equation
    via the inverse inertia (wdot = invJ·(tau_act + tau_dist − …)), so the
    angular-velocity block equals ∂wdot/∂τ and the quaternion / RW-momentum
    blocks are zero. Validated against a finite difference of the dynamics
    w.r.t. an injected disturbance torque (gendist_torque). This Jacobian feeds
    the disturbance-aware TVLQR (McKeen 2025, eq. 7.40).
    """
    fixture = TestSatelliteDynamicsFixture()
    fixture.setup_method()

    step = 50
    sat = fixture.sat
    AV = saltro_py.Satellite.AV_INDEX
    R, B, S, V = (fixture.R[:, step], fixture.B[:, step],
                  fixture.S[:, step], fixture.V[:, step])

    x = np.zeros(sat.stateDim)
    x[AV:AV + 3] = np.array([0.05, 0.02, 0.01])
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    u = np.zeros(sat.controlDim)
    u[0] = 0.05

    _, _, jac_dist = sat.dynamicsJacobians(x, u, saltro_py.DisturbanceConfig(), R, B, S, V)
    assert jac_dist.shape == (sat.stateDim, 3)
    assert np.all(np.isfinite(jac_dist))

    # Finite-difference ∂f/∂τ by injecting a constant body torque via gendist.
    def f_with_torque(i, sign):
        d = saltro_py.DisturbanceConfig()
        d.plan_for_gendist = True
        t = np.zeros(3)
        t[i] = sign * 1e-3
        d.gendist_torque = t
        return np.array(sat.dynamics(x, u, d, R, B, S, V, 0))

    fd = np.zeros((sat.stateDim, 3))
    for i in range(3):
        fd[:, i] = (f_with_torque(i, +1) - f_with_torque(i, -1)) / 2e-3

    # Angular-velocity block (the dominant, physically-primary term) must match
    # the finite difference exactly.
    assert np.allclose(jac_dist[AV:AV + 3, :], fd[AV:AV + 3, :], atol=1e-6)
    assert np.linalg.norm(jac_dist[AV:AV + 3, :]) > 1e-9
    # jac_dist models only that block; attitude and RW-momentum are zero. The
    # true leakage into those blocks (a wheel-momentum coupling) is several
    # orders below the angular-velocity term and is intentionally omitted; the
    # check below bounds it at <1e-3 of that term.
    assert np.allclose(jac_dist[AV + 3:, :], 0.0, atol=1e-12)
    av_scale = np.max(np.abs(fd[AV:AV + 3, :]))
    assert np.max(np.abs(fd[AV + 3:, :])) < 1e-3 * av_scale


# ============================================================================
# TEST SECTION 16: Symmetries
# ============================================================================

def test_dynamics_torque_free_invariant_under_quaternion_rotation():
    """In a torque-free, disturbance-free, B-zero setting, ω̇ depends only
    on (ω, h_rw, J), not on q. Rotating q must not change ω̇.
    """
    fixture = _diag_J_fixture()

    omega = np.array([0.05, 0.03, 0.02])
    base = np.zeros(fixture.sat.stateDim)
    base[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3] = omega

    quats = [
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.707, 0.707, 0.0, 0.0]),
        np.array([0.5, 0.5, 0.5, 0.5]),
        np.array([0.6, -0.4, 0.5, 0.48]),
    ]

    u = np.zeros(fixture.sat.controlDim)
    dist = saltro_py.DisturbanceConfig()  # all off

    omega_dot_ref = None
    for q in quats:
        x = base.copy()
        x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = q / np.linalg.norm(q)
        dxdt = fixture.sat.dynamics(x, u, dist,
                                    fixture.R[:, 0], np.zeros(3), fixture.S[:, 0],
                                    fixture.V[:, 0], 0)
        omega_dot = dxdt[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3]
        if omega_dot_ref is None:
            omega_dot_ref = omega_dot
        else:
            assert np.allclose(omega_dot, omega_dot_ref, atol=1e-12), (
                f"ω̇ depends on q (should not, for torque-free dynamics): "
                f"q={q}, ω̇={omega_dot}, ref={omega_dot_ref}"
            )


def quaternion_error_short_way(q, q_target):
    """Proper quaternion error: q_err = q_target^{-1} ⊗ q, with short-way
    sign flip if the scalar part is negative. Returns the 3-vector part."""
    qt0, qt1, qt2, qt3 = q_target
    q0, q1, q2, q3 = q
    qe0 = qt0 * q0 + qt1 * q1 + qt2 * q2 + qt3 * q3
    qe1 = qt0 * q1 - qt1 * q0 - qt2 * q3 + qt3 * q2
    qe2 = qt0 * q2 + qt1 * q3 - qt2 * q0 - qt3 * q1
    qe3 = qt0 * q3 - qt1 * q2 + qt2 * q1 - qt3 * q0
    if qe0 < 0.0:
        qe1, qe2, qe3 = -qe1, -qe2, -qe3
    return np.array([qe1, qe2, qe3])


# ============================================================================
# TEST SECTION 13: Quaternion error formula utility
# ============================================================================

def test_quaternion_error_short_way_at_identity_target_matches_vector_part():
    """For q_target = identity, the proper short-way error equals q[1:]
    (when q[0] >= 0). Documents the regime where the legacy
    `pd_controller` helper's q[1:] - q_target[1:] simplification is
    valid."""
    q = np.array([0.99, 0.05, 0.03, 0.02])
    q /= np.linalg.norm(q)
    q_target = np.array([1.0, 0.0, 0.0, 0.0])

    err = quaternion_error_short_way(q, q_target)

    assert np.allclose(err, q[1:], atol=1e-15)


def test_quaternion_error_short_way_flips_sign_when_scalar_negative():
    """When q is on the far hemisphere (q[0] < 0 after q_target_inv ⊗ q),
    the short-way flip must invert the vector part — otherwise the
    controller drives the long way around."""
    q = np.array([-0.5, 0.5, 0.5, 0.5])  # 240° rotation, far hemisphere
    q /= np.linalg.norm(q)
    q_target = np.array([1.0, 0.0, 0.0, 0.0])

    err = quaternion_error_short_way(q, q_target)

    # After flip, err = -q[1:]
    assert np.allclose(err, -q[1:], atol=1e-15)


def test_quaternion_error_short_way_zero_at_target():
    """err(q_target, q_target) == 0 exactly."""
    rng = np.random.default_rng(20260515)
    for _ in range(5):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        err = quaternion_error_short_way(q, q)
        assert np.allclose(err, 0.0, atol=1e-15)
