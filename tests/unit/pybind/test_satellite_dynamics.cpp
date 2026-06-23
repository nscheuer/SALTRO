#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Dense>
#include <array>

#include <saltro/pybind/satellite.h>
#include <saltro/orbit_generation/generate_orbit.h>
#include <saltro/math/integrators/rk4.h>
#include <saltro/math/quaternion.h>
#include <saltro/limits.h>

using namespace saltro;

// Test fixture for satellite dynamics tests
class SatelliteDynamicsFixture {
public:
    Eigen::Matrix3d J;
    PlannerSettings settings;
    Satellite sat;
    
    // Orbit data
    static constexpr int n_steps = 100;
    static constexpr double dt = 10.0;
    Eigen::Matrix<double, 1, limits::MAX_LENGTH_TRAJ> jtime;
    Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> R, V, B, S;
    Eigen::Matrix<double, 1, limits::MAX_LENGTH_TRAJ> rho;
    
    SatelliteDynamicsFixture() 
        : J((Eigen::Matrix3d() << 0.067, 0.0, 0.0,
                                   0.0, 0.067, 0.0,
                                   0.0, 0.0, 0.067).finished()),
          settings(),
          sat(J, settings) 
    {
        // Add actuators
        sat.addMTQ(Eigen::Vector3d(1, 0, 0), 0.2);
        sat.addMTQ(Eigen::Vector3d(0, 1, 0), 0.2);
        sat.addMTQ(Eigen::Vector3d(0, 0, 1), 0.2);
        
        sat.addRW(Eigen::Vector3d(1, 0, 0), 0.001, 1e-5, 0.0, 0.01);
        sat.addRW(Eigen::Vector3d(0, 1, 0), 0.001, 1e-5, 0.0, 0.01);
        sat.addRW(Eigen::Vector3d(0, 0, 1), 0.001, 1e-5, 0.0, 0.01);
        
        // Generate orbit
        generateOrbit();
    }
    
    void generateOrbit() {
        Eigen::Vector3d r0(7000e3, 0.0, 0.0);
        Eigen::Vector3d v0(0.0, 7.5e3, 0.0);
        
        for (int i = 0; i < n_steps; ++i) {
            jtime(i) = i * dt;
        }
        
        bool success = orbits::generate_orbit(
            r0, v0, jtime, n_steps,
            0, 0, 0, 0, 0,
            R, V, B, S, rho
        );
        REQUIRE(success);
    }
    
    // Helper to propagate dynamics one step
    Satellite::VecX propagateStep(const Satellite::VecX& x, const Satellite::VecX& u, 
                                   int step_idx, double dt_step = dt) {
        DisturbanceConfig dist;
        Eigen::Vector3d B_eci = B.col(std::min(step_idx, n_steps-1));
        Eigen::Vector3d S_eci = S.col(std::min(step_idx, n_steps-1));
        int rho_idx = static_cast<int>(rho(std::min(step_idx, n_steps-1)));
        
        auto dynamics_func = [&](double t, const Satellite::VecX& x_in, Satellite::VecX& dxdt) {
            dxdt = sat.dynamics(x_in, u, dist, R.col(std::min(step_idx, n_steps-1)), B_eci, S_eci, V.col(std::min(step_idx, n_steps-1)), rho_idx);
        };
        
        Satellite::VecX x_next(sat.stateDim());
        rk4_step(dynamics_func, x, step_idx * dt, dt_step, x_next);
        
        // Normalize quaternion
        Eigen::Vector4d q = x_next.segment<4>(Satellite::QUAT_INDEX);
        q.normalize();
        x_next.segment<4>(Satellite::QUAT_INDEX) = q;
        
        return x_next;
    }
    
    // Helper to propagate multiple steps
    Satellite::VecX propagateSteps(const Satellite::VecX& x0, const Satellite::VecX& u, 
                                    int num_steps) {
        Satellite::VecX x = x0;
        for (int i = 0; i < num_steps; ++i) {
            x = propagateStep(x, u, i);
        }
        return x;
    }
    
    // PD controller
    Satellite::VecX pdController(const Satellite::VecX& x, const Eigen::Vector4d& q_target, 
                                  double kp, double kd) {
        Satellite::VecX u = Satellite::VecX::Zero(sat.controlDim());
        
        Eigen::Vector3d w = x.segment<3>(Satellite::AV_INDEX);
        Eigen::Vector4d q = x.segment<4>(Satellite::QUAT_INDEX);
        
        // Simplified attitude error (vector part difference)
        Eigen::Vector3d attitude_error = q.tail<3>() - q_target.tail<3>();
        
        // PD control: τ = -kp * e - kd * w
        Eigen::Vector3d desired_torque = -kp * attitude_error - kd * w;
        
        // Map to RW commands
        for (int i = 0; i < sat.numRW(); ++i) {
            const RW& rw = sat.getRW(i);
            double torque_component = rw.axis().dot(desired_torque);
            u(sat.numMTQ() + i) = std::max(-rw.u_max(), std::min(rw.u_max(), torque_component));
        }
        
        return u;
    }
};

// ============================================================================
// TEST SECTION 1: Basic Dynamics Properties
// ============================================================================

TEST_CASE_METHOD(SatelliteDynamicsFixture, "Dynamics with zero state and zero control returns zero", 
                 "[dynamics][basic]") {
    Satellite::VecX x = Satellite::VecX::Zero(sat.stateDim());
    x.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1, 0, 0, 0); // Identity quaternion
    
    Satellite::VecX u = Satellite::VecX::Zero(sat.controlDim());
    
    DisturbanceConfig dist;
    Eigen::Vector3d B_eci = Eigen::Vector3d::Zero();
    Eigen::Vector3d S_eci = Eigen::Vector3d::Zero();
    
    Satellite::VecX dxdt = sat.dynamics(x, u, dist, Eigen::Vector3d::Zero(), B_eci, S_eci, Eigen::Vector3d::Zero(), 0);
    
    // Angular velocity derivative should be zero (no torques)
    REQUIRE_THAT(dxdt.segment<3>(Satellite::AV_INDEX).norm(), 
                 Catch::Matchers::WithinAbs(0.0, 1e-10));
}

TEST_CASE_METHOD(SatelliteDynamicsFixture, "Dynamics output has correct dimensions", 
                 "[dynamics][basic]") {
    Satellite::VecX x = Satellite::VecX::Zero(sat.stateDim());
    x.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1, 0, 0, 0);
    
    Satellite::VecX u = Satellite::VecX::Zero(sat.controlDim());
    DisturbanceConfig dist;
    
    Satellite::VecX dxdt = sat.dynamics(x, u, dist, R.col(0), B.col(0), S.col(0), V.col(0), 0);
    
    REQUIRE(dxdt.size() == sat.stateDim());
}

// ============================================================================
// TEST SECTION 2: Conservation Properties (Free-Body Rotation)
// ============================================================================

TEST_CASE_METHOD(SatelliteDynamicsFixture, "Zero initial spin remains stable under zero control", 
                 "[dynamics][conservation]") {
    Satellite::VecX x0 = Satellite::VecX::Zero(sat.stateDim());
    x0.segment<3>(Satellite::AV_INDEX) = Eigen::Vector3d::Zero();
    x0.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1, 0, 0, 0);
    
    Satellite::VecX u_zero = Satellite::VecX::Zero(sat.controlDim());
    Satellite::VecX x_final = propagateSteps(x0, u_zero, 10);
    
    // Angular velocity should remain near zero
    REQUIRE_THAT(x_final.segment<3>(Satellite::AV_INDEX).norm(), 
                 Catch::Matchers::WithinAbs(0.0, 1e-6));
    
    // Quaternion should remain near identity
    REQUIRE_THAT(x_final(Satellite::QUAT_INDEX), 
                 Catch::Matchers::WithinAbs(1.0, 1e-6));
}

TEST_CASE_METHOD(SatelliteDynamicsFixture, "Angular momentum is conserved in torque-free motion", 
                 "[dynamics][conservation]") {
    Satellite::VecX x0 = Satellite::VecX::Zero(sat.stateDim());
    x0.segment<3>(Satellite::AV_INDEX) = Eigen::Vector3d(0.1, 0.05, 0.02);
    x0.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1, 0, 0, 0);
    
    Eigen::Vector3d L0 = J * x0.segment<3>(Satellite::AV_INDEX);
    
    Satellite::VecX u_zero = Satellite::VecX::Zero(sat.controlDim());
    Satellite::VecX x_final = propagateSteps(x0, u_zero, 50);
    
    Eigen::Vector3d L_final = J * x_final.segment<3>(Satellite::AV_INDEX);
    
    // Angular momentum magnitude should be conserved (allowing for numerical drift)
    REQUIRE_THAT(L_final.norm(), 
                 Catch::Matchers::WithinRel(L0.norm(), 0.01)); // 1% tolerance
}

TEST_CASE_METHOD(SatelliteDynamicsFixture, "Quaternion remains normalized during propagation", 
                 "[dynamics][quaternion]") {
    Satellite::VecX x0 = Satellite::VecX::Zero(sat.stateDim());
    x0.segment<3>(Satellite::AV_INDEX) = Eigen::Vector3d(0.2, -0.1, 0.15);
    x0.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1, 0, 0, 0);
    
    Satellite::VecX u_zero = Satellite::VecX::Zero(sat.controlDim());
    
    for (int i = 0; i < 50; ++i) {
        Satellite::VecX x_next = propagateStep(x0, u_zero, i);
        double quat_norm = x_next.segment<4>(Satellite::QUAT_INDEX).norm();
        
        REQUIRE_THAT(quat_norm, Catch::Matchers::WithinAbs(1.0, 1e-10));
        x0 = x_next;
    }
}

// ============================================================================
// TEST SECTION 3: Quaternion Kinematics
// ============================================================================

TEST_CASE_METHOD(SatelliteDynamicsFixture, "Quaternion derivative follows kinematics equation", 
                 "[dynamics][quaternion]") {
    Satellite::VecX x = Satellite::VecX::Zero(sat.stateDim());
    Eigen::Vector3d omega(0.1, 0.0, 0.0); // Rotation about X-axis
    x.segment<3>(Satellite::AV_INDEX) = omega;
    x.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1, 0, 0, 0);
    
    Satellite::VecX u_zero = Satellite::VecX::Zero(sat.controlDim());
    DisturbanceConfig dist;
    
    Satellite::VecX dxdt = sat.dynamics(x, u_zero, dist, R.col(0), B.col(0), S.col(0), V.col(0), 0);
    
    // Quaternion derivative should follow: q_dot = 0.5 * Omega(w) * q
    Eigen::Vector4d q = x.segment<4>(Satellite::QUAT_INDEX);
    Eigen::Vector4d q_dot_expected;
    q_dot_expected << -0.5 * omega.dot(q.tail<3>()),
                       0.5 * (omega(0) * q(0) + omega.cross(q.tail<3>())(0)),
                       0.5 * (omega(1) * q(0) + omega.cross(q.tail<3>())(1)),
                       0.5 * (omega(2) * q(0) + omega.cross(q.tail<3>())(2));
    
    Eigen::Vector4d q_dot_actual = dxdt.segment<4>(Satellite::QUAT_INDEX);
    
    REQUIRE_THAT(q_dot_actual(0), Catch::Matchers::WithinAbs(q_dot_expected(0), 1e-10));
}

TEST_CASE_METHOD(SatelliteDynamicsFixture, "Pure X-axis rotation changes quaternion correctly", 
                 "[dynamics][quaternion]") {
    Satellite::VecX x0 = Satellite::VecX::Zero(sat.stateDim());
    x0.segment<3>(Satellite::AV_INDEX) = Eigen::Vector3d(0.1, 0.0, 0.0); // X-axis rotation
    x0.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1, 0, 0, 0);
    
    Satellite::VecX u_zero = Satellite::VecX::Zero(sat.controlDim());
    Satellite::VecX x_final = propagateSteps(x0, u_zero, 10);
    
    // For rotation about X-axis, q_x component should increase
    REQUIRE(std::abs(x_final(Satellite::QUAT_INDEX + 1)) > 1e-3);
    // q_y and q_z should remain near zero
    REQUIRE_THAT(x_final(Satellite::QUAT_INDEX + 2), Catch::Matchers::WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(x_final(Satellite::QUAT_INDEX + 3), Catch::Matchers::WithinAbs(0.0, 1e-6));
}

// ============================================================================
// TEST SECTION 4: Actuator Torques
// ============================================================================

TEST_CASE_METHOD(SatelliteDynamicsFixture, "RW control produces angular acceleration", 
                 "[dynamics][actuators][rw]") {
    Satellite::VecX x = Satellite::VecX::Zero(sat.stateDim());
    x.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1, 0, 0, 0);
    
    // Apply RW torque about X-axis
    Satellite::VecX u = Satellite::VecX::Zero(sat.controlDim());
    u(sat.numMTQ() + 0) = 0.001; // Max torque on X-axis RW
    
    DisturbanceConfig dist;
    Satellite::VecX dxdt = sat.dynamics(x, u, dist, R.col(0), B.col(0), S.col(0), V.col(0), 0);
    
    // Should produce angular acceleration about X-axis
    double alpha_x = dxdt(Satellite::AV_INDEX);
    REQUIRE(std::abs(alpha_x) > 1e-6); // Non-zero acceleration
}

TEST_CASE_METHOD(SatelliteDynamicsFixture, "RW momentum accumulates with constant torque", 
                 "[dynamics][actuators][rw]") {
    Satellite::VecX x0 = Satellite::VecX::Zero(sat.stateDim());
    x0.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1, 0, 0, 0);
    
    // Apply small constant RW torque for short time
    Satellite::VecX u = Satellite::VecX::Zero(sat.controlDim());
    u(sat.numMTQ() + 0) = 0.0001; // Small torque to avoid saturation
    
    Satellite::VecX x_final = propagateSteps(x0, u, 5); // Only 5 steps = 50s
    
    // RW momentum should have accumulated: Δh ≈ τ×Δt = 0.0001 × 50 = 0.005
    double h_final = x_final(Satellite::RW_MOMENTUM_INDEX);
    REQUIRE(std::abs(h_final) > 0.001); // Should have accumulated
    REQUIRE(std::abs(h_final) < sat.getRW(0).momentumMax()); // Should not saturate (< 0.01)
}

TEST_CASE_METHOD(SatelliteDynamicsFixture, "MTQ torque depends on magnetic field", 
                 "[dynamics][actuators][mtq]") {
    Satellite::VecX x = Satellite::VecX::Zero(sat.stateDim());
    x.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1, 0, 0, 0);
    
    // Apply MTQ dipole moment
    Satellite::VecX u = Satellite::VecX::Zero(sat.controlDim());
    u(0) = 0.1; // X-axis MTQ
    
    DisturbanceConfig dist;
    
    // Test with zero B-field
    Eigen::Vector3d B_zero = Eigen::Vector3d::Zero();
    Satellite::VecX dxdt_zero = sat.dynamics(x, u, dist, R.col(0), B_zero, S.col(0), V.col(0), 0);
    
    // Test with non-zero B-field
    Eigen::Vector3d B_nonzero(0.0, 0.0, 3e-5); // Z-axis field
    Satellite::VecX dxdt_nonzero = sat.dynamics(x, u, dist, R.col(0), B_nonzero, S.col(0), V.col(0), 0);
    
    // Torque should be different with B-field present
    double alpha_diff = (dxdt_nonzero.segment<3>(Satellite::AV_INDEX) - 
                         dxdt_zero.segment<3>(Satellite::AV_INDEX)).norm();
    REQUIRE(alpha_diff > 1e-6);
}

// ============================================================================
// TEST SECTION 5: Control Performance - Spin Stabilization
// ============================================================================

TEST_CASE_METHOD(SatelliteDynamicsFixture, "PD controller reduces small angular velocity", 
                 "[dynamics][control][pd]") {
    Satellite::VecX x0 = Satellite::VecX::Zero(sat.stateDim());
    x0.segment<3>(Satellite::AV_INDEX) = Eigen::Vector3d(0.05, 0.03, 0.02);
    x0.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1, 0, 0, 0);
    
    double omega_initial = x0.segment<3>(Satellite::AV_INDEX).norm();
    
    Eigen::Vector4d q_target(1, 0, 0, 0);
    double kp = 0.00001;
    double kd = 0.0001;
    
    Satellite::VecX x = x0;
    for (int i = 0; i < 100; ++i) {
        Satellite::VecX u = pdController(x, q_target, kp, kd);
        x = propagateStep(x, u, i);
    }
    
    double omega_final = x.segment<3>(Satellite::AV_INDEX).norm();
    
    // Angular velocity should be reduced
    REQUIRE(omega_final < omega_initial);
    REQUIRE(omega_final < 0.01); // Should be significantly reduced
}

TEST_CASE_METHOD(SatelliteDynamicsFixture, "PD controller reduces large angular velocity", 
                 "[dynamics][control][pd]") {
    Satellite::VecX x0 = Satellite::VecX::Zero(sat.stateDim());
    x0.segment<3>(Satellite::AV_INDEX) = Eigen::Vector3d(0.2, -0.15, 0.1);
    x0.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1, 0, 0, 0);
    
    double omega_initial = x0.segment<3>(Satellite::AV_INDEX).norm();
    
    Eigen::Vector4d q_target(1, 0, 0, 0);
    double kp = 0.00001;
    double kd = 0.0001;
    
    Satellite::VecX x = x0;
    for (int i = 0; i < 200; ++i) {
        Satellite::VecX u = pdController(x, q_target, kp, kd);
        x = propagateStep(x, u, i);
    }
    
    double omega_final = x.segment<3>(Satellite::AV_INDEX).norm();
    
    // Should achieve significant reduction
    REQUIRE(omega_final < 0.5 * omega_initial);
}

TEST_CASE_METHOD(SatelliteDynamicsFixture, "PD controller with different gains affects convergence rate", 
                 "[dynamics][control][pd]") {
    Satellite::VecX x0 = Satellite::VecX::Zero(sat.stateDim());
    x0.segment<3>(Satellite::AV_INDEX) = Eigen::Vector3d(0.1, 0.05, 0.0);
    x0.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1, 0, 0, 0);
    
    Eigen::Vector4d q_target(1, 0, 0, 0);
    
    // Low gains
    Satellite::VecX x_low = x0;
    for (int i = 0; i < 100; ++i) {
        Satellite::VecX u = pdController(x_low, q_target, 0.00001, 0.00005);
        x_low = propagateStep(x_low, u, i);
    }
    
    // High gains
    Satellite::VecX x_high = x0;
    for (int i = 0; i < 100; ++i) {
        Satellite::VecX u = pdController(x_high, q_target, 0.00002, 0.0002);
        x_high = propagateStep(x_high, u, i);
    }
    
    double omega_low = x_low.segment<3>(Satellite::AV_INDEX).norm();
    double omega_high = x_high.segment<3>(Satellite::AV_INDEX).norm();
    
    // Higher gains should converge faster
    REQUIRE(omega_high < omega_low);
}

TEST_CASE_METHOD(SatelliteDynamicsFixture, "PD controller stabilizes multi-axis rotation", 
                 "[dynamics][control][pd]") {
    Satellite::VecX x0 = Satellite::VecX::Zero(sat.stateDim());
    x0.segment<3>(Satellite::AV_INDEX) = Eigen::Vector3d(0.1, 0.1, 0.1); // Equal rotation on all axes
    x0.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1, 0, 0, 0);
    
    Eigen::Vector4d q_target(1, 0, 0, 0);
    double kp = 0.00001;
    double kd = 0.0001;
    
    Satellite::VecX x = x0;
    for (int i = 0; i < 150; ++i) {
        Satellite::VecX u = pdController(x, q_target, kp, kd);
        x = propagateStep(x, u, i);
    }
    
    // All angular velocity components should be reduced
    Eigen::Vector3d omega_final = x.segment<3>(Satellite::AV_INDEX);
    REQUIRE(std::abs(omega_final(0)) < 0.05);
    REQUIRE(std::abs(omega_final(1)) < 0.05);
    REQUIRE(std::abs(omega_final(2)) < 0.05);
}

// ============================================================================
// TEST SECTION 6: Control Saturation and Limits
// ============================================================================

TEST_CASE_METHOD(SatelliteDynamicsFixture, "RW torque respects saturation limits", 
                 "[dynamics][control][limits]") {
    Satellite::VecX x0 = Satellite::VecX::Zero(sat.stateDim());
    x0.segment<3>(Satellite::AV_INDEX) = Eigen::Vector3d(1.0, 0.0, 0.0); // Very high spin
    x0.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1, 0, 0, 0);
    
    Eigen::Vector4d q_target(1, 0, 0, 0);
    double kp = 1.0; // Very high gains to force saturation
    double kd = 1.0;
    
    Satellite::VecX u = pdController(x0, q_target, kp, kd);
    
    // Check that RW commands are within limits
    for (int i = 0; i < sat.numRW(); ++i) {
        double u_rw = u(sat.numMTQ() + i);
        REQUIRE(std::abs(u_rw) <= sat.getRW(i).u_max() + 1e-10);
    }
}

TEST_CASE_METHOD(SatelliteDynamicsFixture, "Zero control produces expected free-body motion", 
                 "[dynamics][validation]") {
    // Principal axis rotation (should be stable)
    Satellite::VecX x0 = Satellite::VecX::Zero(sat.stateDim());
    x0.segment<3>(Satellite::AV_INDEX) = Eigen::Vector3d(0.0, 0.0, 0.1); // Z-axis (principal)
    x0.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1, 0, 0, 0);
    
    Satellite::VecX u_zero = Satellite::VecX::Zero(sat.controlDim());
    Satellite::VecX x_final = propagateSteps(x0, u_zero, 50);
    
    // Angular velocity magnitude should be approximately constant
    double omega_initial = x0.segment<3>(Satellite::AV_INDEX).norm();
    double omega_final = x_final.segment<3>(Satellite::AV_INDEX).norm();
    
    REQUIRE_THAT(omega_final, Catch::Matchers::WithinRel(omega_initial, 0.05));
}

// ============================================================================
// TEST SECTION 7: RW Momentum Management
// ============================================================================

TEST_CASE_METHOD(SatelliteDynamicsFixture, "RW momentum stays within bounds during control", 
                 "[dynamics][control][rw]") {
    Satellite::VecX x0 = Satellite::VecX::Zero(sat.stateDim());
    x0.segment<3>(Satellite::AV_INDEX) = Eigen::Vector3d(0.15, 0.1, 0.05);
    x0.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1, 0, 0, 0);
    
    Eigen::Vector4d q_target(1, 0, 0, 0);
    double kp = 0.00001;
    double kd = 0.0001;
    
    Satellite::VecX x = x0;
    for (int i = 0; i < 200; ++i) {
        Satellite::VecX u = pdController(x, q_target, kp, kd);
        x = propagateStep(x, u, i);
        
        // Check RW momentum limits (allow small integration overshoot)
        for (int j = 0; j < sat.numRW(); ++j) {
            double h = x(Satellite::RW_MOMENTUM_INDEX + j);
            REQUIRE(std::abs(h) <= sat.getRW(j).momentumMax() * 1.1); // 10% tolerance for integration
        }
    }
}

TEST_CASE_METHOD(SatelliteDynamicsFixture, "Control drives satellite to near-zero angular velocity", 
                 "[dynamics][control][convergence]") {
    Satellite::VecX x0 = Satellite::VecX::Zero(sat.stateDim());
    x0.segment<3>(Satellite::AV_INDEX) = Eigen::Vector3d(0.08, -0.06, 0.04);
    x0.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1, 0, 0, 0);
    
    double omega_initial = x0.segment<3>(Satellite::AV_INDEX).norm();
    
    Eigen::Vector4d q_target(1, 0, 0, 0);
    double kp = 0.00001;
    double kd = 0.0001;
    
    Satellite::VecX x = x0;
    for (int i = 0; i < 300; ++i) {
        Satellite::VecX u = pdController(x, q_target, kp, kd);
        x = propagateStep(x, u, i);
    }
    
    double omega_final = x.segment<3>(Satellite::AV_INDEX).norm();
    
    // Should achieve low angular velocity (significant reduction from initial)
    REQUIRE(omega_final < 0.01); // Less than 10 millirad/s
    REQUIRE(omega_final < 0.1 * omega_initial); // At least 90% reduction
}

// ============================================================================
// TEST SECTION 8: Different Initial Conditions
// ============================================================================

TEST_CASE_METHOD(SatelliteDynamicsFixture, "Dynamics handles various spin rates correctly", 
                 "[dynamics][robustness]") {
    Eigen::Vector4d q0(1, 0, 0, 0);
    Satellite::VecX u_zero = Satellite::VecX::Zero(sat.controlDim());
    
    std::vector<double> spin_rates = {0.01, 0.05, 0.1, 0.2, 0.5};
    
    for (double omega_mag : spin_rates) {
        DYNAMIC_SECTION("Spin rate: " << omega_mag << " rad/s") {
            Satellite::VecX x0 = Satellite::VecX::Zero(sat.stateDim());
            x0.segment<3>(Satellite::AV_INDEX) = Eigen::Vector3d(omega_mag, 0, 0);
            x0.segment<4>(Satellite::QUAT_INDEX) = q0;
            
            // Should propagate without errors
            REQUIRE_NOTHROW(propagateSteps(x0, u_zero, 10));
            
            Satellite::VecX x_final = propagateSteps(x0, u_zero, 10);
            
            // Quaternion should remain normalized
            REQUIRE_THAT(x_final.segment<4>(Satellite::QUAT_INDEX).norm(), 
                         Catch::Matchers::WithinAbs(1.0, 1e-10));
        }
    }
}

TEST_CASE_METHOD(SatelliteDynamicsFixture, "Dynamics handles different quaternion orientations", 
                 "[dynamics][robustness]") {
    Satellite::VecX u_zero = Satellite::VecX::Zero(sat.controlDim());
    Eigen::Vector3d omega(0.05, 0.0, 0.0);
    
    // Different initial quaternions (all normalized)
    std::vector<Eigen::Vector4d> quaternions = {
        Eigen::Vector4d(1, 0, 0, 0),                                    // Identity
        Eigen::Vector4d(0.707, 0.707, 0, 0),                           // 90° about X
        Eigen::Vector4d(0.707, 0, 0.707, 0),                           // 90° about Y
        Eigen::Vector4d(0.707, 0, 0, 0.707),                           // 90° about Z
        Eigen::Vector4d(0.5, 0.5, 0.5, 0.5)                            // Mixed
    };
    
    for (const auto& q0 : quaternions) {
        Satellite::VecX x0 = Satellite::VecX::Zero(sat.stateDim());
        x0.segment<3>(Satellite::AV_INDEX) = omega;
        x0.segment<4>(Satellite::QUAT_INDEX) = q0;
        
        REQUIRE_NOTHROW(propagateSteps(x0, u_zero, 10));
        
        Satellite::VecX x_final = propagateSteps(x0, u_zero, 10);
        REQUIRE_THAT(x_final.segment<4>(Satellite::QUAT_INDEX).norm(), 
                     Catch::Matchers::WithinAbs(1.0, 1e-10));
    }
}

// ============================================================================
// TEST SECTION 9: Dynamics Jacobians - Dimensions and Basic Checks
// ============================================================================

TEST_CASE_METHOD(SatelliteDynamicsFixture, "Jacobians have correct dimensions", 
                 "[jacobians][dimensions]") {
    Satellite::VecX x = Satellite::VecX::Zero(sat.stateDim());
    x.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1, 0, 0, 0);  // Identity quaternion
    
    Satellite::VecX u = Satellite::VecX::Zero(sat.controlDim());
    DisturbanceConfig dist;
    
    // Test with environmental vectors at mid-orbit
    size_t step = 50;
    Eigen::Vector3d R_eci = R.col(step);
    Eigen::Vector3d B_eci = B.col(step);
    Eigen::Vector3d S_eci = S.col(step);
    Eigen::Vector3d V_eci = V.col(step);
    
    auto [jac_x, jac_u, jac_dist] = sat.dynamicsJacobians(
        x, u, dist, R_eci, B_eci, S_eci, V_eci
    );
    
    int nx = sat.stateDim();
    int nu = sat.controlDim();
    
    // jac_x should be (nx x nx)
    REQUIRE(jac_x.rows() == nx);
    REQUIRE(jac_x.cols() == nx);
    
    // jac_u should be (nx x nu)
    REQUIRE(jac_u.rows() == nx);
    REQUIRE(jac_u.cols() == nu);
    
    // jac_dist should be (nx x 3) for disturbance effects
    REQUIRE(jac_dist.rows() == nx);
    REQUIRE(jac_dist.cols() == 3);
}

TEST_CASE_METHOD(SatelliteDynamicsFixture, "Jacobian blocks are finite and not all NaN", 
                 "[jacobians][sanity]") {
    // Use state from middle of orbit for stable environment
    size_t step = 50;
    
    Satellite::VecX x = Satellite::VecX::Zero(sat.stateDim());
    x.segment<3>(Satellite::AV_INDEX) << 0.05, 0.02, 0.01;  // Angular velocity
    x.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1, 0, 0, 0);  // Identity quaternion
    
    Satellite::VecX u = Satellite::VecX::Zero(sat.controlDim());
    u(0) = 0.001;  // small MTQ command
    
    // Enable all disturbances
    DisturbanceConfig dist;
    dist.plan_for_gg = true;
    dist.plan_for_aero = true;
    dist.plan_for_srp = true;
    
    Eigen::Vector3d R_eci = R.col(step);
    Eigen::Vector3d B_eci = B.col(step);
    Eigen::Vector3d S_eci = S.col(step);
    Eigen::Vector3d V_eci = V.col(step);
    
    auto [jac_x, jac_u, jac_dist] = sat.dynamicsJacobians(
        x, u, dist, R_eci, B_eci, S_eci, V_eci
    );
    
    // Check that Jacobians are finite (no NaN or inf)
    bool jac_x_finite = jac_x.allFinite();
    bool jac_u_finite = jac_u.allFinite();
    bool jac_dist_finite = jac_dist.allFinite();
    
    CHECK(jac_x_finite);
    CHECK(jac_u_finite);
    CHECK(jac_dist_finite);
    
    // Check that Jacobians have some non-zero content
    REQUIRE(jac_x.norm() > 0.0);
    REQUIRE(jac_u.norm() > 0.0);
}

// ============================================================================
// TEST SECTION 10: Dynamics Jacobians - Finite Difference Validation
// ============================================================================

TEST_CASE_METHOD(SatelliteDynamicsFixture, "Jacobian w.r.t. state matches finite differences", 
                 "[jacobians][finite-diff]") {
    // Use mid-orbit point for stable environment
    size_t step = 50;
    
    Satellite::VecX x = Satellite::VecX::Zero(sat.stateDim());
    x.segment<3>(Satellite::AV_INDEX) << 0.05, 0.02, 0.01;  // Angular velocity
    x.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1, 0, 0, 0);  // Identity quaternion
    
    Satellite::VecX u = Satellite::VecX::Zero(sat.controlDim());
    DisturbanceConfig dist;
    dist.plan_for_gg = false;
    dist.plan_for_aero = false;
    dist.plan_for_srp = false;
    
    Eigen::Vector3d R_eci = R.col(step);
    Eigen::Vector3d B_eci = B.col(step);
    Eigen::Vector3d S_eci = S.col(step);
    Eigen::Vector3d V_eci = V.col(step);
    
    auto [jac_x_analytical, jac_u, jac_dist] = sat.dynamicsJacobians(
        x, u, dist, R_eci, B_eci, S_eci, V_eci
    );
    
    // Compute Jacobian via finite differences
    const double eps = 1e-6;
    int nx = sat.stateDim();
    Satellite::MatX jac_x_numerical = Satellite::MatX::Zero(nx, nx);
    
    for (int j = 0; j < nx; ++j) {
        Satellite::VecX x_plus = x;
        Satellite::VecX x_minus = x;
        
        x_plus(j) += eps;
        x_minus(j) -= eps;
        
        // No external renormalization: dynamics() normalizes q internally, and the
        // analytical Jacobian already accounts for the normalization projection.
        
        Satellite::VecX f_plus = sat.dynamics(x_plus, u, dist, R_eci, B_eci, S_eci, V_eci, 0);
        Satellite::VecX f_minus = sat.dynamics(x_minus, u, dist, R_eci, B_eci, S_eci, V_eci, 0);
        
        jac_x_numerical.col(j) = (f_plus - f_minus) / (2.0 * eps);
    }
    
    // Compare analytical vs numerical
    // FD truncation error is O(eps^2) ~ 1e-12; round-off O(eps_mach/eps) ~ 1e-10.
    const double rel_tol = 1e-5;
    const double abs_tol = 1e-9;
    
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < nx; ++j) {
            double analytical = jac_x_analytical(i, j);
            double numerical = jac_x_numerical(i, j);
            
            double rel_err = std::abs(numerical) > abs_tol ? 
                std::abs(analytical - numerical) / std::abs(numerical) : 0.0;
            double abs_err = std::abs(analytical - numerical);
            
            REQUIRE((rel_err <= rel_tol || abs_err <= abs_tol));
        }
    }
}

TEST_CASE_METHOD(SatelliteDynamicsFixture, "Jacobian w.r.t. control matches finite differences", 
                 "[jacobians][finite-diff]") {
    // Use mid-orbit point
    size_t step = 50;
    
    Satellite::VecX x = Satellite::VecX::Zero(sat.stateDim());
    x.segment<3>(Satellite::AV_INDEX) << 0.05, 0.02, 0.01;  // Angular velocity
    x.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1, 0, 0, 0);  // Identity quaternion
    
    Satellite::VecX u = Satellite::VecX::Zero(sat.controlDim());
    u(0) = 0.01;  // Non-zero control input
    
    DisturbanceConfig dist;
    dist.plan_for_gg = false;
    dist.plan_for_aero = false;
    dist.plan_for_srp = false;
    
    Eigen::Vector3d R_eci = R.col(step);
    Eigen::Vector3d B_eci = B.col(step);
    Eigen::Vector3d S_eci = S.col(step);
    Eigen::Vector3d V_eci = V.col(step);
    
    auto [jac_x, jac_u_analytical, jac_dist] = sat.dynamicsJacobians(
        x, u, dist, R_eci, B_eci, S_eci, V_eci
    );
    
    // Compute Jacobian via finite differences
    const double eps = 1e-6;
    int nx = sat.stateDim();
    int nu = sat.controlDim();
    Satellite::MatX jac_u_numerical = Satellite::MatX::Zero(nx, nu);
    
    for (int j = 0; j < nu; ++j) {
        Satellite::VecX u_plus = u;
        Satellite::VecX u_minus = u;
        
        u_plus(j) += eps;
        u_minus(j) -= eps;
        
        Satellite::VecX f_plus = sat.dynamics(x, u_plus, dist, R_eci, B_eci, S_eci, V_eci, 0);
        Satellite::VecX f_minus = sat.dynamics(x, u_minus, dist, R_eci, B_eci, S_eci, V_eci, 0);
        
        jac_u_numerical.col(j) = (f_plus - f_minus) / (2.0 * eps);
    }
    
    // Compare analytical vs numerical
    const double rel_tol = 1e-5;
    const double abs_tol = 1e-9;
    
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < nu; ++j) {
            double analytical = jac_u_analytical(i, j);
            double numerical = jac_u_numerical(i, j);
            
            double rel_err = std::abs(numerical) > abs_tol ? 
                std::abs(analytical - numerical) / std::abs(numerical) : 0.0;
            double abs_err = std::abs(analytical - numerical);
            
            REQUIRE((rel_err <= rel_tol || abs_err <= abs_tol));
        }
    }
}

TEST_CASE_METHOD(SatelliteDynamicsFixture, "Jacobian w.r.t. state with disturbances enabled", 
                 "[jacobians][finite-diff][disturbances]") {
    // Use mid-orbit point with disturbances
    size_t step = 50;
    
    Satellite::VecX x = Satellite::VecX::Zero(sat.stateDim());
    x.segment<3>(Satellite::AV_INDEX) << 0.05, 0.02, 0.01;  // Angular velocity
    x.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1, 0, 0, 0);  // Identity quaternion
    
    Satellite::VecX u = Satellite::VecX::Zero(sat.controlDim());
    DisturbanceConfig dist;
    dist.plan_for_gg = true;
    dist.plan_for_aero = true;
    dist.plan_for_srp = true;
    
    Eigen::Vector3d R_eci = R.col(step);
    Eigen::Vector3d B_eci = B.col(step);
    Eigen::Vector3d S_eci = S.col(step);
    Eigen::Vector3d V_eci = V.col(step);
    
    auto [jac_x_analytical, jac_u, jac_dist] = sat.dynamicsJacobians(
        x, u, dist, R_eci, B_eci, S_eci, V_eci
    );
    
    // Compute Jacobian via finite differences
    const double eps = 1e-6;
    int nx = sat.stateDim();
    Satellite::MatX jac_x_numerical = Satellite::MatX::Zero(nx, nx);
    
    for (int j = 0; j < nx; ++j) {
        Satellite::VecX x_plus = x;
        Satellite::VecX x_minus = x;
        
        x_plus(j) += eps;
        x_minus(j) -= eps;
        
        // No external renormalization: dynamics() normalizes q internally.
        
        Satellite::VecX f_plus = sat.dynamics(x_plus, u, dist, R_eci, B_eci, S_eci, V_eci, 0);
        Satellite::VecX f_minus = sat.dynamics(x_minus, u, dist, R_eci, B_eci, S_eci, V_eci, 0);
        
        jac_x_numerical.col(j) = (f_plus - f_minus) / (2.0 * eps);
    }
    
    // Compare analytical vs numerical
    const double rel_tol = 1e-5;
    const double abs_tol = 1e-9;
    
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < nx; ++j) {
            double analytical = jac_x_analytical(i, j);
            double numerical = jac_x_numerical(i, j);
            
            double rel_err = std::abs(numerical) > abs_tol ? 
                std::abs(analytical - numerical) / std::abs(numerical) : 0.0;
            double abs_err = std::abs(analytical - numerical);
            
            REQUIRE((rel_err <= rel_tol || abs_err <= abs_tol));
        }
    }
}

TEST_CASE_METHOD(SatelliteDynamicsFixture, "Residual dipole state Jacobian matches finite differences",
                 "[jacobians][finite-diff][resdipole]") {
    const int step = 50;
    const Eigen::Vector3d R_eci = R.col(step);
    const Eigen::Vector3d B_eci = B.col(step);
    const Eigen::Vector3d S_eci = S.col(step);
    const Eigen::Vector3d V_eci = V.col(step);
    REQUIRE(B_eci.norm() > 0.0);

    Satellite::VecX u = Satellite::VecX::Zero(sat.controlDim());
    DisturbanceConfig dist;
    dist.plan_for_resdipole = true;
    dist.res_dipole = Eigen::Vector3d(0.05, -0.02, 0.03);

    const std::array<Eigen::Vector4d, 2> attitudes = {
        Eigen::Vector4d(1.0, 0.0, 0.0, 0.0),
        Eigen::Vector4d(0.9, 0.2, -0.3, 0.1).normalized()
    };

    for (const Eigen::Vector4d& q : attitudes) {
        Satellite::VecX x = Satellite::VecX::Zero(sat.stateDim());
        x.segment<3>(Satellite::AV_INDEX) << 0.05, 0.02, 0.01;
        x.segment<4>(Satellite::QUAT_INDEX) = q;

        auto [jac_x_analytical, jac_u, jac_dist] = sat.dynamicsJacobians(
            x, u, dist, R_eci, B_eci, S_eci, V_eci
        );
        REQUIRE(jac_x_analytical.block(Satellite::AV_INDEX, Satellite::QUAT_INDEX, 3, 4).norm() > 0.0);

        const double eps = 1e-6;
        Satellite::MatX jac_x_numerical = Satellite::MatX::Zero(sat.stateDim(), sat.stateDim());
        for (int j = 0; j < sat.stateDim(); ++j) {
            Satellite::VecX x_plus = x;
            Satellite::VecX x_minus = x;
            x_plus(j) += eps;
            x_minus(j) -= eps;
            const Satellite::VecX f_plus = sat.dynamics(x_plus, u, dist, R_eci, B_eci, S_eci, V_eci, 0);
            const Satellite::VecX f_minus = sat.dynamics(x_minus, u, dist, R_eci, B_eci, S_eci, V_eci, 0);
            jac_x_numerical.col(j) = (f_plus - f_minus) / (2.0 * eps);
        }

        const double rel_tol = 1e-5;
        const double abs_tol = 1e-9;
        for (int i = 0; i < sat.stateDim(); ++i) {
            for (int j = 0; j < sat.stateDim(); ++j) {
                const double analytical = jac_x_analytical(i, j);
                const double numerical = jac_x_numerical(i, j);
                const double abs_err = std::abs(analytical - numerical);
                const double rel_err = std::abs(numerical) > abs_tol
                    ? abs_err / std::abs(numerical) : 0.0;
                CAPTURE(q.transpose(), i, j, analytical, numerical, abs_err, rel_err);
                REQUIRE((rel_err <= rel_tol || abs_err <= abs_tol));
            }
        }
    }
}

TEST_CASE_METHOD(SatelliteDynamicsFixture, "Generic disturbance state Jacobian matches finite differences",
                 "[jacobians][finite-diff][gendist]") {
    const int step = 50;
    const Eigen::Vector3d R_eci = R.col(step);
    const Eigen::Vector3d B_eci = B.col(step);
    const Eigen::Vector3d S_eci = S.col(step);
    const Eigen::Vector3d V_eci = V.col(step);

    Satellite::VecX x = Satellite::VecX::Zero(sat.stateDim());
    x.segment<3>(Satellite::AV_INDEX) << 0.05, 0.02, 0.01;
    x.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1, 0, 0, 0);
    Satellite::VecX u = Satellite::VecX::Zero(sat.controlDim());

    DisturbanceConfig dist_on;
    dist_on.plan_for_gendist = true;
    dist_on.gendist_torque = Eigen::Vector3d(-2.0e-5, 3.0e-5, 1.0e-5);
    DisturbanceConfig dist_off;

    auto [jac_x_on, jac_u_on, jac_dist_on] = sat.dynamicsJacobians(
        x, u, dist_on, R_eci, B_eci, S_eci, V_eci
    );
    auto [jac_x_off, jac_u_off, jac_dist_off] = sat.dynamicsJacobians(
        x, u, dist_off, R_eci, B_eci, S_eci, V_eci
    );
    REQUIRE((jac_x_on.array() == jac_x_off.array()).all());
    REQUIRE((jac_u_on.array() == jac_u_off.array()).all());

    const double eps = 1e-6;
    Satellite::MatX jac_x_numerical = Satellite::MatX::Zero(sat.stateDim(), sat.stateDim());
    for (int j = 0; j < sat.stateDim(); ++j) {
        Satellite::VecX x_plus = x;
        Satellite::VecX x_minus = x;
        x_plus(j) += eps;
        x_minus(j) -= eps;
        const Satellite::VecX f_plus = sat.dynamics(x_plus, u, dist_on, R_eci, B_eci, S_eci, V_eci, 0);
        const Satellite::VecX f_minus = sat.dynamics(x_minus, u, dist_on, R_eci, B_eci, S_eci, V_eci, 0);
        jac_x_numerical.col(j) = (f_plus - f_minus) / (2.0 * eps);
    }

    const double rel_tol = 1e-5;
    const double abs_tol = 1e-9;
    for (int i = 0; i < sat.stateDim(); ++i) {
        for (int j = 0; j < sat.stateDim(); ++j) {
            const double analytical = jac_x_on(i, j);
            const double numerical = jac_x_numerical(i, j);
            const double abs_err = std::abs(analytical - numerical);
            const double rel_err = std::abs(numerical) > abs_tol
                ? abs_err / std::abs(numerical) : 0.0;
            CAPTURE(i, j, analytical, numerical, abs_err, rel_err);
            REQUIRE((rel_err <= rel_tol || abs_err <= abs_tol));
        }
    }
}

// ============================================================================
// TEST SECTION 11: Dynamics Hessians - Dimensions and Basic Checks
// ============================================================================

TEST_CASE_METHOD(SatelliteDynamicsFixture, "Hessians have correct dimensions", 
                 "[hessians][dimensions]") {
    Satellite::VecX x = Satellite::VecX::Zero(sat.stateDim());
    x(3) = 1.0;  // q0 (identity quaternion)
    
    Satellite::VecX u = Satellite::VecX::Zero(sat.controlDim());
    DisturbanceConfig dist;
    
    auto [hess_xx, hess_ux, hess_uu] = sat.dynamicsHessians(
        x, u, dist, R.col(0), B.col(0), S.col(0), V.col(0)
    );
    
    // The Tensor3 types use compile-time MAX sizes for fixed-size storage.
    // Runtime nx/nu must be <= MAX sizes; slices have the compile-time dimensions.
    constexpr int nx_max = saltro::limits::MAX_STATE_DIM;
    constexpr int nu_max = saltro::limits::MAX_CTRL_DIM;
    int nx = sat.stateDim();
    int nu = sat.controlDim();
    
    REQUIRE(nx <= nx_max);
    REQUIRE(nu <= nu_max);
    
    // hess_xx: Tensor3<MAX_STATE_DIM, MAX_STATE_DIM, MAX_STATE_DIM>
    // Each slice is a (MAX_STATE_DIM x MAX_STATE_DIM) fixed-size matrix.
    for (int i = 0; i < nx; ++i) {
        REQUIRE(hess_xx.slice(i).rows() == nx_max);
        REQUIRE(hess_xx.slice(i).cols() == nx_max);
    }
    
    // hess_ux: Tensor3<MAX_CTRL_DIM, MAX_STATE_DIM, MAX_STATE_DIM>
    // Each slice is a (MAX_CTRL_DIM x MAX_STATE_DIM) fixed-size matrix.
    for (int i = 0; i < nx; ++i) {
        REQUIRE(hess_ux.slice(i).rows() == nu_max);
        REQUIRE(hess_ux.slice(i).cols() == nx_max);
    }
    
    // hess_uu: Tensor3<MAX_CTRL_DIM, MAX_CTRL_DIM, MAX_STATE_DIM>
    // Each slice is a (MAX_CTRL_DIM x MAX_CTRL_DIM) fixed-size matrix.
    for (int i = 0; i < nx; ++i) {
        REQUIRE(hess_uu.slice(i).rows() == nu_max);
        REQUIRE(hess_uu.slice(i).cols() == nu_max);
    }
}

TEST_CASE_METHOD(SatelliteDynamicsFixture, "Hessian elements are finite", 
                 "[hessians][sanity]") {
    // Use mid-orbit point
    size_t step = 50;
    
    Satellite::VecX x = Satellite::VecX::Zero(sat.stateDim());
    x.segment<3>(Satellite::AV_INDEX) << 0.05, 0.02, 0.01;  // Angular velocity
    x.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1, 0, 0, 0);  // Identity quaternion
    
    Satellite::VecX u = Satellite::VecX::Zero(sat.controlDim());
    DisturbanceConfig dist;
    dist.plan_for_gg = true;
    dist.plan_for_aero = true;
    dist.plan_for_srp = true;
    
    auto [hess_xx, hess_ux, hess_uu] = sat.dynamicsHessians(
        x, u, dist, R.col(step), B.col(step), S.col(step), V.col(step)
    );
    
    int nx = sat.stateDim();
    
    // Check all Hessian blocks are finite
    for (int i = 0; i < nx; ++i) {
        REQUIRE(hess_xx.slice(i).allFinite());
        REQUIRE(hess_ux.slice(i).allFinite());
        REQUIRE(hess_uu.slice(i).allFinite());
    }
}

TEST_CASE_METHOD(SatelliteDynamicsFixture, "Hessians are symmetric where expected", 
                 "[hessians][symmetry]") {
    // Use mid-orbit point
    size_t step = 50;
    
    Satellite::VecX x = Satellite::VecX::Zero(sat.stateDim());
    x.segment<3>(Satellite::AV_INDEX) << 0.05, 0.02, 0.01;  // Angular velocity
    x.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1, 0, 0, 0);  // Identity quaternion
    
    Satellite::VecX u = Satellite::VecX::Zero(sat.controlDim());
    DisturbanceConfig dist;
    dist.plan_for_gg = true;
    dist.plan_for_aero = true;
    dist.plan_for_srp = true;
    
    auto [hess_xx, hess_ux, hess_uu] = sat.dynamicsHessians(
        x, u, dist, R.col(step), B.col(step), S.col(step), V.col(step)
    );
    
    int nx = sat.stateDim();
    
    // For smooth dynamics, hess_xx should be symmetric (within numerical tolerance)
    const double tol = 1e-6;
    for (int i = 0; i < nx; ++i) {
        Satellite::MatX diff = hess_xx.slice(i) - hess_xx.slice(i).transpose();
        REQUIRE(diff.norm() < tol);
    }
}

// ============================================================================
// TEST SECTION 12: Dynamics Hessians - Finite Difference Validation
// ============================================================================

TEST_CASE_METHOD(SatelliteDynamicsFixture, "Hessian w.r.t. state matches finite differences (single component)", 
                 "[hessians][finite-diff]") {
    // Use mid-orbit point
    size_t step = 50;
    
    Satellite::VecX x = Satellite::VecX::Zero(sat.stateDim());
    x.segment<3>(Satellite::AV_INDEX) << 0.05, 0.02, 0.01;  // Angular velocity
    x.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1, 0, 0, 0);  // Identity quaternion
    
    Satellite::VecX u = Satellite::VecX::Zero(sat.controlDim());
    DisturbanceConfig dist;
    dist.plan_for_gg = true;
    dist.plan_for_aero = true;
    dist.plan_for_srp = true;
    
    Eigen::Vector3d R_eci = R.col(step);
    Eigen::Vector3d B_eci = B.col(step);
    Eigen::Vector3d S_eci = S.col(step);
    Eigen::Vector3d V_eci = V.col(step);
    
    auto [hess_xx, hess_ux, hess_uu] = sat.dynamicsHessians(
        x, u, dist, R_eci, B_eci, S_eci, V_eci
    );
    
    const int out_idx = Satellite::AV_INDEX;
    const double eps = 1e-5;
    const int nx = sat.stateDim();
    
    Satellite::MatX hess_numerical = Satellite::MatX::Zero(nx, nx);
    
    for (int j1 = 0; j1 < nx; ++j1) {
        for (int j2 = j1; j2 < nx; ++j2) {  // Use symmetry
            // Four-point stencil for second derivative
            Satellite::VecX x_pp = x, x_pm = x, x_mp = x, x_mm = x;
            x_pp(j1) += eps; x_pp(j2) += eps;
            x_pm(j1) += eps; x_pm(j2) -= eps;
            x_mp(j1) -= eps; x_mp(j2) += eps;
            x_mm(j1) -= eps; x_mm(j2) -= eps;
            
            // No external renormalization: dynamics() normalizes q internally.
            
            double f_pp = sat.dynamics(x_pp, u, dist, R_eci, B_eci, S_eci, V_eci, 0)(out_idx);
            double f_pm = sat.dynamics(x_pm, u, dist, R_eci, B_eci, S_eci, V_eci, 0)(out_idx);
            double f_mp = sat.dynamics(x_mp, u, dist, R_eci, B_eci, S_eci, V_eci, 0)(out_idx);
            double f_mm = sat.dynamics(x_mm, u, dist, R_eci, B_eci, S_eci, V_eci, 0)(out_idx);
            
            double second_deriv = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps * eps);
            
            hess_numerical(j1, j2) = second_deriv;
            hess_numerical(j2, j1) = second_deriv;
        }
    }
    
    // Compare: 4-point stencil truncation error O(eps^2)~1e-10, round-off O(eps_mach/eps^2)~2e-6.
    const double rel_tol = 5e-3;
    const double abs_tol = 1e-6;
    
    for (int j1 = 0; j1 < nx; ++j1) {
        for (int j2 = 0; j2 < nx; ++j2) {
            double analytical = hess_xx.slice(out_idx)(j1, j2);
            double numerical = hess_numerical(j1, j2);
            
            double rel_err = std::abs(numerical) > abs_tol ? 
                std::abs(analytical - numerical) / std::abs(numerical) : 0.0;
            double abs_err = std::abs(analytical - numerical);
            
            REQUIRE((rel_err <= rel_tol || abs_err <= abs_tol));
        }
    }
}

TEST_CASE_METHOD(SatelliteDynamicsFixture, "Hessian w.r.t. control matches finite differences (single component)", 
                 "[hessians][finite-diff]") {
    // Use mid-orbit point
    size_t step = 50;
    
    Satellite::VecX x = Satellite::VecX::Zero(sat.stateDim());
    x.segment<3>(Satellite::AV_INDEX) << 0.05, 0.02, 0.01;  // Angular velocity
    x.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1, 0, 0, 0);  // Identity quaternion
    
    Satellite::VecX u = Satellite::VecX::Zero(sat.controlDim());
    u(0) = 0.001;
    
    DisturbanceConfig dist;
    dist.plan_for_gg = true;
    dist.plan_for_aero = true;
    dist.plan_for_srp = true;
    
    Eigen::Vector3d R_eci = R.col(step);
    Eigen::Vector3d B_eci = B.col(step);
    Eigen::Vector3d S_eci = S.col(step);
    Eigen::Vector3d V_eci = V.col(step);
    
    auto [hess_xx, hess_ux, hess_uu] = sat.dynamicsHessians(
        x, u, dist, R_eci, B_eci, S_eci, V_eci
    );
    
    const int out_idx = Satellite::AV_INDEX;
    const double eps = 1e-5;
    const int nx = sat.stateDim();
    const int nu = sat.controlDim();
    
    Satellite::MatX hess_numerical = Satellite::MatX::Zero(nu, nx);
    
    for (int j_u = 0; j_u < nu; ++j_u) {
        for (int j_x = 0; j_x < nx; ++j_x) {
            // Mixed partial derivative: d²f/du_j_u dx_j_x
            Satellite::VecX u_p = u, u_m = u;
            u_p(j_u) += eps;
            u_m(j_u) -= eps;
            
            Satellite::VecX x_p = x, x_m = x;
            x_p(j_x) += eps;
            x_m(j_x) -= eps;
            
            double f_pp = sat.dynamics(x_p, u_p, dist, R_eci, B_eci, S_eci, V_eci, 0)(out_idx);
            double f_pm = sat.dynamics(x_p, u_m, dist, R_eci, B_eci, S_eci, V_eci, 0)(out_idx);
            double f_mp = sat.dynamics(x_m, u_p, dist, R_eci, B_eci, S_eci, V_eci, 0)(out_idx);
            double f_mm = sat.dynamics(x_m, u_m, dist, R_eci, B_eci, S_eci, V_eci, 0)(out_idx);
            
            double mixed_deriv = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps * eps);
            hess_numerical(j_u, j_x) = mixed_deriv;
        }
    }
    
    const double rel_tol = 5e-3;
    const double abs_tol = 1e-9;
    
    for (int j_u = 0; j_u < nu; ++j_u) {
        for (int j_x = 0; j_x < nx; ++j_x) {
            double analytical = hess_ux.slice(out_idx)(j_u, j_x);
            double numerical = hess_numerical(j_u, j_x);
            
            double rel_err = std::abs(numerical) > abs_tol ? 
                std::abs(analytical - numerical) / std::abs(numerical) : 0.0;
            double abs_err = std::abs(analytical - numerical);
            
            REQUIRE((rel_err <= rel_tol || abs_err <= abs_tol));
        }
    }
}

TEST_CASE_METHOD(SatelliteDynamicsFixture, "Hessian w.r.t. control-control matches finite differences (single component)", 
                 "[hessians][finite-diff]") {
    // Use mid-orbit point
    size_t step = 50;
    
    Satellite::VecX x = Satellite::VecX::Zero(sat.stateDim());
    x.segment<3>(Satellite::AV_INDEX) << 0.05, 0.02, 0.01;  // Angular velocity
    x.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1, 0, 0, 0);  // Identity quaternion
    
    Satellite::VecX u = Satellite::VecX::Zero(sat.controlDim());
    u(0) = 0.001;
    u(1) = 0.0005;
    
    DisturbanceConfig dist;
    dist.plan_for_gg = true;
    dist.plan_for_aero = true;
    dist.plan_for_srp = true;
    
    Eigen::Vector3d R_eci = R.col(step);
    Eigen::Vector3d B_eci = B.col(step);
    Eigen::Vector3d S_eci = S.col(step);
    Eigen::Vector3d V_eci = V.col(step);
    
    auto [hess_xx, hess_ux, hess_uu] = sat.dynamicsHessians(
        x, u, dist, R_eci, B_eci, S_eci, V_eci
    );
    
    const int out_idx = Satellite::AV_INDEX;
    const double eps = 1e-5;
    const int nu = sat.controlDim();
    
    Satellite::MatX hess_numerical = Satellite::MatX::Zero(nu, nu);
    
    for (int j1 = 0; j1 < nu; ++j1) {
        for (int j2 = j1; j2 < nu; ++j2) {
            Satellite::VecX u_pp = u, u_pm = u, u_mp = u, u_mm = u;
            u_pp(j1) += eps; u_pp(j2) += eps;
            u_pm(j1) += eps; u_pm(j2) -= eps;
            u_mp(j1) -= eps; u_mp(j2) += eps;
            u_mm(j1) -= eps; u_mm(j2) -= eps;
            
            double f_pp = sat.dynamics(x, u_pp, dist, R_eci, B_eci, S_eci, V_eci, 0)(out_idx);
            double f_pm = sat.dynamics(x, u_pm, dist, R_eci, B_eci, S_eci, V_eci, 0)(out_idx);
            double f_mp = sat.dynamics(x, u_mp, dist, R_eci, B_eci, S_eci, V_eci, 0)(out_idx);
            double f_mm = sat.dynamics(x, u_mm, dist, R_eci, B_eci, S_eci, V_eci, 0)(out_idx);
            
            double second_deriv = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps * eps);
            
            hess_numerical(j1, j2) = second_deriv;
            hess_numerical(j2, j1) = second_deriv;
        }
    }
    
    // All hess_uu entries are analytically zero (dynamics linear in u).
    // FD noise floor: ~f*eps_mach/eps^2 ~ 0.001*2e-16/1e-10 ~ 2e-9. Use abs_tol > noise.
    const double rel_tol = 5e-3;
    const double abs_tol = 1e-7;
    
    for (int j1 = 0; j1 < nu; ++j1) {
        for (int j2 = 0; j2 < nu; ++j2) {
            double analytical = hess_uu.slice(out_idx)(j1, j2);
            double numerical = hess_numerical(j1, j2);
            
            double rel_err = std::abs(numerical) > abs_tol ? 
                std::abs(analytical - numerical) / std::abs(numerical) : 0.0;
            double abs_err = std::abs(analytical - numerical);
            
            REQUIRE((rel_err <= rel_tol || abs_err <= abs_tol));
        }
    }
}
