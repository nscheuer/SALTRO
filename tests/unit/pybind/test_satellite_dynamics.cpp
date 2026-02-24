#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Dense>

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
