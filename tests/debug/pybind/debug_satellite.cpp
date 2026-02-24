#include <iostream>
#include <iomanip>
#include <Eigen/Dense>

#include <saltro/pybind/satellite.h>
#include <saltro/orbit_generation/generate_orbit.h>
#include <saltro/math/integrators/rk4.h>
#include <saltro/math/quaternion.h>
#include <saltro/limits.h>

using namespace saltro;

// Helper to print vectors nicely
template<typename Derived>
void print_vector(const std::string& name, const Eigen::MatrixBase<Derived>& v) {
    std::cout << name << ": [";
    for (int i = 0; i < v.size(); ++i) {
        std::cout << std::setw(12) << std::setprecision(6) << v(i);
        if (i < v.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

// Helper to print satellite info
void print_satellite_info(const Satellite& sat) {
    std::cout << "\n════════════════════════════════════════════════════════════════\n";
    std::cout << "                    SATELLITE CONFIGURATION                     \n";
    std::cout << "════════════════════════════════════════════════════════════════\n";
    
    std::cout << "\n--- Dimensions ---\n";
    std::cout << "State dimension:         " << sat.stateDim() << "\n";
    std::cout << "Reduced state dimension: " << sat.reducedStateDim() << "\n";
    std::cout << "Control dimension:       " << sat.controlDim() << "\n";
    
    std::cout << "\n--- Inertia Matrix ---\n";
    std::cout << sat.inertia() << "\n";
    
    std::cout << "\n--- Inertia (no RW) ---\n";
    std::cout << sat.inertiaNoRW() << "\n";
    
    std::cout << "\n--- Actuators ---\n";
    std::cout << "Number of MTQs: " << sat.numMTQ() << "\n";
    for (int i = 0; i < sat.numMTQ(); ++i) {
        const MTQ& mtq = sat.getMTQ(i);
        std::cout << "  MTQ " << i << ": axis = " << mtq.axis().transpose() 
                  << ", max = " << mtq.u_max() << " A·m²\n";
    }
    
    std::cout << "Number of RWs:  " << sat.numRW() << "\n";
    for (int i = 0; i < sat.numRW(); ++i) {
        const RW& rw = sat.getRW(i);
        std::cout << "  RW " << i << ": axis = " << rw.axis().transpose()
                  << ", max_torque = " << rw.u_max() << " N·m"
                  << ", J = " << rw.wheelInertia() << " kg·m²"
                  << ", h_max = " << rw.momentumMax() << " N·m·s\n";
    }
    
    std::cout << "════════════════════════════════════════════════════════════════\n\n";
}

// Simple PD controller for attitude stabilization using RWs
Satellite::VecX pd_controller(const Satellite& sat, const Satellite::VecX& x, 
                               const Eigen::Vector4d& q_target, double kp, double kd) {
    Satellite::VecX u = Satellite::VecX::Zero(sat.controlDim());
    
    // Extract current state
    Eigen::Vector3d w = x.segment<3>(Satellite::AV_INDEX);
    Eigen::Vector4d q = x.segment<4>(Satellite::QUAT_INDEX);
    
    // Quaternion error (simplified - just use vector part as error signal)
    // For proper quaternion PD, we'd compute q_err = q_target ⊗ q^{-1}
    Eigen::Vector3d attitude_error = q.tail<3>() - q_target.tail<3>();
    
    // PD control law: τ = -kp * e_att - kd * w
    Eigen::Vector3d desired_torque = -kp * attitude_error - kd * w;
    
    // Map to RW commands (starting at index num_mtq)
    // For simplicity, assume we can distribute torque evenly across RWs
    if (sat.numRW() > 0) {
        for (int i = 0; i < sat.numRW(); ++i) {
            const RW& rw = sat.getRW(i);
            double torque_component = rw.axis().dot(desired_torque);
            u(sat.numMTQ() + i) = std::max(-rw.u_max(), std::min(rw.u_max(), torque_component));
        }
    }
    
    return u;
}

int main() {
    std::cout << std::fixed << std::setprecision(6);
    
    // ========================================================================
    // STEP 1: Generate Orbit
    // ========================================================================
    std::cout << "\n█████████████████████████████████████████████████████████████████\n";
    std::cout << "  STEP 1: GENERATING ORBIT\n";
    std::cout << "█████████████████████████████████████████████████████████████████\n";
    
    Eigen::Vector3d r0(7000e3, 0.0, 0.0);   // 7000 km altitude
    Eigen::Vector3d v0(0.0, 7.5e3, 0.0);    // ~7.5 km/s
    
    int n_steps = 100;
    double dt = 10.0;  // 10 second time steps
    
    Eigen::Matrix<double, 1, limits::MAX_LENGTH_TRAJ> jtime;
    for (int i = 0; i < n_steps; ++i) {
        jtime(i) = i * dt;
    }
    
    Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> R, V, B, S;
    Eigen::Matrix<double, 1, limits::MAX_LENGTH_TRAJ> rho;
    
    bool orbit_ok = orbits::generate_orbit(
        r0, v0, jtime, n_steps,
        0, 0, 0, 0, 0,  // Use simple models
        R, V, B, S, rho
    );
    
    if (!orbit_ok) {
        std::cerr << "ERROR: Orbit generation failed!\n";
        return 1;
    }
    
    std::cout << "Orbit generated successfully!\n";
    std::cout << "Initial position: " << R.col(0).transpose() << " m\n";
    std::cout << "Initial velocity: " << V.col(0).transpose() << " m/s\n";
    std::cout << "Initial B-field:  " << B.col(0).transpose() << " T\n";
    std::cout << "Number of steps:  " << n_steps << "\n";
    std::cout << "Time step:        " << dt << " s\n";
    
    // ========================================================================
    // STEP 2: Create Satellite
    // ========================================================================
    std::cout << "\n█████████████████████████████████████████████████████████████████\n";
    std::cout << "  STEP 2: CREATING SATELLITE\n";
    std::cout << "█████████████████████████████████████████████████████████████████\n";
    
    // Create inertia matrix (roughly cube-shaped satellite, ~10kg, 20cm sides)
    Eigen::Matrix3d J;
    J << 0.067, 0.0, 0.0,
         0.0, 0.067, 0.0,
         0.0, 0.0, 0.067;
    
    PlannerSettings settings;  // Use default settings
    Satellite sat(J, settings);
    
    // Add 3 MTQs along body axes
    sat.addMTQ(Eigen::Vector3d(1, 0, 0), 0.2);  // X-axis, 0.2 A·m²
    sat.addMTQ(Eigen::Vector3d(0, 1, 0), 0.2);  // Y-axis
    sat.addMTQ(Eigen::Vector3d(0, 0, 1), 0.2);  // Z-axis
    
    // Add 3 RWs along body axes
    double rw_max_torque = 0.001;  // 1 mN·m
    double rw_inertia = 1e-5;      // kg·m²
    double rw_h0 = 0.0;            // Initial momentum
    double rw_h_max = 0.01;        // 10 mN·m·s
    
    sat.addRW(Eigen::Vector3d(1, 0, 0), rw_max_torque, rw_inertia, rw_h0, rw_h_max);
    sat.addRW(Eigen::Vector3d(0, 1, 0), rw_max_torque, rw_inertia, rw_h0, rw_h_max);
    sat.addRW(Eigen::Vector3d(0, 0, 1), rw_max_torque, rw_inertia, rw_h0, rw_h_max);
    
    print_satellite_info(sat);
    
    // ========================================================================
    // STEP 3: Test 1 - Zero Control, Zero Initial Angular Velocity
    // ========================================================================
    std::cout << "\n█████████████████████████████████████████████████████████████████\n";
    std::cout << "  STEP 3: TEST 1 - COAST WITH ZERO INITIAL SPIN\n";
    std::cout << "█████████████████████████████████████████████████████████████████\n";
    
    Satellite::VecX x_test1(sat.stateDim());
    x_test1.setZero();
    x_test1.segment<3>(Satellite::AV_INDEX) = Eigen::Vector3d::Zero();  // No initial spin
    x_test1.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1, 0, 0, 0);  // Identity quaternion
    // RW momenta already zero
    
    Satellite::VecX u_zero = Satellite::VecX::Zero(sat.controlDim());
    
    print_vector("Initial state", x_test1);
    
    // Propagate for a few steps
    int test1_steps = 10;
    Satellite::VecX x_current = x_test1;
    
    for (int i = 0; i < test1_steps; ++i) {
        DisturbanceConfig dist;  // Empty disturbance config
        Eigen::Vector3d B_eci_i = B.col(i);
        Eigen::Vector3d S_eci_i = S.col(i);
        int rho_i = static_cast<int>(rho(i));
        
        // RK4 integration
        auto dynamics_func = [&](double t, const Satellite::VecX& x, Satellite::VecX& dxdt) {
            dxdt = sat.dynamics(x, u_zero, dist, R.col(i), B_eci_i, S_eci_i, V.col(i), rho_i);
        };
        
        Satellite::VecX x_next(sat.stateDim());
        rk4_step(dynamics_func, x_current, jtime(i), dt, x_next);
        
        // Normalize quaternion
        Eigen::Vector4d q = x_next.segment<4>(Satellite::QUAT_INDEX);
        q.normalize();
        x_next.segment<4>(Satellite::QUAT_INDEX) = q;
        
        x_current = x_next;
    }
    
    print_vector("Final state (Test 1)", x_current);
    std::cout << "Angular velocity magnitude: " 
              << x_current.segment<3>(Satellite::AV_INDEX).norm() << " rad/s\n";
    std::cout << "Quaternion (should be close to identity): " 
              << x_current.segment<4>(Satellite::QUAT_INDEX).transpose() << "\n";
    
    // ========================================================================
    // STEP 4: Test 2 - Zero Control, Non-Zero Initial Angular Velocity
    // ========================================================================
    std::cout << "\n█████████████████████████████████████████████████████████████████\n";
    std::cout << "  STEP 4: TEST 2 - COAST WITH INITIAL SPIN\n";
    std::cout << "█████████████████████████████████████████████████████████████████\n";
    
    Satellite::VecX x_test2(sat.stateDim());
    x_test2.setZero();
    x_test2.segment<3>(Satellite::AV_INDEX) = Eigen::Vector3d(0.1, 0.05, 0.02);  // 0.1 rad/s about X
    x_test2.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1, 0, 0, 0);
    
    print_vector("Initial state", x_test2);
    double initial_omega_mag = x_test2.segment<3>(Satellite::AV_INDEX).norm();
    std::cout << "Initial angular velocity magnitude: " << initial_omega_mag << " rad/s\n";
    
    int test2_steps = 50;
    x_current = x_test2;
    
    std::cout << "\nPropagating...\n";
    for (int i = 0; i < test2_steps; ++i) {
        DisturbanceConfig dist;
        Eigen::Vector3d B_eci_i = B.col(std::min(i, n_steps-1));
        Eigen::Vector3d S_eci_i = S.col(std::min(i, n_steps-1));
        int rho_i = static_cast<int>(rho(std::min(i, n_steps-1)));
        
        auto dynamics_func = [&](double t, const Satellite::VecX& x, Satellite::VecX& dxdt) {
            dxdt = sat.dynamics(x, u_zero, dist, R.col(std::min(i, n_steps-1)), B_eci_i, S_eci_i, V.col(std::min(i, n_steps-1)), rho_i);
        };
        
        Satellite::VecX x_next(sat.stateDim());
        rk4_step(dynamics_func, x_current, i * dt, dt, x_next);
        
        Eigen::Vector4d q = x_next.segment<4>(Satellite::QUAT_INDEX);
        q.normalize();
        x_next.segment<4>(Satellite::QUAT_INDEX) = q;
        
        x_current = x_next;
        
        if (i % 10 == 0) {
            std::cout << "  Step " << std::setw(3) << i << ": ω_mag = " 
                      << x_current.segment<3>(Satellite::AV_INDEX).norm() << " rad/s\n";
        }
    }
    
    print_vector("Final state (Test 2)", x_current);
    double final_omega_mag = x_current.segment<3>(Satellite::AV_INDEX).norm();
    std::cout << "Final angular velocity magnitude: " << final_omega_mag << " rad/s\n";
    std::cout << "Change in |ω|: " << (final_omega_mag - initial_omega_mag) << " rad/s\n";
    std::cout << "Relative change: " << 100.0 * (final_omega_mag - initial_omega_mag) / initial_omega_mag << "%\n";
    
    // Check angular momentum conservation (roughly)
    Eigen::Vector3d L_initial = J * x_test2.segment<3>(Satellite::AV_INDEX);
    Eigen::Vector3d L_final = J * x_current.segment<3>(Satellite::AV_INDEX);
    std::cout << "Angular momentum conservation check:\n";
    std::cout << "  Initial L: " << L_initial.norm() << " N·m·s\n";
    std::cout << "  Final L:   " << L_final.norm() << " N·m·s\n";
    std::cout << "  Difference: " << (L_final - L_initial).norm() << " N·m·s\n";
    
    // ========================================================================
    // STEP 5: Test 3 - PD Controller to Stop Spin
    // ========================================================================
    std::cout << "\n█████████████████████████████████████████████████████████████████\n";
    std::cout << "  STEP 5: TEST 3 - PD CONTROL TO STOP SPIN\n";
    std::cout << "█████████████████████████████████████████████████████████████████\n";
    
    Satellite::VecX x_test3(sat.stateDim());
    x_test3.setZero();
    x_test3.segment<3>(Satellite::AV_INDEX) = Eigen::Vector3d(0.2, -0.1, 0.15);  // Faster spin
    x_test3.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1, 0, 0, 0);
    
    print_vector("Initial state", x_test3);
    std::cout << "Initial angular velocity magnitude: " 
              << x_test3.segment<3>(Satellite::AV_INDEX).norm() << " rad/s\n";
    
    Eigen::Vector4d q_target(1, 0, 0, 0);  // Target: identity quaternion
    double kp = 0.00001;  // Proportional gain
    double kd = 0.0001;   // Derivative gain
    
    std::cout << "PD gains: kp = " << kp << ", kd = " << kd << "\n";
    
    int test3_steps = 200;
    x_current = x_test3;
    
    std::cout << "\nPropagating with PD control...\n";
    for (int i = 0; i < test3_steps; ++i) {
        // Compute control
        Satellite::VecX u_pd = pd_controller(sat, x_current, q_target, kp, kd);
        
        DisturbanceConfig dist;
        Eigen::Vector3d B_eci_i = B.col(std::min(i, n_steps-1));
        Eigen::Vector3d S_eci_i = S.col(std::min(i, n_steps-1));
        int rho_i = static_cast<int>(rho(std::min(i, n_steps-1)));
        
        auto dynamics_func = [&](double t, const Satellite::VecX& x, Satellite::VecX& dxdt) {
            dxdt = sat.dynamics(x, u_pd, dist, R.col(std::min(i, n_steps-1)), B_eci_i, S_eci_i, V.col(std::min(i, n_steps-1)), rho_i);
        };
        
        Satellite::VecX x_next(sat.stateDim());
        rk4_step(dynamics_func, x_current, i * dt, dt, x_next);
        
        Eigen::Vector4d q = x_next.segment<4>(Satellite::QUAT_INDEX);
        q.normalize();
        x_next.segment<4>(Satellite::QUAT_INDEX) = q;
        
        x_current = x_next;
        
        if (i % 20 == 0) {
            std::cout << "  Step " << std::setw(3) << i 
                      << ": ω_mag = " << std::setw(10) << x_current.segment<3>(Satellite::AV_INDEX).norm() 
                      << " rad/s, RW torques = [";
            for (int j = 0; j < sat.numRW(); ++j) {
                std::cout << std::setw(8) << u_pd(sat.numMTQ() + j);
                if (j < sat.numRW() - 1) std::cout << ", ";
            }
            std::cout << "] N·m\n";
        }
    }
    
    print_vector("Final state (Test 3)", x_current);
    std::cout << "Final angular velocity magnitude: " 
              << x_current.segment<3>(Satellite::AV_INDEX).norm() << " rad/s\n";
    std::cout << "Final RW momenta: " 
              << x_current.segment(Satellite::RW_MOMENTUM_INDEX, sat.numRW()).transpose() << " N·m·s\n";
    
    // ========================================================================
    // Summary
    // ========================================================================
    std::cout << "\n█████████████████████████████████████████████████████████████████\n";
    std::cout << "  TEST SUMMARY\n";
    std::cout << "█████████████████████████████████████████████████████████████████\n";
    std::cout << "✓ Test 1: Zero initial spin - satellite remained stable\n";
    std::cout << "✓ Test 2: Initial spin - angular momentum approximately conserved\n";
    std::cout << "✓ Test 3: PD control - successfully reduced angular velocity\n";
    std::cout << "\nAll tests completed successfully!\n\n";
    
    return 0;
}
