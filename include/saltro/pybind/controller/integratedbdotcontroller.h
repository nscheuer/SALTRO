/**
 * @file integratedbdotcontroller.h
 * @brief B-dot detumbling and PD attitude control law with integrated magnetic derivatives.
 */
#pragma once

#include <vector>

#include <saltro/pybind/controller/controller.h>

namespace saltro::controller {

/**
 * @brief Integrated B-dot controller for satellite detumbling and attitude control.
 *
 * Implements a dual-mode control law combining:
 * 1. **Magnetorquer B-dot control**: Opposes the time rate of change of the magnetic field
 *    to dissipate rotational energy. Uses integrated magnetic field derivative approximation.
 * 2. **Reaction wheel PD control**: Provides precise attitude tracking after initial detumbling.
 *
 * The magnetorquer control law is:
 * \f[
 * \mathbf{m}_{\text{mtq}} = -k_p \mathbf{C}(\mathbf{q})^T \mathbf{B}_{\text{ECI}} 
 * - k_d \frac{d}{dt}[\mathbf{C}(\mathbf{q})^T \mathbf{B}_{\text{ECI}}]
 * \f]
 * where the derivative is approximated using integrated (accumulated) magnetic field changes.
 *
 * The reaction wheel control law uses quaternion-based PD feedback:
 * \f[
 * \boldsymbol{\tau}_{\text{rw}} = -k_p \mathbf{q}_{\text{err,vec}} - k_d \boldsymbol{\omega}
 * \f]
 * where \f$\mathbf{q}_{\text{err,vec}}\f$ is the vector part of the quaternion error
 * \f$\mathbf{q}_{\text{goal}}^{-1} \otimes \mathbf{q}\f$.
 *
 * The controller automatically tunes gains based on satellite inertia and expected
 * magnetic field strength in LEO (35 µT nominal).
 */
class IntegratedBdotController final : public Controller {
public:
    /**
     * @brief Construct an integrated B-dot controller with auto-tuned gains.
     *
     * Initializes the controller and automatically computes proportional and derivative
     * gains for both magnetorquers and reaction wheels based on satellite properties.
     *
     * @param satellite Satellite model with actuator configuration and inertia tensor
     */
    explicit IntegratedBdotController(const Satellite& satellite);

    /**
     * @brief Compute control input using B-dot and PD feedback.
     *
     * Implements the dual-mode control law combining magnetorquer B-dot damping with
     * reaction wheel PD attitude control. The control is saturated by actuator limits
     * defined in the satellite model.
     *
     * @param x Attitude state vector (7+nRW × 1): [ω(3), q(4), h_rw(nRW)]
     * @param B_eci Magnetic field vector in ECI frame (3 × 1), Tesla
     * @param q_goal Goal quaternion (4 × 1), unit quaternion
     * @param boresight_body Desired boresight direction in body frame (3 × 1), unit vector
     * @return Control vector (nu × 1): [m_mtq, τ_rw, τ_magic]
     */
    Satellite::VecX find_u(
        const Satellite::VecX& x,
        const Eigen::Vector3d& B_eci,
        const Eigen::Vector4d& q_goal,
        const Eigen::Vector3d& boresight_body
    ) const override;

protected:
    /**
     * @brief Automatically tune controller gains based on satellite properties.
     *
     * Computes proportional and derivative gains for magnetorquers and reaction wheels
     * using scaling laws based on inertia eigenvalues, actuator capabilities, and
     * expected magnetic field strength. Uses damping ratio ζ ≈ 0.7 for critically
     * damped response.
     */
    void autoTuneGains() override;

private:
    /** Maximum reference angular rate for gain normalization (rad/s) */
    double max_rate_ref_ = 5.0 * 3.14159265358979323846 / 180.0;

    /** Proportional gains for each magnetorquer axis (A·m²/T) */
    std::vector<double> mtq_kp_;
    
    /** Derivative gains for each magnetorquer axis (A·m²·s/T) */
    std::vector<double> mtq_kd_;
    
    /** Proportional gains for each reaction wheel (N·m/rad) */
    std::vector<double> rw_kp_;
    
    /** Derivative gains for each reaction wheel (N·m·s/rad) */
    std::vector<double> rw_kd_;
};

}
