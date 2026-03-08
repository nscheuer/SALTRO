/**
 * @file excitationcontroller.h
 * @brief Persistent excitation controller for system identification and parameter estimation.
 */
#pragma once

#include <saltro/pybind/controller/controller.h>

namespace saltro::controller {

/**
 * @brief Excitation controller for persistent excitation in parameter estimation.
 *
 * Generates control commands that persistently excite the satellite dynamics to enable
 * accurate parameter estimation (e.g., inertia tensor identification, actuator calibration).
 * The controller adds bounded pseudo-random excitation signals to the control while
 * maintaining rate damping to prevent excessive angular velocities.
 *
 * The control law is:
 * \f[
 * \mathbf{u}_{\text{mtq}} = f_{\text{mtq}} \cdot \text{rand}(-1, 1) \cdot 
 * \mathbf{u}_{\text{max,mtq}} - k_d^{\text{mtq}} \boldsymbol{\omega}
 * \f]
 * \f[
 * \mathbf{u}_{\text{rw}} = f_{\text{rw}} \cdot \text{rand}(-1, 1) \cdot 
 * \mathbf{u}_{\text{max,rw}} - k_d^{\text{rw}} \boldsymbol{\omega}
 * \f]
 * where:
 * - \f$f_{\text{mtq}}, f_{\text{rw}}\f$: excitation fractions (typical: 4%)
 * - \f$\text{rand}(-1, 1)\f$: uniform random signal in [-1, 1]
 * - \f$k_d\f$: rate damping gains to prevent runaway angular velocities
 *
 * The excitation is clipped to command fraction limits (typical: 10-15% of max) to
 * ensure safe operation. Damping gains are auto-tuned based on satellite inertia.
 */
class ExcitationController final : public Controller {
public:
    /**
     * @brief Construct an excitation controller with default parameters.
     *
     * Initializes excitation fractions, command limits, and damping gains to default
     * values suitable for typical LEO satellite system identification.
     *
     * @param satellite Satellite model with actuator limits and inertia properties
     */
    explicit ExcitationController(const Satellite& satellite);

    /**
     * @brief Compute excitation control input with rate damping.
     *
     * Generates pseudo-random excitation signals for magnetorquers and reaction wheels,
     * bounded by command fraction limits, with additional rate damping to prevent
     * excessive angular velocities.
     *
     * @param x Attitude state vector (7+nRW × 1): [ω(3), q(4), h_rw(nRW)]
     * @param B_eci Magnetic field vector in ECI frame (unused for excitation)
     * @param q_goal Goal quaternion (unused for excitation)
     * @param boresight_body Boresight constraint (unused for excitation)
     * @return Control vector (nu × 1): [m_mtq, τ_rw, τ_magic] with persistent excitation
     */
    Satellite::VecX find_u(
        const Satellite::VecX& x,
        const Eigen::Vector3d& B_eci,
        const Eigen::Vector4d& q_goal,
        const Eigen::Vector3d& boresight_body
    ) const override;

protected:
    /**
     * @brief Automatically tune rate damping gains.
     *
     * Computes damping gains based on satellite inertia and expected angular rates
     * to prevent excessive tumbling under persistent excitation.
     */
    void autoTuneGains() override;

private:
    /** Fraction of max RW torque used for excitation (default: 4%) */
    double rw_excitation_fraction_ = 0.04;
    
    /** Fraction of max MTQ dipole used for excitation (default: 4%) */
    double mtq_excitation_fraction_ = 0.04;
    
    /** Maximum allowed fraction of RW command (default: 10%) */
    double rw_command_fraction_limit_ = 0.10;
    
    /** Maximum allowed fraction of MTQ command (default: 15%) */
    double mtq_command_fraction_limit_ = 0.15;
    
    /** Rate damping gain for reaction wheels (N·m·s/rad) */
    double rw_rate_damping_gain_ = 0.0;
    
    /** Rate damping gain for magnetorquers (A·m²·s/rad) */
    double mtq_rate_damping_gain_ = 0.0;
    
    /** Reference angular rate for gain normalization (rad/s), default: 5°/s */
    double omega_ref_rad_s_ = 5.0 * 3.14159265358979323846 / 180.0;
};

}
