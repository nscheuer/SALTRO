/**
 * @file controller.h
 * @brief Abstract base class for satellite attitude control laws.
 */
#pragma once

#include <Eigen/Dense>

#include <saltro/pybind/satellite.h>

namespace saltro::controller {

/**
 * @brief Abstract base class for satellite attitude controllers.
 *
 * Defines the interface for control laws that compute actuator commands given the
 * current attitude state, magnetic field, goal orientation, and boresight constraints.
 *
 * The controller computes a combined control vector:
 * \f[
 * \mathbf{u} = [\mathbf{m}_{\text{mtq}}^T, \boldsymbol{\tau}_{\text{rw}}^T, 
 * \boldsymbol{\tau}_{\text{magic}}^T]^T
 * \f]
 * where:
 * - \f$\mathbf{m}_{\text{mtq}}\f$: magnetic dipole moments (A·m²) for each MTQ axis
 * - \f$\boldsymbol{\tau}_{\text{rw}}\f$: commanded torques (N·m) for each reaction wheel
 * - \f$\boldsymbol{\tau}_{\text{magic}}\f$: unconstrained "magic" torque (N·m) in body frame
 *
 * Derived classes implement specific control laws such as B-dot detumbling, PD control,
 * or excitation signals for system identification.
 */
class Controller {
public:
	/**
	 * @brief Construct a controller for the given satellite model.
	 *
	 * @param satellite Satellite model with actuator configuration and inertia properties
	 */
	Controller(const Satellite& satellite);
	
	virtual ~Controller() = default;

	/**
	 * @brief Compute control input for current state and environment.
	 *
	 * Pure virtual method implemented by derived classes to define the control law.
	 * The controller maps the current state, magnetic field, goal orientation, and
	 * boresight constraint to an actuator command vector.
	 *
	 * @param x Attitude state vector (7+nRW × 1): [ω(3), q(4), h_rw(nRW)]
	 *          - ω: angular velocity in body frame (rad/s)
	 *          - q: unit quaternion (scalar-first)
	 *          - h_rw: reaction wheel angular momentum (N·m·s)
	 * @param B_eci Magnetic field vector in ECI frame (3 × 1), Tesla
	 * @param q_goal Goal quaternion (4 × 1), unit quaternion (scalar-first)
	 * @param boresight_body Desired boresight direction in body frame (3 × 1), unit vector
	 * @return Control vector (nu × 1) with actuator commands
	 */
	virtual Satellite::VecX find_u(
		const Satellite::VecX& x,
		const Eigen::Vector3d& B_eci,
		const Eigen::Vector4d& q_goal,
		const Eigen::Vector3d& boresight_body
	) const = 0;

protected:
	/**
	 * @brief Automatically tune controller gains based on satellite properties.
	 *
	 * Pure virtual method implemented by derived classes to set control gains
	 * (e.g., proportional/derivative coefficients) based on satellite inertia,
	 * actuator capabilities, and expected operating conditions.
	 */
	virtual void autoTuneGains() = 0;

	/** Reference to satellite model */
	const Satellite& satellite_;

	/** Expected magnetic field magnitude in LEO (Tesla), used for gain tuning */
	double expected_b_field_leo_ = 35e-6;
};

}

