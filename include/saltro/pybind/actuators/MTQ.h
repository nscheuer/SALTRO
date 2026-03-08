#pragma once

#include <saltro/pybind/actuators/actuator.h>
#include <Eigen/Dense>

/**
 * @brief Magnetorquer (MTQ) actuator class.
 * 
 * Represents a magnetic torque coil that produces torque through interaction
 * with Earth's magnetic field. The torque is:
 * \f[
 * \boldsymbol{\tau} = \mathbf{m} \times \mathbf{B}
 * \f]
 * where \f$\mathbf{m} = u \cdot \text{axis}\f$ is the magnetic dipole moment (A·m²)
 * and \f$\mathbf{B}\f$ is the magnetic field in body frame.
 * 
 * Unlike reaction wheels, MTQs have no internal momentum state. The torque
 * is always reaction-free (momentum is transferred to the Earth's field).
 * 
 * @see Actuator for base class interface
 */
class MTQ final : public Actuator {
public:
    using Vec3 = Actuator::Vec3;
    using BaseState = Actuator::BaseState;

    using Mat13 = Actuator::Mat13;
    using Mat73 = Actuator::Mat73;

    using T113 = Actuator::T113;
    using T173 = Actuator::T173;
    using T773 = Actuator::T773;

    // Keep base overloads visible to avoid hiding virtual interface methods.
    using Actuator::torque;
    using Actuator::dtorq_du;
    using Actuator::dtorq_dbasestate;
    using Actuator::ddtorq_dudu;
    using Actuator::ddtorq_dudbasestate;
    using Actuator::ddtorq_dbasestatedbasestate;

    /**
     * @brief Construct a magnetorquer with specified parameters.
     * 
     * @param axis Unit vector specifying the coil direction in body frame.
     * @param max_dipole Maximum magnetic dipole moment magnitude (A·m²).
     */
    MTQ(const Vec3 &axis, double max_dipole);

    /**
     * @brief Compute the magnetic torque produced by this MTQ.
     * 
     * The torque is the cross product of magnetic dipole moment and field:
     * \f[
     * \boldsymbol{\tau} = (u \cdot \text{axis}) \times \mathbf{B}_{\text{body}}
     * \f]
     * 
     * @param u Control input (dipole moment magnitude, A·m²).
     * @param x Base state (contains quaternion for ECI→body transformation).
     * @param B_body Magnetic field vector in body frame (Tesla).
     * @return 3D torque vector (Newton-meters).
     */
    Vec3 torque(double u, const BaseState& x, const Vec3& B_body) const;

    /**
     * @brief Jacobian of torque with respect to control input.
     * 
     * @param u Control input.
     * @param x Base state.
     * @param B_body Magnetic field in body frame.
     * @return 1×3 Jacobian matrix.
     */
    Mat13 dtorq_du(double u, const BaseState& x, const Vec3& B_body) const;
    
    /**
     * @brief Jacobian of torque with respect to base state.
     * 
     * Account for the fact that magnetic field depends on attitude (quaternion).
     * 
     * @param u Control input.
     * @param x Base state.
     * @param B_body Magnetic field in body frame.
     * @param dB_dq Jacobian of magnetic field with respect to quaternion (4×3).
     * @return 7×3 Jacobian matrix.
     */
    Mat73 dtorq_dbasestate(double u, const BaseState& x, const Vec3& B_body, 
                           const Eigen::Matrix<double, 4, 3>& dB_dq) const;

    /**
     * @brief Hessian of torque control input (second derivatives).
     * 
     * @param u Control input.
     * @param x Base state.
     * @param B_body Magnetic field in body frame.
     * @return Tensor3 with 3 slices of 1×1 matrices.
     */
    T113 ddtorq_dudu(double u, const BaseState& x, const Vec3& B_body) const;

    /**
     * @brief Hessian of mixed torque derivatives (control-state).
     * 
     * @param u Control input.
     * @param x Base state.
     * @param B_body Magnetic field in body frame.
     * @param dB_dq Jacobian of magnetic field with respect to quaternion.
     * @return Tensor3 with 3 slices of 1×7 matrices.
     */
    T173 ddtorq_dudbasestate(double u, const BaseState& x, const Vec3& B_body, 
                             const Eigen::Matrix<double,4,3>& dB_dq) const;

    /**
     * @brief Hessian of state-state torque derivatives.
     * 
     * @param u Control input.
     * @param x Base state.
     * @param dB_dq Jacobian of magnetic field with respect to quaternion.
     * @param d2B_dq2 Array of three 4×4 Hessian matrices for magnetic field.
     * @return Tensor3 with 3 slices of 7×7 matrices.
     */
    T773 ddtorq_dbasestatedbasestate(double u, const BaseState& x, 
                                      const Eigen::Matrix<double,4,3>& dB_dq, 
                                      const std::array<Eigen::Matrix<double,4,4>,3>& d2B_dq2) const;
};