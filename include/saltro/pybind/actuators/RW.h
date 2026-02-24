#pragma once

#include <saltro/pybind/actuators/actuator.h>
#include <Eigen/Dense>

/**
 * @brief Reaction wheel (RW) actuator class.
 * 
 * Represents a momentum-exchange device that produces torque by changing
 * its angular momentum. Inherits from Actuator and extends functionality to
 * track and constrain the wheel's angular momentum state.
 * 
 * Dynamics:
 * The torque applied by the RW depends on the desired rate of momentum change:
 * \f[
 * \boldsymbol{\tau} = u \cdot \text{axis}
 * \f]
 * 
 * The wheel momentum evolves as:
 * \f[
 * \frac{dh}{dt} = -u \quad \text{(reaction principle)}
 * \f]
 * 
 * @see Actuator for base class interface
 */
class RW final : public Actuator {
public:
    using Vec3 = Actuator::Vec3;
    using BaseState = Actuator::BaseState;

    using Mat13 = Actuator::Mat13;
    using Mat73 = Actuator::Mat73;
    using Mat11 = Eigen::Matrix<double, 1, 1>;
    using Mat17 = Eigen::Matrix<double, 1, 7>;
    using Mat71 = Eigen::Matrix<double, 7, 1>;

    using T113 = Actuator::T113;
    using T173 = Actuator::T173;
    using T773 = Actuator::T773;

    /**
     * @brief Construct a reaction wheel with specified parameters.
     * 
     * @param axis Unit vector specifying torque direction in body frame.
     * @param max_torque Maximum torque magnitude (Nm).
     * @param J Wheel moment of inertia (kg·m²).
     * @param h0 Initial wheel momentum (N·m·s).
     * @param h_max Maximum allowable wheel momentum (N·m·s).
     */
    RW(const Vec3& axis, double max_torque, double J, double h0, double h_max);
    ~RW() override = default;

    /**
     * @brief Get the wheel's moment of inertia.
     * 
     * @return J value in kg·m².
     */
    double wheelInertia() const noexcept;
    
    /**
     * @brief Get the current wheel momentum magnitude.
     * 
     * @return Current momentum in N·m·s.
     */
    double momentum() const noexcept;
    
    /**
     * @brief Get the maximum allowable wheel momentum magnitude.
     * 
     * @return h_max in N·m·s.
     */
    double momentumMax() const noexcept;
    
    /**
     * @brief Set the wheel momentum to a new value.
     * 
     * @param h New momentum value (N·m·s).
     */
    void setMomentum(double h) noexcept;

    /**
     * @brief Compute the torque produced by the reaction wheel.
     * 
     * Overrides base class. Torque depends on control input and wheel inertia:
     * \f[
     * \boldsymbol{\tau} = u \cdot \text{axis}
     * \f]
     * 
     * @param u Control input (torque command rate, rad/s²·J).
     * @param x Base state.
     * @return 3D torque vector.
     */
    Vec3 torque(double u, const BaseState& x) const override;

    /**
     * @brief Storage torque (momentum rate).
     * 
     * Returns the rate of change of wheel momentum:
     * \f[
     * \frac{dh}{dt} = -u
     * \f]
     * 
     * @param u Control input.
     * @param x Base state.
     * @return 1×1 matrix containing momentum rate.
     */
    Mat11 storageTorque(double u, const BaseState& x) const;

    /**
     * @brief Jacobian of torque with respect to control input.
     * 
     * @param u Control input.
     * @param x Base state.
     * @return 1×3 Jacobian.
     */
    Mat13 dtorq_du(double u, const BaseState& x) const override;
    
    /**
     * @brief Jacobian of torque with respect to base state.
     * 
     * @param u Control input.
     * @param x Base state.
     * @return 7×3 Jacobian.
     */
    Mat73 dtorq_dbasestate(double u, const BaseState& x) const override;

    /**
     * @brief Jacobian of storage torque with respect to control input.
     * 
     * @param u Control input.
     * @param x Base state.
     * @return 1×1 Jacobian.
     */
    Mat11 dstor_torq_du(double u, const BaseState& x) const;
    
    /**
     * @brief Jacobian of storage torque with respect to base state.
     * 
     * @param u Control input.
     * @param x Base state.
     * @return 7×1 Jacobian.
     */
    Mat71 dstor_torq_dbasestate(double u, const BaseState& x) const;

    // Hessian methods inherited from base class
    using Actuator::ddtorq_dudu;
    using Actuator::ddtorq_dudbasestate;
    using Actuator::ddtorq_dbasestatedbasestate;

private:
    static constexpr int kBaseStateDim = 7;
    static constexpr int kStorageDim = 1;

    double J_{0.0};      ///< Wheel moment of inertia (kg·m²)
    double h_{0.0};      ///< Current momentum (N·m·s)
    double h_max_{0.0};  ///< Maximum momentum (N·m·s)
};