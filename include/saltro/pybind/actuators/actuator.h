#pragma once

#include <Eigen/Dense>
#include <saltro/math/tensor.h>

/**
 * @brief Base class for spacecraft actuators (torque-producing devices).
 * 
 * Abstract base class defining the interface for actuators (e.g., reaction wheels,
 * magnetorquers). Each actuator produces torque in a fixed direction (axis) with
 * bounded control authority (u_max).
 * 
 * Derived classes must implement methods for:
 * - Torque computation: computing the torque vector from a scalar control input
 * - Jacobians: first derivatives of torque with respect to state and control
 * - Hessians: second derivatives for optimization algorithms
 */
class Actuator {
public:
    using Vec3 = Eigen::Matrix<double, 3, 1>;
    using BaseState = Eigen::Matrix<double, 7, 1>;

    using Mat13 = Eigen::Matrix<double, 1, 3>;
    using Mat73 = Eigen::Matrix<double, 7, 3>;

    using T113 = saltro::math::Tensor3<1, 1, 3>;
    using T173 = saltro::math::Tensor3<1, 7, 3>;
    using T773 = saltro::math::Tensor3<7, 7, 3>;

    static constexpr int input_len = 1;

    Actuator() = delete;
    
    /**
     * @brief Construct an actuator with specified axis and control limit.
     * 
     * @param axis Unit vector specifying the torque direction in body frame.
     * @param u_max Control input magnitude limit (amplitude constraint).
     */
    Actuator(const Vec3 &axis, double u_max);
    virtual ~Actuator() = default;

    /**
     * @brief Get the torque axis vector.
     * 
     * @return Const reference to the unit vector along the torque axis.
     */
    const Vec3& axis() const noexcept;
    
    /**
     * @brief Get the maximum control input magnitude.
     * 
     * @return The control input saturation limit.
     */
    double u_max() const noexcept;

    /**
     * @brief Clamp a control input to the saturation limits.
     * 
     * Returns \f$\text{clamp}(u, -u_{\max}, u_{\max})\f$.
     * 
     * @param u Input control value.
     * @return Saturated control input in \f$[-u_{\max}, u_{\max}]\f$.
     */
    double clamp(double u) const noexcept;

    /**
     * @brief Compute the torque vector produced by this actuator.
     * 
     * Returns the torque vector as a function of control input and spacecraft state.
     * 
     * @param u Control input (typically in [-u_max, u_max]).
     * @param x Base state vector (7D: [av; q; ...]).
     * @return 3D torque vector in body frame (Newton-meters).
     */
    virtual Vec3 torque(double u, const BaseState& x) const;

    /**
     * @brief Jacobian of torque with respect to control input.
     * 
     * Returns \f$\frac{\partial \boldsymbol{\tau}}{\partial u}\f$ as a 1×3 row vector.
     * 
     * @param u Control input.
     * @param x Base state.
     * @return 1×3 Jacobian matrix.
     */
    virtual Mat13 dtorq_du(double u, const BaseState& x) const;
    
    /**
     * @brief Jacobian of torque with respect to base state.
     * 
     * Returns \f$\frac{\partial \boldsymbol{\tau}}{\partial \mathbf{x}}\f$ as a 7×3 matrix.
     * 
     * @param u Control input.
     * @param x Base state (7D).
     * @return 7×3 Jacobian matrix.
     */
    virtual Mat73 dtorq_dbasestate(double u, const BaseState& x) const;

    /**
     * @brief Hessian of torque with respect to control input (second derivatives).
     * 
     * Returns a 3-slice tensor of \f$\frac{\partial^2 \tau_i}{\partial u^2}\f$.
     * 
     * @param u Control input.
     * @param x Base state.
     * @return Tensor3 with 3 slices of 1×1 matrices (one per torque component).
     */
    virtual T113 ddtorq_dudu(double u, const BaseState& x) const;
    
    /**
     * @brief Hessian of torque with respect to control input and base state.
     * 
     * Returns mixed partial derivatives \f$\frac{\partial^2 \tau_i}{\partial u \partial x_j}\f$.
     * 
     * @param u Control input.
     * @param x Base state.
     * @return Tensor3 with 3 slices of 1×7 matrices.
     */
    virtual T173 ddtorq_dudbasestate(double u, const BaseState& x) const;
    
    /**
     * @brief Hessian of torque with respect to base state (full second derivatives).
     * 
     * Returns \f$\frac{\partial^2 \tau_i}{\partial x_j \partial x_k}\f$ for each component i.
     * 
     * @param u Control input.
     * @param x Base state.
     * @return Tensor3 with 3 slices of 7×7 matrices.
     */
    virtual T773 ddtorq_dbasestatedbasestate(double u, const BaseState& x) const;

protected:
    /**
     * @brief Normalize a vector to unit length (utility function).
     * 
     * @param v Input vector.
     * @return Normalized unit vector.
     */
    static Vec3 normalize(const Vec3& v);

    Vec3 axis_;        ///< Torque axis (unit vector)
    double u_max_;     ///< Control input magnitude limit
};