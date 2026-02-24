#pragma once

#include <Eigen/Dense>
#include <saltro/math/tensor.h>
#include <saltro/pybind/plannersettings.h>

/**
 * @brief Abstract base class for disturbance models.
 * 
 * Represents physical disturbances acting on the spacecraft (e.g., aerodynamic drag,
 * solar radiation pressure, gravity gradient). Each disturbance computes a torque
 * vector and its derivatives with respect to the spacecraft state.
 * 
 * Derived classes must implement:
 * - `torque()`: compute the disturbance torque vector
 * - Optionally: Jacobians and Hessians for optimization algorithms
 */
class Disturbance {
public:
    using Vec3 = Eigen::Vector3d;
    using Vec4 = Eigen::Vector4d;
    using BaseState = Eigen::Matrix<double, 7, 1>;
    using Mat34 = Eigen::Matrix<double, 3, 4>;
    using Mat44 = Eigen::Matrix<double, 4, 4>;
    using T443 = saltro::math::Tensor3<4, 4, 3>;  // 3 slices of 4×4 matrices for Hessians

    virtual ~Disturbance() = default;

    /**
     * @brief Compute the disturbance torque.
     * 
     * Pure virtual method accepting the base spacecraft state and disturbance
     * configuration. Must be implemented by derived classes.
     * 
     * @param x Base state vector (7D: angular velocity + quaternion).
     * @param dist_cfg Disturbance configuration (flags, coefficients, etc.).
     * @return 3D disturbance torque vector in body frame (Newton-meters).
     */
    virtual Vec3 torque(const BaseState& x, const DisturbanceConfig& dist_cfg) const = 0;

    /**
     * @brief Jacobian of disturbance torque with respect to quaternion.
     * 
     * Computes \f$\frac{\partial \boldsymbol{\tau}}{\partial \mathbf{q}}\f$ as a 3×4 matrix.
     * Default implementation returns zero (for disturbances independent of attitude).
     * 
     * @param x Base state.
     * @param dist_cfg Disturbance configuration.
     * @return 3×4 Jacobian matrix (zero by default).
     */
    virtual Mat34 dtorque_dq(const BaseState& x, const DisturbanceConfig& dist_cfg) const {
        return Mat34::Zero();
    }

    /**
     * @brief Hessian of disturbance torque with respect to quaternion.
     * 
     * Computes second derivatives \f$\frac{\partial^2 \tau_i}{\partial q_j \partial q_k}\f$ 
     * as a 3-slice tensor of 4×4 matrices.
     * Default implementation returns zero.
     * 
     * @param x Base state.
     * @param dist_cfg Disturbance configuration.
     * @return Tensor3 with 3 slices of 4×4 matrices (zero by default).
     */
    virtual T443 ddtorque_dqdq(const BaseState& x, const DisturbanceConfig& dist_cfg) const {
        return T443::Zero();
    }

    /**
     * @brief Enable or disable this disturbance for computations.
     * 
     * @param active True to enable, false to disable.
     */
    void setActive(bool active) { active_ = active; }
    
    /**
     * @brief Check if this disturbance is active.
     * 
     * @return True if active, false otherwise.
     */
    bool isActive() const { return active_; }

protected:
    bool active_ = true;  ///< Activation flag
};