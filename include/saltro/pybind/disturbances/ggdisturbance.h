#pragma once

#include <array>
#include <Eigen/Dense>
#include <saltro/pybind/disturbances/disturbance.h>

namespace saltro::disturbances {

/**
 * @brief Gravity gradient (GG) disturbance model.
 * 
 * Computes torque due to the differential gravitational force (gravity gradient)
 * acting on a spatially extended spacecraft. The torque arises because the spacecraft's
 * geometry extends over a distance comparable to the gravitational field gradient.
 * 
 * The gravity gradient torque is approximated as:
 * \f[
 * \boldsymbol{\tau}_{\text{gg}} = 3 \frac{\mu}{r^5} \mathbf{r} \times (\mathbf{J}\,\mathbf{r})
 * \f]
 * 
 * where:
 * - \f$\mu\f$ is Earth's gravitational parameter
 * - \f$\mathbf{r}\f$ is the spacecraft position (body-fixed frame)
 * - \f$\mathbf{J}\f$ is the spacecraft's moment of inertia tensor
 * 
 * This effect is typically small but important for high-precision attitude control
 * and for large spacecraft with substantial moment of inertia asymmetry.
 */
class GGDisturbance : public ::Disturbance {
public:
    using Vec3 = Eigen::Vector3d;
    using Mat33 = Eigen::Matrix3d;
    using BaseState = ::Disturbance::BaseState;
    using Mat34 = ::Disturbance::Mat34;
    using Mat44 = ::Disturbance::Mat44;
    using T443 = ::Disturbance::T443;

    using ::Disturbance::dtorque_dq;
    using ::Disturbance::ddtorque_dqdq;

    /**
     * @brief Default constructor; initializes with identity inertia.
     */
    GGDisturbance() = default;
    
    /**
     * @brief Construct a gravity gradient disturbance with specified inertia.
     * 
     * @param inertia 3×3 moment of inertia tensor (kg·m²).
     */
    explicit GGDisturbance(const Mat33& inertia);

    /**
     * @brief Set the spacecraft's moment of inertia tensor.
     * 
     * @param inertia 3×3 inertia matrix.
     */
    void setInertia(const Mat33& inertia) { inertia_ = inertia; }
    
    /**
     * @brief Get the spacecraft's moment of inertia tensor.
     * 
     * @return Const reference to the 3×3 inertia matrix.
     */
    const Mat33& inertia() const { return inertia_; }

    /**
     * @brief Compute gravity gradient torque.
     * 
     * The position vector must be in the body-fixed reference frame for correct
     * computation of the cross product with the inertia tensor.
     * 
     * @param x Base state (quaternion for frame transformations).
     * @param dist_cfg Disturbance configuration (unused for GG).
     * @return 3D gravity gradient torque.
     */
    Vec3 torque(const BaseState& x, const DisturbanceConfig& dist_cfg) const override;

    /**
     * @brief Compute gravity gradient torque with explicit position.
     * 
     * Extended interface allowing direct position input in body frame.
     * 
     * @param x Base state.
     * @param dist_cfg Disturbance configuration.
     * @param r_body Position vector in body frame (m).
     * @param J Inertia tensor (kg·m²).
     * @return 3D gravity gradient torque.
     */
    Vec3 torque(const BaseState& x, const DisturbanceConfig& dist_cfg, const Vec3& r_body,
                const Mat33& J) const;
    
    /**
     * @brief Jacobian of gravity gradient torque with respect to quaternion.
     * 
     * @param x Base state.
     * @param dist_cfg Disturbance configuration.
     * @param r_body Position in body frame.
     * @param J Inertia tensor.
     * @param dr_dq Jacobian of position with respect to quaternion (4×3).
     * @return 3×4 Jacobian matrix.
     */
    Mat34 dtorque_dq(const BaseState& x, const DisturbanceConfig& dist_cfg, const Vec3& r_body,
                     const Mat33& J, const Mat34& dr_dq) const;
    
    /**
     * @brief Hessian of gravity gradient torque with respect to quaternion.
     * 
     * @param x Base state.
     * @param dist_cfg Disturbance configuration.
     * @param r_body Position in body frame.
     * @param J Inertia tensor.
     * @param dr_dq Jacobian of position with respect to quaternion.
     * @param d2r_dq2 Array of three 4×4 Hessian matrices for position.
     * @return Tensor3 with 3 slices of 4×4 matrices.
     */
    T443 ddtorque_dqdq(const BaseState& x, const DisturbanceConfig& dist_cfg, const Vec3& r_body,
                       const Mat33& J, const Mat34& dr_dq,
                       const std::array<Mat44, 3>& d2r_dq2) const;

private:
    Mat33 inertia_ = Mat33::Identity();  ///< Moment of inertia tensor
};

}  // namespace saltro::disturbances
