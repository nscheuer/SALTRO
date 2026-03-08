#pragma once

#include <array>
#include <Eigen/Dense>
#include <saltro/pybind/disturbances/disturbance.h>
#include <saltro/pybind/disturbances/geometryconfig.h>

namespace saltro::disturbances {

/**
 * @brief Aerodynamic drag disturbance model.
 * 
 * Computes torque due to aerodynamic drag acting on the spacecraft's surfaces.
 * The drag force depends on atmospheric density (which varies along the orbit)
 * and the spacecraft's velocity relative to the atmosphere.
 * 
 * The total drag torque is computed as the sum of torques from all surface faces:
 * \f[
 * \boldsymbol{\tau}_{\text{drag}} = \sum_i C_D^i \rho A_i (\mathbf{v} \cdot \mathbf{n}_i) \mathbf{r}_i \times \mathbf{v}
 * \f]
 * where:
 * - \f$C_D^i\f$ is the drag coefficient of face \f$i\f$
 * - \f$A_i\f$ is the surface area
 * - \f$\mathbf{n}_i\f$ is the surface normal
 * - \f$\mathbf{r}_i\f$ is the centroid position relative to center of mass
 * 
 * @see GeometryConfig for satellite geometry definition
 */
class DragDisturbance : public ::Disturbance {
public:
    using Vec3 = Eigen::Vector3d;
    using BaseState = ::Disturbance::BaseState;
    using Mat34 = ::Disturbance::Mat34;
    using Mat44 = ::Disturbance::Mat44;
    using T443 = ::Disturbance::T443;

    using ::Disturbance::dtorque_dq;
    using ::Disturbance::ddtorque_dqdq;

    /**
     * @brief Default constructor; creates disturbance with empty geometry.
     */
    DragDisturbance() = default;
    
    /**
     * @brief Construct a drag disturbance with specified geometry.
     * 
     * @param config Satellite geometry configuration (faces and their properties).
     */
    explicit DragDisturbance(const GeometryConfig& config);

    /**
     * @brief Set the satellite geometry configuration.
     * 
     * @param config New geometry.
     */
    void setGeometryConfig(const GeometryConfig& config) { config_ = config; }
    
    /**
     * @brief Get the geometry configuration.
     * 
     * @return Const reference to geometry.
     */
    const GeometryConfig& geometryConfig() const { return config_; }

    /**
     * @brief Compute drag torque.
     * 
     * Evaluates drag disturbance given the state and configuration.
     * The velocity vector must be provided separately.
     * 
     * @param x Base state (primarily for quaternion).
     * @param dist_cfg Disturbance configuration (drag coefficient scaling).
     * @return 3D drag torque in body frame.
     */
    Vec3 torque(const BaseState& x, const DisturbanceConfig& dist_cfg) const override;

    /**
     * @brief Compute drag torque with explicit velocity.
     * 
     * Extended interface allowing direct velocity input.
     * 
     * @param x Base state.
     * @param dist_cfg Disturbance configuration.
     * @param v_body Velocity vector in body frame (m/s).
     * @return 3D drag torque.
     */
    Vec3 torque(const BaseState& x, const DisturbanceConfig& dist_cfg, const Vec3& v_body) const;
    
    /**
     * @brief Jacobian of drag torque with respect to quaternion.
     * 
     * @param x Base state.
     * @param dist_cfg Disturbance configuration.
     * @param v_body Velocity in body frame.
     * @param dV_dq Jacobian of velocity with respect to quaternion (4×3).
     * @return 3×4 Jacobian matrix.
     */
    Mat34 dtorque_dq(const BaseState& x, const DisturbanceConfig& dist_cfg, const Vec3& v_body,
                     const Mat34& dV_dq) const;
    
    /**
     * @brief Hessian of drag torque with respect to quaternion.
     * 
     * @param x Base state.
     * @param dist_cfg Disturbance configuration.
     * @param v_body Velocity in body frame.
     * @param dV_dq Jacobian of velocity with respect to quaternion.
     * @param d2V_dq2 Array of three 4×4 Hessian matrices for velocity.
     * @return Tensor3 with 3 slices of 4×4 matrices.
     */
    T443 ddtorque_dqdq(const BaseState& x, const DisturbanceConfig& dist_cfg, const Vec3& v_body,
                       const Mat34& dV_dq, const std::array<Mat44, 3>& d2V_dq2) const;

private:
    GeometryConfig config_;  ///< Satellite geometry
};

}  // namespace saltro::disturbances
