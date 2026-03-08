#pragma once

#include <array>
#include <Eigen/Dense>
#include <saltro/pybind/disturbances/disturbance.h>
#include <saltro/pybind/disturbances/geometryconfig.h>

namespace saltro::disturbances {

/**
 * @brief Solar radiation pressure (SRP) disturbance model.
 * 
 * Computes torque due to solar radiation pressure acting on the spacecraft's surfaces.
 * The SRP force depends on solar irradiance and the reflective/absorptive properties
 * of each surface element.
 * 
 * For each face, the SRP force is:
 * \f[
 * \mathbf{F}_i = \frac{I}{c} A_i \left[\eta_a \mathbf{s} + 
 * 2(\eta_s + \eta_d \cos\theta) (\mathbf{s} \cdot \mathbf{n}_i) \mathbf{n}_i\right]
 * \f]
 * 
 * where:
 * - \f$I\f$ is solar irradiance (W/m²)
 * - \f$c\f$ is speed of light
 * - \f$\eta_a, \eta_s, \eta_d\f$ are absorptivity and reflection coefficients
 * - \f$\mathbf{s}\f$ is the unit sun direction
 * - \f$\theta\f$ is the incident angle
 * 
 * The torque is the sum of all face torques: \f$\boldsymbol{\tau} = \sum_i \mathbf{r}_i \times \mathbf{F}_i\f$
 * 
 * @see GeometryConfig for satellite geometry definition
 */
class SRPDisturbance : public ::Disturbance {
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
     * @brief Default constructor; creates disturbance with empty geometry.
     */
    SRPDisturbance() = default;
    
    /**
     * @brief Construct an SRP disturbance with specified geometry.
     * 
     * @param config Satellite geometry configuration (faces and optical properties).
     */
    explicit SRPDisturbance(const GeometryConfig& config);

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
     * @brief Compute SRP torque.
     * 
     * Evaluates solar radiation pressure disturbance given the state and configuration.
     * The sun direction vector must be provided separately.
     * 
     * @param x Base state (contains quaternion for frame transformations).
     * @param dist_cfg Disturbance configuration (SRP coefficient scaling).
     * @return 3D SRP torque in body frame.
     */
    Vec3 torque(const BaseState& x, const DisturbanceConfig& dist_cfg) const override;

    /**
     * @brief Compute SRP torque with explicit sun direction.
     * 
     * Extended interface allowing direct sun direction input.
     * 
     * @param x Base state.
     * @param dist_cfg Disturbance configuration.
     * @param v_body Sun direction vector in body frame (unit vector).
     * @return 3D SRP torque.
     */
    Vec3 torque(const BaseState& x, const DisturbanceConfig& dist_cfg, const Vec3& v_body) const;
    
    /**
     * @brief Jacobian of SRP torque with respect to quaternion.
     * 
     * @param x Base state.
     * @param dist_cfg Disturbance configuration.
     * @param v_body Sun direction in body frame.
     * @param dV_dq Jacobian of sun direction with respect to quaternion (4×3).
     * @return 3×4 Jacobian matrix.
     */
    Mat34 dtorque_dq(const BaseState& x, const DisturbanceConfig& dist_cfg, const Vec3& v_body,
                     const Mat34& dV_dq) const;
    
    /**
     * @brief Hessian of SRP torque with respect to quaternion.
     * 
     * @param x Base state.
     * @param dist_cfg Disturbance configuration.
     * @param v_body Sun direction in body frame.
     * @param dV_dq Jacobian of sun direction with respect to quaternion.
     * @param d2V_dq2 Array of three 4×4 Hessian matrices for sun direction.
     * @return Tensor3 with 3 slices of 4×4 matrices.
     */
    T443 ddtorque_dqdq(const BaseState& x, const DisturbanceConfig& dist_cfg, const Vec3& v_body,
                       const Mat34& dV_dq, const std::array<Mat44, 3>& d2V_dq2) const;

private:
    GeometryConfig config_;  ///< Satellite geometry
};

}  // namespace saltro::disturbances
