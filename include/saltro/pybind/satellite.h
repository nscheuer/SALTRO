#pragma once

#include <array>
#include <memory>
#include <tuple>
#include <Eigen/Dense>

#include <saltro/pybind/actuators/actuator.h>
#include <saltro/pybind/actuators/RW.h>
#include <saltro/pybind/actuators/MTQ.h>
#include <saltro/limits.h>
#include <saltro/pybind/plannersettings.h>
#include <saltro/math/angles.h>
#include <saltro/math/quaternion.h>
#include <saltro/pybind/disturbances/geometryconfig.h>

/**
 * @brief Central spacecraft model integrating dynamics, actuators, and disturbances.
 * 
 * The Satellite class encapsulates the complete spacecraft model including:
 * - Attitude dynamics (quaternion and angular velocity)
 * - Reaction wheel momentum states
 * - Actuator (magnetorquer and reaction wheel) models
 * - Disturbance models (drag, SRP, gravity gradient)
 * - Cost and constraint functions for trajectory optimization
 * 
 * State vector structure:
 * \f[
 * \mathbf{x} = \begin{bmatrix}
 * \boldsymbol{\omega} \\
 * \mathbf{q} \\
 * \mathbf{h}
 * \end{bmatrix}
 * \quad \text{where} \quad
 * \boldsymbol{\omega} = \text{angular velocity (3D)},
 * \mathbf{q} = \text{attitude quaternion (4D)},
 * \mathbf{h} = \text{reaction wheel momentum}
 * \f]
 * 
 * Control input structure:
 * \f[
 * \mathbf{u} = \begin{bmatrix}
 * \mathbf{m} \\
 * \boldsymbol{\tau}_{rw}
 * \end{bmatrix}
 * \f]
 * where \f$\mathbf{m}\f$ are magnetorquer dipoles and \f$\boldsymbol{\tau}_{rw}\f$ are RW torques.
 */
class Satellite {
public:
    // ── Fixed-size type aliases ──────────────────────────────────────────
    using Vec3 = Eigen::Vector3d;
    using Vec4 = Eigen::Vector4d;
    using Vec7 = Eigen::Matrix<double, 7, 1>;
    using Mat33 = Eigen::Matrix3d;
    using Mat34 = Eigen::Matrix<double, 3, 4>;
    using Mat43 = Eigen::Matrix<double, 4, 3>;
    using Mat73 = Eigen::Matrix<double, 7, 3>;
    using Mat77 = Eigen::Matrix<double, 7, 7>;

    static constexpr int MaxDim = saltro::limits::MAX_CONSTRAINT_DIM;

    using VecX  = Eigen::Matrix<double, Eigen::Dynamic, 1, 0, MaxDim, 1>;

    using MatX  = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0,
                                MaxDim, MaxDim>;

    using DynHessXX = saltro::math::Tensor3<saltro::limits::MAX_STATE_DIM,
                              saltro::limits::MAX_STATE_DIM,
                              saltro::limits::MAX_STATE_DIM>;
    using DynHessUX = saltro::math::Tensor3<saltro::limits::MAX_CTRL_DIM,
                              saltro::limits::MAX_STATE_DIM,
                              saltro::limits::MAX_STATE_DIM>;
    using DynHessUU = saltro::math::Tensor3<saltro::limits::MAX_CTRL_DIM,
                              saltro::limits::MAX_CTRL_DIM,
                              saltro::limits::MAX_STATE_DIM>;

    /**
     * @brief Index of angular velocity in state vector.
     * 
     * Angular velocity \f$\boldsymbol{\omega}\f$ occupies indices 0–2.
     */
    static constexpr int AV_INDEX = 0;
    
    /**
     * @brief Index of quaternion in state vector.
     * 
     * Quaternion \f$\mathbf{q}\f$ occupies indices 3–6.
     */
    static constexpr int QUAT_INDEX = 3;
    
    /**
     * @brief Index of first reaction wheel momentum in state vector.
     * 
     * Reaction wheel momenta start at index 7 (for each added RW).
     */
    static constexpr int RW_MOMENTUM_INDEX = 7;

    /**
     * @brief Default constructor; creates zero inertia and empty actuators.
     */
    Satellite();
    
    /**
     * @brief Construct a satellite with specified inertia and planner settings.
     * 
     * @param Jcom_in 3×3 inertia matrix (center of mass, body frame).
     * @param settings Planner settings (cost/constraint configurations).
     */
    Satellite(const Mat33& Jcom_in, const PlannerSettings& settings);

    ~Satellite() = default;

    /**
     * @brief Set the satellite's moment of inertia.
     * 
     * Updates inertia and recomputes inertia without reaction wheels if needed.
     * 
     * @param Jcom_in 3×3 inertia matrix.
     */
    void setInertia(const Mat33& Jcom_in);
    
    /**
     * @brief Get total inertia (including reaction wheel contribution).
     * 
     * @return Const reference to inertia matrix.
     */
    const Mat33& inertia() const { return Jcom_; }
    
    /**
     * @brief Get inverse of total inertia.
     * 
     * @return Const reference to inverse inertia matrix.
     */
    const Mat33& invInertia() const { return invJcom_; }
    
    /**
     * @brief Get inertia contribution excluding reaction wheels.
     * 
     * @return Const reference to spacecraft-body inertia.
     */
    const Mat33& inertiaNoRW() const { return Jcom_noRW_; }
    
    /**
     * @brief Get inverse inertia excluding reaction wheels.
     * 
     * @return Const reference to inverse inertia (no RW).
     */
    const Mat33& invInertiaNoRW() const { return invJcom_noRW_; }

    /**
     * @brief Set satellite geometry configuration.
     * 
     * Used by disturbance models (drag, SRP) for surface calculations.
     * 
     * @param config Geometry configuration with faces.
     */
    void setGeometryConfig(const saltro::disturbances::GeometryConfig& config);
    
    /**
     * @brief Get geometry configuration (const).
     * 
     * @return Const reference to geometry.
     */
    const saltro::disturbances::GeometryConfig& geometryConfig() const { return geometry_config_; }
    
    /**
     * @brief Get geometry configuration (mutable).
     * 
     * @return Reference to geometry.
     */
    saltro::disturbances::GeometryConfig& geometryConfig() { return geometry_config_; }

    /**
     * @brief Add a magnetorquer to the satellite.
     * 
     * @param axis Unit vector specifying coil direction (body frame).
     * @param max_dipole Maximum dipole moment (A·m²).
     * @throws std::runtime_error If max MTQ count exceeded.
     */
    void addMTQ(const Vec3& axis, double max_dipole);
    
    /**
     * @brief Add a reaction wheel to the satellite.
     * 
     * @param axis Unit vector specifying wheel spin axis (body frame).
     * @param max_torque Maximum torque output (N·m).
     * @param J Wheel moment of inertia (kg·m²).
     * @param h0 Initial momentum (N·m·s).
     * @param h_max Maximum momentum (N·m·s).
     * @throws std::runtime_error If max RW count exceeded.
     */
    void addRW(const Vec3& axis, double max_torque, double J, double h0, double h_max);
    
    /**
     * @brief Get number of magnetorquers.
     * 
     * @return Number of MTQs.
     */
    int numMTQ() const { return num_mtq_; }
    
    /**
     * @brief Get number of reaction wheels.
     * 
     * @return Number of RWs.
     */
    int numRW() const { return num_rw_; }

    /**
     * @brief Get total control input dimension.
     * 
     * Equals \f$\text{numMTQ} + \text{numRW}\f$.
     * 
     * @return Control dimension.
     */
    int controlDim() const { return num_mtq_ + num_rw_; }
    
    /**
     * @brief Get full state dimension.
     * 
     * Equals \f$7 + \text{numRW}\f$ (AV + Q + RW momenta).
     * 
     * @return State dimension.
     */
    int stateDim() const { return 7 + num_rw_; }
    
    /**
     * @brief Get reduced state dimension for optimization.
     * 
     * Equals \f$6 + \text{numRW}\f$ (without full attitude).
     * 
     * @return Reduced state dimension.
     */
    int reducedStateDim() const { return 6 + num_rw_; }

    /**
     * @brief Get a magnetorquer by index (const).
     * 
     * @param i MTQ index.
     * @return Const reference to MTQ.
     */
    const MTQ& getMTQ(int i) const;
    
    /**
     * @brief Get a reaction wheel by index (const).
     * 
     * @param i RW index.
     * @return Const reference to RW.
     */
    const RW& getRW(int i) const;

    /**
     * @brief Get a magnetorquer by index (mutable).
     * 
     * @param i MTQ index.
     * @return Reference to MTQ.
     */
    MTQ& getMTQ(int i);
    
    /**
     * @brief Get a reaction wheel by index (mutable).
     * 
     * @param i RW index.
     * @return Reference to RW.
     */
    RW& getRW(int i);

    /**
     * @brief Set planner settings.
     * 
     * @param settings Settings object.
     */
    void setSettings(const PlannerSettings& settings) { settings_ = settings; }
    
    /**
     * @brief Get planner settings (const).
     * 
     * @return Const reference to settings.
     */
    const PlannerSettings& settings() const { return settings_; }

    /**
     * @brief Compute total actuator torque.
     * 
     * Sums torques from all MTQs and RWs given control input and state.
     * 
     * @param x State vector (contains quaternion for ECI→body transformation).
     * @param u Control input.
     * @param B_eci Magnetic field in ECI frame (Tesla).
     * @return 3D actuator torque in body frame.
     */
    Vec3 actuatorTorque(const VecX& x, const VecX& u, const Vec3& B_eci) const;
    
    /**
     * @brief Compute total disturbance torque.
     * 
     * Sums torques from all enabled disturbance models.
     * 
     * @param x State vector.
     * @param dist Disturbance configuration.
     * @param R_eci Position in ECI frame (meters).
     * @param B_eci Magnetic field in ECI (Tesla).
     * @param S_eci Spacecraft-to-Sun in ECI (meters).
     * @param V_eci Velocity in ECI (m/s).
     * @param rho Atmospheric density (kg/m³).
     * @return 3D disturbance torque in body frame.
     */
    Vec3 disturbanceTorque(const VecX& x, const DisturbanceConfig& dist, 
                          const Vec3& R_eci, const Vec3& B_eci, const Vec3& S_eci, 
                          const Vec3& V_eci, const int rho) const;
    
    /**
     * @brief Compute spacecraft dynamics.
     * 
     * Returns state derivative:
     * \f[
     * \dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, \mathbf{u}, \text{env})
     * \f]
     * 
     * @param x State vector.
     * @param u Control input.
     * @param dist Disturbance configuration.
     * @param R_eci Orbital position (ECI).
     * @param B_eci Magnetic field (ECI).
     * @param S_eci Sun direction (ECI).
     * @param V_eci Velocity (ECI).
     * @param rho Atmospheric density.
     * @return State derivative vector.
     */
    VecX dynamics(const VecX& x, const VecX& u, const DisturbanceConfig& dist, 
                 const Vec3& R_eci, const Vec3& B_eci, const Vec3& S_eci, 
                 const Vec3& V_eci, const int rho) const;

    /**
     * @brief Compute dynamics Jacobians.
     * 
     * Returns \f$\frac{\partial \mathbf{f}}{\partial \mathbf{x}}\f$, 
     * \f$\frac{\partial \mathbf{f}}{\partial \mathbf{u}}\f$, and 
     * \f$\frac{\partial \mathbf{f}}{\partial \text{disturbances}}\f$.
     * 
     * @param x State.
     * @param u Control.
     * @param dist Disturbances.
     * @return Tuple of Jacobian matrices (Fx, Fu, Fd).
     */
    std::tuple<MatX, MatX, MatX> dynamicsJacobians(const VecX& x, const VecX& u, 
                                                   const DisturbanceConfig& dist) const;
    
    /**
     * @brief Compute dynamics Hessians (second derivatives).
     * 
     * @param x State.
     * @param u Control.
     * @param dist Disturbances.
     * @return Tuple of Hessian tensors (Hxx, Hux, Huu).
     */
    std::tuple<DynHessXX, DynHessUX, DynHessUU> dynamicsHessians(const VecX& x, const VecX& u, 
                                                                 const DisturbanceConfig& dist) const;

    /**
     * @brief Compute stage cost (intermediate time step).
     * 
     * Evaluates the cost function at time step k:
     * \f[
     * J_k = c(\mathbf{x}_k, \mathbf{u}_k, k)
     * \f]
     * 
     * @param k Current time step index.
     * @param N Total time steps.
     * @param x State at step k.
     * @param u Control at step k.
     * @param sat_direction Target direction for satellite (body frame).
     * @param eci_target Target quaternion in ECI frame.
     * @param B_eci Magnetic field (ECI).
     * @param cost_cfg Cost configuration.
     * @return Scalar cost value.
     */
    double stageCost(int k, int N, const VecX& x, const VecX& u, 
                    const Vec3& sat_direction, const Vec4& eci_target, 
                    const Vec3& B_eci, const CostConfig& cost_cfg) const;
    
    /**
     * @brief Compute terminal cost (final time step).
     * 
     * @param x Final state.
     * @param sat_direction Target satellite direction.
     * @param eci_target Target quaternion.
     * @param B_eci Magnetic field.
     * @param cost_cfg Cost configuration.
     * @return Scalar terminal cost.
     */
    double terminalCost(const VecX& x, const Vec3& sat_direction, const Vec4& eci_target, 
                       const Vec3& B_eci, const CostConfig& cost_cfg) const;

    /**
     * @brief Compute stage cost Jacobians.
     * 
     * Returns gradients of cost:
     * \f[\frac{\partial J_k}{\partial \mathbf{x}}, \quad \frac{\partial J_k}{\partial \mathbf{u}}\f]
     * 
     * @param k Time step.
     * @param N Total steps.
     * @param x State.
     * @param u Control.
     * @param sat_direction Target direction.
     * @param eci_target Target quaternion.
     * @param B_eci Magnetic field.
     * @param cost_cfg Cost configuration.
     * @return Tuple of (∇_x J, ∇_u J).
     */
    std::tuple<VecX, MatX, MatX> stageCostJacobians(int k, int N, const VecX& x, const VecX& u, 
                                                    const Vec3& sat_direction, const Vec4& eci_target, 
                                                    const Vec3& B_eci, const CostConfig& cost_cfg) const;
    
    /**
     * @brief Compute stage cost Hessians.
     * 
     * @param k Time step.
     * @param N Total steps.
     * @param x State.
     * @param u Control.
     * @param sat_direction Target direction.
     * @param eci_target Target quaternion.
     * @param B_eci Magnetic field.
     * @param cost_cfg Cost configuration.
     * @return Tuple of Hessian matrices (Hxx, Huu, Hxu).
     */
    std::tuple<MatX, MatX, MatX> stageCostHessians(int k, int N, const VecX& x, const VecX& u, 
                                                   const Vec3& sat_direction, const Vec4& eci_target, 
                                                   const Vec3& B_eci, const CostConfig& cost_cfg) const;

    /**
     * @brief Evaluate constraints.
     * 
     * Returns constraint violation vector. Zero means satisfied.
     * Constraints include angular velocity limits, actuator saturations, sun avoidance.
     * 
     * @param k Time step.
     * @param N Total steps.
     * @param x State.
     * @param u Control.
     * @param sun_eci Sun direction in ECI.
     * @param cnst_cfg Constraint configuration.
     * @return Constraint violation vector.
     */
    VecX constraints(int k, int N, const VecX& x, const VecX& u, 
                    const Vec3& sun_eci, const ConstraintConfig& cnst_cfg) const;
    
    /**
     * @brief Compute constraint Jacobians.
     * 
     * @param k Time step.
     * @param N Total steps.
     * @param x State.
     * @param u Control.
     * @param sun_eci Sun in ECI.
     * @param cnst_cfg Constraint configuration.
     * @return Tuple of (∂c/∂x, ∂c/∂u).
     */
    std::tuple<MatX, MatX> constraintJacobians(int k, int N, const VecX& x, const VecX& u, 
                                               const Vec3& sun_eci, const ConstraintConfig& cnst_cfg) const;
    
    /**
     * @brief Compute constraint Hessians.
     * 
     * @param k Time step.
     * @param N Total steps.
     * @param x State.
     * @param u Control.
     * @param sun_eci Sun in ECI.
     * @param cnst_cfg Constraint configuration.
     * @return Tuple of Hessian matrices.
     */
    std::tuple<MatX, MatX, MatX> constraintHessians(int k, int N, const VecX& x, const VecX& u, 
                                                    const Vec3& sun_eci, const ConstraintConfig& cnst_cfg) const;

private:
    Mat33 Jcom_;
    Mat33 invJcom_;
    Mat33 Jcom_noRW_;
    Mat33 invJcom_noRW_;

    std::array<std::unique_ptr<MTQ>, saltro::limits::MAX_NUM_MTQ> mtq_actuators_;
    std::array<std::unique_ptr<RW>, saltro::limits::MAX_NUM_RW> rw_actuators_;
    int num_mtq_ = 0;
    int num_rw_ = 0;

    PlannerSettings settings_;
    saltro::disturbances::GeometryConfig geometry_config_;

    /**
     * @brief Recompute inertia with and without RW contributions.
     */
    void updateInertiaNoRW();

};