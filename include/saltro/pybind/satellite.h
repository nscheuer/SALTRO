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

    static constexpr int AV_INDEX = 0;
    static constexpr int QUAT_INDEX = 3;
    static constexpr int RW_MOMENTUM_INDEX = 7;

    Satellite();
    Satellite(const Mat33& Jcom_in, const PlannerSettings& settings);

    ~Satellite() = default;

    void setInertia(const Mat33& Jcom_in);
    const Mat33& inertia() const { return Jcom_; }
    const Mat33& invInertia() const { return invJcom_; }
    const Mat33& inertiaNoRW() const { return Jcom_noRW_; }
    const Mat33& invInertiaNoRW() const { return invJcom_noRW_; }

    void setGeometryConfig(const saltro::disturbances::GeometryConfig& config);
    const saltro::disturbances::GeometryConfig& geometryConfig() const { return geometry_config_; }
    saltro::disturbances::GeometryConfig& geometryConfig() { return geometry_config_; }

    void addMTQ(const Vec3& axis, double max_dipole);
    void addRW(const Vec3& axis, double max_torque, double J, double h0, double h_max);
    int numMTQ() const { return num_mtq_; }
    int numRW() const { return num_rw_; }

    int controlDim() const { return num_mtq_ + num_rw_; }
    int stateDim() const { return 7 + num_rw_; }
    int reducedStateDim() const { return 6 + num_rw_; }

    const MTQ& getMTQ(int i) const;
    const RW& getRW(int i) const;

    MTQ& getMTQ(int i);
    RW& getRW(int i);

    void setSettings(const PlannerSettings& settings) { settings_ = settings; }
    const PlannerSettings& settings() const { return settings_; }

    Vec3 actuatorTorque(const VecX& x, const VecX& u, const Vec3& B_eci) const;
    Vec3 disturbanceTorque(const VecX& x, const DisturbanceConfig& dist, const Vec3& B_eci, const Vec3& S_eci, const Vec3& V_eci, const int rho) const;
    VecX dynamics(const VecX& x, const VecX& u, const DisturbanceConfig& dist, const Vec3& B_eci, const Vec3& S_eci, const Vec3& V_eci, const int rho) const;

    std::tuple<MatX, MatX, MatX> dynamicsJacobians(const VecX& x, const VecX& u, const DisturbanceConfig& dist) const;
    std::tuple<DynHessXX, DynHessUX, DynHessUU> dynamicsHessians(const VecX& x, const VecX& u, const DisturbanceConfig& dist) const;

    double stageCost(int k, int N, const VecX& x, const VecX& u, const Vec3& sat_direction, const Vec4& eci_target, const Vec3& B_eci, const CostConfig& cost_cfg) const;
    double terminalCost(const VecX& x, const Vec3& sat_direction, const Vec4& eci_target, const Vec3& B_eci, const CostConfig& cost_cfg) const;

    std::tuple<VecX, MatX, MatX> stageCostJacobians(int k, int N, const VecX& x, const VecX& u, const Vec3& sat_direction, const Vec4& eci_target, const Vec3& B_eci, const CostConfig& cost_cfg) const;
    std::tuple<MatX, MatX, MatX> stageCostHessians(int k, int N, const VecX& x, const VecX& u, const Vec3& sat_direction, const Vec4& eci_target, const Vec3& B_eci, const CostConfig& cost_cfg) const;

    VecX constraints(int k, int N, const VecX& x, const VecX& u, const Vec3& sun_eci, const ConstraintConfig& cnst_cfg) const;
    std::tuple<MatX, MatX> constraintJacobians(int k, int N, const VecX& x, const VecX& u, const Vec3& sun_eci, const ConstraintConfig& cnst_cfg) const;
    std::tuple<MatX, MatX, MatX> constraintHessians(int k, int N, const VecX& x, const VecX& u, const Vec3& sun_eci, const ConstraintConfig& cnst_cfg) const;

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

    void updateInertiaNoRW();

};
