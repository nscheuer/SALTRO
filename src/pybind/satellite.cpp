#include <saltro/pybind/satellite.h>
#include <saltro/pybind/disturbances/dragdisturbance.h>
#include <saltro/pybind/disturbances/ggdisturbance.h>
#include <saltro/pybind/disturbances/srpdisturbance.h>
#include <saltro/math/integrators/rk4.h>

#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <limits>

using std::invalid_argument;
using std::out_of_range;

namespace {

inline double safeAbs(double x) {
    return std::abs(x);
}

inline double safeSign(double x) {
    return (x >= 0.0) ? 1.0 : -1.0;
}

inline double clampUnit(double x) {
    return std::clamp(x, -1.0, 1.0);
}

inline double safeNorm(const Eigen::Vector3d& v) {
    const double n = v.norm();
    return std::isfinite(n) ? n : 0.0;
}

inline bool safeUnitQuat(const Eigen::Vector4d& q_raw, Eigen::Vector4d& q_unit) {
    const double n = q_raw.norm();
    if (!std::isfinite(n) || n < 1e-12) {
        q_unit = Eigen::Vector4d::Zero();
        return false;
    }
    q_unit = q_raw / n;
    return true;
}

/**
 * @brief Helper to check if first element is NaN (indicating ECI format).
 */
inline bool isECIFormat(const Eigen::Vector4d& vec) {
    return std::isnan(vec(0));
}

}

Satellite::Satellite() {
    Jcom_.setIdentity();
    invJcom_.setIdentity();
    Jcom_noRW_.setIdentity();
    invJcom_noRW_.setIdentity();
}

Satellite::Satellite(const Mat33& Jcom_in, const PlannerSettings& settings) : settings_(settings) {
    setInertia(Jcom_in);
}

void Satellite::setInertia(const Mat33& Jcom_in) {
    if (!Jcom_in.allFinite()) {
        throw invalid_argument("Inertia matrix must have finite entries.");
    }
    if (std::abs(Jcom_in.determinant()) < 1e-12) {
        throw invalid_argument("Inertia matrix must be invertible.");
    }

    Jcom_ = Jcom_in;
    invJcom_ = Jcom_.inverse();

    updateInertiaNoRW();
}

void Satellite::setGeometryConfig(const saltro::disturbances::GeometryConfig& config) {
    geometry_config_ = config;
}

void Satellite::addMTQ(const Vec3& axis, double max_dipole) {
    if (num_mtq_ >= saltro::limits::MAX_NUM_MTQ) {
        throw out_of_range("Exceeded MAX_NUM_MTQ.");
    }
    mtq_actuators_[num_mtq_] = std::make_unique<MTQ>(axis, max_dipole);

    ++num_mtq_;
}

void Satellite::addRW(const Vec3& axis, double max_torque, double J, double h0, double h_max) {
    if (num_rw_ >= saltro::limits::MAX_NUM_RW) {
        throw out_of_range("Exceeded MAX_NUM_RW.");
    }

    rw_actuators_[num_rw_] = std::make_unique<RW>(axis, max_torque, J, h0, h_max);

    ++num_rw_;
    updateInertiaNoRW();
}

void Satellite::addMagic(const Vec3& axis, double max_torque) {
    if (num_magic_ >= saltro::limits::MAX_NUM_MAGIC) {
        throw out_of_range("Exceeded MAX_NUM_MAGIC.");
    }
    magic_actuators_[num_magic_] = std::make_unique<Magic>(axis, max_torque);

    ++num_magic_;
}

const MTQ& Satellite::getMTQ(int i) const {
    if (i < 0 || i >= num_mtq_) {
        throw out_of_range("MTQ index out of range.");
    }
    return *mtq_actuators_[i];
}

MTQ& Satellite::getMTQ(int i) {
    if (i < 0 || i >= num_mtq_) {
        throw out_of_range("MTQ index out of range.");
    }
    return *mtq_actuators_[i];
}

const RW& Satellite::getRW(int i) const {
    if (i < 0 || i >= num_rw_) {
        throw out_of_range("RW index out of range.");
    }
    return *rw_actuators_[i];
}

RW& Satellite::getRW(int i) {
    if (i < 0 || i >= num_rw_) {
        throw out_of_range("RW index out of range.");
    }
    return *rw_actuators_[i];
}

const Magic& Satellite::getMagic(int i) const {
    if (i < 0 || i >= num_magic_) {
        throw out_of_range("Magic actuator index out of range.");
    }
    return *magic_actuators_[i];
}

Magic& Satellite::getMagic(int i) {
    if (i < 0 || i >= num_magic_) {
        throw out_of_range("Magic actuator index out of range.");
    }
    return *magic_actuators_[i];
}

void Satellite::updateInertiaNoRW() {
    Jcom_noRW_ = Jcom_;
    
    for (int i = 0; i < num_rw_; ++i) {
        const RW& rw = getRW(i);
        Vec3 axis = rw.axis();
        double J_rw = rw.wheelInertia();
        Jcom_noRW_ -= J_rw * axis * axis.transpose();
    }
    
    invJcom_noRW_ = Jcom_noRW_.inverse();
}

std::pair<Satellite::Vec4, bool> Satellite::processAttitudeTarget(
    const Vec4& attitude_target, const Vec3& boresight_body) const {
    // "No goal" sentinels:
    // 1) ECI sentinel [nan, 0, 0, 0]
    // 2) Quaternion sentinel [0, 0, 0, 0]
    // Return identity quaternion and mark as special-target mode so downstream
    // cost code can disable attitude terms.
    if (attitude_target.allFinite() && attitude_target.squaredNorm() < 1e-18) {
        return std::make_pair(Vec4(1.0, 0.0, 0.0, 0.0), true);
    }
    
    // Check if this is ECI format: first element is NaN
    if (isECIFormat(attitude_target)) {
        // ECI goal vector format: [nan, x, y, z]
        // In this case, we want to align boresight_body with the ECI vector
        Vec3 eci_vec = attitude_target.tail(3);
        if (!eci_vec.allFinite() || eci_vec.squaredNorm() < 1e-18) {
            return std::make_pair(Vec4(1.0, 0.0, 0.0, 0.0), true);
        }
        Vec3 eci_normalized = eci_vec.normalized();
        Vec3 sat_dir_normalized = boresight_body.normalized();
        
        // Compute the rotation from sat_direction to eci_vec
        // For simplicity, return a quaternion that represents this alignment
        // The quaternion is used to compute alignment cost
        Vec3 axis = sat_dir_normalized.cross(eci_normalized);
        double axis_norm = axis.norm();
        
        if (axis_norm < 1e-8) {
            // Vectors are parallel
            double dot_prod = sat_dir_normalized.dot(eci_normalized);
            if (dot_prod > 0) {
                // Already aligned
                return std::make_pair(Vec4(1.0, 0.0, 0.0, 0.0), true);
            } else {
                // Opposite direction - need 180 degree rotation around arbitrary perpendicular
                Vec3 perp = (std::abs(sat_dir_normalized(0)) < 0.9) 
                    ? Vec3(1.0, 0.0, 0.0) 
                    : Vec3(0.0, 1.0, 0.0);
                perp = perp - (perp.dot(sat_dir_normalized)) * sat_dir_normalized;
                perp = perp.normalized();
                return std::make_pair(Vec4(0.0, perp(0), perp(1), perp(2)), true);
            }
        } else {
            // General case: compute quaternion for rotation
            axis = axis.normalized();
            double angle = std::acos(std::clamp(sat_dir_normalized.dot(eci_normalized), -1.0, 1.0));
            double half_angle = angle / 2.0;
            Vec4 q_goal;
            q_goal(0) = std::cos(half_angle);
            q_goal.tail(3) = std::sin(half_angle) * axis;
            return std::make_pair(q_goal.normalized(), true);
        }
    } else {
        // Quaternion goal vector format: [q0, qx, qy, qz]
        return std::make_pair(attitude_target.normalized(), false);
    }
}

Satellite::Vec3 Satellite::actuatorTorque(const VecX& x, const VecX& u, const Vec3& B_eci) const {
    Vec3 torque = Vec3::Zero();
    
    // Extract base state (first 7 elements: w + q)
    Vec4 q;
    if (!safeUnitQuat(x.segment<4>(QUAT_INDEX), q)) {
        return Vec3::Zero();
    }
    Vec7 x_base = x.head<7>();
    x_base.segment<4>(QUAT_INDEX) = q;

    Mat33 R_T = saltro::math::rotationMatrix(q).transpose();
    Vec3 B_body = R_T * B_eci;
    
    if (num_mtq_ > 0) {
        for (int i = 0; i < num_mtq_; ++i) {
            double u_i = u(i);
            const MTQ& mtq = getMTQ(i);
            torque += mtq.torque(u_i, x_base, B_body);
        }
    }
    
    if (num_rw_ > 0) {
        for (int i = 0; i < num_rw_; ++i) {
            int ctrl_idx = num_mtq_ + i;
            double u_i = u(ctrl_idx);
            const RW& rw = getRW(i);
            torque += rw.torque(u_i, x_base);
        }
    }

    if (num_magic_ > 0) {
        for (int i = 0; i < num_magic_; ++i) {
            int ctrl_idx = num_mtq_ + num_rw_ + i;
            double u_i = u(ctrl_idx);
            const Magic& magic = getMagic(i);
            torque += magic.torque(u_i, x_base);
        }
    }

    return torque;
}

Satellite::Vec3 Satellite::disturbanceTorque(const VecX& x, const DisturbanceConfig& dist, const Vec3& R_eci, const Vec3& B_eci, const Vec3& S_eci, const Vec3& V_eci, const int rho) const {
    Vec3 torque = Vec3::Zero();
    Vec4 q;
    if (!safeUnitQuat(x.segment<4>(QUAT_INDEX), q)) {
        return Vec3::Zero();
    }
    Vec7 x_base = x.head<7>();
    x_base.segment<4>(QUAT_INDEX) = q;

    Mat33 R_T = saltro::math::rotationMatrix(q).transpose();
    Vec3 V_body = R_T * V_eci;
    Vec3 R_body = R_T * R_eci;
    // S_eci is spacecraft-to-Sun; keep eclipse zeroing intact in body frame.
    Vec3 S_body = R_T * S_eci;
    (void)B_eci;
    (void)rho;

    if (dist.plan_for_aero) {
        saltro::disturbances::DragDisturbance drag(geometry_config_);
        torque += drag.torque(x_base, dist, V_body);
    }
    
    if (dist.plan_for_gg) {
        saltro::disturbances::GGDisturbance gg(Jcom_);
        torque += gg.torque(x_base, dist, R_body, Jcom_);
    }

    if (dist.plan_for_srp) {
        saltro::disturbances::SRPDisturbance srp(geometry_config_);
        torque += srp.torque(x_base, dist, S_body);
    }

    // Constant body-fixed propulsion torque (e.g. off-axis thruster). The
    // DisturbanceConfig struct already exposes plan_for_prop and prop_torque,
    // but the dispatch in disturbanceTorque() previously had no branch that
    // applied them — they were dead settings. ∂τ/∂x = 0 and ∂τ/∂u = 0 so no
    // dynamicsJacobians/dynamicsHessians changes are required.
    if (dist.plan_for_prop) {
        torque += dist.prop_torque;
    }

    return torque;
}

Satellite::VecX Satellite::dynamics(const VecX& x, const VecX& u, const DisturbanceConfig& dist, const Vec3& R_eci, const Vec3& B_eci, const Vec3& S_eci, const Vec3& V_eci, const int rho) const {
    Vec3 w = x.segment<3>(AV_INDEX);
    Vec4 q;
    if (!safeUnitQuat(x.segment<4>(QUAT_INDEX), q)) {
        return VecX::Constant(stateDim(), std::numeric_limits<double>::quiet_NaN());
    }
    
    Vec3 tau_act = actuatorTorque(x, u, B_eci);
    Vec3 tau_dist = disturbanceTorque(x, dist, R_eci, B_eci, S_eci, V_eci, rho);
    Vec3 h_rw = Vec3::Zero();
    if (num_rw_ > 0) {
        for (int i = 0; i < num_rw_; ++i) {
            double h_i = x(RW_MOMENTUM_INDEX + i);
            h_rw += h_i * getRW(i).axis();
        }
    }
    Vec3 wdot = invJcom_noRW_ * (tau_act + tau_dist - w.cross(Jcom_ * w + h_rw));
    Mat43 W = saltro::math::findWMat(q);
    Vec4 qdot = 0.5 * W * w;
    VecX xdot(stateDim());
    xdot.segment<3>(AV_INDEX) = wdot;
    xdot.segment<4>(QUAT_INDEX) = qdot;
    if (num_rw_ > 0) {
        for (int i = 0; i < num_rw_; ++i) {
            int ctrl_idx = num_mtq_ + i;
            double tau_rw_cmd = (ctrl_idx < controlDim()) ? u(ctrl_idx) : 0.0;
            
            const RW& rw = getRW(i);
            Vec3 axis = rw.axis();
            double J_rw = rw.wheelInertia();
            double hdot_i = -tau_rw_cmd - J_rw * axis.dot(wdot);
            xdot(RW_MOMENTUM_INDEX + i) = hdot_i;
        }
    }
    return xdot;
}

Satellite::VecX Satellite::dynamicsStepRK4(
    const VecX& x, const VecX& u, double dt,
    const DisturbanceConfig& dist,
    const Vec3& R_eci, const Vec3& B_eci,
    const Vec3& S_eci, const Vec3& V_eci,
    int rho
) const {
    VecX x_next;
    saltro::math::rk4_step<VecX>(
        [&](double /*t*/, const VecX& x_state, VecX& dxdt) {
            dxdt = dynamics(x_state, u, dist, R_eci, B_eci, S_eci, V_eci, rho);
        },
        x, 0.0, dt, x_next
    );

    // Renormalize quaternion to counteract integration drift.
    if (x_next.size() >= QUAT_INDEX + 4) {
        const Vec4 q = x_next.segment<4>(QUAT_INDEX);
        const double qn = q.norm();
        if (std::isfinite(qn) && qn > 1e-10) {
            x_next.segment<4>(QUAT_INDEX) = q / qn;
        }
    }

    return x_next;
}

std::tuple<Satellite::MatX, Satellite::MatX, Satellite::MatX> Satellite::dynamicsJacobians(const VecX& x, const VecX& u,
                                                   const DisturbanceConfig& dist,
                                                   const Vec3& R_eci, const Vec3& B_eci,
                                                   const Vec3& S_eci, const Vec3& V_eci) const {
    const int nx = stateDim();
    const int nu = controlDim();
    
    MatX jac_x(nx, nx);
    MatX jac_u(nx, nu);
    MatX jac_dist(nx, 3);  // jacobian w.r.t. disturbance effects
    
    jac_x.setZero();
    jac_u.setZero();
    jac_dist.setZero();
    
    // Extract state components
    Vec3 w = x.segment<3>(AV_INDEX);
    Vec4 q = x.segment<4>(QUAT_INDEX).normalized();
    Vec7 x_base = x.head<7>();
    x_base.segment<4>(QUAT_INDEX) = q;

    // Compute rotation matrix and body-frame vectors (flight-safe)
    Mat33 R = saltro::math::rotationMatrix(q);
    Mat33 R_T = R.transpose();
    Vec3 B_body = R_T * B_eci;
    Vec3 R_body = R_T * R_eci;
    Vec3 S_body = R_T * S_eci;
    Vec3 V_body = R_T * V_eci;
    
    // =========================================================================
    // Angular velocity Jacobian: ∂wdot/∂x
    // wdot = invJcom_noRW_ * (tau_act + tau_dist - w × (Jcom*w + h_rw))
    // =========================================================================
    
    // Compute RW momentum vector
    Vec3 h_rw = Vec3::Zero();
    if (num_rw_ > 0) {
        for (int i = 0; i < num_rw_; ++i) {
            double h_i = x(RW_MOMENTUM_INDEX + i);
            h_rw += h_i * getRW(i).axis();
        }
    }
    Vec3 angular_mom = Jcom_ * w + h_rw;
    
    // ∂wdot/∂w: from gyroscopic term -(w × (Jcom*w + h_rw))
    // d/dw of [w × v] where v = Jcom*w + h_rw
    // = skew(v) - skew(w)*Jcom (cleaner form)
    Mat33 gyro_term = saltro::math::skewSymmetric(angular_mom) - saltro::math::skewSymmetric(w) * Jcom_;
    
    jac_x.block<3, 3>(AV_INDEX, AV_INDEX) = invJcom_noRW_ * gyro_term;
    
    // =========================================================================
    // ∂wdot/∂q: Actuator and disturbance coupling through rotation
    // =========================================================================
    Mat34 dwdot_dq = Mat34::Zero();
    
    // MTQ contribution: τ_mtq = cross(Σ(axis_i * u_i), R^T*B)
    // ∂τ_mtq/∂q = skew(magvec) * ∂(R^T*B)/∂q
    if (num_mtq_ > 0 && B_eci.norm() > 1e-12) {
        // Compute dB_body/dq = d(R^T*B_eci)/dq (returns Mat43: 4x3)
        Mat43 dB_dq = saltro::math::drotmatTvecdq(q, B_eci);
        
        // Accumulate MTQ torque derivatives
        for (int i = 0; i < num_mtq_; ++i) {
            const MTQ& mtq = getMTQ(i);
            double u_i = u(i);
            
            // Get ∂τ_i/∂q from MTQ class (includes cross product with dB/dq)
            Mat73 dtau_dx = mtq.dtorq_dbasestate(u_i, x_base, B_body, dB_dq);
            // Extract quaternion block (rows 3-6, 4 rows) and transpose to match dwdot_dq (3x4)
            dwdot_dq += dtau_dx.block<4, 3>(3, 0).transpose();
        }
    }
    
    // Gravity gradient contribution (flight-safe: check if enabled and R_eci valid)
    if (dist.plan_for_gg && R_eci.norm() > 1e-6) {
        try {
            saltro::disturbances::GGDisturbance gg(Jcom_);
            Mat43 dR_dq = saltro::math::drotmatTvecdq(q, R_eci);
            Mat34 gg_jac = gg.dtorque_dq(x_base, dist, R_body, Jcom_, dR_dq.transpose());
            
            // Check for valid output (flight-safe)
            if (gg_jac.allFinite()) {
                dwdot_dq += gg_jac;
            }
        } catch (...) {
            // Silently ignore disturbance derivative errors (flight-safe)
        }
    }
    
    // Drag contribution (flight-safe: check if enabled and V_eci valid)
    if (dist.plan_for_aero && V_eci.norm() > 1e-6) {
        try {
            saltro::disturbances::DragDisturbance drag(geometry_config_);
            Mat43 dV_dq = saltro::math::drotmatTvecdq(q, V_eci);
            Mat34 drag_jac = drag.dtorque_dq(x_base, dist, V_body, dV_dq.transpose());
            
            // Check for valid output (flight-safe)
            if (drag_jac.allFinite()) {
                dwdot_dq += drag_jac;
            }
        } catch (...) {
            // Silently ignore disturbance derivative errors (flight-safe)
        }
    }
    
    // SRP contribution (flight-safe: check if enabled and S_eci valid)
    if (dist.plan_for_srp && S_eci.norm() > 1e-6) {
        try {
            saltro::disturbances::SRPDisturbance srp(geometry_config_);
            Mat43 dS_dq = saltro::math::drotmatTvecdq(q, S_eci);
            Mat34 srp_jac = srp.dtorque_dq(x_base, dist, S_body, dS_dq.transpose());
            
            // Check for valid output (flight-safe)
            if (srp_jac.allFinite()) {
                dwdot_dq += srp_jac;
            }
        } catch (...) {
            //Silently ignore disturbance derivative errors (flight-safe)
        }
    }
    
    // Apply accumulated quaternion derivatives to Jacobian
    jac_x.block<3, 4>(AV_INDEX, QUAT_INDEX) = invJcom_noRW_ * dwdot_dq;
    
    // ∂wdot/∂h: from gyroscopic term -(w × h_rw) where h_rw = sum(h_i * axis_i)
    if (num_rw_ > 0) {
        for (int i = 0; i < num_rw_; ++i) {
            Vec3 axis_i = getRW(i).axis();
            // ∂wdot/∂h_i = -invJcom_noRW * ∂(w × h_rw)/∂h_i
            //            = -invJcom_noRW * (w × axis_i)
            Vec3 gyro_coupling = w.cross(axis_i);
            jac_x.block<3, 1>(AV_INDEX, RW_MOMENTUM_INDEX + i) = -invJcom_noRW_ * gyro_coupling;
        }
    }
    
    // ∂wdot/∂u: from tau_act = sum of MTQ and RW torques
    // MTQ contribution: tau_mtq = -(B_body × axis) * u_mtq
    for (int i = 0; i < num_mtq_; ++i) {
        const MTQ& mtq = getMTQ(i);
        double u_i = u(i);
        // dtorque/du for MTQ
        Mat13 dtau_du = mtq.dtorq_du(u_i, x_base, B_body);
        jac_u.block<3, 1>(AV_INDEX, i) = invJcom_noRW_ * dtau_du.transpose();
    }
    
    // RW contribution: tau_rw = axis * u_rw
    for (int i = 0; i < num_rw_; ++i) {
        int ctrl_idx = num_mtq_ + i;
        const RW& rw = getRW(i);
        Vec3 axis_i = rw.axis();
        jac_u.block<3, 1>(AV_INDEX, ctrl_idx) = invJcom_noRW_ * axis_i;
    }

    // Magic actuator contribution: tau_magic = axis * u_magic
    // Same column shape as RW but no h-state coupling and no Newton-3 back-reaction.
    for (int i = 0; i < num_magic_; ++i) {
        int ctrl_idx = num_mtq_ + num_rw_ + i;
        const Magic& magic = getMagic(i);
        Vec3 axis_i = magic.axis();
        jac_u.block<3, 1>(AV_INDEX, ctrl_idx) = invJcom_noRW_ * axis_i;
    }
    
    // =========================================================================
    // Quaternion Jacobian: ∂qdot/∂x
    // qdot = 0.5 * W(q) * w
    // =========================================================================
    
    Mat43 W = saltro::math::findWMat(q);
    
    // ∂qdot/∂w = 0.5 * W(q)
    jac_x.block<4, 3>(QUAT_INDEX, AV_INDEX) = 0.5 * W;
    
    // ∂qdot/∂q: from ∂W/∂q * w
    // W matrix derivatives w.r.t. each quaternion component:
    // W = 0.5 * [[-q1, -q2, -q3],      ∂W/∂q_0 = 0.5 * [[0,  0,  0],
    //            [ q0, -q3,  q2],                       [1,  0,  0],
    //            [ q3,  q0, -q1],                       [0,  1,  0],
    //            [-q2,  q1,  q0]]                       [0,  0,  1]]
    // (and similar for other components)
    
    for (int j = 0; j < 4; ++j) {
        Vec4 dW_col = Vec4::Zero();
        
        // Build ∂W/∂q_j matrix and multiply by w to get contribution
        if (j == 0) {
            // ∂W/∂q_0: affects rows 1,2,3 (q_0 coefficients)
            dW_col(0) = 0.0;           // row 0: -q1, -q2, -q3 (no q_0 dependence)
            dW_col(1) = w(0);          // row 1: q_0 coefficient in column 0
            dW_col(2) = w(1);          // row 2: q_0 coefficient in column 1
            dW_col(3) = w(2);          // row 3: q_0 coefficient in column 2
        } else if (j == 1) {
            // ∂W/∂q_1: -q_1 in row 0, affects row 0 in all columns
            dW_col(0) = -w(0);         // row 0: -q_1 in column 0
            dW_col(1) = 0.0;           // row 1: no q_1 in first position
            dW_col(2) = -w(2);         // row 2: -q_1 in column 2
            dW_col(3) = w(1);          // row 3: q_1 in column 1
        } else if (j == 2) {
            // ∂W/∂q_2: -q_2 in row 0, affects row 0
            dW_col(0) = -w(1);         // row 0: -q_2 in column 1
            dW_col(1) = w(2);          // row 1: q_2 in column 2
            dW_col(2) = 0.0;           // row 2: no q_2 in first position
            dW_col(3) = -w(0);         // row 3: -q_2 in column 0 (row3 = -q2*w0+q1*w1+q0*w2)
        } else if (j == 3) {
            // ∂W/∂q_3: -q_3 in row 0, affects row 0
            dW_col(0) = -w(2);         // row 0: -q_3 in column 2
            dW_col(1) = -w(1);         // row 1: -q_3 in column 1
            dW_col(2) = w(0);          // row 2: q_3 in column 0
            dW_col(3) = 0.0;           // row 3: no q_3 in first position
        }
        
        jac_x.block<4, 1>(QUAT_INDEX, QUAT_INDEX + j) = 0.5 * dW_col;
    }
    
    // =========================================================================
    // Reaction Wheel Momentum Jacobian: ∂hdot/∂x
    // hdot_i = -tau_rw_cmd - J_rw * axis_i · wdot
    // =========================================================================
    if (num_rw_ > 0) {
        for (int i = 0; i < num_rw_; ++i) {
            const RW& rw = getRW(i);
            Vec3 axis_i = rw.axis();
            double J_rw = rw.wheelInertia();
            int state_idx = RW_MOMENTUM_INDEX + i;
            
            // ∂hdot_i/∂w = -J_rw * axis_i · (∂wdot/∂w)
            jac_x.block<1, 3>(state_idx, AV_INDEX) = -J_rw * axis_i.transpose() * jac_x.block<3, 3>(AV_INDEX, AV_INDEX);
            
            // ∂hdot_i/∂q = -J_rw * axis_i · (∂wdot/∂q) - coupling through quaternion
            jac_x.block<1, 4>(state_idx, QUAT_INDEX) = -J_rw * axis_i.transpose() * jac_x.block<3, 4>(AV_INDEX, QUAT_INDEX);
            
            // ∂hdot_i/∂h_k = -J_rw * axis_i · (∂wdot/∂h_k)
            for (int k = 0; k < num_rw_; ++k) {
                jac_x(state_idx, RW_MOMENTUM_INDEX + k) = 
                    -J_rw * axis_i.dot(jac_x.block<3, 1>(AV_INDEX, RW_MOMENTUM_INDEX + k));
            }
        }
    }
    
    // =========================================================================
    // Control Jacobian: ∂xdot/∂u
    // =========================================================================
    
    // ∂wdot/∂u: already computed above
    
    // ∂qdot/∂u = 0 (quaternion rate doesn't depend on control input directly)
    
    // ∂hdot_i/∂u_j
    if (num_rw_ > 0) {
        for (int i = 0; i < num_rw_; ++i) {
            const RW& rw = getRW(i);
            Vec3 axis_i = rw.axis();
            double J_rw = rw.wheelInertia();
            int state_idx = RW_MOMENTUM_INDEX + i;
            int rw_ctrl_idx = num_mtq_ + i;
            
            for (int j = 0; j < controlDim(); ++j) {
                if (j == rw_ctrl_idx) {
                    // Direct control term: ∂(-tau_rw_cmd)/∂u_rw = -1
                    // Plus coupling term: -J_rw * axis_i · (∂wdot/∂u_rw)
                    jac_u(state_idx, j) = -1.0 - J_rw * axis_i.dot(jac_u.block<3, 1>(AV_INDEX, j));
                } else {
                    // Indirect coupling through wdot: -J_rw * axis_i · (∂wdot/∂u_j)
                    jac_u(state_idx, j) = -J_rw * axis_i.dot(jac_u.block<3, 1>(AV_INDEX, j));
                }
            }
        }
    }
    
    // =========================================================================
    // Normalization projection for quaternion input columns
    // =========================================================================
    // dynamics() normalizes q internally (q → q/|q|), so the effective Jacobian
    // w.r.t. a raw quaternion perturbation δq_k is:
    //   ∂f/∂q_k_raw = Σ_l (∂f/∂q_l_norm) * (I - q*q^T)_{lk}
    // i.e., right-multiply the quaternion input columns by (I - q*q^T).
    {
        using Mat44 = Eigen::Matrix<double, 4, 4>;
        Mat44 proj_q = Mat44::Identity() - q * q.transpose();
        jac_x.block(0, QUAT_INDEX, nx, 4) =
            jac_x.block(0, QUAT_INDEX, nx, 4) * proj_q;
    }

    return std::make_tuple(jac_x, jac_u, jac_dist);
}

std::tuple<Satellite::DynHessXX, Satellite::DynHessUX, Satellite::DynHessUU> Satellite::dynamicsHessians(const VecX& x, const VecX& u, 
                                                                 const DisturbanceConfig& dist,
                                                                 const Vec3& R_eci, const Vec3& B_eci,
                                                                 const Vec3& S_eci, const Vec3& V_eci) const {
    const int nx = stateDim();

    DynHessXX hess_xx;  // (state, state, state) - indexed by output equation
    DynHessUX hess_ux;  // (control, state, state) - indexed by output equation
    DynHessUU hess_uu;  // (control, control, state) - indexed by output equation
    
    hess_xx.setZero();
    hess_ux.setZero();
    hess_uu.setZero();

    // Extract state components (ω intentionally not extracted: the corrected
    // ∂²ω̇/∂ω² block below is ω-independent — it's the Hessian of a quadratic).
    Vec4 q = x.segment<4>(QUAT_INDEX).normalized();
    Vec7 x_base = x.head<7>();
    x_base.segment<4>(QUAT_INDEX) = q;

    // Compute rotation matrix and body-frame vectors (flight-safe)
    Mat33 R_T = saltro::math::rotationMatrix(q).transpose();
    Vec3 B_body = R_T * B_eci;
    Vec3 R_body = R_T * R_eci;
    Vec3 S_body = R_T * S_eci;
    Vec3 V_body = R_T * V_eci;
    
    // Compute RW momentum vector
    Vec3 h_rw = Vec3::Zero();
    if (num_rw_ > 0) {
        for (int i = 0; i < num_rw_; ++i) {
            double h_i = x(RW_MOMENTUM_INDEX + i);
            h_rw += h_i * getRW(i).axis();
        }
    }

    // =========================================================================
    // Angular velocity Hessian: ∂²wdot_i/∂x_j∂x_k (indexed by output i = 0,1,2)
    // =========================================================================
    // wdot = invJcom_noRW_ * (tau - w × (Jcom*w + h_rw))
    //
    // ∂²ω̇_i/∂ω_j∂ω_k: derived from g(ω) = ω × (J·ω). Since g is quadratic in ω,
    // its Hessian is a CONSTANT tensor (independent of ω). In index form:
    //   ∂²g_i/∂ω_j∂ω_k = ε_{ijn} J_{nk} + ε_{ikn} J_{nj}
    // Equivalently, with M(v) := J^{-1}·skew(v)·J,
    //   ∂²ω̇_i/∂ω_j∂ω_k = -M(e_j)_{ik} - M(e_k)_{ij}
    // Sanity: for J = c·I, M(e) = skew(e), giving ε_{ijk}+ε_{ikj}=0 — i.e.,
    // ω × (cω) = 0 has zero Hessian. ✓
    // (Prior implementation multiplied a constant "d2cross" by ω, which gave
    //  w-linear output instead of a constant and was also missing terms.)
    {
        Mat33 M[3];  // M[v_idx] = invJcom_noRW · skew(e_{v_idx}) · Jcom
        for (int v = 0; v < 3; ++v) {
            Vec3 ev = Vec3::Zero(); ev(v) = 1.0;
            M[v] = invJcom_noRW_ * saltro::math::skewSymmetric(ev) * Jcom_;
        }
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                    const double hess_val = -M[j](i, k) - M[k](i, j);
                    hess_xx.slice(AV_INDEX + i)(AV_INDEX + j, AV_INDEX + k) = hess_val;
                }
            }
        }
    }

    // (Continue with mixed and pure-w blocks that depend on w/h/u.)
    for (int i = 0; i < 3; ++i) {
        Vec3 ei = Vec3::Zero();
        ei(i) = 1.0;
        
        // ∂²wdot_i/∂w_j∂h_k: from d/dw_j of [invJcom_noRW * (-w × h_rw)]
        // = -d/dw_j of [w × (axis_k * h_k)] = -skew(ej) * axis_k
        if (num_rw_ > 0) {
            for (int j = 0; j < 3; ++j) {
                Vec3 ej = Vec3::Zero();
                ej(j) = 1.0;
                
                for (int k = 0; k < num_rw_; ++k) {
                    Vec3 axis_k = getRW(k).axis();
                    Vec3 result = -saltro::math::skewSymmetric(ej) * axis_k;
                    double hess_val = ei.dot(invJcom_noRW_ * result);
                    hess_xx.slice(AV_INDEX + i)(AV_INDEX + j, RW_MOMENTUM_INDEX + k) = hess_val;
                    hess_xx.slice(AV_INDEX + i)(RW_MOMENTUM_INDEX + k, AV_INDEX + j) = hess_val;  // symmetry
                }
            }
        }
        
        // ∂²wdot_i/∂u_j∂w_k: from actuator second derivatives through tau_act
        // For RW: rw_torque = axis * u is linear in u, so second deriv in u vanishes
        // For MTQ: computed via MTQ class methods (flight-safe)
        if (num_mtq_ > 0 && B_eci.norm() > 1e-12) {
            try {
                Mat43 dB_dq = saltro::math::drotmatTvecdq(q, B_eci);
                
                for (int j = 0; j < num_mtq_; ++j) {
                    const MTQ& mtq = getMTQ(j);
                    double u_j = u(j);
                    
                    // Get mixed Hessian ∂²τ/∂u∂q from MTQ class
                    // H_mtq_ux is T173 = Tensor3<1,7,3>: each slice is 1x7.
                    // Row index is always 0 (one control input per per-actuator call).
                    auto H_mtq_ux = mtq.ddtorq_dudbasestate(u_j, x_base, B_body, dB_dq);
                    
                    // Accumulate ∂²wdot_i/∂u_j∂q_k for all 4 quaternion components
                    for (int qk = 0; qk < 4; ++qk) {
                        double contrib = 0.0;
                        for (int m = 0; m < 3; ++m) {
                            double val = H_mtq_ux.slice(m)(0, QUAT_INDEX + qk);
                            if (std::isfinite(val)) {
                                contrib += invJcom_noRW_(i, m) * val;
                            }
                        }
                        hess_ux.slice(AV_INDEX + i)(j, QUAT_INDEX + qk) += contrib;
                    }
                }
            } catch (...) {
                // Silently ignore MTQ Hessian errors (flight-safe)
            }
        }
        
        // ∂²wdot_i/∂u_j∂u_k: from actuator control Hessians
        // For RW and MTQ actuators, control is linear, so this is zero
    }
    
    // =========================================================================
    // ∂²wdot/∂q²: Disturbance Hessian contributions (flight-safe)
    // =========================================================================
    // Add disturbance second derivatives to complete angular velocity Hessian
    
    // Gravity gradient Hessian
    if (dist.plan_for_gg && R_eci.norm() > 1e-6) {
        try {
            saltro::disturbances::GGDisturbance gg(Jcom_);
            Mat34 dR_dq = saltro::math::drotmatTvecdq(q, R_eci).transpose();
            auto d2R_dq2 = saltro::math::ddrotmatTvecdqdq(q, R_eci);
            
            auto gg_hess = gg.ddtorque_dqdq(x_base, dist, R_body, Jcom_, dR_dq, d2R_dq2);
            
            // Accumulate contributions for each output component
            for (int i = 0; i < 3; ++i) {
                if (gg_hess.slice(i).allFinite()) {
                    // Apply invJcom_noRW and add to existing Hessian
                    for (int j = 0; j < 4; ++j) {
                        for (int k = 0; k < 4; ++k) {
                            double contrib = 0.0;
                            for (int m = 0; m < 3; ++m) {
                                contrib += invJcom_noRW_(i, m) * gg_hess.slice(m)(j, k);
                            }
                            hess_xx.slice(AV_INDEX + i)(QUAT_INDEX + j, QUAT_INDEX + k) += contrib;
                        }
                    }
                }
            }
        } catch (...) {
            // Silently ignore GG Hessian errors (flight-safe)
        }
    }
    
    // Drag Hessian
    if (dist.plan_for_aero && V_eci.norm() > 1e-6) {
        try {
            saltro::disturbances::DragDisturbance drag(geometry_config_);
            Mat34 dV_dq = saltro::math::drotmatTvecdq(q, V_eci).transpose();
            auto d2V_dq2 = saltro::math::ddrotmatTvecdqdq(q, V_eci);
            
            auto drag_hess = drag.ddtorque_dqdq(x_base, dist, V_body, dV_dq, d2V_dq2);
            
            for (int i = 0; i < 3; ++i) {
                if (drag_hess.slice(i).allFinite()) {
                    for (int j = 0; j < 4; ++j) {
                        for (int k = 0; k < 4; ++k) {
                            double contrib = 0.0;
                            for (int m = 0; m < 3; ++m) {
                                contrib += invJcom_noRW_(i, m) * drag_hess.slice(m)(j, k);
                            }
                            hess_xx.slice(AV_INDEX + i)(QUAT_INDEX + j, QUAT_INDEX + k) += contrib;
                        }
                    }
                }
            }
        } catch (...) {
            // Silently ignore drag Hessian errors (flight-safe)
        }
    }
    
    // SRP Hessian
    if (dist.plan_for_srp && S_eci.norm() > 1e-6) {
        try {
            saltro::disturbances::SRPDisturbance srp(geometry_config_);
            Mat34 dS_dq = saltro::math::drotmatTvecdq(q, S_eci).transpose();
            auto d2S_dq2 = saltro::math::ddrotmatTvecdqdq(q, S_eci);
            
            auto srp_hess = srp.ddtorque_dqdq(x_base, dist, S_body, dS_dq, d2S_dq2);
            
            for (int i = 0; i < 3; ++i) {
                if (srp_hess.slice(i).allFinite()) {
                    for (int j = 0; j < 4; ++j) {
                        for (int k = 0; k < 4; ++k) {
                            double contrib = 0.0;
                            for (int m = 0; m < 3; ++m) {
                                contrib += invJcom_noRW_(i, m) * srp_hess.slice(m)(j, k);
                            }
                            hess_xx.slice(AV_INDEX + i)(QUAT_INDEX + j, QUAT_INDEX + k) += contrib;
                        }
                    }
                }
            }
        } catch (...) {
            // Silently ignore SRP Hessian errors (flight-safe)
        }
    }
    
    // =========================================================================
    // Quaternion Hessian: ∂²qdot_i/∂x_j∂x_k (indexed by output i = 0,1,2,3)
    // =========================================================================
    // qdot = 0.5 * W(q) * w where W depends on q
    // ∂qdot_i/∂w_j = 0.5 * W_ij
    // ∂qdot_i/∂q_j involves ∂W/∂q_j
    
    // ∂²qdot/∂w∂w = 0 (qdot is linear in w)
    // ∂²qdot/∂q∂q = 0 (W is linear in q, so second q-derivatives vanish)
    //
    // ∂²qdot_r/(∂q_j ∂w_k) = 0.5 * ∂W[r,k]/∂q_j  (NON-ZERO due to bilinear structure)
    //
    // W matrix:
    //   W = [[-q1, -q2, -q3],
    //        [ q0, -q3,  q2],
    //        [ q3,  q0, -q1],
    //        [-q2,  q1,  q0]]
    //
    // Each entry W[r,k] = sign * q[j], encoded as W_qidx[r][k]=j, W_sign[r][k]=sign:
    static const int    W_qidx[4][3] = {{1,2,3},{0,3,2},{3,0,1},{2,1,0}};
    static const double W_sign[4][3] = {{-1,-1,-1},{+1,-1,+1},{+1,+1,-1},{-1,+1,+1}};

    for (int r = 0; r < 4; ++r) {
        for (int k = 0; k < 3; ++k) {
            int j = W_qidx[r][k];
            double val = 0.5 * W_sign[r][k];
            hess_xx.slice(QUAT_INDEX + r)(QUAT_INDEX + j, AV_INDEX + k) = val;
            hess_xx.slice(QUAT_INDEX + r)(AV_INDEX + k, QUAT_INDEX + j) = val;
        }
    }
    
    // =========================================================================
    // Reaction Wheel Momentum Hessian: ∂²hdot_i/∂x_j∂x_k
    // =========================================================================
    // hdot_i = -tau_rw_cmd - J_rw * axis_i · wdot
    // ∂hdot_i/∂w_j = -J_rw * axis_i · (∂wdot/∂w_j)
    // ∂²hdot_i/∂w_j∂w_k = -J_rw * axis_i · (∂²wdot/∂w_j∂w_k)
    
    if (num_rw_ > 0) {
        for (int i = 0; i < num_rw_; ++i) {
            const RW& rw = getRW(i);
            Vec3 axis_i = rw.axis();
            double J_rw = rw.wheelInertia();
            int state_idx = RW_MOMENTUM_INDEX + i;
            
            // ∂²hdot_i/∂w_j∂w_k through second derivatives of wdot
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                    // wdot depends on w,q,h,u; we compute derivatives
                    // Using chain rule: ∂²hdot/∂w_j∂w_k = -J_rw * axis_i · (∂²wdot/∂w_j∂w_k)
                    double sum_val = 0.0;
                    for (int m = 0; m < 3; ++m) {
                        sum_val += axis_i(m) * hess_xx.slice(AV_INDEX + m)(AV_INDEX + j, AV_INDEX + k);
                    }
                    hess_xx.slice(state_idx)(AV_INDEX + j, AV_INDEX + k) = -J_rw * sum_val;
                }
            }
            
            // ∂²hdot_i/∂w_j∂h_k through second derivatives of wdot
            if (num_rw_ > 0) {
                for (int j = 0; j < 3; ++j) {
                    for (int k = 0; k < num_rw_; ++k) {
                        double sum_val = 0.0;
                        for (int m = 0; m < 3; ++m) {
                            sum_val += axis_i(m) * hess_xx.slice(AV_INDEX + m)(AV_INDEX + j, RW_MOMENTUM_INDEX + k);
                        }
                        hess_xx.slice(state_idx)(AV_INDEX + j, RW_MOMENTUM_INDEX + k) = -J_rw * sum_val;
                        hess_xx.slice(state_idx)(RW_MOMENTUM_INDEX + k, AV_INDEX + j) = -J_rw * sum_val;  // symmetry
                    }
                }
            }
            
            // ∂²hdot_i/∂w_j∂q_k through second derivatives of wdot (quaternion coupling)
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 4; ++k) {
                    double sum_val = 0.0;
                    for (int m = 0; m < 3; ++m) {
                        sum_val += axis_i(m) * hess_xx.slice(AV_INDEX + m)(AV_INDEX + j, QUAT_INDEX + k);
                    }
                    hess_xx.slice(state_idx)(AV_INDEX + j, QUAT_INDEX + k) = -J_rw * sum_val;
                    hess_xx.slice(state_idx)(QUAT_INDEX + k, AV_INDEX + j) = -J_rw * sum_val;  // symmetry
                }
            }
            
            // ∂²hdot_i/∂q_j∂q_k through second derivatives of wdot (quaternion-quaternion coupling)
            for (int j = 0; j < 4; ++j) {
                for (int k = 0; k < 4; ++k) {
                    double sum_val = 0.0;
                    for (int m = 0; m < 3; ++m) {
                        sum_val += axis_i(m) * hess_xx.slice(AV_INDEX + m)(QUAT_INDEX + j, QUAT_INDEX + k);
                    }
                    hess_xx.slice(state_idx)(QUAT_INDEX + j, QUAT_INDEX + k) = -J_rw * sum_val;
                }
            }
            
            // ∂²hdot_i/∂q_j∂h_k through second derivatives of wdot (quaternion-momentum coupling)
            if (num_rw_ > 0) {
                for (int j = 0; j < 4; ++j) {
                    for (int k = 0; k < num_rw_; ++k) {
                        double sum_val = 0.0;
                        for (int m = 0; m < 3; ++m) {
                            sum_val += axis_i(m) * hess_xx.slice(AV_INDEX + m)(QUAT_INDEX + j, RW_MOMENTUM_INDEX + k);
                        }
                        hess_xx.slice(state_idx)(QUAT_INDEX + j, RW_MOMENTUM_INDEX + k) = -J_rw * sum_val;
                        hess_xx.slice(state_idx)(RW_MOMENTUM_INDEX + k, QUAT_INDEX + j) = -J_rw * sum_val;  // symmetry
                    }
                }
            }
            
            // ∂²hdot_i/∂u_j∂w_k through wdot derivatives
            for (int j = 0; j < controlDim(); ++j) {
                for (int k = 0; k < 3; ++k) {
                    // ∂²hdot_i/∂u_j∂w_k = -J_rw * axis_i · (∂²wdot/∂u_j∂w_k)
                    // This is small for flight-safety purposes
                    hess_ux.slice(state_idx)(j, AV_INDEX + k) = 0.0;
                }
            }
            
            // ∂²hdot_i/∂u_j∂q_k through wdot derivatives (quaternion coupling)
            for (int j = 0; j < controlDim(); ++j) {
                for (int k = 0; k < 4; ++k) {
                    // ∂²hdot_i/∂u_j∂q_k = -J_rw * axis_i · (∂²wdot/∂u_j∂q_k)
                    double sum_val = 0.0;
                    for (int m = 0; m < 3; ++m) {
                        sum_val += axis_i(m) * hess_ux.slice(AV_INDEX + m)(j, QUAT_INDEX + k);
                    }
                    hess_ux.slice(state_idx)(j, QUAT_INDEX + k) = -J_rw * sum_val;
                }
            }
            
            // ∂²hdot_i/∂u_j∂h_k through wdot derivatives (momentum coupling)
            if (num_rw_ > 0) {
                for (int j = 0; j < controlDim(); ++j) {
                    for (int k = 0; k < num_rw_; ++k) {
                        double sum_val = 0.0;
                        for (int m = 0; m < 3; ++m) {
                            sum_val += axis_i(m) * hess_ux.slice(AV_INDEX + m)(j, RW_MOMENTUM_INDEX + k);
                        }
                        hess_ux.slice(state_idx)(j, RW_MOMENTUM_INDEX + k) = -J_rw * sum_val;
                    }
                }
            }
            
            // ∂²hdot_i/∂u_j∂u_k: direct term and through wdot
            int rw_ctrl_idx = num_mtq_ + i;
            for (int j = 0; j < controlDim(); ++j) {
                for (int k = 0; k < controlDim(); ++k) {
                    if (j == rw_ctrl_idx && k == rw_ctrl_idx) {
                        // ∂²/∂u_rw² of (-u_rw) = 0 (linear)
                        // Plus contribution from -J_rw * axis_i · (∂²wdot/∂u_rw∂u_rw)
                        hess_uu.slice(state_idx)(j, k) = 0.0;
                    } else if (j == rw_ctrl_idx || k == rw_ctrl_idx) {
                        // Mixed terms are small
                        hess_uu.slice(state_idx)(j, k) = 0.0;
                    } else {
                        // Through wdot cross-couplings
                        hess_uu.slice(state_idx)(j, k) = 0.0;
                    }
                }
            }
        }
    }
    
    // =========================================================================
    // Normalization projection for quaternion input directions
    // =========================================================================
    // Mirrors the projection applied in dynamicsJacobians().
    // For each output-component slice, apply (I - q*q^T) to:
    //   - quaternion columns of hess_xx (right multiply)
    //   - quaternion rows   of hess_xx (left  multiply)
    //   - quaternion columns of hess_ux  (right multiply)
    {
        constexpr int NX = saltro::limits::MAX_STATE_DIM;
        constexpr int NU = saltro::limits::MAX_CTRL_DIM;
        using Mat44 = Eigen::Matrix<double, 4, 4>;
        const Mat44 proj_q = Mat44::Identity() - q * q.transpose();
        for (int si = 0; si < nx; ++si) {
            auto& Hxx = hess_xx.slice(si);
            // Right-project quaternion columns
            Hxx.template block<NX, 4>(0, QUAT_INDEX) =
                Hxx.template block<NX, 4>(0, QUAT_INDEX) * proj_q;
            // Left-project quaternion rows
            Hxx.template block<4, NX>(QUAT_INDEX, 0) =
                proj_q * Hxx.template block<4, NX>(QUAT_INDEX, 0);
            // Right-project quaternion columns of hess_ux
            hess_ux.slice(si).template block<NU, 4>(0, QUAT_INDEX) =
                hess_ux.slice(si).template block<NU, 4>(0, QUAT_INDEX) * proj_q;
        }
    }

    return std::make_tuple(hess_xx, hess_ux, hess_uu);
}

namespace {

// ===========================================================================
// Vector-pointing attitude cost — PhD-planner-style reduced-space formulation.
//
// The attitude cost is h(q) = f(c), with c = bs·R(q)ᵀ·r̂ ∈ [-1, 1] the cosine
// of the boresight-to-target angle.  Following the Generalized_ADCS PhD
// planner (cost2angQ / ddvTRTudqQ in GeneralUtil.cpp), the cost shape f and
// the geometry of c are factored into two small helpers, and every attitude-
// cost derivative is taken in the 3-D attitude tangent space.
// ===========================================================================

// Cost shape f(c) and its first two derivatives w.r.t. c.
//   0: 1−c                       linear; f'=−1; f''=0 (no curvature for GN)
//   1: ½(1−c)²                   convex quadratic; f'=−(1−c); f''=1
//   2: acos(c)                   ABSOLUTE angle; f''<0 for c>0 (NOT PSD)
//   3: ½·acos(c)²                squared angle; f'' diverges at c=±1
//   4: (1−c)²                    PSD f''=2 but f'(c=1)=0 — flat near alignment
//   5: (1−c) + (1−c)² = (1−c)(2−c)
//        linear-plus-quadratic blend with the failure mode of each piece
//        removed:
//          • Like 0, has constant non-vanishing pull at alignment:
//            f'(c=1) = −1.
//          • Like 4, has bounded constant Hessian f''=2 (PSD by construction,
//            no singularity at c=±1).
//        The linear term keeps the gradient alive when the quadratic flattens
//        out near c=1, so iLQR retains drive toward exact alignment even
//        with a strong warm start that is already close. The quadratic term
//        provides Gauss-Newton curvature info for fine convergence.
// fpp ≥ 0 for types 0,1,3,4,5; type 2 (acos) is the exception (fpp < 0 for c > 0).
struct AngCostShape { double f, fp, fpp; };

AngCostShape angCostShape(double c, int type) {
    const double omc2 = std::max(1.0 - c * c, 1e-12);  // 1 − c²  (floored)
    const double s = std::sqrt(omc2);                  // √(1 − c²)
    switch (type) {
        case 0: return { 1.0 - c, -1.0, 0.0 };
        case 1: { const double e = 1.0 - c; return { 0.5 * e * e, -e, 1.0 }; }
        case 2: return { std::acos(c), -1.0 / s, -c / (omc2 * s) };
        case 3: {
            const double phi = std::acos(c);
            return { 0.5 * phi * phi, -phi / s,
                     1.0 / omc2 - phi * c / (omc2 * s) };
        }
        case 4: { const double e = 1.0 - c; return { e * e, -2.0 * e, 2.0 }; }
        case 5: {
            // f  = (1-c) + (1-c)^2
            // f' = -1 - 2(1-c) = 2c - 3
            // f''= 2
            const double e = 1.0 - c;
            return { e + e * e, 2.0 * c - 3.0, 2.0 };
        }
        default: return { std::acos(c), -1.0 / s, -c / (omc2 * s) };
    }
}

// Reduced-space geometry of c = bs·R(q)ᵀ·r̂.  The gradient and Hessian are
// expressed in the attitude tangent space (basis W = findWMat(q)); `ddc`
// carries the "Planning with Attitude" manifold-curvature correction
// −(∂c/∂q·q)·I₃, exactly as the PhD planner's ddvTRTudqQ.  This is the
// vector-pointing analogue of cost2angQ.
struct VecPointingGeom {
    double c;             // clamped cos(pointing error)
    Eigen::Vector3d dc;   // ∂c/∂θ   (reduced gradient)
    Eigen::Matrix3d ddc;  // ∂²c/∂θ² (reduced Hessian, manifold-corrected)
};

VecPointingGeom vecPointingGeom(const Eigen::Vector4d& q,
                                const Eigen::Vector3d& bs_unit,
                                const Eigen::Vector3d& r_eci) {
    const auto W = saltro::math::findWMat(q);  // 4×3 attitude tangent basis

    // Ambient (ℝ⁴) gradient and Hessian of c.
    const Eigen::Vector4d dc_amb =
        saltro::math::drotmatTvecdq(q, r_eci) * bs_unit;
    const auto H_RTv = saltro::math::ddrotmatTvecdqdq(q, r_eci);
    Eigen::Matrix4d d2c_amb = Eigen::Matrix4d::Zero();
    for (int b = 0; b < 3; ++b) d2c_amb += bs_unit(b) * H_RTv[b];

    VecPointingGeom g;
    g.c  = std::clamp(bs_unit.dot(saltro::math::rotationMatrix(q).transpose()
                                  * r_eci), -1.0, 1.0);
    g.dc = W.transpose() * dc_amb;
    // Project ambient → tangent, then subtract the manifold correction
    // −(∂c/∂q·q)·I₃.  (∂c/∂q·q = 2c since c is degree-2 homogeneous in q.)
    g.ddc = W.transpose() * d2c_amb * W
          - dc_amb.dot(q) * Eigen::Matrix3d::Identity();
    return g;
}

}  // namespace

double Satellite::stageCost(int k, int N, const VecX& x, const VecX& u,
                            const Vec3& boresight_body, const Vec4& attitude_target,
                            const Vec3& B_eci, const CostConfig& cost_cfg) const {
    if (N <= 0) {
        throw invalid_argument("N must be positive in stageCost().");
    }
    if (k < 0 || k >= N) {
        throw out_of_range("Time index k out of range in stageCost().");
    }
    if (x.size() < stateDim()) {
        throw invalid_argument("State vector has insufficient dimension in stageCost().");
    }
    if (u.size() < controlDim()) {
        throw invalid_argument("Control vector has insufficient dimension in stageCost().");
    }

    const bool terminal = (k >= N - 1);
    const double w_ang = terminal ? cost_cfg.angle_N : cost_cfg.angle;
    const double w_av = terminal ? cost_cfg.ang_vel_N : cost_cfg.ang_vel;
    const double w_avmag = terminal ? cost_cfg.ang_vel_mag_N : cost_cfg.ang_vel_mag;
    const double w_avang = terminal ? cost_cfg.ang_vel_err_dir_N : cost_cfg.ang_vel_err_dir;
    const double w_u_mult = terminal ? 0.0 : cost_cfg.control_mult;

    const Vec3 w = x.segment<3>(AV_INDEX);
    const Vec4 q = x.segment<4>(QUAT_INDEX).normalized();
    
    // Process attitude_target to handle both ECI vector and quaternion formats
    auto [q_goal, is_eci_format] = processAttitudeTarget(attitude_target, boresight_body);
    
    // Disable angle cost if ECI target is invalid
    double w_ang_eff = w_ang;
    if (is_eci_format && attitude_target.tail(3).norm() < 1e-9) {
        w_ang_eff = 0.0;
    }

    const double qdot = q_goal.dot(q);

    // ===== CRITICAL FIX: Handle quaternion double-cover =====
    // Ensure q_goal and q are on the same hemisphere to avoid sign ambiguity
    Vec4 q_goal_aligned = q_goal;
    if (qdot < 0.0) {
        q_goal_aligned = -q_goal;
    }
    const double qdot_aligned = safeAbs(q_goal_aligned.dot(q));

    double ang_cost = 0.0;
    if (is_eci_format) {
        // Vector-pointing: cost h = f(c), c = bs·R(q)ᵀ·r̂_eci ∈ [-1, 1].
        // 2-DOF (roll about bs is free); no double-cover ambiguity.  The cost
        // shape lives in the angCostShape helper (shared with the Jacobian /
        // Hessian), mirroring the PhD planner's veccost path.
        const Vec3 bs_unit = boresight_body.normalized();
        const Vec3 r_eci = attitude_target.tail(3).normalized();
        const double c_val = std::clamp(
            bs_unit.dot(saltro::math::rotationMatrix(q).transpose() * r_eci),
            -1.0, 1.0);
        ang_cost = angCostShape(c_val, cost_cfg.ang_cost_func_type).f;
    } else {
        switch (cost_cfg.ang_cost_func_type) {
            case 0:
                ang_cost = 1.0 - qdot_aligned;
                break;
            case 1: {
                const double err = 1.0 - qdot_aligned;
                ang_cost = 0.5 * err * err;
                break;
            }
            case 2:
                ang_cost = std::acos(qdot_aligned);
                break;
            case 3: {
                const double phi = std::acos(qdot_aligned);
                ang_cost = 0.5 * phi * phi;
                break;
            }
            case 4: {
                // (1-d)² with d = qdot_aligned ∈ [0,1] (post-hemisphere-flip).
                // Matches vec mode case 4 shape. Convex Hessian (f''=2),
                // f'(d) = -2(1-d). Replaces former `1-d²` which had concave
                // Hessian (f''=-2) and zero gradient at d=0 (90° error) —
                // unusable in practice.
                const double err = 1.0 - qdot_aligned;
                ang_cost = err * err;
                break;
            }
            case 5: {
                // (1-d) + (1-d)² = (1-d)(2-d). Matches vec mode case 5 shape.
                // PSD f''=2 like case 4 but with non-vanishing gradient at
                // d=1 (alignment): f'(d) = -1 - 2(1-d) = 2d - 3, f'(1) = -1.
                const double err = 1.0 - qdot_aligned;
                ang_cost = err + err * err;
                break;
            }
            default:
                ang_cost = std::acos(qdot_aligned);
                break;
        }
    }

    const Mat43 W = saltro::math::findWMat(q);

    // ω-related terms.  Two paths:
    //  - back-compat: when `ang_vel_err_dir(_N)` is nonzero, use the legacy
    //    `−sign(qdot)·(q_g^T·W·ω)·w_avang` cross-cost and uniform |ω|² cost.
    //    No feedforward (ω_ref ignored).
    //  - new path: ω_ff = R(q_e)·ω_ref with q_e = q^*·q_g_aligned. Quadratic
    //    becomes ½·w_av·|ω−ω_ff|². If `ang_vel_err_dir_ratio` > 0, add a
    //    PSD-bounded Lyapunov crossterm α·err_dir^T·(ω−ω_ff) where
    //    err_dir = −sign(qdot)·(W^T·q_g_aligned), α = β·√(angle·ang_vel).
    //    With ω_ref=0 and ratio=0 (defaults), this reduces to ½·w_av·|ω|².
    double quad_omega_cost = 0.0;
    double cross_cost = 0.0;
    // Legacy `w_avang` cross-cost is well-defined only in quaternion mode (uses
    // q_goal_aligned, which in vec mode is a synthetic min-rotation quaternion
    // — semantically incorrect). In vec mode, route w_avang through the
    // new path with alpha = w_avang, err_dir = (R^T·r̂)×bs.  This matches
    // PhD's veccostJacobians linear form `w_avang · ω · (R^T·r̂×bs)`.
    //
    // NOTE: ω_ff feed-forward tracking was removed (was always zero in
    // production — BP never passed nonzero `omega_ref`).  All cross-cost
    // forms are linear in ω with ω_ff = 0.
    const bool quat_legacy = (w_avang != 0.0) && !is_eci_format;
    if (quat_legacy) {
        quad_omega_cost = 0.5 * w_av * w.squaredNorm();
        cross_cost = -safeSign(qdot_aligned) * (q_goal_aligned.transpose() * W * w)(0) * w_avang;
    } else {
        // New path. Mode-aware err_dir; ω_ff = 0.
        Vec3 err_dir  = Vec3::Zero();
        double alpha  = 0.0;
        if (is_eci_format) {
            // Vector mode.
            //   err_dir = (R^T · r̂_eci) × bs  (tangent descent, perp to bs).
            //   α from raw `w_avang` (legacy coefficient) when set, else from
            //   `β · √(angle · λ_min(W_ω))` (PSD-bounded auto coefficient).
            //   λ_min(W_ω) = w_av · roll_ratio.
            const Vec3 bs_unit = boresight_body.normalized();
            const Mat33 R_T = saltro::math::rotationMatrix(q).transpose();
            if (w_avang != 0.0) {
                alpha = w_avang;
                const Vec3 r_body = R_T * attitude_target.tail(3).normalized();
                err_dir = r_body.cross(bs_unit);
            } else if (cost_cfg.ang_vel_err_dir_ratio > 0.0 && w_av > 0.0 && w_ang_eff > 0.0) {
                const double lam_min = w_av * cost_cfg.ang_vel_roll_ratio;
                alpha = cost_cfg.ang_vel_err_dir_ratio * std::sqrt(w_ang_eff * lam_min);
                const Vec3 r_body = R_T * attitude_target.tail(3).normalized();
                err_dir = r_body.cross(bs_unit);
            }
        } else {
            // Quaternion mode.
            //   err_dir = -sign(qdot) · (W^T · q_g_aligned).
            //   α = β · √(angle · ang_vel)  (W_ω = w_av · I, λ_min = w_av).
            if (cost_cfg.ang_vel_err_dir_ratio > 0.0 && w_av > 0.0 && w_ang_eff > 0.0) {
                alpha = cost_cfg.ang_vel_err_dir_ratio * std::sqrt(w_ang_eff * w_av);
                err_dir = -safeSign(qdot_aligned) * (W.transpose() * q_goal_aligned).head<3>();
            }
        }
        quad_omega_cost = 0.5 * w_av * w.squaredNorm();
        if (alpha > 0.0) {
            cross_cost = alpha * err_dir.dot(w);
        }
    }

    // Vector-mode axis-aware ω-cost reduction. Boresight pointing is 2-DOF;
    // roll about bs is unconstrained, so reduce W_ω's eigenvalue along bs
    // by (1 − roll_ratio). Default roll_ratio=1 → no-op (uniform |ω|²).
    //   W_ω = w_av · (roll · bs·bsᵀ + (I − bs·bsᵀ))
    //       = w_av · I − w_av · (1 − roll) · bs·bsᵀ
    //   ½·ωᵀ·W_ω·ω = ½·w_av·|ω|² − ½·w_av·(1 − roll)·(bs·ω)²
    // The reduction targets ω directly (not dw): roll about bs is irrelevant
    // for boresight pointing regardless of any tracking ω_ff.
    if (is_eci_format && cost_cfg.ang_vel_roll_ratio < 1.0) {
        const double roll_factor = 1.0 - cost_cfg.ang_vel_roll_ratio;
        const Vec3 bs_unit = boresight_body.normalized();
        const double bs_dot_w = bs_unit.dot(w);
        quad_omega_cost -= 0.5 * w_av * roll_factor * bs_dot_w * bs_dot_w;
    }

    double state_mag_cost = 0.0;
    const double b_norm = safeNorm(B_eci);
    if (b_norm > 1e-12) {
        const Mat33 R_T = saltro::math::rotationMatrix(q).transpose();
        const Vec3 b_body = R_T * (B_eci / b_norm);
        state_mag_cost = w_avmag * std::abs(w.dot(b_body));
    }

    double control_cost = 0.0;
    if (!terminal && controlDim() > 0) {
        for (int i = 0; i < num_mtq_; ++i) {
            const double lim = std::max(1e-9, std::abs(getMTQ(i).u_max()));
            const double normalized = u(i) / lim;
            control_cost += 0.5 * w_u_mult * cost_cfg.mtq_control_weight * normalized * normalized;
        }
        for (int i = 0; i < num_rw_; ++i) {
            const int ctrl_idx = num_mtq_ + i;
            const double lim = std::max(1e-9, std::abs(getRW(i).u_max()));
            const double normalized = u(ctrl_idx) / lim;
            control_cost += 0.5 * w_u_mult * cost_cfg.rw_control_weight * normalized * normalized;
        }
        for (int i = 0; i < num_magic_; ++i) {
            const int ctrl_idx = num_mtq_ + num_rw_ + i;
            const double lim = std::max(1e-9, std::abs(getMagic(i).u_max()));
            const double normalized = u(ctrl_idx) / lim;
            control_cost += 0.5 * w_u_mult * cost_cfg.magic_control_weight * normalized * normalized;
        }
    }

    double rw_momentum_cost = 0.0;
    double rw_stiction_cost = 0.0;
    for (int i = 0; i < num_rw_; ++i) {
        const double h = x(RW_MOMENTUM_INDEX + i);
        const double z = std::abs(h);
        const double h_max = std::max(1e-9, std::abs(getRW(i).momentumMax()));

        const double h_thresh = std::clamp(cost_cfg.RWh_max_mult, 0.0, 1.0) * h_max;
        const double denom_high = std::max(1e-9, h_max - h_thresh);
        if (z > h_thresh) {
            const double over = (z - h_thresh) / denom_high;
            rw_momentum_cost += 0.5 * cost_cfg.rw_AM_weight * over * over;
        } else {
            const double scaled = z / h_max;
            rw_momentum_cost += 0.5 * cost_cfg.rw_AM_weight * cost_cfg.RWh_ok_mult * scaled * scaled;
        }

        const double h_stic = std::clamp(cost_cfg.RWh_stiction_mult, 0.0, 1.0) * h_max;
        if (h_stic > 1e-12 && z < h_stic) {
            const double near_zero = (h_stic - z) / h_stic;
            rw_stiction_cost += 0.5 * cost_cfg.rw_stic_weight * near_zero * near_zero;
        }
    }

    const double state_cost = quad_omega_cost + w_ang_eff * ang_cost;
    return state_cost + cross_cost + state_mag_cost + control_cost + rw_momentum_cost + rw_stiction_cost;
}

double Satellite::terminalCost(const VecX& x, const Vec3& boresight_body, const Vec4& attitude_target,
                               const Vec3& B_eci, const CostConfig& cost_cfg) const {
    const VecX u_zero = VecX::Zero(controlDim());
    return stageCost(0, 1, x, u_zero, boresight_body, attitude_target, B_eci, cost_cfg);
}

std::tuple<Satellite::VecX, Satellite::MatX, Satellite::MatX> Satellite::stageCostJacobians(
    int k, int N, const VecX& x, const VecX& u,
    const Vec3& boresight_body, const Vec4& attitude_target,
    const Vec3& B_eci, const CostConfig& cost_cfg) const {
    if (N <= 0) {
        throw invalid_argument("N must be positive in stageCostJacobians().");
    }
    if (k < 0 || k >= N) {
        throw out_of_range("Time index k out of range in stageCostJacobians().");
    }
    if (x.size() < stateDim()) {
        throw invalid_argument("State vector has insufficient dimension in stageCostJacobians().");
    }
    if (u.size() < controlDim()) {
        throw invalid_argument("Control vector has insufficient dimension in stageCostJacobians().");
    }

    const int nx = stateDim();
    const int nu = controlDim();
    
    VecX lx = VecX::Zero(nx);
    VecX lu = VecX::Zero(nu);
    MatX lux = MatX::Zero(nu, nx);

    const bool terminal = (k >= N - 1);
    const double w_ang = terminal ? cost_cfg.angle_N : cost_cfg.angle;
    const double w_av = terminal ? cost_cfg.ang_vel_N : cost_cfg.ang_vel;
    const double w_avmag = terminal ? cost_cfg.ang_vel_mag_N : cost_cfg.ang_vel_mag;
    const double w_avang = terminal ? cost_cfg.ang_vel_err_dir_N : cost_cfg.ang_vel_err_dir;
    const double w_u_mult = terminal ? 0.0 : cost_cfg.control_mult;

    const Vec3 w = x.segment<3>(AV_INDEX);
    const Vec4 q = x.segment<4>(QUAT_INDEX).normalized();
    
    // Process attitude_target to handle both ECI vector and quaternion formats
    auto [q_goal, is_eci_format] = processAttitudeTarget(attitude_target, boresight_body);
    
    // Disable angle cost if ECI target is invalid
    double w_ang_eff = w_ang;
    if (is_eci_format && attitude_target.tail(3).norm() < 1e-9) {
        w_ang_eff = 0.0;
    }

    const double qdot = q_goal.dot(q);

    // ===== CRITICAL FIX: Handle quaternion double-cover =====
    // Quaternions q and -q represent the same rotation.
    // Ensure q_goal is on the same hemisphere as q to avoid sign flip issues.
    // This is essential for proper gradient computation when |qdot| is large.
    Vec4 q_goal_aligned = q_goal;
    double qdot_aligned = qdot;
    if (qdot < 0.0) {
        // If dot product is negative, q_goal points to the opposite hemisphere
        // Flip q_goal to the same hemisphere: (-q_goal) · q > 0
        q_goal_aligned = -q_goal;
        qdot_aligned = -qdot;
    }
    // Now qdot_aligned >= 0 always, and q_goal_aligned is on the same hemisphere as q

    // =====================================================================
    // Gradient w.r.t. ω and q from the ω-related stage cost terms.
    // Two paths:
    //  - back-compat (`w_avang != 0`): legacy formulas. ω_ff ignored.
    //  - new path: full chain rule through ω_ff(q) and the optional
    //    α · err_dir^T · (ω − ω_ff) crossterm. Reduces to legacy default
    //    (½·w_av·|ω|², no crossterm) when ω_ref=0 and ratio=0.
    // The ω-contribution `g_w_om` is assigned to `lx[AV]` here; the q-side
    // `g_q_om` is added to `lx[QUAT]` after the ang_cost gradient is set.
    // =====================================================================
    const Mat43 W = saltro::math::findWMat(q);
    Vec3 g_w_om;
    Vec4 g_q_om;
    {
        // Legacy back-compat applies only in quat mode (uses q_goal_aligned).
        // In vec mode, route w_avang through the new path's vec branch with
        // alpha = w_avang, ω_ff = 0 — matches PhD's linear veccostJacobians.
        const bool back_compat = (w_avang != 0.0) && !is_eci_format;
        if (back_compat) {
            const Vec3 cross_grad_w =
                -safeSign(qdot_aligned) * (W.transpose() * q_goal_aligned).head<3>() * w_avang;
            g_w_om = w_av * w + cross_grad_w;
            // ∂(cross_cost)/∂q via d_qgoal_W_w_dq:
            //   cross = -sign · (q_g^T · W(q) · w) · w_avang
            //   ∂/∂q_k = -sign · w_avang · q_g^T · (∂W/∂q_k) · w
            const double g0 = q_goal_aligned(0), g1 = q_goal_aligned(1);
            const double g2 = q_goal_aligned(2), g3 = q_goal_aligned(3);
            const double ww0 = w(0), ww1 = w(1), ww2 = w(2);
            Vec4 d_qgoal_W_w_dq;
            d_qgoal_W_w_dq(0) =  g1*ww0 + g2*ww1 + g3*ww2;
            d_qgoal_W_w_dq(1) = -g0*ww0 - g2*ww2 + g3*ww1;
            d_qgoal_W_w_dq(2) = -g0*ww1 + g1*ww2 - g3*ww0;
            d_qgoal_W_w_dq(3) = -g0*ww2 - g1*ww1 + g2*ww0;
            g_q_om = -safeSign(qdot_aligned) * w_avang * d_qgoal_W_w_dq;
        } else {
            // New path. Mode-aware err_dir(q) and its q-Jacobian; ω_ff = 0.
            Vec3 err_dir  = Vec3::Zero();
            double alpha  = 0.0;
            Eigen::Matrix<double, 3, 4> derr_dir_dq  = Eigen::Matrix<double, 3, 4>::Zero();

            if (is_eci_format) {
                // Vector mode. err_dir = (R^T·r̂) × bs.
                //   derr_dir_dq = -S · (drotmatTvecdq(q, r̂))^T,  S = skew(bs).
                const Vec3 bs_unit = boresight_body.normalized();
                const Mat33 S = saltro::math::skewSymmetric(bs_unit);
                const Mat33 R_T = saltro::math::rotationMatrix(q).transpose();
                if (w_avang != 0.0) {
                    // PhD veccostJacobians form: α = w_avang.
                    alpha = w_avang;
                } else if (cost_cfg.ang_vel_err_dir_ratio > 0.0 && w_av > 0.0 && w_ang_eff > 0.0) {
                    const double lam_min = w_av * cost_cfg.ang_vel_roll_ratio;
                    alpha = cost_cfg.ang_vel_err_dir_ratio * std::sqrt(w_ang_eff * lam_min);
                }
                if (alpha > 0.0) {
                    const Vec3 r_eci_norm = attitude_target.tail(3).normalized();
                    err_dir = (R_T * r_eci_norm).cross(bs_unit);
                    const Mat43 J_rhat = saltro::math::drotmatTvecdq(q, r_eci_norm);
                    derr_dir_dq = -S * J_rhat.transpose();
                }
            } else {
                // Quaternion mode.
                //   err_dir = -sign(qdot) · (W^T · q_g_aligned).
                //   α = β · √(angle · ang_vel)  (W_ω = w_av · I, λ_min = w_av).
                if (cost_cfg.ang_vel_err_dir_ratio > 0.0 && w_av > 0.0 && w_ang_eff > 0.0) {
                    alpha = cost_cfg.ang_vel_err_dir_ratio * std::sqrt(w_ang_eff * w_av);
                    err_dir = -safeSign(qdot_aligned) * (W.transpose() * q_goal_aligned).head<3>();
                    // ∂err_dir/∂q via the analytic W-derivative.
                    const double s = safeSign(qdot_aligned);
                    const double g0 = q_goal_aligned(0), g1 = q_goal_aligned(1);
                    const double g2 = q_goal_aligned(2), g3 = q_goal_aligned(3);
                    derr_dir_dq << -s*g1,  s*g0,  s*g3, -s*g2,
                                   -s*g2, -s*g3,  s*g0,  s*g1,
                                   -s*g3,  s*g2, -s*g1,  s*g0;
                }
            }
            // Linear-in-ω cross + quadratic-in-ω cost.  ω_ff = 0.
            //   ∂(quad)/∂ω = w_av · ω
            //   ∂(cross)/∂ω = α · err_dir
            //   ∂(cross)/∂q = α · (∂err_dir/∂q)^T · ω
            g_w_om = w_av * w + alpha * err_dir;
            g_q_om.setZero();
            if (alpha > 0.0) {
                g_q_om = alpha * (derr_dir_dq.transpose() * w);
            }
        }
    }

    // Vector-mode axis-aware ω-grad reduction (mirrors stageCost addend).
    // ∂/∂ω of −½·w_av·(1−roll)·(bs·ω)² = −w_av·(1−roll)·(bs·ω)·bs.
    // bs doesn't depend on q, so no q-grad contribution.
    if (is_eci_format && cost_cfg.ang_vel_roll_ratio < 1.0) {
        const double roll_factor = 1.0 - cost_cfg.ang_vel_roll_ratio;
        const Vec3 bs_unit = boresight_body.normalized();
        const double bs_dot_w = bs_unit.dot(w);
        g_w_om -= w_av * roll_factor * bs_dot_w * bs_unit;
    }

    lx.segment<3>(AV_INDEX) = g_w_om;

    // From state_mag_cost = w_avmag * |w · b_body|
    const double b_norm = safeNorm(B_eci);
    if (b_norm > 1e-12) {
        const Mat33 R_T = saltro::math::rotationMatrix(q).transpose();
        const Vec3 b_body = R_T * (B_eci / b_norm);
        const double w_dot_b = w.dot(b_body);
        const double sign_w_dot_b = safeSign(w_dot_b);
        lx.segment<3>(AV_INDEX) += w_avmag * sign_w_dot_b * b_body;
    }

    // =====================================================================
    // Gradient w.r.t. quaternion q: ∂L/∂q
    // =====================================================================
    
    // Compute derivative of attitude cost w.r.t. q
    if (is_eci_format) {
        // Vector mode: h = f(c), c = bs·R(q)ᵀ·r̂.  Work in the attitude tangent
        // space (PhD-style): ∂h/∂θ = f'(c)·∂c/∂θ.  Lift the reduced 3-vector
        // back to the ambient q-block with W; the backward pass re-projects
        // with Wᵀ (WᵀW = I₃, so the lift round-trips exactly).
        const Vec3 bs_unit = boresight_body.normalized();
        const Vec3 r_eci = attitude_target.tail(3).normalized();
        const VecPointingGeom geom = vecPointingGeom(q, bs_unit, r_eci);
        const AngCostShape f = angCostShape(geom.c, cost_cfg.ang_cost_func_type);
        lx.segment<4>(QUAT_INDEX) =
            w_ang_eff * f.fp * (saltro::math::findWMat(q) * geom.dc);
    } else {
        double d_ang_cost_dqdot = 0.0;  // ∂(ang_cost)/∂(qdot)
        switch (cost_cfg.ang_cost_func_type) {
            case 0:  // ang_cost = 1 - |qdot|
                d_ang_cost_dqdot = -1.0;  // Always negative since qdot_aligned >= 0
                break;
            case 1: {  // ang_cost = 0.5 * (1 - |qdot|)^2
                const double err = 1.0 - qdot_aligned;  // Use aligned value
                d_ang_cost_dqdot = -err;  // Always negative
                break;
            }
            case 2: {  // ang_cost = acos(|qdot|)
                const double denom = std::sqrt(1.0 - qdot_aligned * qdot_aligned + 1e-12);
                d_ang_cost_dqdot = -1.0 / denom;  // Always negative
                break;
            }
            case 3: {  // ang_cost = 0.5 * acos(|qdot|)^2
                const double phi = std::acos(qdot_aligned);
                const double denom = std::sqrt(1.0 - qdot_aligned * qdot_aligned + 1e-12);
                d_ang_cost_dqdot = -phi / denom;  // Always negative
                break;
            }
            case 4:  // ang_cost = (1 - |qdot|)^2
                d_ang_cost_dqdot = -2.0 * (1.0 - qdot_aligned);
                break;
            case 5:  // ang_cost = (1 - |qdot|) + (1 - |qdot|)^2 = (1 - d)(2 - d)
                //   d/dd [ (1-d) + (1-d)² ] = -1 - 2(1-d) = 2d - 3
                d_ang_cost_dqdot = 2.0 * qdot_aligned - 3.0;
                break;
            default:
                d_ang_cost_dqdot = -1.0 / std::sqrt(1.0 - qdot_aligned * qdot_aligned + 1e-12);
                break;
        }

        // ∂(qdot)/∂q where qdot = q_goal · q  →  ∂(qdot)/∂q = q_goal (as col).
        const Vec4 dqdot_dq = q_goal_aligned;
        // ∂L/∂q from attitude cost.
        lx.segment<4>(QUAT_INDEX) = w_ang_eff * d_ang_cost_dqdot * dqdot_dq;
    }

    // ω-related q-grad contribution (already computed above as `g_q_om`).
    lx.segment<4>(QUAT_INDEX) += g_q_om;

    // ∂(state_mag_cost)/∂q = ∂(w_avmag * |w · b_body|)/∂q
    // = w_avmag * sign(w · b_body) * ∂(w · b_body)/∂q
    // where b_body = R^T * (B_eci / |B_eci|)
    // ∂b_body/∂q = ∂(R^T * b_unit)/∂q
    if (b_norm > 1e-12) {
        const Vec3 b_unit = B_eci / b_norm;
        const Mat43 db_body_dq = saltro::math::drotmatTvecdq(q, b_unit);
        const Mat33 R_T = saltro::math::rotationMatrix(q).transpose();
        const Vec3 b_body = R_T * b_unit;
        const double w_dot_b = w.dot(b_body);
        const double sign_w_dot_b = safeSign(w_dot_b);
        
        // ∂(w · b_body)/∂q = J * w (as a 4D vector)
        // where J is 4x3 (quaternion rows, vector cols)
        Vec4 dw_dot_b_dq = db_body_dq * w;
        lx.segment<4>(QUAT_INDEX) += w_avmag * sign_w_dot_b * dw_dot_b_dq;
    }

    // NOTE: Quaternion projection is applied at the END (see below), not here.

    // =====================================================================
    // Gradient w.r.t. RW momentum h: ∂L/∂h
    // =====================================================================
    for (int i = 0; i < num_rw_; ++i) {
        const double h = x(RW_MOMENTUM_INDEX + i);
        const double z = std::abs(h);
        const double h_max = std::max(1e-9, std::abs(getRW(i).momentumMax()));

        // RW momentum penalty: soft penalty in saturation and stiction regions
        const double h_thresh = std::clamp(cost_cfg.RWh_max_mult, 0.0, 1.0) * h_max;
        const double denom_high = std::max(1e-9, h_max - h_thresh);
        const double sign_h = safeSign(h);
        
        if (z > h_thresh) {
            const double over = (z - h_thresh) / denom_high;
            lx(RW_MOMENTUM_INDEX + i) += cost_cfg.rw_AM_weight * sign_h * over / denom_high;
        } else {
            const double scaled = z / h_max;
            lx(RW_MOMENTUM_INDEX + i) += cost_cfg.rw_AM_weight * cost_cfg.RWh_ok_mult * sign_h * scaled / h_max;
        }

        // Stiction penalty
        const double h_stic = std::clamp(cost_cfg.RWh_stiction_mult, 0.0, 1.0) * h_max;
        if (h_stic > 1e-12 && z < h_stic) {
            const double near_zero = (h_stic - z) / h_stic;
            lx(RW_MOMENTUM_INDEX + i) += cost_cfg.rw_stic_weight * (-sign_h) * near_zero / h_stic;
        }
    }

    // =====================================================================
    // Gradient w.r.t. control u: ∂L/∂u
    // =====================================================================
    if (!terminal && w_u_mult > 1e-12) {
        // MTQ control costs
        for (int i = 0; i < num_mtq_; ++i) {
            const double lim = std::max(1e-9, std::abs(getMTQ(i).u_max()));
            const double normalized = u(i) / lim;
            lu(i) = w_u_mult * cost_cfg.mtq_control_weight * normalized / lim;
        }

        // RW control costs
        for (int i = 0; i < num_rw_; ++i) {
            const int ctrl_idx = num_mtq_ + i;
            const double lim = std::max(1e-9, std::abs(getRW(i).u_max()));
            const double normalized = u(ctrl_idx) / lim;
            lu(ctrl_idx) = w_u_mult * cost_cfg.rw_control_weight * normalized / lim;
        }

        // Magic actuator control costs
        for (int i = 0; i < num_magic_; ++i) {
            const int ctrl_idx = num_mtq_ + num_rw_ + i;
            const double lim = std::max(1e-9, std::abs(getMagic(i).u_max()));
            const double normalized = u(ctrl_idx) / lim;
            lu(ctrl_idx) = w_u_mult * cost_cfg.magic_control_weight * normalized / lim;
        }
    }

    // =========================================================================
    // Apply Quaternion Normalization Projection to Gradient
    // =========================================================================
    // Since cost function normalizes q, gradient must lie in tangent space
    // grad_projected = (I - q*q^T) * grad_raw
    {
        using Mat44 = Eigen::Matrix<double, 4, 4>;
        const Mat44 proj_q = Mat44::Identity() - q * q.transpose();
        lx.segment<4>(QUAT_INDEX) = proj_q * lx.segment<4>(QUAT_INDEX);
    }

    // Reshape lu as a 1 × nu matrix for return type compatibility
    MatX Lu_mat = lu.transpose();
    
    return std::make_tuple(lx, Lu_mat, lux);
}

std::tuple<Satellite::VecX, Satellite::MatX, Satellite::MatX> Satellite::terminalCostJacobians(
    const VecX& x, const Vec3& boresight_body, const Vec4& attitude_target,
    const Vec3& B_eci, const CostConfig& cost_cfg) const {
    const VecX u_zero = VecX::Zero(controlDim());
    return stageCostJacobians(0, 1, x, u_zero, boresight_body, attitude_target, B_eci, cost_cfg);
}

std::tuple<Satellite::MatX, Satellite::MatX, Satellite::MatX> Satellite::stageCostHessians(
    int k, int N, const VecX& x, const VecX& u,
    const Vec3& boresight_body, const Vec4& attitude_target,
    const Vec3& B_eci, const CostConfig& cost_cfg) const {
    if (N <= 0) {
        throw invalid_argument("N must be positive in stageCostHessians().");
    }
    if (k < 0 || k >= N) {
        throw out_of_range("Time index k out of range in stageCostHessians().");
    }
    if (x.size() < stateDim()) {
        throw invalid_argument("State vector has insufficient dimension in stageCostHessians().");
    }
    if (u.size() < controlDim()) {
        throw invalid_argument("Control vector has insufficient dimension in stageCostHessians().");
    }

    const int nx = stateDim();
    const int nu = controlDim();
    
    MatX lxx = MatX::Zero(nx, nx);
    MatX luu = MatX::Zero(nu, nu);
    MatX lux = MatX::Zero(nu, nx);

    const bool terminal = (k >= N - 1);
    const double w_ang = terminal ? cost_cfg.angle_N : cost_cfg.angle;
    const double w_av = terminal ? cost_cfg.ang_vel_N : cost_cfg.ang_vel;
    const double w_avmag = terminal ? cost_cfg.ang_vel_mag_N : cost_cfg.ang_vel_mag;
    const double w_avang = terminal ? cost_cfg.ang_vel_err_dir_N : cost_cfg.ang_vel_err_dir;
    const double w_u_mult = terminal ? 0.0 : cost_cfg.control_mult;

    const Vec3 w = x.segment<3>(AV_INDEX);
    const Vec4 q = x.segment<4>(QUAT_INDEX).normalized();

    // Process attitude_target to handle both ECI vector and quaternion formats
    auto [q_goal, is_eci_format] = processAttitudeTarget(attitude_target, boresight_body);
    
    // Disable angle cost if ECI target is invalid
    double w_ang_eff = w_ang;
    if (is_eci_format && attitude_target.tail(3).norm() < 1e-9) {
        w_ang_eff = 0.0;
    }

    // Quaternion alignment: ensure q_goal is on the same hemisphere as q
    const double qdot = q_goal.dot(q);
    Vec4 q_goal_aligned = q_goal;
    if (qdot < 0.0) {
        q_goal_aligned = -q_goal;
    }
    const double qdot_aligned = std::abs(qdot);  // always >= 0

    // KNOWN GAP: state-magnitude cost `w_avmag · |ω · b_body|` (gradient is
    // computed in stageCostJacobians but the (ω,ω) and (q,q) Hessian addends
    // are NOT implemented here.  When `use_cost_hess=true` and `ang_vel_mag`
    // is nonzero, the BP's quadratic model is missing this term's curvature.
    // Acceptable in current production (default `ang_vel_mag = 0`); fix if
    // ever activated.
    (void)w_avmag;
    (void)B_eci;

    // If cost Hessians are disabled, keep only control quadratic curvature.
    // This avoids unstable second-order state terms (notably quaternion terms)
    // while preserving positive curvature in control for backward pass stability.
    if (!cost_cfg.use_cost_hess) {
        if (!terminal && w_u_mult > 1e-12) {
            for (int i = 0; i < num_mtq_; ++i) {
                const double lim = std::max(1e-9, std::abs(getMTQ(i).u_max()));
                luu(i, i) += w_u_mult * cost_cfg.mtq_control_weight / (lim * lim);
            }

            for (int i = 0; i < num_rw_; ++i) {
                const int ctrl_idx = num_mtq_ + i;
                const double lim = std::max(1e-9, std::abs(getRW(i).u_max()));
                luu(ctrl_idx, ctrl_idx) += w_u_mult * cost_cfg.rw_control_weight / (lim * lim);
            }

            for (int i = 0; i < num_magic_; ++i) {
                const int ctrl_idx = num_mtq_ + num_rw_ + i;
                const double lim = std::max(1e-9, std::abs(getMagic(i).u_max()));
                luu(ctrl_idx, ctrl_idx) += w_u_mult * cost_cfg.magic_control_weight / (lim * lim);
            }
        }

        return std::make_tuple(lxx, luu, lux);
    }

    // =====================================================================
    // Hessian w.r.t. angular velocity: ∂²L/∂w²
    // =====================================================================
    // From 0.5 * w_av * ‖w‖² → ∂²L/∂w² = w_av * I_3
    lxx.block<3, 3>(AV_INDEX, AV_INDEX) += w_av * Mat33::Identity();

    // Vector-mode axis-aware ω-Hessian reduction (mirrors stageCost +
    // stageCostJacobians addends).
    //   ∂²/∂ω² of −½·w_av·(1 − roll)·(bs·ω)² = −w_av·(1 − roll)·bs·bsᵀ.
    if (is_eci_format && cost_cfg.ang_vel_roll_ratio < 1.0) {
        const double roll_factor = 1.0 - cost_cfg.ang_vel_roll_ratio;
        const Vec3 bs_unit = boresight_body.normalized();
        lxx.block<3, 3>(AV_INDEX, AV_INDEX) -= w_av * roll_factor
            * (bs_unit * bs_unit.transpose());
    }

    // =====================================================================
    // Hessian w.r.t. quaternion: ∂²L/∂q² (attitude terms) — ANALYTIC
    // =====================================================================
    // Per Jackson/Tracy/Manchester "Planning with Attitude" (2020), for a
    // scalar cost h(q) on S³ with q ∈ R⁴, the manifold Hessian is
    //   ∂²h_red/∂q² = G^T · ∂²h_amb/∂q² · G − I_3 · (∂h_amb/∂q · q)
    // We compute the ambient (R⁴) Hessian and the (∂h_amb/∂q · q) scalar;
    // the G-projection is implicit when this lxx feeds into the BP's
    // reduced-state machinery, and the −(grad·q)·I_4 correction is applied
    // here on the q-block.
    if (is_eci_format) {
        // Vector mode: reduced-space angle Hessian, PhD-style.  c = bs·R(q)ᵀ·r̂,
        // h = f(c).  In the attitude tangent space the Hessian splits cleanly:
        //
        //   Gauss-Newton:  H = f''(c)·(∂c/∂θ)(∂c/∂θ)ᵀ
        //                  — a rank-1 outer product, PSD by construction
        //                    wherever f'' ≥ 0 (types 0,1,3,4).
        //   Full (GN off): H += f'(c)·∂²c/∂θ²
        //                  — the chain-rule term.  ∂²c/∂θ² (geom.ddc) already
        //                    carries the Planning-with-Attitude manifold
        //                    correction, so this branch is the exact Hessian.
        //
        // The cost_hess_gauss_newton flag selects between them: with the flag
        // OFF the full (exact) Hessian is returned; ON gives the GN form.
        const Vec3 bs_unit = boresight_body.normalized();
        const Vec3 r_eci = attitude_target.tail(3).normalized();
        const VecPointingGeom geom = vecPointingGeom(q, bs_unit, r_eci);
        const AngCostShape f = angCostShape(geom.c, cost_cfg.ang_cost_func_type);

        Eigen::Matrix3d H_red = f.fpp * (geom.dc * geom.dc.transpose());
        if (!cost_cfg.cost_hess_gauss_newton) {
            H_red += f.fp * geom.ddc;  // full exact Hessian
        }
        // Lift reduced 3×3 → ambient q-block; the BP re-projects with Wᵀ.
        const auto W = saltro::math::findWMat(q);
        lxx.block<4, 4>(QUAT_INDEX, QUAT_INDEX) +=
            w_ang_eff * (W * H_red * W.transpose());
    } else {
        // Quaternion mode: h(q) = f(d) where d = q_g_aligned · q.
        //   ∂h/∂q   = f'(d) · q_g
        //   ∂²h/∂q² = f''(d) · q_g · q_g^T
        //   ∂h/∂q · q = f'(d) · d   (Euler deg-1)
        const double d = qdot_aligned;
        const double d2 = d * d;
        const double one_minus_d2 = std::max(1.0 - d2, 1e-12);
        const double sqrt_omd2 = std::sqrt(one_minus_d2);

        double d2h_dd2 = 0.0;
        switch (cost_cfg.ang_cost_func_type) {
            case 0:
                d2h_dd2 = 0.0;
                break;
            case 1:
                d2h_dd2 = 1.0;
                break;
            case 2:
                d2h_dd2 = -d / (one_minus_d2 * sqrt_omd2);
                break;
            case 3: {
                const double phi = std::acos(std::clamp(d, 0.0, 1.0));
                d2h_dd2 = 1.0 / one_minus_d2 - phi * d / (one_minus_d2 * sqrt_omd2);
                break;
            }
            case 4:
                d2h_dd2 = 2.0;  // h = (1-d)² → d²h/dd² = 2 (was -2 for old 1-d² form)
                break;
            case 5:
                d2h_dd2 = 2.0;  // h = (1-d) + (1-d)² → d²h/dd² = 2 (linear term contributes 0)
                break;
            default:
                d2h_dd2 = -d / (one_minus_d2 * sqrt_omd2);
                break;
        }
        lxx.block<4, 4>(QUAT_INDEX, QUAT_INDEX) += w_ang_eff * d2h_dd2
            * (q_goal_aligned * q_goal_aligned.transpose());

        double dh_dd = 0.0;
        switch (cost_cfg.ang_cost_func_type) {
            case 0: dh_dd = -1.0; break;
            case 1: dh_dd = -(1.0 - d); break;
            case 2: dh_dd = -1.0 / sqrt_omd2; break;
            case 3: {
                const double phi2 = std::acos(std::clamp(d, 0.0, 1.0));
                dh_dd = -phi2 / sqrt_omd2;
                break;
            }
            case 4: dh_dd = -2.0 * (1.0 - d); break;  // h = (1-d)² → dh/dd = -2(1-d)
            case 5: dh_dd = 2.0 * d - 3.0; break;     // h = (1-d) + (1-d)² → dh/dd = 2d - 3
            default: dh_dd = -1.0 / sqrt_omd2; break;
        }
        // Quat mode: PwA correction always applied (it's the manifold-
        // curvature term, PSD when f'·d < 0 which is the aligned-hemisphere
        // case for our cost shapes). GN flag has no effect here because
        // quat mode has no `f'·d²d/dq²` chain term to drop (d is linear in q).
        const double grad_dot_q = dh_dd * d;
        lxx.block<4, 4>(QUAT_INDEX, QUAT_INDEX) -= w_ang_eff * grad_dot_q
            * Eigen::Matrix<double, 4, 4>::Identity();
    }

    // =====================================================================
    // ω-related Hessian contributions — branched on path.
    //
    // Quat-mode legacy (`w_avang != 0` AND quaternion mode): cross_cost is
    // bilinear in (q, ω). (q,q) Hessian is identically zero.  The (ω,q)
    // cross-Hessian is analytically nonzero but intentionally skipped here to
    // preserve existing behavior.
    //
    // All other paths (vec mode regardless of w_avang, or quat mode with
    // w_avang == 0): full Hessian of
    //   ½·w_av·|ω-ω_ff(q)|² + α·err_dir(q)^T·(ω-ω_ff(q)).
    //   In vec-mode legacy (w_avang != 0 + is_eci_format), α = w_avang and
    //   ω_ff = 0 (matches PhD's veccostJacobians linear cross).
    //   (ω,ω): w_av·I_3 — already added above.
    //   (ω,q): -w_av·∂ω_ff/∂q + α·∂err_dir/∂q.
    //   (q,q):  w_av·(∂ω_ff/∂q)^T·(∂ω_ff/∂q)
    //         - Σ_l (w_av·dw_l + α·err_dir_l) · ∂²ω_ff_l/∂q²
    //         - α · ((∂err_dir/∂q)^T·(∂ω_ff/∂q) + (∂ω_ff/∂q)^T·(∂err_dir/∂q))
    // =====================================================================
    if (w_avang == 0.0 || is_eci_format) {
        // Mode-aware err_dir(q) and its first/second q-derivatives.
        // ω_ff = 0 — feed-forward tracking removed (was always zero in production).
        // The cost is `½·w_av·|ω|² + α·err_dir(q)·ω`, so:
        //   (ω,ω): w_av·I_3  (already added above)
        //   (ω,q): α · ∂err_dir/∂q
        //   (q,q): α · Σ_b ω_b · ∂²err_dir_b/∂q²  (nonzero only in vector mode)
        Vec3 err_dir  = Vec3::Zero();
        double alpha  = 0.0;
        Eigen::Matrix<double, 3, 4> derr_dir_dq  = Eigen::Matrix<double, 3, 4>::Zero();
        std::array<Eigen::Matrix4d, 3> dderr_dir_dqdq;
        for (auto& M : dderr_dir_dqdq)  M.setZero();

        if (is_eci_format) {
            // Vector mode. err_dir = (R^T·r̂) × bs.
            //   derr_dir_dq = -S · (drotmatTvecdq(q, r̂))^T,  S = skew(bs).
            //   ∂²err_dir_b/∂q² = -Σ_c S[b,c] · ddrotmatTvecdqdq(q, r̂)[c].
            const Vec3 bs_unit = boresight_body.normalized();
            const Mat33 S = saltro::math::skewSymmetric(bs_unit);
            const Mat33 R_T = saltro::math::rotationMatrix(q).transpose();
            if (w_avang != 0.0) {
                alpha = w_avang;
            } else if (cost_cfg.ang_vel_err_dir_ratio > 0.0 && w_av > 0.0 && w_ang_eff > 0.0) {
                const double lam_min = w_av * cost_cfg.ang_vel_roll_ratio;
                alpha = cost_cfg.ang_vel_err_dir_ratio * std::sqrt(w_ang_eff * lam_min);
            }
            if (alpha > 0.0) {
                const Vec3 r_eci_norm = attitude_target.tail(3).normalized();
                err_dir = (R_T * r_eci_norm).cross(bs_unit);
                const Mat43 J_rhat = saltro::math::drotmatTvecdq(q, r_eci_norm);
                derr_dir_dq = -S * J_rhat.transpose();
                const std::array<Eigen::Matrix4d, 3> H_rhat =
                    saltro::math::ddrotmatTvecdqdq(q, r_eci_norm);
                for (int b = 0; b < 3; ++b) {
                    for (int c = 0; c < 3; ++c) {
                        dderr_dir_dqdq[b] -= S(b, c) * H_rhat[c];
                    }
                }
            }
        } else {
            // Quaternion mode (err_dir is linear in q ⇒ dderr_dir_dqdq = 0).
            if (cost_cfg.ang_vel_err_dir_ratio > 0.0 && w_av > 0.0 && w_ang_eff > 0.0) {
                alpha = cost_cfg.ang_vel_err_dir_ratio * std::sqrt(w_ang_eff * w_av);
                const Mat43 W = saltro::math::findWMat(q);
                err_dir = -safeSign(qdot_aligned) * (W.transpose() * q_goal_aligned).head<3>();
                const double s = safeSign(qdot_aligned);
                const double g0 = q_goal_aligned(0), g1 = q_goal_aligned(1);
                const double g2 = q_goal_aligned(2), g3 = q_goal_aligned(3);
                derr_dir_dq << -s*g1,  s*g0,  s*g3, -s*g2,
                               -s*g2, -s*g3,  s*g0,  s*g1,
                               -s*g3,  s*g2, -s*g1,  s*g0;
            }
        }

        if (alpha > 0.0) {
            // (ω, q) cross-Hessian.
            const Eigen::Matrix<double, 3, 4> mix_omega_q = alpha * derr_dir_dq;
            lxx.block<3, 4>(AV_INDEX, QUAT_INDEX) += mix_omega_q;
            lxx.block<4, 3>(QUAT_INDEX, AV_INDEX) += mix_omega_q.transpose();

            // (q, q) Hessian addend: α · Σ_b ω_b · ∂²err_dir_b/∂q²  (vec mode only).
            Eigen::Matrix4d Hqq_om = Eigen::Matrix4d::Zero();
            for (int b = 0; b < 3; ++b) {
                Hqq_om += alpha * w(b) * dderr_dir_dqdq[b];
            }
            lxx.block<4, 4>(QUAT_INDEX, QUAT_INDEX) += Hqq_om;
        }
    }

    // =====================================================================
    // Hessian w.r.t. RW momentum: ∂²L/∂h²
    // =====================================================================
    for (int i = 0; i < num_rw_; ++i) {
        const double h = x(RW_MOMENTUM_INDEX + i);
        const double z = std::abs(h);
        const double h_max = std::max(1e-9, std::abs(getRW(i).momentumMax()));

        // Angular momentum saturation penalty
        const double h_thresh = std::clamp(cost_cfg.RWh_max_mult, 0.0, 1.0) * h_max;
        const double denom_high = std::max(1e-9, h_max - h_thresh);
        if (z > h_thresh) {
            const double d2_over_dz2 = 1.0 / (denom_high * denom_high);
            lxx(RW_MOMENTUM_INDEX + i, RW_MOMENTUM_INDEX + i) += 
                cost_cfg.rw_AM_weight * d2_over_dz2;
        } else {
            const double d2_scaled_dz2 = 1.0 / (h_max * h_max);
            lxx(RW_MOMENTUM_INDEX + i, RW_MOMENTUM_INDEX + i) += 
                cost_cfg.rw_AM_weight * cost_cfg.RWh_ok_mult * d2_scaled_dz2;
        }

        // Stiction penalty second derivative
        const double h_stic = std::clamp(cost_cfg.RWh_stiction_mult, 0.0, 1.0) * h_max;
        if (h_stic > 1e-12 && z < h_stic) {
            const double d2_near_zero_dz2 = 1.0 / (h_stic * h_stic);
            lxx(RW_MOMENTUM_INDEX + i, RW_MOMENTUM_INDEX + i) += 
                cost_cfg.rw_stic_weight * d2_near_zero_dz2;
        }
    }

    // =====================================================================
    // Hessian w.r.t. control: ∂²L/∂u²
    // =====================================================================
    if (!terminal && w_u_mult > 1e-12) {
        // MTQ control costs: diagonal contributions
        for (int i = 0; i < num_mtq_; ++i) {
            const double lim = std::max(1e-9, std::abs(getMTQ(i).u_max()));
            luu(i, i) += w_u_mult * cost_cfg.mtq_control_weight / (lim * lim);
        }

        // RW control costs: diagonal contributions
        for (int i = 0; i < num_rw_; ++i) {
            const int ctrl_idx = num_mtq_ + i;
            const double lim = std::max(1e-9, std::abs(getRW(i).u_max()));
            luu(ctrl_idx, ctrl_idx) += w_u_mult * cost_cfg.rw_control_weight / (lim * lim);
        }

        // Magic actuator control costs: diagonal contributions
        for (int i = 0; i < num_magic_; ++i) {
            const int ctrl_idx = num_mtq_ + num_rw_ + i;
            const double lim = std::max(1e-9, std::abs(getMagic(i).u_max()));
            luu(ctrl_idx, ctrl_idx) += w_u_mult * cost_cfg.magic_control_weight / (lim * lim);
        }
    }

    // =====================================================================
    // Apply Quaternion Normalization Projection
    // =====================================================================
    // For normalized quaternions, gradients lie in the constraint manifold.
    // Hessians must be projected: H_proj = (I - q·q^T) H (I - q·q^T)
    // Note: q is already normalized (line 1388), so q_norm = 1
    {
        using Mat44 = Eigen::Matrix<double, 4, 4>;
        const Mat44 proj_q = Mat44::Identity() - q * q.transpose();
        // Left projection
        lxx.block(QUAT_INDEX, 0, 4, nx) = proj_q * lxx.block(QUAT_INDEX, 0, 4, nx);
        // Right projection
        lxx.block(0, QUAT_INDEX, nx, 4) = lxx.block(0, QUAT_INDEX, nx, 4) * proj_q;
    }

    return std::make_tuple(lxx, luu, lux);
}

std::tuple<Satellite::MatX, Satellite::MatX, Satellite::MatX> Satellite::terminalCostHessians(
    const VecX& x, const Vec3& boresight_body, const Vec4& attitude_target,
    const Vec3& B_eci, const CostConfig& cost_cfg) const {
    const VecX u_zero = VecX::Zero(controlDim());
    return stageCostHessians(0, 1, x, u_zero, boresight_body, attitude_target, B_eci, cost_cfg);
}

double Satellite::totalCost(const Eigen::Ref<const Eigen::MatrixXd>& X,
                            const Eigen::Ref<const Eigen::MatrixXd>& U,
                            const Eigen::Ref<const Eigen::MatrixXd>& B,
                            const Eigen::Ref<const Eigen::MatrixXd>& boresight,
                            const Eigen::Ref<const Eigen::MatrixXd>& attitude_target,
                            const CostConfig& cost_cfg) const {
    const int nx = stateDim();
    const int nu = controlDim();
    const int N = static_cast<int>(X.cols());

    if (X.rows() != nx) {
        throw std::invalid_argument(
            "totalCost: X has " + std::to_string(X.rows()) +
            " rows, expected " + std::to_string(nx) + " (nx x N layout)");
    }
    if (U.rows() != nu) {
        throw std::invalid_argument(
            "totalCost: U has " + std::to_string(U.rows()) +
            " rows, expected " + std::to_string(nu) + " (nu x N-1 layout)");
    }
    if (U.cols() != N - 1) {
        throw std::invalid_argument(
            "totalCost: U has " + std::to_string(U.cols()) +
            " columns, expected " + std::to_string(N - 1));
    }
    if (B.rows() != 3) {
        throw std::invalid_argument(
            "totalCost: B has " + std::to_string(B.rows()) +
            " rows, expected 3");
    }
    if (B.cols() != N) {
        throw std::invalid_argument(
            "totalCost: B has " + std::to_string(B.cols()) +
            " columns, expected " + std::to_string(N));
    }
    if (boresight.cols() != N) {
        throw std::invalid_argument(
            "totalCost: boresight has " + std::to_string(boresight.cols()) +
            " columns, expected " + std::to_string(N));
    }
    if (boresight.rows() != 3) {
        throw std::invalid_argument(
            "totalCost: boresight has " + std::to_string(boresight.rows()) +
            " rows, expected 3");
    }
    if (attitude_target.rows() != 4) {
        throw std::invalid_argument(
            "totalCost: attitude_target has " + std::to_string(attitude_target.rows()) +
            " rows, expected 4");
    }
    if (attitude_target.cols() != N) {
        throw std::invalid_argument(
            "totalCost: attitude_target has " + std::to_string(attitude_target.cols()) +
            " columns, expected " + std::to_string(N));
    }
    
    double J_total = 0.0;
    // Sum stage costs for k = 0 to N-2
    for (int k = 0; k < N - 1; ++k) {
        VecX x_k = X.col(k);
        VecX u_k = U.col(k);
        Vec3 B_k = B.col(k);
        Vec3 boresight_k = boresight.col(k);
        Vec4 attitude_target_k = attitude_target.col(k);

        double stage_cost = stageCost(k, N, x_k, u_k, boresight_k, attitude_target_k, B_k, cost_cfg);
        J_total += stage_cost;
    }

    // Terminal cost at k = N-1
    VecX x_final = X.col(N - 1);
    Vec3 B_final = B.col(N - 1);
    Vec3 boresight_final = boresight.col(N - 1);
    Vec4 attitude_target_final = attitude_target.col(N - 1);

    double terminal_cost = terminalCost(x_final, boresight_final, attitude_target_final, B_final, cost_cfg);
    J_total += terminal_cost;
    
    return J_total;
}

Satellite::VecX Satellite::constraints(int k, int N, const VecX& x, const VecX& u,
                                      const Vec3& sun_eci, const ConstraintConfig& cnst_cfg) const {
    if (N <= 0) {
        throw invalid_argument("N must be positive in constraints().");
    }
    if (k < 0 || k >= N) {
        throw out_of_range("Time index k out of range in constraints().");
    }
    if (x.size() < stateDim()) {
        throw invalid_argument("State vector has insufficient dimension in constraints().");
    }
    if (u.size() < controlDim()) {
        throw invalid_argument("Control vector has insufficient dimension in constraints().");
    }

    const bool has_control_constraints = (k < N - 1);
    const int n_constraints = 1 + 1 + (has_control_constraints
                                           ? (2 * num_mtq_ + 5 * num_rw_ + 2 * num_magic_)
                                           : 0);
    VecX c(n_constraints);
    c.setZero();

    int idx = 0;

    const double wmax = std::max(1e-9, std::abs(cnst_cfg.wmax));
    const Vec3 w = x.segment<3>(AV_INDEX);
    c(idx++) = (w.squaredNorm() - wmax * wmax) / (wmax * wmax);

    const double sun_limit = std::clamp(cnst_cfg.sun_limit_angle, 0.0, M_PI);
    const Vec4 q_raw = x.segment<4>(QUAT_INDEX);
    const double q_norm = q_raw.norm();
    const Vec4 q = (q_norm > 1e-12) ? (q_raw / q_norm) : q_raw;
    const Mat33 R_T = saltro::math::rotationMatrix(q).transpose();
    const double sun_norm = sun_eci.norm();
    if (std::isfinite(sun_norm) && sun_norm > 1e-12) {
        const Vec3 sun_body = R_T * (sun_eci / sun_norm);
        c(idx++) = sun_body.x() - std::cos(sun_limit);
    } else {
        c(idx++) = 0.0;
    }

    if (!has_control_constraints) {
        return c;
    }

    auto configured_u_max = [&](int ctrl_idx) -> double {
        if (ctrl_idx < cnst_cfg.u_max.size()) {
            const double v = std::abs(cnst_cfg.u_max(ctrl_idx));
            if (std::isfinite(v) && v > 0.0) {
                return v;
            }
        }
        return 0.0;
    };

    const double scale = (std::isfinite(cnst_cfg.control_limit_scale) && cnst_cfg.control_limit_scale > 0.0)
                             ? cnst_cfg.control_limit_scale
                             : 1.0;

    for (int i = 0; i < num_mtq_; ++i) {
        const double u_cmd = u(i);
        const double lim_from_cfg = configured_u_max(i);
        const double lim_from_act = std::abs(getMTQ(i).u_max());
        const double lim = scale * std::max(1e-9, (lim_from_cfg > 0.0 ? lim_from_cfg : lim_from_act));
        c(idx++) = (u_cmd - lim) / lim;
        c(idx++) = (-u_cmd - lim) / lim;
    }

    for (int i = 0; i < num_rw_; ++i) {
        const int ctrl_idx = num_mtq_ + i;
        const double u_cmd = u(ctrl_idx);
        const double lim_from_cfg = configured_u_max(ctrl_idx);
        const double lim_from_act = std::abs(getRW(i).u_max());
        const double tau_lim = scale * std::max(1e-9, (lim_from_cfg > 0.0 ? lim_from_cfg : lim_from_act));

        c(idx++) = (u_cmd - tau_lim) / tau_lim;
        c(idx++) = (-u_cmd - tau_lim) / tau_lim;

        const double h = x(RW_MOMENTUM_INDEX + i);
        const double h_lim = std::max(1e-9, std::abs(getRW(i).momentumMax()));
        c(idx++) = (h - h_lim) / h_lim;
        c(idx++) = (-h - h_lim) / h_lim;

        c(idx++) = -(u_cmd * h) * (u_cmd * h);
    }

    // Magic-actuator torque bounds (no momentum or stiction terms — no internal state).
    for (int i = 0; i < num_magic_; ++i) {
        const int ctrl_idx = num_mtq_ + num_rw_ + i;
        const double u_cmd = u(ctrl_idx);
        const double lim_from_cfg = configured_u_max(ctrl_idx);
        const double lim_from_act = std::abs(getMagic(i).u_max());
        const double lim = scale * std::max(1e-9, (lim_from_cfg > 0.0 ? lim_from_cfg : lim_from_act));
        c(idx++) = (u_cmd - lim) / lim;
        c(idx++) = (-u_cmd - lim) / lim;
    }

    return c;
}

std::tuple<Satellite::MatX, Satellite::MatX> Satellite::constraintJacobians(
    int k, int N, const VecX& x, const VecX& u,
    const Vec3& sun_eci, const ConstraintConfig& cnst_cfg) const {
    
    if (N <= 0) {
        throw invalid_argument("N must be positive in constraintJacobians().");
    }
    if (k < 0 || k >= N) {
        throw out_of_range("Time index k out of range in constraintJacobians().");
    }
    if (x.size() < stateDim()) {
        throw invalid_argument("State vector has insufficient dimension in constraintJacobians().");
    }
    if (u.size() < controlDim()) {
        throw invalid_argument("Control vector has insufficient dimension in constraintJacobians().");
    }

    const bool has_control_constraints = (k < N - 1);
    const int n_constraints = 1 + 1 + (has_control_constraints
                                           ? (2 * num_mtq_ + 5 * num_rw_ + 2 * num_magic_)
                                           : 0);

    MatX c_u(n_constraints, controlDim());
    MatX c_x(n_constraints, stateDim());
    c_u.setZero();
    c_x.setZero();

    int idx = 0;

    // 1) Angular velocity constraint Jacobian: ∂/∂w of (||w||² - wmax²) / wmax²
    const double wmax = std::max(1e-9, std::abs(cnst_cfg.wmax));
    const Vec3 w = x.segment<3>(AV_INDEX);
    c_x.block<1, 3>(idx, AV_INDEX) = (2.0 / (wmax * wmax)) * w.transpose();
    idx++;

    // 2) Sun constraint Jacobian: ∂/∂q of (R(q)^T * sun).x - cos(limit)
    const double sun_norm = sun_eci.norm();
    if (std::isfinite(sun_norm) && sun_norm > 1e-12) {
        const Vec4 q_raw = x.segment<4>(QUAT_INDEX);
        const double q_norm = q_raw.norm();
        if (q_norm > 1e-12) {
            const Vec4 q = q_raw / q_norm;
            const Vec3 sun_unit = sun_eci / sun_norm;

            // Derivative of (R^T * sun) w.r.t. unit quaternion.
            const Mat34 dRTsun_dq = saltro::math::drotmatTvecdq(q, sun_unit).transpose();

            // Chain rule: q -> q / ||q||.
            const Eigen::Matrix4d Jnorm = (1.0 / q_norm) *
                                           (Eigen::Matrix4d::Identity() -
                                            (q_raw * q_raw.transpose()) / (q_norm * q_norm));
            const Mat34 dRTsun_dqraw = dRTsun_dq * Jnorm;

            // We only care about the x-component, so take first row.
            c_x.block<1, 4>(idx, QUAT_INDEX) = dRTsun_dqraw.row(0);
        }
    }
    idx++;

    if (!has_control_constraints) {
        return std::make_tuple(c_u, c_x);
    }

    // Helper for configured limits
    auto configured_u_max = [&](int ctrl_idx) -> double {
        if (ctrl_idx < cnst_cfg.u_max.size()) {
            const double v = std::abs(cnst_cfg.u_max(ctrl_idx));
            if (std::isfinite(v) && v > 0.0) {
                return v;
            }
        }
        return 0.0;
    };

    const double scale = (std::isfinite(cnst_cfg.control_limit_scale) && cnst_cfg.control_limit_scale > 0.0)
                             ? cnst_cfg.control_limit_scale
                             : 1.0;

    // 3) MTQ control bound Jacobians
    for (int i = 0; i < num_mtq_; ++i) {
        const double lim_from_cfg = configured_u_max(i);
        const double lim_from_act = std::abs(getMTQ(i).u_max());
        const double lim = scale * std::max(1e-9, (lim_from_cfg > 0.0 ? lim_from_cfg : lim_from_act));
        
        // Upper bound: (u - lim) / lim
        c_u(idx, i) = 1.0 / lim;
        idx++;
        
        // Lower bound: (-u - lim) / lim
        c_u(idx, i) = -1.0 / lim;
        idx++;
    }

    // 4) RW constraints
    for (int i = 0; i < num_rw_; ++i) {
        const int ctrl_idx = num_mtq_ + i;
        const int state_idx = RW_MOMENTUM_INDEX + i;
        
        const double lim_from_cfg = configured_u_max(ctrl_idx);
        const double lim_from_act = std::abs(getRW(i).u_max());
        const double tau_lim = scale * std::max(1e-9, (lim_from_cfg > 0.0 ? lim_from_cfg : lim_from_act));
        
        // RW torque upper bound
        c_u(idx, ctrl_idx) = 1.0 / tau_lim;
        idx++;
        
        // RW torque lower bound
        c_u(idx, ctrl_idx) = -1.0 / tau_lim;
        idx++;
        
        const double h_lim = std::max(1e-9, std::abs(getRW(i).momentumMax()));
        
        // RW momentum upper bound
        c_x(idx, state_idx) = 1.0 / h_lim;
        idx++;
        
        // RW momentum lower bound
        c_x(idx, state_idx) = -1.0 / h_lim;
        idx++;
        
        // RW stiction: -(u*h)²
        // ∂/∂u = -2*u*h²
        // ∂/∂h = -2*h*u²
        const double u_i = u(ctrl_idx);
        const double h_i = x(state_idx);
        c_u(idx, ctrl_idx) = -2.0 * u_i * h_i * h_i;
        c_x(idx, state_idx) = -2.0 * h_i * u_i * u_i;
        idx++;
    }

    // 5) Magic-actuator control bound Jacobians (linear in u only; no state coupling).
    for (int i = 0; i < num_magic_; ++i) {
        const int ctrl_idx = num_mtq_ + num_rw_ + i;
        const double lim_from_cfg = configured_u_max(ctrl_idx);
        const double lim_from_act = std::abs(getMagic(i).u_max());
        const double lim = scale * std::max(1e-9, (lim_from_cfg > 0.0 ? lim_from_cfg : lim_from_act));

        // Upper bound:  (u - lim) / lim
        c_u(idx, ctrl_idx) =  1.0 / lim;
        idx++;
        // Lower bound: (-u - lim) / lim
        c_u(idx, ctrl_idx) = -1.0 / lim;
        idx++;
    }

    return std::make_tuple(c_u, c_x);
}

std::tuple<Satellite::ConstrHessUU, Satellite::ConstrHessUX, Satellite::ConstrHessXX> 
Satellite::constraintHessians(
    int k, int N, const VecX& x, const VecX& u,
    const Vec3& sun_eci, const ConstraintConfig& cnst_cfg) const {
    
    if (N <= 0) {
        throw invalid_argument("N must be positive in constraintHessians().");
    }
    if (k < 0 || k >= N) {
        throw out_of_range("Time index k out of range in constraintHessians().");
    }
    if (x.size() < stateDim()) {
        throw invalid_argument("State vector has insufficient dimension in constraintHessians().");
    }
    if (u.size() < controlDim()) {
        throw invalid_argument("Control vector has insufficient dimension in constraintHessians().");
    }

    ConstrHessUU H_uu;
    ConstrHessUX H_ux;
    ConstrHessXX H_xx;
    H_uu.setZero();
    H_ux.setZero();
    H_xx.setZero();

    const bool has_control_constraints = (k < N - 1);
    int idx = 0;

    // 1) Angular velocity constraint Hessian: ∂²/∂w² of (||w||² - wmax²) / wmax²
    const double wmax = std::max(1e-9, std::abs(cnst_cfg.wmax));
    const double scale_av = 2.0 / (wmax * wmax);
    H_xx.slice(idx).block<3, 3>(AV_INDEX, AV_INDEX) = scale_av * Mat33::Identity();
    idx++;

    // 2) Sun constraint Hessian: central finite differences of constraint Jacobian
    // Constraint: c = [R(q)^T * sun_unit]_x - cos(limit)
    // The chain rule through quaternion normalization makes the analytic form fragile,
    // so we use central FD for robustness.  This is only called during the backward
    // pass when use_constraint_hess is enabled.
    const double sun_norm = sun_eci.norm();
    const int sun_idx = idx;
    if (std::isfinite(sun_norm) && sun_norm > 1e-12) {
        const double eps = 1e-7;
        for (int j = 0; j < stateDim(); ++j) {
            VecX xp = x; xp(j) += eps;
            VecX xm = x; xm(j) -= eps;
            if (j >= QUAT_INDEX && j < QUAT_INDEX + 4) {
                xp.segment<4>(QUAT_INDEX).normalize();
                xm.segment<4>(QUAT_INDEX).normalize();
            }

            auto [_, c_xp] = constraintJacobians(k, N, xp, u, sun_eci, cnst_cfg);
            auto [_2, c_xm] = constraintJacobians(k, N, xm, u, sun_eci, cnst_cfg);
            const Eigen::VectorXd fd_row =
                (c_xp.row(sun_idx) - c_xm.row(sun_idx)).transpose() / (2.0 * eps);
            auto col = H_xx.slice(sun_idx).col(j);
            col.setZero();
            col.head(stateDim()) = fd_row;
        }
    }
    idx++;

    if (!has_control_constraints) {
        return std::make_tuple(H_uu, H_ux, H_xx);
    }

    // MTQ bounds have zero Hessians (linear constraints)
    idx += 2 * num_mtq_;

    // RW constraints
    for (int i = 0; i < num_rw_; ++i) {
        const int ctrl_idx = num_mtq_ + i;
        const int state_idx = RW_MOMENTUM_INDEX + i;
        
        // RW torque bounds (linear, zero Hessian)
        idx += 2;
        
        // RW momentum bounds (linear, zero Hessian)
        idx += 2;
        
        // RW stiction: -(u*h)²
        // ∂²/∂u² = -2*h²
        // ∂²/∂h² = -2*u²
        // ∂²/∂u∂h = -4*u*h
        const double u_i = u(ctrl_idx);
        const double h_i = x(state_idx);
        
        H_uu.slice(idx)(ctrl_idx, ctrl_idx) = -2.0 * h_i * h_i;
        H_xx.slice(idx)(state_idx, state_idx) = -2.0 * u_i * u_i;
        H_ux.slice(idx)(ctrl_idx, state_idx) = -4.0 * u_i * h_i;
        idx++;
    }

    // Magic-actuator bounds are linear in u with no state coupling — zero Hessian.
    idx += 2 * num_magic_;

    return std::make_tuple(H_uu, H_ux, H_xx);
}

