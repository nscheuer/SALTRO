#include <saltro/pybind/satellite.h>
#include <saltro/pybind/disturbances/dragdisturbance.h>
#include <saltro/pybind/disturbances/ggdisturbance.h>
#include <saltro/pybind/disturbances/srpdisturbance.h>

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

Satellite::Vec3 Satellite::actuatorTorque(const VecX& x, const VecX& u, const Vec3& B_eci) const {
    Vec3 torque = Vec3::Zero();
    
    // Extract base state (first 7 elements: w + q)
    Vec4 q = x.segment<4>(QUAT_INDEX).normalized();
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
    
    return torque;
}

Satellite::Vec3 Satellite::disturbanceTorque(const VecX& x, const DisturbanceConfig& dist, const Vec3& R_eci, const Vec3& B_eci, const Vec3& S_eci, const Vec3& V_eci, const int rho) const {
    Vec3 torque = Vec3::Zero();
    Vec4 q = x.segment<4>(QUAT_INDEX).normalized();
    Vec7 x_base = x.head<7>();
    x_base.segment<4>(QUAT_INDEX) = q;

    Mat33 R_T = saltro::math::rotationMatrix(q).transpose();
    Vec3 V_body = R_T * V_eci;
    Vec3 R_body = R_T * R_eci;
    // S_eci is spacecraft-to-Sun; keep eclipse zeroing intact in body frame.
    Vec3 S_body = R_T * S_eci;
    Mat34 dV_dq = saltro::math::drotmatTvecdq(q, V_eci).transpose();
    auto d2V_dq2 = saltro::math::ddrotmatTvecdqdq(q, V_eci);
    (void)B_eci;
    (void)rho;
    (void)dV_dq;
    (void)d2V_dq2;

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
    
    return torque;
}

Satellite::VecX Satellite::dynamics(const VecX& x, const VecX& u, const DisturbanceConfig& dist, const Vec3& R_eci, const Vec3& B_eci, const Vec3& S_eci, const Vec3& V_eci, const int rho) const {
    Vec3 w = x.segment<3>(AV_INDEX);
    Vec4 q = x.segment<4>(QUAT_INDEX).normalized();
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
    const int nu = controlDim();
    
    DynHessXX hess_xx;  // (state, state, state) - indexed by output equation
    DynHessUX hess_ux;  // (control, state, state) - indexed by output equation
    DynHessUU hess_uu;  // (control, control, state) - indexed by output equation
    
    hess_xx.setZero();
    hess_ux.setZero();
    hess_uu.setZero();
    
    // Extract state components
    Vec3 w = x.segment<3>(AV_INDEX);
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
    
    Vec3 angular_mom = Jcom_ * w + h_rw;
    
    // =========================================================================
    // Angular velocity Hessian: ∂²wdot_i/∂x_j∂x_k (indexed by output i = 0,1,2)
    // =========================================================================
    // wdot = invJcom_noRW_ * (tau - w × (Jcom*w + h_rw))
    
    // For each output component i of wdot:
    for (int i = 0; i < 3; ++i) {
        Vec3 ei = Vec3::Zero();
        ei(i) = 1.0;
        
        // ∂²wdot_i/∂w_j∂w_k: from d/dw_j of [invJcom_noRW * (-w × (Jcom*w))]_i
        // The cross product w × (Jcom*w) is quadratic in w
        // d/dw_j d/dw_k of (w × (Jcom*w))_i = ei^T * [skew(e_j)*Jcom + Jcom*skew(e_j)^T] * e_k
        // where first term acts on w and second on (Jcom*w)
        
        for (int j = 0; j < 3; ++j) {
            Vec3 ej = Vec3::Zero();
            ej(j) = 1.0;
            
            for (int k = 0; k < 3; ++k) {
                Vec3 ek = Vec3::Zero();
                ek(k) = 1.0;
                
                // Second derivative of cross product: a × b
                // ∂²(a × b)/∂a_j∂a_k for b = Jcom*w
                // = -skew(ek)*Jcom + Jcom*skew(ek) applied to ej
                // ∂²(a × b)/∂a_j∂b_k for a = w, b = Jcom*w
                // = -skew(ej)*skew(Jcom*ek)
                
                Mat33 d2cross = saltro::math::skewSymmetric(ej) * Jcom_ * saltro::math::skewSymmetric(ek)
                              + saltro::math::skewSymmetric(ej) * saltro::math::skewSymmetric(ek) * Jcom_
                              - saltro::math::skewSymmetric(saltro::math::skewSymmetric(ej) * Jcom_ * ek);
                
                // Apply invJcom_noRW and extract component i
                Vec3 result = -d2cross * w;
                double hess_val = ei.dot(invJcom_noRW_ * result);
                hess_xx.slice(AV_INDEX + i)(AV_INDEX + j, AV_INDEX + k) = hess_val;
            }
        }
        
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
    
    Mat43 W = saltro::math::findWMat(q);
    
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

double Satellite::stageCost(int k, int N, const VecX& x, const VecX& u,
                            const Vec3& sat_direction, const Vec4& eci_target,
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

    (void)sat_direction;

    const bool terminal = (k >= N - 1);
    const double w_ang = terminal ? cost_cfg.angle_N : cost_cfg.angle;
    const double w_av = terminal ? cost_cfg.ang_vel_N : cost_cfg.ang_vel;
    const double w_avmag = terminal ? cost_cfg.ang_vel_mag_N : cost_cfg.ang_vel_mag;
    const double w_avang = terminal ? cost_cfg.ang_vel_err_dir_N : cost_cfg.ang_vel_err_dir;
    const double w_u_mult = terminal ? 0.0 : cost_cfg.control_mult;

    const Vec3 w = x.segment<3>(AV_INDEX);
    const Vec4 q = x.segment<4>(QUAT_INDEX).normalized();
    const Vec4 q_goal = eci_target.normalized();

    const double qdot = q_goal.dot(q);
    const double abs_qdot = clampUnit(safeAbs(qdot));

    double ang_cost = 0.0;
    switch (cost_cfg.ang_cost_func_type) {
        case 0:
            ang_cost = 1.0 - abs_qdot;
            break;
        case 1: {
            const double err = 1.0 - abs_qdot;
            ang_cost = 0.5 * err * err;
            break;
        }
        case 2:
            ang_cost = std::acos(abs_qdot);
            break;
        case 3: {
            const double phi = std::acos(abs_qdot);
            ang_cost = 0.5 * phi * phi;
            break;
        }
        case 4:
            ang_cost = 1.0 - abs_qdot * abs_qdot;
            break;
        default:
            ang_cost = std::acos(abs_qdot);
            break;
    }

    const Mat43 W = saltro::math::findWMat(q);
    const double cross_cost = -safeSign(qdot) * (q_goal.transpose() * W * w)(0) * w_avang;

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

    const double state_cost = 0.5 * w_av * w.squaredNorm() + w_ang * ang_cost;
    return state_cost + cross_cost + state_mag_cost + control_cost + rw_momentum_cost + rw_stiction_cost;
}

double Satellite::terminalCost(const VecX& x, const Vec3& sat_direction, const Vec4& eci_target,
                               const Vec3& B_eci, const CostConfig& cost_cfg) const {
    const VecX u_zero = VecX::Zero(controlDim());
    return stageCost(0, 1, x, u_zero, sat_direction, eci_target, B_eci, cost_cfg);
}

std::tuple<Satellite::VecX, Satellite::MatX, Satellite::MatX> Satellite::stageCostJacobians(
    int k, int N, const VecX& x, const VecX& u,
    const Vec3& sat_direction, const Vec4& eci_target,
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

    (void)sat_direction;

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
    const Vec4 q_goal = eci_target.normalized();

    const double qdot = q_goal.dot(q);
    const double abs_qdot = clampUnit(safeAbs(qdot));
    const double sign_qdot = safeSign(qdot);

    // =====================================================================
    // Gradient w.r.t. angular velocity w: ∂L/∂w
    // =====================================================================
    lx.segment<3>(AV_INDEX) = w_av * w;

    // From cross_cost = -sign(qdot) * (q_goal^T * W * w) * w_avang
    // ∂cross_cost/∂w = -sign(qdot) * (W^T * q_goal) * w_avang
    const Mat43 W = saltro::math::findWMat(q);
    lx.segment<3>(AV_INDEX) += -sign_qdot * (W.transpose() * q_goal).head<3>() * w_avang;

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
    double d_ang_cost_dqdot = 0.0;  // ∂(ang_cost)/∂(qdot)
    switch (cost_cfg.ang_cost_func_type) {
        case 0:  // ang_cost = 1 - |qdot|
            d_ang_cost_dqdot = -sign_qdot;
            break;
        case 1: {  // ang_cost = 0.5 * (1 - |qdot|)^2
            const double err = 1.0 - abs_qdot;
            d_ang_cost_dqdot = -sign_qdot * err;
            break;
        }
        case 2: {  // ang_cost = acos(|qdot|)
            const double denom = std::sqrt(1.0 - abs_qdot * abs_qdot + 1e-12);
            d_ang_cost_dqdot = -sign_qdot / denom;
            break;
        }
        case 3: {  // ang_cost = 0.5 * acos(|qdot|)^2
            const double phi = std::acos(abs_qdot);
            const double denom = std::sqrt(1.0 - abs_qdot * abs_qdot + 1e-12);
            d_ang_cost_dqdot = -sign_qdot * phi / denom;
            break;
        }
        case 4:  // ang_cost = 1 - |qdot|^2
            d_ang_cost_dqdot = -2.0 * sign_qdot * abs_qdot;
            break;
        default:
            d_ang_cost_dqdot = -sign_qdot / std::sqrt(1.0 - abs_qdot * abs_qdot + 1e-12);
            break;
    }

    // ∂(qdot)/∂q where qdot = q_goal · q
    // = q_goal (as a row vector, or column in Jacobian context)
    Vec4 dqdot_dq = q_goal;
    
    // ∂L/∂q from attitude cost: w_ang * ∂(ang_cost)/∂(qdot) * ∂(qdot)/∂q
    lx.segment<4>(QUAT_INDEX) = w_ang * d_ang_cost_dqdot * dqdot_dq;

    // ∂(cross_cost)/∂q = ∂/∂q[-sign(qdot) * (q_goal^T * W * w) * w_avang]
    // = -sign(qdot)*w_avang * ∂(q_goal^T * W * w)/∂q
    // where W depends on q
    // W = 0.5*[[-q1, -q2, -q3], [q0, -q3, q2], [q3, q0, -q1], [-q2, q1, q0]]^T (4x3)
    // ∂(W^T * q_goal)/∂q_j is a 3D vector
    // For each component j, compute ∂(W^T * q_goal)/∂q_j · w
    // This is complex, so use finite difference approach or compute analytically
    // Let me compute it analytically by noting:
    // (q_goal^T * W * w) = sum_i (q_goal_i * W_i,: * w) where W_i,: is i-th row of W
    // For W matrix structure:
    Vec4 d_qgoal_W_w_dq = Vec4::Zero();
    {
        // The W matrix is 4x3. We compute how (q_goal^T * W * w) changes with each q component.
        // W(q) is linear in q, so we can compute its derivatives directly.
        // W = 0.5 * [[-q(1), -q(2), -q(3)],
        //            [q(0), -q(3), q(2)],
        //            [q(3), q(0), -q(1)],
        //            [-q(2), q(1), q(0)]]  (in quaternion [q0, q1, q2, q3] form)
        // Each row i of W depends on q as: W_i = 0.5 * (linear combination of q components)
        // ∂(W^T * q_goal)/∂q_j gives a 3D vector
        // We need to compute how (q_goal^T * W * w) changes
        
        // Use numerical differentiation for simplicity and robustness
        const double eps = 1e-8;
        Vec4 q_pert = q;
        double f0 = (q_goal.transpose() * W * w)(0);
        
        for (int j = 0; j < 4; ++j) {
            q_pert = q;
            q_pert(j) += eps;
            q_pert.normalize();
            const Mat43 W_pert = saltro::math::findWMat(q_pert);
            const double f_pert = (q_goal.transpose() * W_pert * w)(0);
            d_qgoal_W_w_dq(j) = (f_pert - f0) / eps;
        }
    }
    lx.segment<4>(QUAT_INDEX) += -sign_qdot * w_avang * d_qgoal_W_w_dq;

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

    // Apply quaternion normalization Jacobian projection
    {
        const double q_norm = q.norm();
        if (q_norm > 1e-12) {
            using Mat44 = Eigen::Matrix<double, 4, 4>;
            const Mat44 proj_q = Mat44::Identity() - q * q.transpose() / (q_norm * q_norm);
            lx.segment<4>(QUAT_INDEX) = proj_q * lx.segment<4>(QUAT_INDEX);
        }
    }

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
    }

    // Reshape lu as a 1 × nu matrix for return type compatibility
    MatX Lu_mat = lu.transpose();
    
    return std::make_tuple(lx, Lu_mat, lux);
}

std::tuple<Satellite::VecX, Satellite::MatX, Satellite::MatX> Satellite::terminalCostJacobians(
    const VecX& x, const Vec3& sat_direction, const Vec4& eci_target,
    const Vec3& B_eci, const CostConfig& cost_cfg) const {
    const VecX u_zero = VecX::Zero(controlDim());
    return stageCostJacobians(0, 1, x, u_zero, sat_direction, eci_target, B_eci, cost_cfg);
}

std::tuple<Satellite::MatX, Satellite::MatX, Satellite::MatX> Satellite::stageCostHessians(
    int k, int N, const VecX& x, const VecX& u,
    const Vec3& sat_direction, const Vec4& eci_target,
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

    (void)sat_direction;

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

    (void)w_ang;  // Unused in numerical Hessian, kept for consistency
    (void)w_avmag;
    (void)w_avang;
    (void)w;
    (void)B_eci;

    // =====================================================================
    // Hessian w.r.t. angular velocity: ∂²L/∂w²
    // =====================================================================
    // From 0.5 * w_av * ‖w‖² → ∂²L/∂w² = w_av * I_3
    lxx.block<3, 3>(AV_INDEX, AV_INDEX) += w_av * Mat33::Identity();

    // =====================================================================
    // Hessian w.r.t. quaternion: ∂²L/∂q² (attitude terms)
    // =====================================================================
    // This is complex and depends on ang_cost_func_type. 
    // For robustness, use numerical differentiation of the gradient.
    {
        const double eps = 1e-8;
        VecX x_pert = x;
        auto lx_base = stageCostJacobians(k, N, x, u, sat_direction, eci_target, B_eci, cost_cfg);
        VecX grad_base = std::get<0>(lx_base);

        for (int j = 0; j < 4; ++j) {
            x_pert = x;
            x_pert(QUAT_INDEX + j) += eps;
            auto lx_pert = stageCostJacobians(k, N, x_pert, u, sat_direction, eci_target, B_eci, cost_cfg);
            VecX grad_pert = std::get<0>(lx_pert);

            VecX d_grad = (grad_pert - grad_base) / eps;
            lxx.col(QUAT_INDEX + j) += d_grad;
        }
    }

    // =====================================================================
    // Hessian w.r.t. RW momentum: ∂²L/∂h²
    // =====================================================================
    for (int i = 0; i < num_rw_; ++i) {
        const double h = x(RW_MOMENTUM_INDEX + i);
        const double z = std::abs(h);
        const double sign_h = safeSign(h);
        const double h_max = std::max(1e-9, std::abs(getRW(i).momentumMax()));

        // Angular momentum saturation penalty
        const double h_thresh = std::clamp(cost_cfg.RWh_max_mult, 0.0, 1.0) * h_max;
        const double denom_high = std::max(1e-9, h_max - h_thresh);
        if (z > h_thresh) {
            const double over = (z - h_thresh) / denom_high;
            const double d2_over_dz2 = 1.0 / (denom_high * denom_high);
            lxx(RW_MOMENTUM_INDEX + i, RW_MOMENTUM_INDEX + i) += 
                cost_cfg.rw_AM_weight * d2_over_dz2;
        } else {
            const double scaled = z / h_max;
            const double d2_scaled_dz2 = 1.0 / (h_max * h_max);
            lxx(RW_MOMENTUM_INDEX + i, RW_MOMENTUM_INDEX + i) += 
                cost_cfg.rw_AM_weight * cost_cfg.RWh_ok_mult * d2_scaled_dz2;
        }

        // Stiction penalty second derivative
        const double h_stic = std::clamp(cost_cfg.RWh_stiction_mult, 0.0, 1.0) * h_max;
        if (h_stic > 1e-12 && z < h_stic) {
            const double near_zero = (h_stic - z) / h_stic;
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
    }

    // =====================================================================
    // Apply Quaternion Normalization Projection
    // =====================================================================
    // For normalized quaternions, gradients lie in the constraint manifold.
    // Hessians must be projected: H_proj = (I - q·q^T) H (I - q·q^T)
    {
        const double q_norm = q.norm();
        if (q_norm > 1e-12) {
            using Mat44 = Eigen::Matrix<double, 4, 4>;
            const Mat44 proj_q = Mat44::Identity() - q * q.transpose() / (q_norm * q_norm);
            // Left projection
            lxx.block(QUAT_INDEX, 0, 4, nx) = proj_q * lxx.block(QUAT_INDEX, 0, 4, nx);
            // Right projection
            lxx.block(0, QUAT_INDEX, nx, 4) = lxx.block(0, QUAT_INDEX, nx, 4) * proj_q;
        }
    }

    return std::make_tuple(lxx, luu, lux);
}

std::tuple<Satellite::MatX, Satellite::MatX, Satellite::MatX> Satellite::terminalCostHessians(
    const VecX& x, const Vec3& sat_direction, const Vec4& eci_target,
    const Vec3& B_eci, const CostConfig& cost_cfg) const {
    const VecX u_zero = VecX::Zero(controlDim());
    return stageCostHessians(0, 1, x, u_zero, sat_direction, eci_target, B_eci, cost_cfg);
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
    const int n_constraints = 1 + 1 + (has_control_constraints ? (2 * num_mtq_ + 5 * num_rw_) : 0);
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
    const int n_constraints = 1 + 1 + (has_control_constraints ? (2 * num_mtq_ + 5 * num_rw_) : 0);
    
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

    // 2) Sun constraint Hessian: computed numerically to match normalization-aware Jacobian
    const double sun_norm = sun_eci.norm();
    const int sun_idx = idx;
    if (std::isfinite(sun_norm) && sun_norm > 1e-12) {
        const double eps = 1e-7;
        for (int j = 0; j < stateDim(); ++j) {
            VecX xp = x; xp(j) += eps;
            VecX xm = x; xm(j) -= eps;
            if (j >= QUAT_INDEX && j < QUAT_INDEX + 4) {
                Vec4 qxp = xp.segment<4>(QUAT_INDEX);
                Vec4 qxm = xm.segment<4>(QUAT_INDEX);
                qxp.normalize();
                qxm.normalize();
                xp.segment<4>(QUAT_INDEX) = qxp;
                xm.segment<4>(QUAT_INDEX) = qxm;
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

    return std::make_tuple(H_uu, H_ux, H_xx);
}

