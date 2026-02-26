#include <saltro/pybind/satellite.h>
#include <saltro/pybind/disturbances/dragdisturbance.h>
#include <saltro/pybind/disturbances/ggdisturbance.h>
#include <saltro/pybind/disturbances/srpdisturbance.h>

#include <stdexcept>
#include <cmath>
#include <algorithm>

using std::invalid_argument;
using std::out_of_range;

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
    Vec7 x_base = x.head<7>();
    
    Vec4 q = x.segment<4>(QUAT_INDEX);
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
    Vec7 x_base = x.head<7>();
    Vec4 q = x.segment<4>(QUAT_INDEX);

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
    Vec4 q = x.segment<4>(QUAT_INDEX);
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
    const Vec4 q = x.segment<4>(QUAT_INDEX);
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
        const Vec4 q = x.segment<4>(QUAT_INDEX);
        const Vec3 sun_unit = sun_eci / sun_norm;
        
        // Derivative of (R^T * sun) w.r.t. q is a 4×3 matrix
        const Mat34 dRTsun_dq = saltro::math::drotmatTvecdq(q, sun_unit).transpose();
        
        // We only care about the x-component, so take first row
        c_x.block<1, 4>(idx, QUAT_INDEX) = dRTsun_dq.row(0);
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

    // 2) Sun constraint Hessian: ∂²/∂q² of (R(q)^T * sun).x - cos(limit)
    const double sun_norm = sun_eci.norm();
    if (std::isfinite(sun_norm) && sun_norm > 1e-12) {
        const Vec4 q = x.segment<4>(QUAT_INDEX);
        const Vec3 sun_unit = sun_eci / sun_norm;
        
        // Second derivative of (R^T * sun) w.r.t. q
        const auto d2RTsun_dq2 = saltro::math::ddrotmatTvecdqdq(q, sun_unit);
        
        // Only x-component matters
        H_xx.slice(idx).block<4, 4>(QUAT_INDEX, QUAT_INDEX) = d2RTsun_dq2[0];
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

