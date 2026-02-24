#include <saltro/pybind/satellite.h>
#include <saltro/pybind/disturbances/dragdisturbance.h>
#include <saltro/pybind/disturbances/ggdisturbance.h>

#include <stdexcept>
#include <cmath>

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
    Mat34 dV_dq = saltro::math::drotmatTvecdq(q, V_eci).transpose();
    auto d2V_dq2 = saltro::math::ddrotmatTvecdqdq(q, V_eci);
    (void)B_eci;
    (void)S_eci;
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

