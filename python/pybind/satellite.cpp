#include <saltro/pybind/satellite.h>

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

void Satellite::setSettings(const PlannerSettings& settings) {
    settings_ = settings;
}

const PlannerSettings& Satellite::settings() const {
    return settings_;
}

