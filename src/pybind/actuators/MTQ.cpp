#include <saltro/pybind/actuators/MTQ.h>

MTQ::MTQ(const Vec3 &axis, double max_dipole) : Actuator(axis, max_dipole) {}

MTQ::Vec3 MTQ::torque(double u, const BaseState& x, const Vec3& B_body) const {
    (void)x;
    return -(B_body.cross(axis_)) * u;
}

MTQ::Mat13 MTQ::dtorq_du(double u, const BaseState& x, const Vec3& B_body) const {
    (void)u;
    (void)x;
    Mat13 J;
    J = -(B_body.cross(axis_)).transpose();
    return J;
}

MTQ::Mat73 MTQ::dtorq_dbasestate(double u, const BaseState&, const Vec3&, const Eigen::Matrix<double,4,3>& dB_dq) const {
    Mat73 J = Mat73::Zero();

    for (int i = 0; i < 4; ++i) {
        Eigen::Vector3d row = -(dB_dq.row(i).transpose().cross(axis_)) * u;
        J.row(3 + i) = row.transpose();
    }

    return J;
}

MTQ::T113 MTQ::ddtorq_dudu(double, const BaseState&, const Vec3&) const {
    return T113::Zero();
}

MTQ::T173 MTQ::ddtorq_dudbasestate(double, const BaseState&, const Vec3&, const Eigen::Matrix<double,4,3>& dB_dq) const {
    T173 H = T173::Zero();

    for (int k = 0; k < 3; ++k) {
        for (int i = 0; i < 4; ++i) {
            Eigen::Vector3d dB_dqi = dB_dq.row(i).transpose();
            double dtau_k = -(dB_dqi.cross(axis_))[k];
            H.slice(k)(0, 3 + i) = dtau_k;
        }
    }

    return H;
}

MTQ::T773 MTQ::ddtorq_dbasestatedbasestate(double u, const BaseState&, const Eigen::Matrix<double,4,3>&, const std::array<Eigen::Matrix<double,4,4>,3>& d2B_dq2) const {
    T773 H = T773::Zero();

    for (int k = 0; k < 3; ++k) {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                double val = -(d2B_dq2[k](i,j)) * u;
                H.slice(k)(3+i, 3+j) = val;
            }
        }
    }

    return H;
}