#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <saltro/math/integrators/rk4.h>
#include <saltro/pybind/satellite.h>

namespace py = pybind11;

namespace {

using MatrixVector = std::vector<Eigen::MatrixXd>;

auto makeJacobianCallback(
    const Satellite& satellite,
    const DisturbanceConfig& disturbances,
    const Eigen::Vector3d& R,
    const Eigen::Vector3d& B,
    const Eigen::Vector3d& S,
    const Eigen::Vector3d& V
) {
    return [&](double /*t*/,
               const Eigen::Ref<const Eigen::VectorXd>& x,
               const Eigen::Ref<const Eigen::VectorXd>& u,
               Eigen::Ref<Eigen::MatrixXd> A,
               Eigen::Ref<Eigen::MatrixXd> B_u,
               Eigen::Ref<Eigen::VectorXd> f) {
        auto [A_c, B_c, disturbance_jacobian] = satellite.dynamicsJacobians(
            x, u, disturbances, R, B, S, V);
        (void)disturbance_jacobian;
        A = A_c;
        B_u = B_c;
        f = satellite.dynamics(x, u, disturbances, R, B, S, V, 0);
    };
}

}  // namespace

void bind_rk4(py::module_& m) {
    m.def(
        "rk4_dynamics_jacobians",
        [](const Satellite& satellite,
           const Eigen::Ref<const Eigen::VectorXd>& x,
           const Eigen::Ref<const Eigen::VectorXd>& u,
           double dt,
           const DisturbanceConfig& disturbances,
           const Eigen::Vector3d& R,
           const Eigen::Vector3d& B,
           const Eigen::Vector3d& S,
           const Eigen::Vector3d& V) {
            Eigen::MatrixXd A = Eigen::MatrixXd::Zero(x.size(), x.size());
            Eigen::MatrixXd B_u = Eigen::MatrixXd::Zero(x.size(), u.size());
            auto callback = makeJacobianCallback(satellite, disturbances, R, B, S, V);
            rk4_jacobians(callback, x, u, 0.0, dt, A, B_u);
            return py::make_tuple(A, B_u);
        },
        py::arg("satellite"), py::arg("x"), py::arg("u"), py::arg("dt"),
        py::arg("disturbances"), py::arg("R"), py::arg("B"), py::arg("S"), py::arg("V"),
        "Compute the discrete RK4 dynamics Jacobians using the C++ integrator."
    );

    m.def(
        "rk4_dynamics_hessians",
        [](const Satellite& satellite,
           const Eigen::Ref<const Eigen::VectorXd>& x,
           const Eigen::Ref<const Eigen::VectorXd>& u,
           double dt,
           const DisturbanceConfig& disturbances,
           const Eigen::Vector3d& R,
           const Eigen::Vector3d& B,
           const Eigen::Vector3d& S,
           const Eigen::Vector3d& V) {
            auto callback = [&](double /*t*/,
                                const Eigen::Ref<const Eigen::VectorXd>& x_local,
                                const Eigen::Ref<const Eigen::VectorXd>& u_local,
                                Eigen::Ref<Eigen::MatrixXd> A,
                                Eigen::Ref<Eigen::MatrixXd> B_u,
                                Eigen::Ref<Eigen::VectorXd> f,
                                MatrixVector& f_xx,
                                MatrixVector& f_ux,
                                MatrixVector& f_uu) {
                auto [A_c, B_c, disturbance_jacobian] = satellite.dynamicsJacobians(
                    x_local, u_local, disturbances, R, B, S, V);
                (void)disturbance_jacobian;
                A = A_c;
                B_u = B_c;
                f = satellite.dynamics(x_local, u_local, disturbances, R, B, S, V, 0);

                auto [H_xx, H_ux, H_uu] = satellite.dynamicsHessians(
                    x_local, u_local, disturbances, R, B, S, V);
                const int nx = static_cast<int>(x_local.size());
                const int nu = static_cast<int>(u_local.size());
                f_xx.assign(static_cast<std::size_t>(nx), Eigen::MatrixXd::Zero(nx, nx));
                f_ux.assign(static_cast<std::size_t>(nx), Eigen::MatrixXd::Zero(nu, nx));
                f_uu.assign(static_cast<std::size_t>(nx), Eigen::MatrixXd::Zero(nu, nu));
                for (int output = 0; output < nx; ++output) {
                    f_xx[static_cast<std::size_t>(output)] = H_xx.slice(output).topLeftCorner(nx, nx);
                    f_ux[static_cast<std::size_t>(output)] = H_ux.slice(output).topLeftCorner(nu, nx);
                    f_uu[static_cast<std::size_t>(output)] = H_uu.slice(output).topLeftCorner(nu, nu);
                }
            };

            MatrixVector F_xx;
            MatrixVector F_ux;
            MatrixVector F_uu;
            rk4_hessians(callback, x, u, 0.0, dt, F_xx, F_ux, F_uu);
            return py::make_tuple(F_xx, F_ux, F_uu);
        },
        py::arg("satellite"), py::arg("x"), py::arg("u"), py::arg("dt"),
        py::arg("disturbances"), py::arg("R"), py::arg("B"), py::arg("S"), py::arg("V"),
        "Compute the discrete RK4 dynamics Hessians using the C++ integrator."
    );
}
