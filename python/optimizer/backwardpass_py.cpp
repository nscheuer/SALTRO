#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <saltro/optimizer/backwardpass.h>

namespace py = pybind11;
using namespace saltro::optimizer;

static py::tuple backward_pass_py(
    const Satellite& satellite,
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const Eigen::Ref<const Eigen::MatrixXd>& U,
    const Eigen::Ref<const Eigen::MatrixXd>& R,
    const Eigen::Ref<const Eigen::MatrixXd>& V,
    const Eigen::Ref<const Eigen::MatrixXd>& B,
    const Eigen::Ref<const Eigen::MatrixXd>& S,
    const Eigen::Ref<const Eigen::MatrixXd>& rho,
    const Eigen::Ref<const Eigen::MatrixXd>& boresight,
    const Eigen::Ref<const Eigen::MatrixXd>& attitude_target,
    const PlannerSettings& settings
)
{
    const int N = static_cast<int>(X.cols());
    const int nx = static_cast<int>(X.rows());
    const int nu = static_cast<int>(U.rows());

    // Allocate outputs
    std::vector<Eigen::MatrixXd> K(std::max(0, N - 1), Eigen::MatrixXd::Zero(nu, nx));
    std::vector<Eigen::VectorXd> d(std::max(0, N - 1), Eigen::VectorXd::Zero(nu));
    Eigen::Vector2d deltaV = Eigen::Vector2d::Zero();

    const bool ok = backwardPass(
        satellite, X, U, R, V, B, S, rho, boresight, attitude_target,
        settings, K, d, deltaV
    );

    // Stack K into shape (nu, nx, N-1)
    py::array_t<double> K_arr({std::max(0, N - 1), nu, nx});
    auto K_buf = K_arr.mutable_unchecked<3>();
    for (int k = 0; k < std::max(0, N - 1); ++k) {
        for (int i = 0; i < nu; ++i) {
            for (int j = 0; j < nx; ++j) {
                K_buf(k, i, j) = K[k](i, j);
            }
        }
    }

    // Stack d into shape (nu, N-1)
    py::array_t<double> d_arr({nu, std::max(0, N - 1)});
    auto d_buf = d_arr.mutable_unchecked<2>();
    for (int k = 0; k < std::max(0, N - 1); ++k) {
        for (int i = 0; i < nu; ++i) {
            d_buf(i, k) = d[k](i);
        }
    }

    py::array_t<double> deltaV_arr({2});
    auto deltaV_buf = deltaV_arr.mutable_unchecked<1>();
    deltaV_buf(0) = deltaV(0);
    deltaV_buf(1) = deltaV(1);

    return py::make_tuple(ok, K_arr, d_arr, deltaV_arr);
}

void bind_backwardpass(py::module_& m)
{
    m.def(
        "backward_pass",
        &backward_pass_py,
        py::arg("satellite"),
        py::arg("X"),
        py::arg("U"),
        py::arg("R"),
        py::arg("V"),
        py::arg("B"),
        py::arg("S"),
        py::arg("rho"),
        py::arg("boresight"),
        py::arg("attitude_target"),
        py::arg("settings"),
        R"doc(
Backward pass for iLQR.

Returns
-------
ok : bool
K : ndarray (N-1, control_dim, state_dim)
d : ndarray (control_dim, N-1)
deltaV : ndarray (2,)
attitude_target : ndarray (4, N)
)doc"
    );
}
