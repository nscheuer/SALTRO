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
    const PlannerSettings& settings,
    const std::vector<Eigen::VectorXd>& lambda_aug,
    const std::vector<Eigen::VectorXd>& mu_aug,
    double reg
)
{
    const int N = static_cast<int>(X.cols());
    const int nx = static_cast<int>(X.rows());
    const int nu = static_cast<int>(U.rows());
    const int nxr = satellite.reducedStateDim(); // 6 + nRW

    // Allocate outputs in reduced state space: K is nu × nxr
    std::vector<Eigen::MatrixXd> K(std::max(0, N - 1), Eigen::MatrixXd::Zero(nu, nxr));
    std::vector<Eigen::VectorXd> d(std::max(0, N - 1), Eigen::VectorXd::Zero(nu));
    Eigen::Vector2d deltaV = Eigen::Vector2d::Zero();

    // Disturbance-aware (eq. 7.40): when settings.tvlqr.disturbance_aware is
    // set (mirroring how trajOpt requests K_dist), also compute the K_tau
    // gains and return [K_x | K_tau] of width nxr + 3.
    const bool dist_aware = settings.tvlqr.disturbance_aware;
    std::vector<Eigen::MatrixXd> K_dist(
        static_cast<std::size_t>(dist_aware ? std::max(0, N - 1) : 0),
        Eigen::MatrixXd::Zero(nu, 3));

    const bool ok = backwardPass(
        satellite, X, U, R, V, B, S, rho, boresight, attitude_target,
        settings, reg, K, d, deltaV, lambda_aug, mu_aug,
        dist_aware ? &K_dist : nullptr
    );

    // Stack K into shape (N-1, nu, nxr) — reduced state columns — or
    // (N-1, nu, nxr + 3) with the K_tau block appended when disturbance-aware.
    const int gain_w = dist_aware ? nxr + 3 : nxr;
    py::array_t<double> K_arr({std::max(0, N - 1), nu, gain_w});
    auto K_buf = K_arr.mutable_unchecked<3>();
    for (int k = 0; k < std::max(0, N - 1); ++k) {
        for (int i = 0; i < nu; ++i) {
            for (int j = 0; j < nxr; ++j) {
                K_buf(k, i, j) = K[k](i, j);
            }
            for (int j = 0; dist_aware && j < 3; ++j) {
                K_buf(k, i, nxr + j) = K_dist[static_cast<std::size_t>(k)](i, j);
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
        py::arg("lambda_aug"),
        py::arg("mu_aug"),
        py::arg("reg"),
        R"doc(
Backward pass for iLQR using reduced state (MRP) representation.

Returns
-------
ok : bool
K : ndarray (N-1, control_dim, reduced_state_dim)
    Feedback gains in reduced state space (6 + num_rw columns). When
    settings.tvlqr.disturbance_aware is set, the last axis widens to
    reduced_state_dim + 3 and each block is [K_x | K_tau] (eq. 7.40).
d : ndarray (control_dim, N-1)
deltaV : ndarray (2,)
)doc"
    );
}
