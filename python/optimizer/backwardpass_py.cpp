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
    double reg,
    bool return_quu
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

    std::vector<Eigen::MatrixXd> Q_uu_hist, Quu_ddp_hist;
    std::vector<Eigen::MatrixXd>* Q_uu_ptr   = return_quu ? &Q_uu_hist   : nullptr;
    std::vector<Eigen::MatrixXd>* Quu_ddp_ptr = return_quu ? &Quu_ddp_hist : nullptr;

    const bool ok = backwardPass(
        satellite, X, U, R, V, B, S, rho, boresight, attitude_target,
        settings, reg, K, d, deltaV, lambda_aug, mu_aug,
        Q_uu_ptr, Quu_ddp_ptr
    );

    // Stack K into shape (N-1, nu, nxr) — reduced state columns
    py::array_t<double> K_arr({std::max(0, N - 1), nu, nxr});
    auto K_buf = K_arr.mutable_unchecked<3>();
    for (int k = 0; k < std::max(0, N - 1); ++k) {
        for (int i = 0; i < nu; ++i) {
            for (int j = 0; j < nxr; ++j) {
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

    if (!return_quu) {
        return py::make_tuple(ok, K_arr, d_arr, deltaV_arr);
    }

    // Stack Q_uu and Quu_ddp into (N-1, nu, nu) numpy arrays.
    auto stack_nu_nu = [&](const std::vector<Eigen::MatrixXd>& v) {
        py::array_t<double> arr({std::max(0, N - 1), nu, nu});
        auto buf = arr.mutable_unchecked<3>();
        for (int k = 0; k < std::max(0, N - 1); ++k) {
            for (int i = 0; i < nu; ++i) {
                for (int j = 0; j < nu; ++j) {
                    buf(k, i, j) = v[static_cast<std::size_t>(k)](i, j);
                }
            }
        }
        return arr;
    };
    py::array_t<double> Q_uu_arr   = stack_nu_nu(Q_uu_hist);
    py::array_t<double> Quu_ddp_arr = stack_nu_nu(Quu_ddp_hist);
    return py::make_tuple(ok, K_arr, d_arr, deltaV_arr, Q_uu_arr, Quu_ddp_arr);
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
        py::arg("return_quu") = false,
        R"doc(
Backward pass for iLQR using reduced state (MRP) representation.

Parameters
----------
return_quu : bool, default False
    When True, also return per-knot Q_uu and Quu_ddp histories.

Returns
-------
If `return_quu=False` (default):
    (ok, K, d, deltaV)
If `return_quu=True`:
    (ok, K, d, deltaV, Q_uu, Quu_ddp)

ok : bool
K : ndarray (N-1, control_dim, reduced_state_dim)
    Feedback gains in reduced state space (6 + num_rw columns).
d : ndarray (control_dim, N-1)
deltaV : ndarray (2,)
Q_uu : ndarray (N-1, control_dim, control_dim)
    Symmetrized un-regularized Q_uu per knot. Diagnostic output.
Quu_ddp : ndarray (N-1, control_dim, control_dim)
    DDP curvature contribution to Q_uu per knot (zero when
    use_dynamics_hess=False). Diagnostic output.
)doc"
    );
}
