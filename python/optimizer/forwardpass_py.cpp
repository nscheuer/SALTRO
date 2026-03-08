#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <saltro/optimizer/forwardpass.h>

namespace py = pybind11;
using namespace saltro::optimizer;

static py::tuple forward_pass_py(
    const Satellite& satellite,
    const Eigen::Ref<const Eigen::MatrixXd>& X_in,
    const Eigen::Ref<const Eigen::MatrixXd>& U_in,
    const std::vector<Eigen::MatrixXd>& K,
    const std::vector<Eigen::VectorXd>& d,
    const Eigen::Ref<const Eigen::Vector2d>& deltaV,
    const Eigen::Ref<const Eigen::MatrixXd>& B,
    const Eigen::Ref<const Eigen::MatrixXd>& R,
    const Eigen::Ref<const Eigen::MatrixXd>& V,
    const Eigen::Ref<const Eigen::MatrixXd>& S,
    const Eigen::Ref<const Eigen::MatrixXd>& rho,
    const Eigen::Ref<const Eigen::MatrixXd>& boresight,
    const Eigen::Ref<const Eigen::MatrixXd>& attitude_target,
    const PlannerSettings& settings,
    const Eigen::Ref<const Eigen::VectorXd>& jtime,
    double J_prev
)
{
    const int nx = static_cast<int>(X_in.rows());
    const int N = static_cast<int>(X_in.cols());
    const int nu = static_cast<int>(U_in.rows());

    if (N <= 0) {
        throw std::runtime_error("X must have at least one column (N>0)");
    }
    if (U_in.cols() < std::max(0, N - 1)) {
        throw std::runtime_error("U must have at least N-1 columns");
    }
    if (static_cast<int>(K.size()) != std::max(0, N - 1)) {
        throw std::runtime_error("K must have size N-1");
    }
    if (static_cast<int>(d.size()) != std::max(0, N - 1)) {
        throw std::runtime_error("d must have size N-1");
    }
    if (B.cols() != N || R.cols() != N || V.cols() != N || S.cols() != N || rho.cols() != N || boresight.cols() != N || attitude_target.cols() != N) {
        throw std::runtime_error("Environment matrices must have N columns");
    }
    if (B.rows() != 3 || R.rows() != 3 || V.rows() != 3 || S.rows() != 3 || boresight.rows() != 3 || rho.rows() != 1 || attitude_target.rows() != 4) {
        throw std::runtime_error("Environment matrices must have shapes (3,N), rho (1,N), attitude_target (4,N)");
    }
    if (jtime.size() != N) {
        throw std::runtime_error("jtime must have length N");
    }

    Eigen::MatrixXd X = X_in;
    Eigen::MatrixXd U = U_in;
    double J_new = J_prev;

    const bool ok = forwardPass(
        satellite,
        X,
        U,
        K,
        d,
        deltaV,
        B,
        R,
        V,
        S,
        rho,
        boresight,
        attitude_target,
        settings,
        jtime,
        J_prev,
        J_new
    );

    return py::make_tuple(ok, X, U, J_new);
}

void bind_forwardpass(py::module_& m)
{
    m.def(
        "forward_pass",
        &forward_pass_py,
        py::arg("satellite"),
        py::arg("X"),
        py::arg("U"),
        py::arg("K"),
        py::arg("d"),
        py::arg("deltaV"),
        py::arg("B"),
        py::arg("R"),
        py::arg("V"),
        py::arg("S"),
        py::arg("rho"),
        py::arg("boresight"),
        py::arg("attitude_target"),
        py::arg("settings"),
        py::arg("jtime"),
        py::arg("J_prev"),
        R"doc(
Forward pass for iLQR with backtracking line search.
Uses MRP (Modified Rodrigues Parameters) for attitude error computation.

Parameters
----------
satellite : Satellite
X : ndarray (nx, N)
U : ndarray (nu, N) or (nu, N-1)
K : list/array of (nu x nxr) gains in reduced state space, length N-1
d : list/array of (nu,) feedforward terms, length N-1
deltaV : ndarray (2,) expected cost change terms
B, R, V, S : ndarray (3, N)
rho : ndarray (1, N)
boresight : ndarray (3, N)
attitude_target : ndarray (4, N)
settings : PlannerSettings
jtime : ndarray (N,)
J_prev : float previous trajectory cost

Returns
-------
ok : bool
X_out : ndarray (nx, N)
U_out : ndarray (nu, N)
J_new : float
)doc"
    );
}
