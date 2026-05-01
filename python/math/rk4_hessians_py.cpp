// PYBIND_DEPENDS: satellite
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <array>
#include <vector>
#include <Eigen/Dense>

#include <saltro/math/integrators/rk4.h>
#include <saltro/math/quaternion.h>
#include <saltro/pybind/satellite.h>

namespace py = pybind11;
using namespace saltro;

namespace {

/**
 * Expose rk4_hessians to Python by binding a fixed satellite callback.
 * Returns (F_xx, F_ux, F_uu) as lists of numpy arrays, one per output
 * equation.
 */
py::tuple rk4_hessians_py(
    const Satellite& satellite,
    const Eigen::Ref<const Eigen::VectorXd>& x,
    const Eigen::Ref<const Eigen::VectorXd>& u,
    const Eigen::Ref<const Eigen::Vector3d>& R_eci,
    const Eigen::Ref<const Eigen::Vector3d>& B_eci,
    const Eigen::Ref<const Eigen::Vector3d>& S_eci,
    const Eigen::Ref<const Eigen::Vector3d>& V_eci,
    double dt
) {
    DisturbanceConfig dist;  // all disturbances default-off
    const int nx = static_cast<int>(x.size());
    const int nu = static_cast<int>(u.size());

    auto dyn_hess_wrapper = [&](double /*t*/,
                                const Eigen::Ref<const Eigen::VectorXd>& x_local,
                                const Eigen::Ref<const Eigen::VectorXd>& u_local,
                                Eigen::Ref<Eigen::MatrixXd> A_out,
                                Eigen::Ref<Eigen::MatrixXd> B_out,
                                Eigen::Ref<Eigen::VectorXd> k_out,
                                std::vector<Eigen::MatrixXd>& fxx_out,
                                std::vector<Eigen::MatrixXd>& fux_out,
                                std::vector<Eigen::MatrixXd>& fuu_out) {
        auto [A_c, B_c, C_unused] = satellite.dynamicsJacobians(
            x_local, u_local, dist, R_eci, B_eci, S_eci, V_eci);
        A_out = A_c;
        B_out = B_c;
        k_out = satellite.dynamics(x_local, u_local, dist,
                                   R_eci, B_eci, S_eci, V_eci, 0);

        auto [hxx, hux, huu] = satellite.dynamicsHessians(
            x_local, u_local, dist, R_eci, B_eci, S_eci, V_eci);
        const int nx_l = static_cast<int>(x_local.size());
        const int nu_l = static_cast<int>(u_local.size());
        fxx_out.assign(static_cast<std::size_t>(nx_l),
                       Eigen::MatrixXd::Zero(nx_l, nx_l));
        fux_out.assign(static_cast<std::size_t>(nx_l),
                       Eigen::MatrixXd::Zero(nu_l, nx_l));
        fuu_out.assign(static_cast<std::size_t>(nx_l),
                       Eigen::MatrixXd::Zero(nu_l, nu_l));
        for (int l = 0; l < nx_l; ++l) {
            fxx_out[static_cast<std::size_t>(l)] =
                hxx.slice(l).topLeftCorner(nx_l, nx_l);
            fux_out[static_cast<std::size_t>(l)] =
                hux.slice(l).topLeftCorner(nu_l, nx_l);
            fuu_out[static_cast<std::size_t>(l)] =
                huu.slice(l).topLeftCorner(nu_l, nu_l);
        }
    };

    std::vector<Eigen::MatrixXd> F_xx, F_ux, F_uu;
    saltro::math::rk4_hessians(dyn_hess_wrapper, x, u, 0.0, dt, F_xx, F_ux, F_uu);

    py::list xx, ux, uu;
    for (const auto& m : F_xx) xx.append(m);
    for (const auto& m : F_ux) ux.append(m);
    for (const auto& m : F_uu) uu.append(m);
    return py::make_tuple(xx, ux, uu);
}

/**
 * One RK4 discrete step driven by satellite dynamics. Useful for Python
 * finite-difference checks against rk4_hessians_discrete.
 */
Eigen::VectorXd rk4_step_py(
    const Satellite& satellite,
    const Eigen::Ref<const Eigen::VectorXd>& x,
    const Eigen::Ref<const Eigen::VectorXd>& u,
    const Eigen::Ref<const Eigen::Vector3d>& R_eci,
    const Eigen::Ref<const Eigen::Vector3d>& B_eci,
    const Eigen::Ref<const Eigen::Vector3d>& S_eci,
    const Eigen::Ref<const Eigen::Vector3d>& V_eci,
    double dt
) {
    DisturbanceConfig dist;
    auto step_rhs = [&](double /*t*/, const Eigen::VectorXd& xk, Eigen::VectorXd& dx) {
        dx = satellite.dynamics(xk, u, dist, R_eci, B_eci, S_eci, V_eci, 0);
    };
    Eigen::VectorXd x_next(x.size());
    Eigen::VectorXd x_norm = x;
    x_norm.segment<4>(3).normalize();
    saltro::math::rk4_step<Eigen::VectorXd>(step_rhs, x_norm, 0.0, dt, x_next);
    x_next.segment<4>(3).normalize();
    return x_next;
}

// State-normalization Jacobian / Hessian bindings — used by Python unit
// tests to FD-check the closed-form formulas in rk4.h. Not needed for
// production runs.
Eigen::MatrixXd stateNormJacobian_py(
    const Eigen::Ref<const Eigen::VectorXd>& x,
    int quat_idx
) {
    return saltro::math::stateNormJacobian(x, static_cast<int>(x.size()), quat_idx);
}

py::list stateNormHessian_py(
    const Eigen::Ref<const Eigen::VectorXd>& x,
    int quat_idx
) {
    auto NH = saltro::math::stateNormHessian(x, static_cast<int>(x.size()), quat_idx);
    py::list out;
    for (const auto& M : NH) out.append(M);
    return out;
}

// Quaternion rotation-derivative bindings — for FD tests of Bug 3 (ddrotmatTvecdqdq)
// and infrastructure validation. Not needed for production.
Eigen::Matrix<double, 4, 3> drotmatTvecdq_py(const Eigen::Vector4d& q, const Eigen::Vector3d& v) {
    return saltro::math::drotmatTvecdq(q, v);
}

py::list ddrotmatTvecdqdq_py(const Eigen::Vector4d& q, const Eigen::Vector3d& v) {
    std::array<Eigen::Matrix<double, 4, 4>, 3> H = saltro::math::ddrotmatTvecdqdq(q, v);
    py::list out;
    for (const auto& Hi : H) {
        out.append(Hi);
    }
    return out;
}

Eigen::Vector3d rotmatTvec_py(const Eigen::Vector4d& q, const Eigen::Vector3d& v) {
    // R^T v where R = rotationMatrix(q). The rotation in body frame.
    return saltro::math::rotationMatrix(q).transpose() * v;
}

/**
 * Non-normalizing R^T v using the raw quaternion formula:
 *   R(q)^T v = (q0² - |qv|²) v + 2 (qv·v) qv + 2 q0 (qv × v)
 * (Hamilton convention, world→body i.e. R^T.)
 *
 * This matches the derivative formulas that `drotmatTvecdq` and
 * `ddrotmatTvecdqdq` compute with respect to the RAW 4-vector q.
 * `rotationMatrix(q)` normalizes q internally, so direct FD against it would
 * project out any perturbation along q itself — producing wrong FD for tests.
 * Exposed for unit-test use.
 */
Eigen::Vector3d rotmatTvec_raw_py(const Eigen::Vector4d& q, const Eigen::Vector3d& v) {
    const double q0 = q(0);
    const Eigen::Vector3d qv = q.tail<3>();
    // R^T v for Hamilton-convention quaternion; sign of cross term is NEGATIVE
    // (R = (q0²-|qv|²) I + 2 qv qv^T + 2 q0 skew(qv), so R^T has -2 q0 skew(qv)).
    return (q0 * q0 - qv.squaredNorm()) * v
         + 2.0 * qv.dot(v) * qv
         - 2.0 * q0 * qv.cross(v);
}

}  // namespace

void bind_rk4_hessians(py::module_& m) {
    m.def("stateNormJacobian", &stateNormJacobian_py,
          py::arg("x"), py::arg("quat_idx") = 3,
          "Jacobian of state normalization (quaternion-block normalize). Returns "
          "(nx, nx) with identity outside the quaternion block and "
          "(I/r - q q^T / r^3) on the quaternion 4x4 block, r = ||q||.");
    m.def("stateNormHessian", &stateNormHessian_py,
          py::arg("x"), py::arg("quat_idx") = 3,
          "Hessian of state normalization. Returns nx slices of (nx, nx); only "
          "slices for the four quaternion outputs have a nonzero quaternion-block.");

    m.def("rotmatTvec", &rotmatTvec_py, py::arg("q"), py::arg("v"),
          "Return R(q)^T v. Uses the library rotationMatrix() which normalizes q.");
    m.def("rotmatTvec_raw", &rotmatTvec_raw_py, py::arg("q"), py::arg("v"),
          "Return R(q)^T v using the raw (non-normalizing) quaternion formula. "
          "Use this for FD cross-checks against drotmatTvecdq / ddrotmatTvecdqdq, "
          "which differentiate w.r.t. the raw 4-vector q.");
    m.def("drotmatTvecdq", &drotmatTvecdq_py, py::arg("q"), py::arg("v"),
          "Jacobian of R(q)^T v with respect to q. Returns 4x3.");
    m.def("ddrotmatTvecdqdq", &ddrotmatTvecdqdq_py, py::arg("q"), py::arg("v"),
          "Hessian tensor of R(q)^T v with respect to q. Returns list of 3 "
          "4x4 matrices, one per output component.");

    m.def("rk4_step_discrete", &rk4_step_py,
          py::arg("satellite"),
          py::arg("x"),
          py::arg("u"),
          py::arg("R_eci"),
          py::arg("B_eci"),
          py::arg("S_eci"),
          py::arg("V_eci"),
          py::arg("dt"),
          R"doc(
Compute one RK4 step for satellite dynamics. Primarily for FD cross-checks.
)doc");

    m.def("rk4_hessians_discrete", &rk4_hessians_py,
          py::arg("satellite"),
          py::arg("x"),
          py::arg("u"),
          py::arg("R_eci"),
          py::arg("B_eci"),
          py::arg("S_eci"),
          py::arg("V_eci"),
          py::arg("dt"),
          R"doc(
Compute discrete-time dynamics Hessians through a single RK4 step.

Returns (F_xx, F_ux, F_uu) as lists of numpy arrays where F_*[l] is the
Hessian of the l-th output equation:
  F_xx[l] : (nx, nx) — ∂²x_{k+1}_l / ∂x² at (x_k, u_k)
  F_ux[l] : (nu, nx) — ∂²x_{k+1}_l / ∂u∂x
  F_uu[l] : (nu, nu) — ∂²x_{k+1}_l / ∂u²

Disturbances are treated as off for the linearization (matches
backward-pass behavior).
)doc");
}
