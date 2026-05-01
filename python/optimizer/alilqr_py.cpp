// PYBIND_DEPENDS: plannersettings satellite

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <stdexcept>
#include <string>

#include <saltro/optimizer/alilqr.h>

namespace py = pybind11;

namespace {

const char* status_to_string(saltro::optimizer::ALILQRStatus status)
{
    using saltro::optimizer::ALILQRStatus;
    switch (status) {
        case ALILQRStatus::Converged:
            return "converged";
        case ALILQRStatus::MaxOuterIterations:
            return "max_outer_iterations";
        case ALILQRStatus::InnerFailed:
            return "inner_failed";
        case ALILQRStatus::PenaltyMaxReached:
            return "penalty_max_reached";
        default:
            return "unknown";
    }
}

py::tuple alilqr_py(
    const PlannerSettings& settings,
    int pass_idx,
    const Satellite& satellite,
    const Eigen::Ref<const Eigen::MatrixXd>& X_in,
    const Eigen::Ref<const Eigen::MatrixXd>& U_in,
    const Eigen::Ref<const Eigen::MatrixXd>& R,
    const Eigen::Ref<const Eigen::MatrixXd>& V,
    const Eigen::Ref<const Eigen::MatrixXd>& B,
    const Eigen::Ref<const Eigen::MatrixXd>& S,
    const Eigen::Ref<const Eigen::MatrixXd>& rho,
    const Eigen::Ref<const Eigen::VectorXd>& jtime,
    const Eigen::Ref<const Eigen::MatrixXd>& boresight,
    const Eigen::Ref<const Eigen::MatrixXd>& attitude_target
)
{
    if (pass_idx < 0 || pass_idx >= settings.num_passes) {
        throw std::runtime_error("pass_idx out of range");
    }

    Eigen::MatrixXd X = X_in;
    Eigen::MatrixXd U = U_in;
    saltro::optimizer::ALILQRStatus status = saltro::optimizer::ALILQRStatus::MaxOuterIterations;
    double max_c = 0.0;

    const bool ok = saltro::optimizer::alilqr(
        settings,
        pass_idx,
        satellite,
        X,
        U,
        R,
        V,
        B,
        S,
        rho,
        jtime,
        boresight,
        attitude_target,
        status,
        max_c
    );

    return py::make_tuple(ok, X, U, std::string(status_to_string(status)), max_c);
}

} // namespace

void bind_alilqr(py::module_& m)
{
    m.def(
        "alilqr",
        &alilqr_py,
        py::arg("settings"),
        py::arg("pass_idx"),
        py::arg("satellite"),
        py::arg("X"),
        py::arg("U"),
        py::arg("R"),
        py::arg("V"),
        py::arg("B"),
        py::arg("S"),
        py::arg("rho"),
        py::arg("jtime"),
        py::arg("boresight"),
        py::arg("attitude_target"),
        R"doc(
Run one AL-iLQR outer solve on a given trajectory guess.

Returns
-------
ok : bool
X : ndarray
U : ndarray
status : str
max_constraint_violation : float
)doc"
    );
}
