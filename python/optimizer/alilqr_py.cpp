// PYBIND_DEPENDS: plannersettings satellite

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

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
        case ALILQRStatus::MaxTotalIterations:
            return "max_total_iterations";
        case ALILQRStatus::PenaltyMaxReached:
            return "penalty_max_reached";
        case ALILQRStatus::InnerFailed:
            return "inner_failed";
        default:
            return "unknown";
    }
}

const char* ilqr_status_to_string(saltro::optimizer::ILQRStatus status)
{
    using saltro::optimizer::ILQRStatus;
    switch (status) {
        case ILQRStatus::Converged:
            return "converged";
        case ILQRStatus::MaxIterations:
            return "max_iterations";
        case ILQRStatus::RegularizationExceeded:
            return "regularization_exceeded";
        case ILQRStatus::Stalled:
            return "stalled";
        default:
            return "unknown";
    }
}

const char* break_reason_to_string(saltro::optimizer::ILQRBreakReason reason)
{
    using saltro::optimizer::ILQRBreakReason;
    switch (reason) {
        case ILQRBreakReason::None:
            return "none";
        case ILQRBreakReason::GradientIntermediate:
            return "gradient_intermediate";
        case ILQRBreakReason::CostIntermediate:
            return "cost_intermediate";
        case ILQRBreakReason::RelCostIntermediate:
            return "rel_cost_intermediate";
        case ILQRBreakReason::GradientStationary:
            return "gradient_stationary";
        case ILQRBreakReason::StrictConjunction:
            return "strict_conjunction";
        case ILQRBreakReason::Stalled:
            return "stalled";
        case ILQRBreakReason::MaxIterations:
            return "max_iterations";
        case ILQRBreakReason::RegularizationExceeded:
            return "regularization_exceeded";
        default:
            return "unknown";
    }
}

const char* converged_via_to_string(saltro::optimizer::ALConvergedVia via)
{
    using saltro::optimizer::ALConvergedVia;
    switch (via) {
        case ALConvergedVia::None:
            return "none";
        case ALConvergedVia::FastPath:
            return "fast_path";
        case ALConvergedVia::Settled:
            return "settled";
        default:
            return "unknown";
    }
}

py::dict telemetry_to_dict(const saltro::optimizer::ALILQRTelemetry& telemetry)
{
    py::dict d;
    d["converged_via"] = converged_via_to_string(telemetry.converged_via);
    d["total_inner_iterations"] = telemetry.total_inner_iterations;
    d["max_c_family"] = telemetry.max_c_family;
    d["nominal_cost"] = telemetry.nominal_cost;
    d["penalty_cost"] = telemetry.penalty_cost;

    py::list outer;
    for (const auto& rec : telemetry.outer) {
        py::dict r;
        r["settle"] = rec.settle;
        r["inner_status"] = ilqr_status_to_string(rec.inner_status);
        r["break_reason"] = break_reason_to_string(rec.inner_break_reason);
        r["inner_iterations"] = rec.inner_iterations;
        r["accepted_steps"] = rec.accepted_steps;
        r["max_c"] = rec.max_c;
        r["final_grad"] = rec.final_grad;
        r["last_delta_J"] = rec.last_delta_J;
        r["min_delta_J"] = rec.min_delta_J;
        outer.append(r);
    }
    d["outer"] = outer;
    return d;
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
    saltro::optimizer::ALILQRTelemetry telemetry;

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
        max_c,
        telemetry
    );

    return py::make_tuple(
        ok, X, U, std::string(status_to_string(status)), max_c, telemetry_to_dict(telemetry));
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
    One of {converged, max_outer_iterations, max_total_iterations,
    penalty_max_reached, inner_failed}.
max_constraint_violation : float
telemetry : dict
    converged_via : str  -- {none, fast_path, settled}
    total_inner_iterations : int
    max_c_family : list[float]  -- final per-family max violation
    nominal_cost : float  -- satellite cost of the returned trajectory
    penalty_cost : float  -- AL penalty share at the final lambda/mu
    outer : list[dict]  -- per-outer-iteration records with keys
        {settle, inner_status, break_reason, inner_iterations,
         accepted_steps, max_c}
)doc"
    );
}
