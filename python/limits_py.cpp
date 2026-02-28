#include <pybind11/pybind11.h>
#include <saltro/limits.h>

namespace py = pybind11;

void bind_limits(py::module_& m)
{
    // Create submodule: saltro_py.limits
    py::module_ limits = m.def_submodule(
        "limits",
        "SALTRO compile-time limits and constants"
    );

    // ---- trajectory limits ----
    limits.attr("MAX_LENGTH_TRAJ") = saltro::limits::MAX_LENGTH_TRAJ;

    // ---- kepler solver ----
    limits.attr("KEPLER_MAX_ITERS") = saltro::limits::KEPLER_MAX_ITERS;
    limits.attr("KEPLER_TOLERANCE") = saltro::limits::KEPLER_TOLERANCE;

    // ---- hardware limits ----
    limits.attr("MAX_NUM_PASSES") = saltro::limits::MAX_NUM_PASSES;
    limits.attr("MAX_NUM_MTQ")    = saltro::limits::MAX_NUM_MTQ;
    limits.attr("MAX_NUM_RW")     = saltro::limits::MAX_NUM_RW;
    limits.attr("MAX_NUM_GEOMETRY_FACES") = saltro::limits::MAX_NUM_GEOMETRY_FACES;
}