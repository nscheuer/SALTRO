#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

#include <saltro/limits.h>
#include <saltro/orbit_generation/orbits/compute_orbit_keplerian.h>

namespace py = pybind11;

bool compute_orbit_keplerian_py(
    const Eigen::Vector3d& r0,
    const Eigen::Vector3d& v0,
    const Eigen::Ref<
        const Eigen::Matrix<double,1,saltro::limits::MAX_LENGTH_TRAJ>
    >& jtime_ref,
    int jtime_length,
    Eigen::Ref<
        Eigen::Matrix<double,3,saltro::limits::MAX_LENGTH_TRAJ>
    > R_ref,
    Eigen::Ref<
        Eigen::Matrix<double,3,saltro::limits::MAX_LENGTH_TRAJ>
    > V_ref
)
{
    Eigen::Matrix<double,1,saltro::limits::MAX_LENGTH_TRAJ> jtime = jtime_ref;

    Eigen::Matrix<double,3,saltro::limits::MAX_LENGTH_TRAJ> R;
    Eigen::Matrix<double,3,saltro::limits::MAX_LENGTH_TRAJ> V;

    bool ok = saltro::orbit::compute_orbit_keplerian(
        r0,
        v0,
        jtime,
        jtime_length,
        R,
        V
    );

    R_ref = R;
    V_ref = V;

    return ok;
}

void bind_compute_orbit(py::module_& m)
{
    m.def(
        "compute_orbit_keplerian",
        &compute_orbit_keplerian_py,
        py::arg("r0"),
        py::arg("v0"),
        py::arg("jtime"),
        py::arg("jtime_length"),
        py::arg("R"),
        py::arg("V")
    );
}