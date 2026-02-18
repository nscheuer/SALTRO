#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <Eigen/Dense>
#include <saltro/limits.h>
#include <saltro/orbit_generation/magnetics/compute_magnetic_dipole.h>

namespace py = pybind11;

bool compute_magnetic_dipole_py(
    const Eigen::Ref<
        const Eigen::Matrix<double,3,saltro::limits::MAX_LENGTH_TRAJ>
    >& R_ref,
    const Eigen::Ref<
        const Eigen::Matrix<double,1,saltro::limits::MAX_LENGTH_TRAJ>
    >& jtime_ref,
    int jtime_length,
    Eigen::Ref<
        Eigen::Matrix<double,3,saltro::limits::MAX_LENGTH_TRAJ>
    > B_ref
) {
    Eigen::Matrix<double,3,saltro::limits::MAX_LENGTH_TRAJ> R = R_ref;
    Eigen::Matrix<double,1,saltro::limits::MAX_LENGTH_TRAJ> jtime = jtime_ref;

    Eigen::Matrix<double,3,saltro::limits::MAX_LENGTH_TRAJ> B;

    const bool ok = saltro::orbit::compute_magnetic_dipole(
        R,
        jtime,
        jtime_length,
        B
    );

    B_ref = B;
    return ok;
}

void bind_compute_magnetic_dipole(py::module_& m) {
    m.def(
        "compute_magnetic_dipole",
        &compute_magnetic_dipole_py,
        py::arg("R"),
        py::arg("jtime"),
        py::arg("jtime_length"),
        py::arg("B")
    );
}