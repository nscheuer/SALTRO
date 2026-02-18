#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

#include <saltro/limits.h>
#include <saltro/orbit_generation/orbits/compute_orbit_keplerian.h>

namespace py = pybind11;

py::tuple compute_orbit_keplerian_py(
    const Eigen::Vector3d& r0,
    const Eigen::Vector3d& v0,
    const Eigen::Ref<const Eigen::RowVectorXd>& jtime_in
)
{
    const int N = jtime_in.size();

    if (N <= 0)
        throw std::runtime_error("jtime must have length > 0");

    if (N > saltro::limits::MAX_LENGTH_TRAJ)
        throw std::runtime_error("Trajectory exceeds MAX_LENGTH_TRAJ");

    // ---- fixed flight buffers ----
    Eigen::Matrix<double,1,saltro::limits::MAX_LENGTH_TRAJ> jtime;
    Eigen::Matrix<double,3,saltro::limits::MAX_LENGTH_TRAJ> R;
    Eigen::Matrix<double,3,saltro::limits::MAX_LENGTH_TRAJ> V;

    jtime.leftCols(N) = jtime_in;

    const bool ok = saltro::orbit::compute_orbit_keplerian(
        r0,
        v0,
        jtime,
        N,
        R,
        V
    );

    Eigen::MatrixXd Rout = R.leftCols(N);
    Eigen::MatrixXd Vout = V.leftCols(N);

    return py::make_tuple(ok, Rout, Vout);
}

void bind_compute_orbit(py::module_& m)
{
    m.def(
        "compute_orbit_keplerian",
        &compute_orbit_keplerian_py,
        py::arg("r0"),
        py::arg("v0"),
        py::arg("jtime"),
        R"doc(
Propagate orbit using Keplerian model.

Parameters
----------
r0 : ndarray (3,)
    Initial position
v0 : ndarray (3,)
    Initial velocity
jtime : ndarray (N,)
    Julian times

Returns
-------
ok : bool
R : ndarray (3,N)
V : ndarray (3,N)
)doc"
    );
}