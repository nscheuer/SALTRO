#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

#include <saltro/limits.h>
#include <saltro/orbit_generation/magnetics/compute_magnetic.h>

namespace py = pybind11;

py::tuple compute_magnetic_py(
    const Eigen::Ref<const Eigen::Matrix<double,3,Eigen::Dynamic>>& R_in,
    const Eigen::Ref<const Eigen::RowVectorXd>& jtime_in,
    const int magnetic_model
)
{
    const int N = jtime_in.size();

    if (N <= 0)
        throw std::runtime_error("jtime must have length > 0");

    if (R_in.cols() != N)
        throw std::runtime_error("R and jtime must have same length");

    if (N > saltro::limits::MAX_LENGTH_TRAJ)
        throw std::runtime_error("Trajectory exceeds MAX_LENGTH_TRAJ");

    Eigen::Matrix<double,3,saltro::limits::MAX_LENGTH_TRAJ> R;
    Eigen::Matrix<double,1,saltro::limits::MAX_LENGTH_TRAJ> jtime;
    Eigen::Matrix<double,3,saltro::limits::MAX_LENGTH_TRAJ> B;

    R.leftCols(N) = R_in;
    jtime.leftCols(N) = jtime_in;

    const bool ok = saltro::orbits::compute_magnetic(
        R,
        jtime,
        N,
        magnetic_model,
        B
    );

    Eigen::MatrixXd Bout = B.leftCols(N);

    return py::make_tuple(ok, Bout);
}

void bind_compute_magnetic(py::module_& m)
{
    m.def(
        "compute_magnetic",
        &compute_magnetic_py,
        py::arg("R"),
        py::arg("jtime"),
        py::arg("magnetic_model"),
        R"doc(
Compute magnetic field along trajectory.

Parameters
----------
R : ndarray (3,N)
    Position vectors
jtime : ndarray (N,)
    Julian times
magnetic_model : int
    0 = tilted dipole
    1 = IGRF8
    2 = IGRF13

Returns
-------
ok : bool
B : ndarray (3,N)
)doc"
    );
}