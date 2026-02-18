#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <Eigen/Dense>
#include <saltro/limits.h>
#include <saltro/orbit_generation/magnetics/compute_magnetic_igrf8.h>

namespace py = pybind11;

py::tuple compute_magnetic_igrf8_py(
    const Eigen::Ref<const Eigen::MatrixXd>& R_in,
    const Eigen::Ref<const Eigen::RowVectorXd>& jtime_in
)
{
    const int N = jtime_in.size();

    if (R_in.rows() != 3)
        throw std::runtime_error("R must be shape (3, N)");

    if (R_in.cols() != N)
        throw std::runtime_error("R and jtime length mismatch");

    if (N > saltro::limits::MAX_LENGTH_TRAJ)
        throw std::runtime_error("Trajectory exceeds MAX_LENGTH_TRAJ");

    // --- fixed flight buffers ---
    Eigen::Matrix<double,3,saltro::limits::MAX_LENGTH_TRAJ> R;
    Eigen::Matrix<double,1,saltro::limits::MAX_LENGTH_TRAJ> jtime;
    Eigen::Matrix<double,3,saltro::limits::MAX_LENGTH_TRAJ> B;

    // copy into fixed buffers
    R.leftCols(N) = R_in;
    jtime.leftCols(N) = jtime_in;

    const bool ok = saltro::orbit::compute_magnetic_igrf8(
        R,
        jtime,
        N,
        B
    );

    Eigen::MatrixXd Bout = B.leftCols(N);

    return py::make_tuple(ok, Bout);
}

void bind_compute_magnetic_igrf8(py::module_& m)
{
    m.def(
        "compute_magnetic_igrf8",
        &compute_magnetic_igrf8_py,
        py::arg("R"),
        py::arg("jtime"),
        R"doc(
Compute magnetic field using IGRF8 model.

Parameters
----------
R : ndarray (3,N)
    Position vectors
jtime : ndarray (N,)
    Julian times

Returns
-------
ok : bool
B : ndarray (3,N)
)doc"
    );
}