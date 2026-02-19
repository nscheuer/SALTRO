#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

#include <saltro/limits.h>
#include <saltro/orbit_generation/sun/compute_sun_nrelspa.h>

namespace py = pybind11;

py::tuple compute_sun_nrelspa_py(
    const Eigen::Ref<const Eigen::Matrix<double,3,Eigen::Dynamic>>& R_in,
    const Eigen::Ref<const Eigen::RowVectorXd>& jtime_in
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
    Eigen::Matrix<double,3,saltro::limits::MAX_LENGTH_TRAJ> S;

    R.leftCols(N) = R_in;
    jtime.leftCols(N) = jtime_in;

    const bool ok = saltro::orbits::compute_sun_nrelspa(
        R,
        jtime,
        N,
        S
    );

    Eigen::MatrixXd Sout = S.leftCols(N);

    return py::make_tuple(ok, Sout);
}

void bind_compute_sun_nrelspa(py::module_& m)
{
    m.def(
        "compute_sun_nrelspa",
        &compute_sun_nrelspa_py,
        py::arg("R"),
        py::arg("jtime"),
        R"doc(
Compute Sun vector along trajectory using NREL SPA model.

Parameters
----------
R : ndarray (3,N)
    Spacecraft position vectors
jtime : ndarray (N,)
    Julian times

Returns
-------
ok : bool
S : ndarray (3,N)
)doc"
    );
}