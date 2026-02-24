#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

#include <saltro/limits.h>
#include <saltro/orbit_generation/eclipse/compute_eclipse.h>

namespace py = pybind11;

py::tuple compute_eclipse_py(
    const Eigen::Ref<const Eigen::Matrix<double,3,Eigen::Dynamic>>& R_in,
    const Eigen::Ref<const Eigen::RowVectorXd>& jtime_in,
    const Eigen::Ref<const Eigen::Matrix<double,3,Eigen::Dynamic>>& S_in,
    const int eclipse_model
)
{
    const int N = jtime_in.size();

    if (N <= 0)
        throw std::runtime_error("jtime must have length > 0");

    if (R_in.cols() != N)
        throw std::runtime_error("R and jtime must have same length");

    if (S_in.cols() != N)
        throw std::runtime_error("S and jtime must have same length");

    if (N > saltro::limits::MAX_LENGTH_TRAJ)
        throw std::runtime_error("Trajectory exceeds MAX_LENGTH_TRAJ");

    Eigen::Matrix<double,3,saltro::limits::MAX_LENGTH_TRAJ> R;
    Eigen::Matrix<double,1,saltro::limits::MAX_LENGTH_TRAJ> jtime;
    Eigen::Matrix<double,3,saltro::limits::MAX_LENGTH_TRAJ> S;

    R.leftCols(N) = R_in;
    jtime.leftCols(N) = jtime_in;
    S.leftCols(N) = S_in;

    const bool ok = saltro::orbits::compute_eclipse(
        R,
        jtime,
        N,
        eclipse_model,
        S
    );

    Eigen::MatrixXd Sout = S.leftCols(N);

    return py::make_tuple(ok, Sout);
}

void bind_compute_eclipse(py::module_& m)
{
    m.def(
        "compute_eclipse",
        &compute_eclipse_py,
        py::arg("R"),
        py::arg("jtime"),
        py::arg("S"),
        py::arg("eclipse_model"),
        R"doc(
Compute Earth eclipse shadow along trajectory.

When satellite is in eclipse, the Sun vector is zeroed out.

Parameters
----------
R : ndarray (3,N)
    Position vectors (meters)
jtime : ndarray (N,)
    Julian time values
S : ndarray (3,N)
    Spacecraft-to-Sun vectors (meters). Modified in-place where eclipsed.
eclipse_model : int
    0 = cylindrical shadow model
    1 = analytical penumbra/umbra shadow cone model

Returns
-------
ok : bool
S : ndarray (3,N)
    Modified Sun vectors (zeroed where eclipsed)
)doc"
    );
}
