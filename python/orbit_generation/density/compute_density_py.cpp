#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

#include <saltro/limits.h>
#include <saltro/orbit_generation/density/compute_density.h>

namespace py = pybind11;

py::tuple compute_density_py(
    const Eigen::Ref<const Eigen::Matrix<double,3,Eigen::Dynamic>>& R_in,
    const Eigen::Ref<const Eigen::Matrix<double,3,Eigen::Dynamic>>& S_in,
    const int density_model
)
{
    const int N = R_in.cols();

    if (N <= 0)
        throw std::runtime_error("R must have at least one column");

    if (S_in.cols() != N)
        throw std::runtime_error("R and S must have same length");

    if (N > saltro::limits::MAX_LENGTH_TRAJ)
        throw std::runtime_error("Trajectory exceeds MAX_LENGTH_TRAJ");

    Eigen::Matrix<double,3,saltro::limits::MAX_LENGTH_TRAJ> R;
    Eigen::Matrix<double,3,saltro::limits::MAX_LENGTH_TRAJ> S;
    Eigen::Matrix<double,1,saltro::limits::MAX_LENGTH_TRAJ> rho;

    R.leftCols(N) = R_in;
    S.leftCols(N) = S_in;

    const bool ok = saltro::orbits::compute_density(
        R,
        S,
        N,
        density_model,
        rho
    );

    Eigen::RowVectorXd rho_out = rho.leftCols(N);

    return py::make_tuple(ok, rho_out);
}

void bind_compute_density(py::module_& m)
{
    m.def(
        "compute_density",
        &compute_density_py,
        py::arg("R"),
        py::arg("S"),
        py::arg("density_model"),
        R"doc(
Compute atmospheric density along trajectory.

Parameters
----------
R : ndarray (3,N)
    Spacecraft position vectors
S : ndarray (3,N)
    Sun vectors relative to spacecraft
density_model : int
    Density model selector

Returns
-------
ok : bool
rho : ndarray (N,)
)doc"
    );
}