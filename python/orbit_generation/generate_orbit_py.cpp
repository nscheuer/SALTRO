#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

#include <saltro/limits.h>
#include <saltro/orbit_generation/generate_orbit.h>

namespace py = pybind11;

py::tuple generate_orbit_py(
    const Eigen::Vector3d& r0,
    const Eigen::Vector3d& v0,
    const Eigen::Ref<const Eigen::RowVectorXd>& jtime_in,
    const int orbit_model,
    const int magnetic_model,
    const int sun_model,
    const int density_model
)
{
    const int N = jtime_in.size();

    if (N <= 0)
        throw std::runtime_error("jtime must have length > 0");

    if (N > saltro::limits::MAX_LENGTH_TRAJ)
        throw std::runtime_error("Trajectory exceeds MAX_LENGTH_TRAJ");

    Eigen::Matrix<double,1,saltro::limits::MAX_LENGTH_TRAJ> jtime;
    Eigen::Matrix<double,3,saltro::limits::MAX_LENGTH_TRAJ> R;
    Eigen::Matrix<double,3,saltro::limits::MAX_LENGTH_TRAJ> V;
    Eigen::Matrix<double,3,saltro::limits::MAX_LENGTH_TRAJ> B;
    Eigen::Matrix<double,3,saltro::limits::MAX_LENGTH_TRAJ> S;
    Eigen::Matrix<double,1,saltro::limits::MAX_LENGTH_TRAJ> rho;

    jtime.leftCols(N) = jtime_in;

    const bool ok = saltro::orbits::generate_orbit(
        r0,
        v0,
        jtime,
        N,
        orbit_model,
        magnetic_model,
        sun_model,
        density_model,
        R,
        V,
        B,
        S,
        rho
    );

    Eigen::MatrixXd Rout = R.leftCols(N);
    Eigen::MatrixXd Vout = V.leftCols(N);
    Eigen::MatrixXd Bout = B.leftCols(N);
    Eigen::MatrixXd Sout = S.leftCols(N);
    Eigen::RowVectorXd rho_out = rho.leftCols(N);

    return py::make_tuple(ok, Rout, Vout, Bout, Sout, rho_out);
}

void bind_generate_orbit(py::module_& m)
{
    m.def(
        "generate_orbit",
        &generate_orbit_py,
        py::arg("r0"),
        py::arg("v0"),
        py::arg("jtime"),
        py::arg("orbit_model"),
        py::arg("magnetic_model"),
        py::arg("sun_model"),
        py::arg("density_model"),
        R"doc(
Generate orbit and environmental data along trajectory.

Parameters
----------
r0 : ndarray (3,)
    Initial position
v0 : ndarray (3,)
    Initial velocity
jtime : ndarray (N,)
    Julian times
orbit_model : int
magnetic_model : int
sun_model : int
density_model : int

Returns
-------
ok : bool
R : ndarray (3,N)
V : ndarray (3,N)
B : ndarray (3,N)
S : ndarray (3,N)
rho : ndarray (N,)
)doc"
    );
}