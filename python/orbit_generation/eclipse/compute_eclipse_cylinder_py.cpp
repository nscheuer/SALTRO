#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <Eigen/Dense>
#include <saltro/limits.h>
#include <saltro/orbit_generation/eclipse/compute_eclipse_cylinder.h>

namespace py = pybind11;

py::tuple compute_eclipse_cylinder_py(
    const Eigen::Ref<const Eigen::Matrix<double,3,Eigen::Dynamic>>& R_in,
    const Eigen::Ref<const Eigen::Matrix<double,3,Eigen::Dynamic>>& S_in
)
{
    const int N = R_in.cols();

    if (R_in.rows() != 3)
        throw std::runtime_error("R must be shape (3, N)");

    if (S_in.rows() != 3)
        throw std::runtime_error("S must be shape (3, N)");

    if (S_in.cols() != N)
        throw std::runtime_error("R and S must have same length");

    if (N > saltro::limits::MAX_LENGTH_TRAJ)
        throw std::runtime_error("Trajectory exceeds MAX_LENGTH_TRAJ");

    // --- fixed-size buffers ---
    Eigen::Matrix<double,3,saltro::limits::MAX_LENGTH_TRAJ> R;
    Eigen::Matrix<double,3,saltro::limits::MAX_LENGTH_TRAJ> S;

    // copy into fixed buffers
    R.leftCols(N) = R_in;
    S.leftCols(N) = S_in;

    const bool ok = saltro::orbits::compute_eclipse_cylinder(
        R,
        N,
        S
    );

    Eigen::MatrixXd Sout = S.leftCols(N);

    return py::make_tuple(ok, Sout);
}

void bind_compute_eclipse_cylinder(py::module_& m)
{
    m.def(
        "compute_eclipse_cylinder",
        &compute_eclipse_cylinder_py,
        py::arg("R"),
        py::arg("S"),
        R"doc(
Compute Earth eclipse shadow using cylindrical shadow model.

The cylindrical shadow model approximates Earth's shadow as a cylinder
perpendicular to the Sun direction. When a satellite is in eclipse
(umbra), the Sun vector is zeroed out.

Parameters
----------
R : ndarray (3,N)
    Position vectors (meters)
S : ndarray (3,N)
    Spacecraft-to-Sun vectors (meters)

Returns
-------
ok : bool
S : ndarray (3,N)
    Modified Sun vectors (zeroed where eclipsed)
)doc"
    );
}
