// PYBIND_DEPENDS: plannersettings satellite

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <algorithm>
#include <stdexcept>

#include <saltro/limits.h>
#include <saltro/optimizer/trajOpt.h>

namespace py = pybind11;

py::tuple trajOpt_py(
	const PlannerSettings& settings,
	const Satellite& satellite,
	const Eigen::Ref<const Eigen::VectorXd>& x0_in,
	const Eigen::Vector3d& r0,
	const Eigen::Vector3d& v0,
	const Eigen::Ref<const Eigen::VectorXd>& jtime,
	const Eigen::Ref<const Eigen::MatrixXd>& q_goal,
	const Eigen::Ref<const Eigen::MatrixXd>& boresight
)
{
	const int N = static_cast<int>(jtime.size());

	if (N <= 0)
		throw std::runtime_error("jtime must have length > 0");

	if (N > saltro::limits::MAX_LENGTH_TRAJ)
		throw std::runtime_error("Trajectory exceeds MAX_LENGTH_TRAJ");

	if (q_goal.cols() != N)
		throw std::runtime_error("q_goal must have N columns matching jtime length");

	if (q_goal.rows() != 4)
		throw std::runtime_error("q_goal must have shape (4, N)");

	if (boresight.cols() != N)
		throw std::runtime_error("boresight must have N columns matching jtime length");

	if (boresight.rows() != 3)
		throw std::runtime_error("boresight must have shape (3, N)");

	const int state_dim = satellite.stateDim();
	const int input_dim = satellite.controlDim();

	if (x0_in.size() != state_dim)
		throw std::runtime_error("x0 must have size satellite.stateDim()");

	if (state_dim > saltro::limits::MAX_STATE_DIM)
		throw std::runtime_error("state_dim exceeds MAX_STATE_DIM");

	if (input_dim > saltro::limits::MAX_CTRL_DIM)
		throw std::runtime_error("input_dim exceeds MAX_CTRL_DIM");

	Satellite::VecX x0(state_dim);
	x0 = x0_in;

	Eigen::MatrixXd X(state_dim, saltro::limits::MAX_LENGTH_TRAJ);
	Eigen::MatrixXd U(input_dim, saltro::limits::MAX_LENGTH_TRAJ);
	Eigen::MatrixXd K(input_dim, state_dim * saltro::limits::MAX_LENGTH_TRAJ);
	int N_out = N;

	const bool ok = saltro::optimizer::trajOpt(
		settings,
		satellite,
		x0,
		r0,
		v0,
		jtime,
		q_goal,
		boresight,
		X,
		U,
		K,
		state_dim,
		input_dim,
		N_out
	);

	const int N_cols = std::max(0, std::min(N_out, saltro::limits::MAX_LENGTH_TRAJ));
	return py::make_tuple(
		ok,
		X.leftCols(N_cols),
		U.leftCols(N_cols),
		K.leftCols(state_dim * N_cols)
	);
}

void bind_trajOpt(py::module_& m)
{
	m.def(
		"trajOpt",
		&trajOpt_py,
		py::arg("settings"),
		py::arg("satellite"),
		py::arg("x0"),
		py::arg("r0"),
		py::arg("v0"),
		py::arg("jtime"),
		py::arg("q_goal"),
		py::arg("boresight"),
		R"doc(
Run trajectory optimization.

Parameters
----------
settings : PlannerSettings
satellite : Satellite
x0 : ndarray (state_dim,)
	Initial state vector
r0 : ndarray (3,)
	Initial ECI position
v0 : ndarray (3,)
	Initial ECI velocity
jtime : ndarray (N,)
	Julian times
q_goal : ndarray (4,N)
	Goal quaternion sequence
boresight : ndarray (3,N)
	Boresight direction sequence

Returns
-------
ok : bool
X : ndarray (state_dim,N)
U : ndarray (input_dim,N)
K : ndarray (input_dim,state_dim*N)
)doc"
	);
}
