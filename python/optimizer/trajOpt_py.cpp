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

	const int state_dim = satellite.stateDim();
	const int input_dim = satellite.controlDim();

	Satellite::VecX x0(state_dim);
	if (x0_in.size() == state_dim) {
		x0 = x0_in;
	} else {
		x0.setZero();
		const int n_copy = std::min(state_dim, static_cast<int>(x0_in.size()));
		x0.head(n_copy) = x0_in.head(n_copy);
	}

	Eigen::MatrixXd X(state_dim, saltro::limits::MAX_LENGTH_TRAJ);
	Eigen::MatrixXd U(input_dim, saltro::limits::MAX_LENGTH_TRAJ);
	const int reduced_state_dim = satellite.reducedStateDim();
	Eigen::MatrixXd K(input_dim, reduced_state_dim * saltro::limits::MAX_LENGTH_TRAJ);
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
		K.leftCols(reduced_state_dim * N_cols)
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
