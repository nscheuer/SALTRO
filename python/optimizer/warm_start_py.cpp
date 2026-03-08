// PYBIND_DEPENDS: plannersettings satellite

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <stdexcept>

#include <saltro/limits.h>
#include <saltro/optimizer/warm_start.h>

namespace py = pybind11;

py::tuple warm_start_py(
	const PlannerSettings& settings,
	const Satellite& satellite,
	const Eigen::Ref<const Eigen::VectorXd>& x0_in,
	const Eigen::Ref<const Eigen::VectorXd>& jtime,
	const Eigen::Ref<const Eigen::MatrixXd>& q_goal,
	const Eigen::Ref<const Eigen::MatrixXd>& boresight,
	const Eigen::Ref<const Eigen::MatrixXd>& R_in,
	const Eigen::Ref<const Eigen::MatrixXd>& V_in,
	const Eigen::Ref<const Eigen::MatrixXd>& B_in,
	const Eigen::Ref<const Eigen::MatrixXd>& S_in,
	const Eigen::Ref<const Eigen::RowVectorXd>& rho_in
)
{
	const int N = static_cast<int>(jtime.size());

	if (N <= 0)
		throw std::runtime_error("jtime must have length > 0");

	if (N > saltro::limits::MAX_LENGTH_TRAJ)
		throw std::runtime_error("Trajectory exceeds MAX_LENGTH_TRAJ");

	if (q_goal.rows() != 4 || q_goal.cols() != N)
		throw std::runtime_error("q_goal must have shape (4, N)");

	if (boresight.rows() != 3 || boresight.cols() != N)
		throw std::runtime_error("boresight must have shape (3, N)");

	if (R_in.rows() != 3 || R_in.cols() != N)
		throw std::runtime_error("R must have shape (3, N)");

	if (V_in.rows() != 3 || V_in.cols() != N)
		throw std::runtime_error("V must have shape (3, N)");

	if (B_in.rows() != 3 || B_in.cols() != N)
		throw std::runtime_error("B must have shape (3, N)");

	if (S_in.rows() != 3 || S_in.cols() != N)
		throw std::runtime_error("S must have shape (3, N)");

	if (rho_in.size() != N)
		throw std::runtime_error("rho must have shape (N,)");

	const int state_dim = satellite.stateDim();
	const int input_dim = satellite.controlDim();

	if (x0_in.size() != state_dim)
		throw std::runtime_error("x0 must have size satellite.stateDim()");

	Satellite::VecX x0(state_dim);
	x0 = x0_in;

	Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ> R;
	Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ> V;
	Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ> B;
	Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ> S;
	Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ> rho;

	R.leftCols(N) = R_in;
	V.leftCols(N) = V_in;
	B.leftCols(N) = B_in;
	S.leftCols(N) = S_in;
	rho.leftCols(N) = rho_in;

	Eigen::MatrixXd X(state_dim, N);
	Eigen::MatrixXd U(input_dim, N);

	const bool ok = saltro::optimizer::warm_start(
		settings,
		satellite,
		x0,
		jtime,
		q_goal,
		boresight,
		N,
		R,
		V,
		B,
		S,
		rho,
		X,
		U
	);

	return py::make_tuple(ok, X, U);
}

void bind_warm_start(py::module_& m)
{
	m.def(
		"warm_start",
		&warm_start_py,
		py::arg("settings"),
		py::arg("satellite"),
		py::arg("x0"),
		py::arg("jtime"),
		py::arg("q_goal"),
		py::arg("boresight"),
		py::arg("R"),
		py::arg("V"),
		py::arg("B"),
		py::arg("S"),
		py::arg("rho"),
		R"doc(
Generate warm-start state/control trajectories.

Parameters
----------
settings : PlannerSettings
satellite : Satellite
x0 : ndarray (state_dim,)
	Initial state vector
jtime : ndarray (N,)
	Julian times
q_goal : ndarray (4,N)
	Goal quaternion sequence
boresight : ndarray (3,N)
	Boresight direction sequence (body frame)
R : ndarray (3,N)
	Orbit position sequence
V : ndarray (3,N)
	Orbit velocity sequence
B : ndarray (3,N)
	Magnetic field sequence
S : ndarray (3,N)
	Sun vector sequence
rho : ndarray (N,)
	Density sequence

Returns
-------
ok : bool
X : ndarray (state_dim,N)
U : ndarray (input_dim,N)
)doc"
	);
}

