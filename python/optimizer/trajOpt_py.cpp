// PYBIND_DEPENDS: plannersettings satellite

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include <stdexcept>
#include <algorithm>

#include <saltro/limits.h>
#include <saltro/optimizer/trajOpt.h>

namespace py = pybind11;

py::tuple trajOpt_py(
	const PlannerSettings& settings,
	const Satellite& satellite,
	const Eigen::Ref<const Eigen::VectorXd>& x0_in,
	const Eigen::Vector3d& r0,
	const Eigen::Vector3d& v0,
	const Eigen::Ref<const Eigen::RowVectorXd>& jtime_in,
	const Eigen::Ref<const Eigen::MatrixXd>& q_goal_in,
	int jtime_length
) {
	if (jtime_length <= 0) {
		throw std::runtime_error("jtime_length must be > 0");
	}
	if (jtime_length > saltro::limits::MAX_LENGTH_TRAJ) {
		throw std::runtime_error("jtime_length exceeds MAX_LENGTH_TRAJ");
	}
	if (jtime_in.size() < jtime_length) {
		throw std::runtime_error("jtime has fewer elements than jtime_length");
	}
	if (q_goal_in.rows() != 4) {
		throw std::runtime_error("q_goal must be a 4xN matrix");
	}
	if (q_goal_in.cols() < jtime_length) {
		throw std::runtime_error("q_goal has fewer columns than jtime_length");
	}

	Satellite::VecX x0 = x0_in;

	// Use dynamic-sized matrices with heap allocation to avoid stack overflow
	// These will be mapped to the fixed-size interface expected by C++ trajOpt
	Eigen::Matrix<double, 1, Eigen::Dynamic> jtime_dyn(1, saltro::limits::MAX_LENGTH_TRAJ);
	Eigen::Matrix<double, 4, Eigen::Dynamic> q_goal_dyn(4, saltro::limits::MAX_LENGTH_TRAJ);
	jtime_dyn.setZero();
	q_goal_dyn.setZero();

	jtime_dyn.leftCols(jtime_length) = jtime_in.leftCols(jtime_length);
	q_goal_dyn.leftCols(jtime_length) = q_goal_in.leftCols(jtime_length);

	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> X_dyn(saltro::limits::MAX_STATE_DIM, saltro::limits::MAX_LENGTH_TRAJ);
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> U_dyn(saltro::limits::MAX_CTRL_DIM, saltro::limits::MAX_LENGTH_TRAJ);
	X_dyn.setZero();
	U_dyn.setZero();
	
	// Tensor3 must be heap-allocated due to its large size
	auto K_ptr = new saltro::math::Tensor3<saltro::limits::MAX_CTRL_DIM,
	                                        saltro::limits::MAX_STATE_DIM,
	                                        saltro::limits::MAX_LENGTH_TRAJ>();

	// Map dynamic matrices to fixed-size views for C++ function call
	Eigen::Map<Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ>> jtime_map(jtime_dyn.data());
	Eigen::Map<Eigen::Matrix<double, 4, saltro::limits::MAX_LENGTH_TRAJ>> q_goal_map(q_goal_dyn.data());
	Eigen::Map<Eigen::Matrix<double, saltro::limits::MAX_STATE_DIM, saltro::limits::MAX_LENGTH_TRAJ>> X_map(X_dyn.data());
	Eigen::Map<Eigen::Matrix<double, saltro::limits::MAX_CTRL_DIM, saltro::limits::MAX_LENGTH_TRAJ>> U_map(U_dyn.data());

	const bool ok = saltro::optimizer::trajOpt(
		settings,
		satellite,
		x0,
		r0,
		v0,
		jtime_map,
		q_goal_map,
		jtime_length,
		X_map,
		U_map,
		*K_ptr
	);

	const int nx = satellite.stateDim();
	const int nu = satellite.controlDim();
	const int N = jtime_length;
	const int N_u = std::max(0, N - 1);

	Eigen::MatrixXd X_out = X_dyn.topRows(nx).leftCols(N);
	Eigen::MatrixXd U_out = U_dyn.topRows(nu).leftCols(N_u);

	py::array_t<double> K_out({N_u, nu, nx});
	auto K_buf = K_out.mutable_unchecked<3>();
	for (int k = 0; k < N_u; ++k) {
		for (int i = 0; i < nu; ++i) {
			for (int j = 0; j < nx; ++j) {
				K_buf(k, i, j) = K_ptr->slice(k)(i, j);
			}
		}
	}
	
	delete K_ptr;

	return py::make_tuple(ok, X_out, U_out, K_out);
}

void bind_trajOpt(py::module_& m) {
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
		py::arg("jtime_length")
	);
}
