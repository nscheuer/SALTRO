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

	Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ> jtime;
	Eigen::Matrix<double, 4, saltro::limits::MAX_LENGTH_TRAJ> q_goal;
	jtime.setZero();
	q_goal.setZero();

	jtime.leftCols(jtime_length) = jtime_in.leftCols(jtime_length);
	q_goal.leftCols(jtime_length) = q_goal_in.leftCols(jtime_length);

	Eigen::Matrix<double, saltro::limits::MAX_STATE_DIM, saltro::limits::MAX_LENGTH_TRAJ> X;
	Eigen::Matrix<double, saltro::limits::MAX_CTRL_DIM, saltro::limits::MAX_LENGTH_TRAJ> U;
	saltro::math::Tensor3<saltro::limits::MAX_CTRL_DIM,
	                     saltro::limits::MAX_STATE_DIM,
	                     saltro::limits::MAX_LENGTH_TRAJ> K;

	const bool ok = saltro::optimizer::trajOpt(
		settings,
		satellite,
		x0,
		r0,
		v0,
		jtime,
		q_goal,
		jtime_length,
		X,
		U,
		K
	);

	const int nx = satellite.stateDim();
	const int nu = satellite.controlDim();
	const int N = jtime_length;
	const int N_u = std::max(0, N - 1);

	Eigen::MatrixXd X_out = X.topRows(nx).leftCols(N);
	Eigen::MatrixXd U_out = U.topRows(nu).leftCols(N_u);

	py::array_t<double> K_out({N_u, nu, nx});
	auto K_buf = K_out.mutable_unchecked<3>();
	for (int k = 0; k < N_u; ++k) {
		const auto& Kk = K.slice(k);
		for (int i = 0; i < nu; ++i) {
			for (int j = 0; j < nx; ++j) {
				K_buf(k, i, j) = Kk(i, j);
			}
		}
	}

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
