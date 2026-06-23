#include <saltro/math/matrix.h>

namespace saltro::math {

void psd_clip(Eigen::MatrixXd& M) {
	const Eigen::MatrixXd Msym = 0.5 * (M + Eigen::MatrixXd(M.transpose()));
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Msym);
	const Eigen::VectorXd lam = es.eigenvalues().cwiseMax(0.0);
	M = es.eigenvectors() * lam.asDiagonal() * es.eigenvectors().transpose();
}

}  // namespace saltro::math
