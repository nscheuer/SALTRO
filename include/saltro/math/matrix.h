#pragma once

#include <Eigen/Dense>

namespace saltro::math {

void psd_clip(Eigen::MatrixXd& M);

}  // namespace saltro::math
