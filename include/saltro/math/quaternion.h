#pragma once

#include <Eigen/Dense>

namespace saltro::math {

using Vec3 = Eigen::Vector3d;
using Vec4 = Eigen::Vector4d;
using Mat33 = Eigen::Matrix3d;
using Mat43 = Eigen::Matrix<double, 4, 3>;

Vec4 normalizeQuat(const Vec4& q);

Mat33 rotationMatrix(const Vec4& q);

Mat43 findWMat(const Vec4& q);

Mat43 quatNormJacobian(const Vec4& q);

Mat33 skewSymmetric(const Vec3& v);

}
