#include <saltro/math/quaternion.h>
#include <cmath>
#include <stdexcept>

namespace saltro::math {

Vec4 normalizeQuat(const Vec4& q) {
    double norm = q.norm();
    if (norm < 1e-12) {
        throw std::runtime_error("Quaternion norm is too small to normalize");
    }
    return q / norm;
}

Mat33 rotationMatrix(const Vec4& q) {
    Vec4 qn = normalizeQuat(q);
    double q0 = qn(0), q1 = qn(1), q2 = qn(2), q3 = qn(3);
    
    Mat33 R;
    R(0, 0) = 1.0 - 2.0 * (q2 * q2 + q3 * q3);
    R(0, 1) = 2.0 * (q1 * q2 - q0 * q3);
    R(0, 2) = 2.0 * (q1 * q3 + q0 * q2);
    
    R(1, 0) = 2.0 * (q1 * q2 + q0 * q3);
    R(1, 1) = 1.0 - 2.0 * (q1 * q1 + q3 * q3);
    R(1, 2) = 2.0 * (q2 * q3 - q0 * q1);
    
    R(2, 0) = 2.0 * (q1 * q3 - q0 * q2);
    R(2, 1) = 2.0 * (q2 * q3 + q0 * q1);
    R(2, 2) = 1.0 - 2.0 * (q1 * q1 + q2 * q2);
    
    return R;
}

Mat43 findWMat(const Vec4& q) {
    Mat43 W;
    W(0, 0) = -q(1);  W(0, 1) = -q(2);  W(0, 2) = -q(3);
    W(1, 0) =  q(0);  W(1, 1) = -q(3);  W(1, 2) =  q(2);
    W(2, 0) =  q(3);  W(2, 1) =  q(0);  W(2, 2) = -q(1);
    W(3, 0) = -q(2);  W(3, 1) =  q(1);  W(3, 2) =  q(0);
    return W;
}

Mat43 quatNormJacobian(const Vec4& q) {
    double qn = q.norm();
    if (qn < 1e-12) {
        throw std::runtime_error("Quaternion norm is too small");
    }
    
    double qn3 = qn * qn * qn;
    Mat43 J = Mat43::Zero();
    
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (i < 3) {
                J(i, j) = (i == j ? 1.0 : 0.0) / qn - q(i) * q(j) / qn3;
            } else {
                J(i, j) = -q(i) * q(j) / qn3;
            }
        }
    }
    
    return J;
}

Mat33 skewSymmetric(const Vec3& v) {
    Mat33 S;
    S << 0.0, -v(2), v(1),
         v(2), 0.0, -v(0),
         -v(1), v(0), 0.0;
    return S;
}

}
