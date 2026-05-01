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

Vec4 quatConj(const Vec4& q) {
    return {q(0), -q(1), -q(2), -q(3)};
}

Vec4 quatMult(const Vec4& q1, const Vec4& q2) {
    // Hamilton product q1 ⊗ q2 = L(q1) · q2, where L(q1) = [q1 | W(q1)].
    // Splitting by scalar/vector part of q2:
    //   L(q1) · q2 = q2(0) · q1 + W(q1) · q2_vec
    return q2(0) * q1 + findWMat(q1) * q2.tail<3>();
}

double quatAngle(const Vec4& q_err) {
    const double w = std::clamp(std::abs(q_err(0)), 0.0, 1.0);
    return 2.0 * std::acos(w);
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

Mat43 drotmatTvecdq(const Vec4& q, const Vec3& v) {
    const double q0 = q(0);
    const Vec3 qv = q.tail<3>();

    Mat43 J;
    const Vec3 top = q0 * v - qv.cross(v);
    J.row(0) = (2.0 * top).transpose();

    const Mat33 bottom = Mat33::Identity() * qv.dot(v)
                         - qv * v.transpose()
                         + v * qv.transpose()
                         - q0 * skewSymmetric(v);
    J.block<3, 3>(1, 0) = 2.0 * bottom;

    return J;
}

std::array<Mat44, 3> ddrotmatTvecdqdq(const Vec4& /*q*/, const Vec3& v) {
    std::array<Mat44, 3> output = {Mat44::Zero(), Mat44::Zero(), Mat44::Zero()};

    Mat43 stack;
    stack.row(0) = v.transpose();
    stack.block<3, 3>(1, 0) = -skewSymmetric(v);

    for (int k = 0; k < 3; ++k) {
        for (int j = 0; j < 4; ++j) {
            const double val = 2.0 * stack(j, k);
            output[static_cast<size_t>(k)](0, j) = val;
            output[static_cast<size_t>(k)](j, 0) = val;
        }
    }

    // q_v × q_v block: ∂²(R^T v)_k / ∂(q_v)_a ∂(q_v)_b.
    // Deriving from R^T v = (q_0² - |q_v|²) v + 2 (q_v · v) q_v - 2 q_0 (q_v × v):
    //   (1) (q_0² - Σ (q_v)_c²) v_k            →  -2 δ_{ab} v_k
    //   (2) 2 (Σ (q_v)_c v_c) (q_v)_k           →  2 v_a δ_{bk} + 2 v_b δ_{ak}
    //   (3) -2 q_0 ε_{kcd} (q_v)_c v_d          →  0 (linear in q_v)
    // Total:  H_k[1+a, 1+b] = -2 δ_{ab} v_k + 2 v_a δ_{bk} + 2 v_b δ_{ak}
    // Fixed 2026-04-23: prior implementation had only the (a==k)·2v_b term,
    // missing the -2 δ_{ab} v_k diagonal and the 2 v_a δ_{bk} symmetric partner.
    for (int k = 0; k < 3; ++k) {
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                double val = 0.0;
                if (a == b) val -= 2.0 * v(k);
                if (b == k) val += 2.0 * v(a);
                if (a == k) val += 2.0 * v(b);
                output[static_cast<size_t>(k)](1 + a, 1 + b) += val;
            }
        }
    }

    return output;
}

}
