#pragma once

#include <utility>
#include <vector>
#include <tuple>
#include <Eigen/Dense>

/**
 * @brief Perform a single step of fourth-order Runge-Kutta (RK4) integration.
 *
 * This function performs one time step of the explicit fourth-order Runge-Kutta
 * method for solving ordinary differential equations of the form:
 * \f[
 * \frac{d\mathbf{x}}{dt} = \mathbf{f}(t, \mathbf{x})
 * \f]
 *
 * The RK4 method is one of the most widely used numerical integration schemes due to
 * its excellent balance of accuracy and efficiency. It uses four weighted function
 * evaluations per step:
 * \f[
 * \mathbf{k}_1 = \mathbf{f}(t, \mathbf{x}) \\
 * \mathbf{k}_2 = \mathbf{f}\left(t + \frac{\Delta t}{2}, \mathbf{x} + \frac{\Delta t}{2}\mathbf{k}_1\right) \\
 * \mathbf{k}_3 = \mathbf{f}\left(t + \frac{\Delta t}{2}, \mathbf{x} + \frac{\Delta t}{2}\mathbf{k}_2\right) \\
 * \mathbf{k}_4 = \mathbf{f}(t + \Delta t, \mathbf{x} + \Delta t \, \mathbf{k}_3) \\
 * \mathbf{x}_{\text{out}} = \mathbf{x} + \frac{\Delta t}{6}(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4)
 * \f]
 *
 * RK4 has a local truncation error of \f$O(\Delta t^5)\f$ and is fourth-order accurate,
 * making it suitable for high-precision orbit propagation and other dynamical systems.
 *
 * @tparam State Type of the state vector (must support scalar multiplication and addition).
 * @tparam DerivFunc Type of the derivative function/functor.
 *
 * @param f Derivative function with signature: `void f(double t, const State& x, State& dxdt)`.
 *           Computes the derivative at time `t` and state `x`, storing the result in `dxdt`.
 * @param x Current state vector.
 * @param t Current time.
 * @param dt Time step size.
 * @param x_out Output state vector after one RK4 step.
 * @param quat_idx Starting index of a unit quaternion in the state, or a
 *        negative value (default) to disable on-manifold normalization. When
 *        non-negative this realizes "convention B": the input is normalized
 *        first (m = norm(x)), every RK4 substage quaternion is renormalized
 *        before the derivative evaluation, and the output is normalized. This
 *        makes the integrated discrete map exactly the one that rk4_jacobians
 *        and rk4_hessians linearize. Callers without a quaternion (or that
 *        normalize externally) leave this at the default and get the classic
 *        un-normalized RK4 step, byte-for-byte as before.
 */

template <typename State, typename DerivFunc>
void rk4_step(
    DerivFunc&& f,
    const State& x,
    double t,
    double dt,
    State& x_out,
    int quat_idx = -1
) {
    auto renorm = [quat_idx](State& s) {
        if (quat_idx >= 0) {
            s.template segment<4>(quat_idx).normalize();
        }
    };

    State k1, k2, k3, k4, xtemp;

    // Convention B: normalize the input so every "+state" leading term is the
    // unit quaternion m = norm(x).
    State m = x;
    renorm(m);

    f(t, m, k1);

    xtemp = m + 0.5 * dt * k1;
    renorm(xtemp);
    f(t + 0.5 * dt, xtemp, k2);

    xtemp = m + 0.5 * dt * k2;
    renorm(xtemp);
    f(t + 0.5 * dt, xtemp, k3);

    xtemp = m + dt * k3;
    renorm(xtemp);
    f(t + dt, xtemp, k4);

    x_out = m + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    renorm(x_out);
}

/**
 * @brief Compute exact discrete-time Jacobians for RK4 integration.
 *
 * This function computes the first-order sensitivities of the RK4 integrator,
 * providing exact discrete-time Jacobians A and B such that:
 * \f[
 * \mathbf{x}_{k+1} \approx \mathbf{x}_k + \mathbf{A}(\mathbf{x}_k - \mathbf{x}_{\text{ref}}) 
 *                   + \mathbf{B}(\mathbf{u}_k - \mathbf{u}_{\text{ref}})
 * \f]
 * where A = ∂x_{k+1}/∂x_k and B = ∂x_{k+1}/∂u_k are evaluated at the linearization point.
 *
 * The computation uses the chain rule through all four RK4 stages:
 * - ∂k_1/∂x = f_x(t, x, u)
 * - ∂k_2/∂x = f_x(...) · (I + dt/2 · ∂k_1/∂x)
 * - ∂k_3/∂x = f_x(...) · (I + dt/2 · ∂k_2/∂x)
 * - ∂k_4/∂x = f_x(...) · (I + dt · ∂k_3/∂x)
 * - A = I + (dt/6)(∂k_1/∂x + 2·∂k_2/∂x + 2·∂k_3/∂x + ∂k_4/∂x)
 *
 * This is significantly more accurate than first-order Euler discretization for iLQR,
 * as it ensures the backward pass linearization exactly matches the forward pass integration.
 *
 * @param dynamics_jac Function that computes continuous-time Jacobians f_x and f_u.
 *                     Signature: void(double t, const VecX& x, const VecX& u, MatXX& A_c, MatXU& B_c)
 * @param x Current state vector (n × 1)
 * @param u Current control vector (m × 1)
 * @param t Current time
 * @param dt Time step size
 * @param A_discrete Output: discrete-time state Jacobian ∂x_{k+1}/∂x_k (n × n)
 * @param B_discrete Output: discrete-time control Jacobian ∂x_{k+1}/∂u_k (n × m)
 *
 * @tparam DynamicsJacFunc Function type for computing continuous Jacobians
 */
/**
 * @brief Compute the state normalization Jacobian.
 *
 * Returns an nx×nx identity matrix with the quaternion block (4×4)
 * replaced by (I/||q|| - q*q^T/||q||^3).  For unit quaternions this
 * is approximately (I - q*q^T), the tangent-space projector on S³.
 *
 * @param x  Full state vector (quaternion at indices quat_idx..quat_idx+3)
 * @param nx State dimension
 * @param quat_idx Starting index of the quaternion in the state vector
 */
inline Eigen::MatrixXd stateNormJacobian(
    const Eigen::Ref<const Eigen::VectorXd>& x,
    int nx,
    int quat_idx = 3
) {
    Eigen::MatrixXd J = Eigen::MatrixXd::Identity(nx, nx);
    const Eigen::Vector4d q = x.segment<4>(quat_idx);
    const double qn = q.norm();
    if (qn < 1e-10) return J;
    const double qn3 = qn * qn * qn;
    J.block<4, 4>(quat_idx, quat_idx) =
        Eigen::Matrix4d::Identity() / qn - (q * q.transpose()) / qn3;
    return J;
}

/**
 * @brief Second derivative of the state normalization map.
 *
 * For the normalization map norm_a(q) = q_a / ‖q‖ (a = 0..3), this returns the
 * full Hessian as nx slices, one per output component. Only the four
 * quaternion-output slices (indices quat_idx+a) are non-zero, and within those
 * the non-zero block is the quat×quat block:
 *   ∂²norm_a/∂q_m∂q_j = 3 q_a q_m q_j / r⁵
 *                       − (δ_am q_j + δ_aj q_m + δ_mj q_a) / r³,   r = ‖q‖.
 * Every other state component is an identity passthrough, so its second
 * derivative is zero. This is the "Planning with Attitude" eq.-15 curvature
 * term of the dynamics retraction (the term PR #71 was missing).
 *
 * @param x        Full state vector (quaternion at quat_idx..quat_idx+3).
 * @param nx       State dimension.
 * @param quat_idx Starting index of the quaternion.
 */
inline std::vector<Eigen::MatrixXd> stateNormHessian(
    const Eigen::Ref<const Eigen::VectorXd>& x,
    int nx,
    int quat_idx = 3
) {
    std::vector<Eigen::MatrixXd> H(static_cast<std::size_t>(nx),
                                   Eigen::MatrixXd::Zero(nx, nx));
    const Eigen::Vector4d q = x.segment<4>(quat_idx);
    const double r = q.norm();
    if (r < 1e-10) return H;
    const double r3 = r * r * r;
    const double r5 = r3 * r * r;
    for (int a = 0; a < 4; ++a) {
        Eigen::MatrixXd& slice = H[static_cast<std::size_t>(quat_idx + a)];
        for (int m = 0; m < 4; ++m) {
            for (int j = 0; j < 4; ++j) {
                const double delta_am = (a == m) ? 1.0 : 0.0;
                const double delta_aj = (a == j) ? 1.0 : 0.0;
                const double delta_mj = (m == j) ? 1.0 : 0.0;
                const double val =
                    3.0 * q(a) * q(m) * q(j) / r5
                    - (delta_am * q(j) + delta_aj * q(m) + delta_mj * q(a)) / r3;
                slice(quat_idx + m, quat_idx + j) = val;
            }
        }
    }
    return H;
}

template <typename DynamicsJacFunc>
void rk4_jacobians(
    DynamicsJacFunc&& dynamics_jac,
    const Eigen::Ref<const Eigen::VectorXd>& x,
    const Eigen::Ref<const Eigen::VectorXd>& u,
    double t,
    double dt,
    Eigen::Ref<Eigen::MatrixXd> A_discrete,
    Eigen::Ref<Eigen::MatrixXd> B_discrete,
    int quat_idx = 3
) {
    const int nx = static_cast<int>(x.size());
    const int nu = static_cast<int>(u.size());

    const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(nx, nx);

    Eigen::MatrixXd A_c1(nx, nx), A_c2(nx, nx), A_c3(nx, nx), A_c4(nx, nx);
    Eigen::MatrixXd B_c1(nx, nu), B_c2(nx, nu), B_c3(nx, nu), B_c4(nx, nu);
    // dkᵢ_dm: derivative of stage kᵢ w.r.t. the normalized base m = norm(x).
    Eigen::MatrixXd dk1_dm(nx, nx), dk2_dm(nx, nx), dk3_dm(nx, nx), dk4_dm(nx, nx);
    Eigen::MatrixXd dk1_du(nx, nu), dk2_du(nx, nu), dk3_du(nx, nu), dk4_du(nx, nu);
    Eigen::VectorXd x2(nx), x3(nx), x4(nx);
    Eigen::VectorXd k1(nx), k2(nx), k3(nx), k4(nx);

    // Convention B: the discrete map is F(x) = norm(Φ(m)), m = norm(x). We first
    // build Φ and dΦ/dm with the unit quaternion m as the leading "+state" term,
    // then chain the input normalization dm/dx = N0 at the very end.
    const Eigen::MatrixXd N0 = stateNormJacobian(x, nx, quat_idx);
    Eigen::VectorXd m = x;
    m.segment<4>(quat_idx).normalize();

    // ========================================================================
    // Stage 1: k1 = f(t, m, u)
    // ========================================================================
    dynamics_jac(t, m, u, A_c1, B_c1, k1);
    // m is the base, so dk1/dm = A_c1 (no N0 here — N0 is chained outside).
    dk1_dm = A_c1;
    dk1_du = B_c1;

    // ========================================================================
    // Stage 2: k2 = f(t + dt/2, norm(m + dt/2 * k1), u)
    // ========================================================================
    x2 = m + 0.5 * dt * k1;
    const Eigen::MatrixXd N2 = stateNormJacobian(x2, nx, quat_idx);
    x2.segment<4>(quat_idx).normalize();
    dynamics_jac(t + 0.5 * dt, x2, u, A_c2, B_c2, k2);
    // d(norm(m+½dt·k1))/dm = N2·(I + ½dt·dk1_dm); dk2/dm = A_c2·N2·(I + ½dt·dk1_dm)
    dk2_dm = A_c2 * N2 * (I + 0.5 * dt * dk1_dm);
    dk2_du = A_c2 * N2 * (0.5 * dt * dk1_du) + B_c2;

    // ========================================================================
    // Stage 3: k3 = f(t + dt/2, norm(m + dt/2 * k2), u)
    // ========================================================================
    x3 = m + 0.5 * dt * k2;
    const Eigen::MatrixXd N3 = stateNormJacobian(x3, nx, quat_idx);
    x3.segment<4>(quat_idx).normalize();
    dynamics_jac(t + 0.5 * dt, x3, u, A_c3, B_c3, k3);
    dk3_dm = A_c3 * N3 * (I + 0.5 * dt * dk2_dm);
    dk3_du = A_c3 * N3 * (0.5 * dt * dk2_du) + B_c3;

    // ========================================================================
    // Stage 4: k4 = f(t + dt, norm(m + dt * k3), u)
    // ========================================================================
    x4 = m + dt * k3;
    const Eigen::MatrixXd N4 = stateNormJacobian(x4, nx, quat_idx);
    x4.segment<4>(quat_idx).normalize();
    dynamics_jac(t + dt, x4, u, A_c4, B_c4, k4);
    dk4_dm = A_c4 * N4 * (I + dt * dk3_dm);
    dk4_du = A_c4 * N4 * (dt * dk3_du) + B_c4;

    // ========================================================================
    // Assemble discrete Jacobians
    // ========================================================================
    // Φ(m)      = norm(m + (dt/6)(k1+2k2+2k3+k4))
    // dΦ/dm     = N_next·(I + (dt/6)Σ dkᵢ_dm)
    // F(x)      = norm(Φ(norm(x)))  ⇒  A_discrete = dΦ/dm · N0
    Eigen::VectorXd phi_raw = m + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    const Eigen::MatrixXd N_next = stateNormJacobian(phi_raw, nx, quat_idx);

    const Eigen::MatrixXd dPhi_dm =
        N_next * (I + (dt / 6.0) * (dk1_dm + 2.0 * dk2_dm + 2.0 * dk3_dm + dk4_dm));

    A_discrete = dPhi_dm * N0;
    // u does NOT chain through N0 (m = norm(x) is control-independent).
    B_discrete = N_next * ((dt / 6.0) * (dk1_du + 2.0 * dk2_du + 2.0 * dk3_du + dk4_du));
}

// ============================================================================
// rk4_hessians — discrete-time dynamics Hessians for the RK4 integrator step
// ============================================================================
namespace saltro_rk4_detail {

// All "cubes" below are std::vector<Eigen::MatrixXd>: a list of L slices, each
// slice being one output equation's matrix. These helpers contract / combine
// such cubes the way the DDP composition chain rule requires.

// out[l] = M · cube[l]   (left-multiply every slice by M)
inline std::vector<Eigen::MatrixXd> matTimesCube(
    const Eigen::MatrixXd& M, const std::vector<Eigen::MatrixXd>& cube) {
    std::vector<Eigen::MatrixXd> out;
    out.reserve(cube.size());
    for (const auto& s : cube) out.emplace_back(M * s);
    return out;
}

// out[l] = cube[l] · M   (right-multiply every slice by M)
inline std::vector<Eigen::MatrixXd> cubeTimesMat(
    const std::vector<Eigen::MatrixXd>& cube, const Eigen::MatrixXd& M) {
    std::vector<Eigen::MatrixXd> out;
    out.reserve(cube.size());
    for (const auto& s : cube) out.emplace_back(s * M);
    return out;
}

// out[l] = M · cube[l]^T  (left-multiply the TRANSPOSE of each slice by M)
inline std::vector<Eigen::MatrixXd> matTimesCubeT(
    const Eigen::MatrixXd& M, const std::vector<Eigen::MatrixXd>& cube) {
    std::vector<Eigen::MatrixXd> out;
    out.reserve(cube.size());
    for (const auto& s : cube) out.emplace_back(M * s.transpose());
    return out;
}

// out[i] = Σ_l A(i,l) · cube[l]   (contract a leading index of the cube with a
// matrix A; used to "lift" a per-output cube through a stage Jacobian A so that
// the result is indexed by A's output rows). All slices must be the same size.
inline std::vector<Eigen::MatrixXd> matOverCube(
    const Eigen::MatrixXd& A, const std::vector<Eigen::MatrixXd>& cube) {
    const int n_out = static_cast<int>(A.rows());
    const int n_in = static_cast<int>(A.cols());
    const int r = cube.empty() ? 0 : static_cast<int>(cube[0].rows());
    const int c = cube.empty() ? 0 : static_cast<int>(cube[0].cols());
    std::vector<Eigen::MatrixXd> out(static_cast<std::size_t>(n_out),
                                     Eigen::MatrixXd::Zero(r, c));
    for (int i = 0; i < n_out; ++i) {
        for (int l = 0; l < n_in; ++l) {
            const double a = A(i, l);
            if (a != 0.0) out[static_cast<std::size_t>(i)].noalias() += a * cube[static_cast<std::size_t>(l)];
        }
    }
    return out;
}

// out[l] = a[l] + b[l]
inline std::vector<Eigen::MatrixXd> cubeAdd(
    const std::vector<Eigen::MatrixXd>& a, const std::vector<Eigen::MatrixXd>& b) {
    std::vector<Eigen::MatrixXd> out;
    out.reserve(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) out.emplace_back(a[i] + b[i]);
    return out;
}

// out[l] = s · cube[l]
inline std::vector<Eigen::MatrixXd> cubeScale(
    double s, const std::vector<Eigen::MatrixXd>& cube) {
    std::vector<Eigen::MatrixXd> out;
    out.reserve(cube.size());
    for (const auto& m : cube) out.emplace_back(s * m);
    return out;
}

// Add the normalization-curvature contribution of a stage whose input is
// norm(g), where g(m,u) has Jacobians Jx = ∂g/∂m and Ju = ∂g/∂u. The map
// norm(·) has second derivative d2N (nx slices, one per output component; only
// the four quaternion-output slices are non-zero). By the chain rule, the part
// of ∂²norm(g)/∂(·)∂(·) coming from norm's own curvature is, per output l:
//   Hxx[l] += Jxᵀ·d2N[l]·Jx
//   Hux[l] += Juᵀ·d2N[l]·Jx
//   Huu[l] += Juᵀ·d2N[l]·Ju
// (Only the four quat-output slices l = quat_idx..quat_idx+3 are touched, since
//  d2N is zero on every other slice.)
inline void addNormHessianTerm(
    std::vector<Eigen::MatrixXd>& Hxx,
    std::vector<Eigen::MatrixXd>& Hux,
    std::vector<Eigen::MatrixXd>& Huu,
    const std::vector<Eigen::MatrixXd>& d2N,
    const Eigen::MatrixXd& Jx,
    const Eigen::MatrixXd& Ju,
    int quat_idx) {
    for (int a = 0; a < 4; ++a) {
        const std::size_t l = static_cast<std::size_t>(quat_idx + a);
        const Eigen::MatrixXd& S = d2N[l];
        Hxx[l].noalias() += Jx.transpose() * S * Jx;
        Hux[l].noalias() += Ju.transpose() * S * Jx;
        Huu[l].noalias() += Ju.transpose() * S * Ju;
    }
}

}  // namespace saltro_rk4_detail

/**
 * @brief Compute discrete-time dynamics Hessians for an RK4 integrator step.
 *
 * Given a continuous-time dynamics callback that supplies Jacobians
 * (A_c = ∂f/∂x, B_c = ∂f/∂u) and Hessians (f_xx = ∂²f/∂x², f_ux = ∂²f/∂u∂x,
 * f_uu = ∂²f/∂u²), this composes them through all four RK4 stages to produce
 * the discrete-time Hessians of x_{k+1} w.r.t. x_k and u_k:
 *   F_xx[l] = ∂²x_{k+1}_l / ∂x_k∂x_k      (nx slices, each nx×nx)
 *   F_ux[l] = ∂²x_{k+1}_l / ∂u_k∂x_k      (nx slices, each nu×nx)
 *   F_uu[l] = ∂²x_{k+1}_l / ∂u_k∂u_k      (nx slices, each nu×nu)
 *
 * Convention B (consistent with rk4_jacobians/rk4_step): the discrete map is
 * F(x) = norm(Φ(m)), m = norm(x), where Φ is RK4 with per-substage and output
 * renormalization applied to the already-normalized base m. This routine builds
 * Φ_x/Φ_xx/Φ_ux/Φ_uu w.r.t. m and then chains the input normalization
 * (dm/dx = N0, d²m/dx² = d²N0). Unlike earlier drafts, the SECOND derivative of
 * the normalization map IS modeled — both at every substage/output (the
 * Jᵀ·d²N·J curvature term) and at the input (Σ_a Φ_x[l,a]·d²N0[a], the
 * "Planning with Attitude" eq.-15 term). This makes the discrete second-order
 * model exact in the quaternion block.
 *
 * The callback signature matches `rk4_jacobians` extended with three Hessian
 * output cubes (std::vector<Eigen::MatrixXd>, nx slices each).
 *
 * @tparam DynHessFunc callback type.
 * @param dynamics_hess Function returning continuous-time Jacobians + Hessians.
 * @param x   Current state (nx).
 * @param u   Current control (nu).
 * @param t   Current time.
 * @param dt  Step size.
 * @param[out] F_xx discrete-time ∂²x_{k+1}/∂x² (nx slices, each nx×nx).
 * @param[out] F_ux discrete-time ∂²x_{k+1}/∂u∂x (nx slices, each nu×nx).
 * @param[out] F_uu discrete-time ∂²x_{k+1}/∂u² (nx slices, each nu×nu).
 */
template <typename DynHessFunc>
void rk4_hessians(
    DynHessFunc&& dynamics_hess,
    const Eigen::Ref<const Eigen::VectorXd>& x,
    const Eigen::Ref<const Eigen::VectorXd>& u,
    double t,
    double dt,
    std::vector<Eigen::MatrixXd>& F_xx,
    std::vector<Eigen::MatrixXd>& F_ux,
    std::vector<Eigen::MatrixXd>& F_uu,
    int quat_idx = 3
) {
    using MatXd = Eigen::MatrixXd;
    using VecXd = Eigen::VectorXd;
    using saltro_rk4_detail::matTimesCube;
    using saltro_rk4_detail::cubeTimesMat;
    using saltro_rk4_detail::matTimesCubeT;
    using saltro_rk4_detail::matOverCube;
    using saltro_rk4_detail::cubeAdd;
    using saltro_rk4_detail::cubeScale;
    using saltro_rk4_detail::addNormHessianTerm;

    const int nx = static_cast<int>(x.size());
    const int nu = static_cast<int>(u.size());

    const MatXd I = MatXd::Identity(nx, nx);

    auto makeXX = [nx]() {
        return std::vector<MatXd>(static_cast<std::size_t>(nx), MatXd::Zero(nx, nx));
    };
    auto makeUX = [nx, nu]() {
        return std::vector<MatXd>(static_cast<std::size_t>(nx), MatXd::Zero(nu, nx));
    };
    auto makeUU = [nx, nu]() {
        return std::vector<MatXd>(static_cast<std::size_t>(nx), MatXd::Zero(nu, nu));
    };

    // Compose first/second derivatives of k_i = f(x_in, u) through a stage input
    // (x_in, u), where x_in has Jacobians (Mx, Mu) and Hessians (H_xx, H_ux,
    // H_uu) w.r.t. the original (x_raw, u). Returns d²k_i/d(·)d(·).
    //   Kxx[l] = Mx^T·fxx[l]·Mx + (A·H_xx)[l]
    //   Kux[l] = Mu^T·fxx[l]·Mx + fux[l]·Mx + (A·H_ux)[l]
    //   Kuu[l] = Mu^T·fxx[l]·Mu + fux[l]·Mu + (fux[l]·Mu)^T + fuu[l] + (A·H_uu)[l]
    auto composeF = [&](const MatXd& A,
                        const std::vector<MatXd>& fxx,
                        const std::vector<MatXd>& fux,
                        const std::vector<MatXd>& fuu,
                        const MatXd& Mx, const MatXd& Mu,
                        const std::vector<MatXd>& H_xx,
                        const std::vector<MatXd>& H_ux,
                        const std::vector<MatXd>& H_uu)
    {
        auto Kxx = cubeAdd(
            matTimesCube(Mx.transpose(), cubeTimesMat(fxx, Mx)),
            matOverCube(A, H_xx)
        );
        auto Kux = cubeAdd(
            cubeAdd(
                matTimesCube(Mu.transpose(), cubeTimesMat(fxx, Mx)),
                cubeTimesMat(fux, Mx)
            ),
            matOverCube(A, H_ux)
        );
        auto Kuu = cubeAdd(
            cubeAdd(
                matTimesCube(Mu.transpose(), cubeTimesMat(fxx, Mu)),
                cubeAdd(
                    cubeTimesMat(fux, Mu),
                    matTimesCubeT(Mu.transpose(), fux)
                )
            ),
            cubeAdd(fuu, matOverCube(A, H_uu))
        );
        return std::make_tuple(std::move(Kxx), std::move(Kux), std::move(Kuu));
    };

    // Convention B: we build Φ and its first/second derivatives w.r.t. the
    // normalized base m = norm(x), then chain the input normalization
    // dm/dx = N0 (and its curvature d²N0) at the very end. Hence the stage-0
    // input is m itself: Mx_in = I, Mu_in = 0, and NO input curvature here.
    const MatXd N0 = stateNormJacobian(x, nx, quat_idx);
    const std::vector<MatXd> d2N0 = stateNormHessian(x, nx, quat_idx);
    VecXd m = x;
    m.segment<4>(quat_idx).normalize();

    MatXd Mx_in = I;
    MatXd Mu_in = MatXd::Zero(nx, nu);
    auto Hxx_in = makeXX();
    auto Hux_in = makeUX();
    auto Huu_in = makeUU();

    // Stage 1: k1 = f(m, u)
    MatXd A1(nx, nx), B1(nx, nu);
    VecXd k1(nx);
    auto fxx1 = makeXX(); auto fux1 = makeUX(); auto fuu1 = makeUU();
    dynamics_hess(t, m, u, A1, B1, k1, fxx1, fux1, fuu1);
    auto K1 = composeF(A1, fxx1, fux1, fuu1, Mx_in, Mu_in, Hxx_in, Hux_in, Huu_in);
    auto& Kxx1 = std::get<0>(K1); auto& Kux1 = std::get<1>(K1); auto& Kuu1 = std::get<2>(K1);
    // Derivatives w.r.t. m (base): dk1/dm = A1, dk1/du = B1.
    const MatXd dk1_dm = A1 * Mx_in;
    const MatXd dk1_du = A1 * Mu_in + B1;

    // Stage 2 input: x_1 = norm(m + 0.5·dt·k1). The inner arg g_1 = m + ½dt·k1
    // has Jacobians Jx1 = I+½dt·dk1_dm, Ju1 = ½dt·dk1_du w.r.t. (m,u). The
    // normalization contributes both its first-deriv chain (via N1) and its own
    // curvature (addNormHessianTerm with d²N at g_1).
    {
        const VecXd g_1 = m + 0.5 * dt * k1;
        const MatXd N1 = stateNormJacobian(g_1, nx, quat_idx);
        const MatXd Jx1 = I + 0.5 * dt * dk1_dm;
        const MatXd Ju1 = 0.5 * dt * dk1_du;
        Mx_in = N1 * Jx1;
        Mu_in = N1 * Ju1;
        Hxx_in = matOverCube(N1, cubeScale(0.5 * dt, Kxx1));
        Hux_in = matOverCube(N1, cubeScale(0.5 * dt, Kux1));
        Huu_in = matOverCube(N1, cubeScale(0.5 * dt, Kuu1));
        addNormHessianTerm(Hxx_in, Hux_in, Huu_in,
                           stateNormHessian(g_1, nx, quat_idx), Jx1, Ju1, quat_idx);
    }
    VecXd x_1 = m + 0.5 * dt * k1;
    x_1.segment<4>(quat_idx).normalize();

    // Stage 2: k2 = f(x_1, u)
    MatXd A2(nx, nx), B2(nx, nu);
    VecXd k2(nx);
    auto fxx2 = makeXX(); auto fux2 = makeUX(); auto fuu2 = makeUU();
    dynamics_hess(t + 0.5 * dt, x_1, u, A2, B2, k2, fxx2, fux2, fuu2);
    auto K2 = composeF(A2, fxx2, fux2, fuu2, Mx_in, Mu_in, Hxx_in, Hux_in, Huu_in);
    auto& Kxx2 = std::get<0>(K2); auto& Kux2 = std::get<1>(K2); auto& Kuu2 = std::get<2>(K2);
    const MatXd dk2_dm = A2 * Mx_in;
    const MatXd dk2_du = A2 * Mu_in + B2;

    // Stage 3 input: x_2 = norm(m + 0.5·dt·k2)
    {
        const VecXd g_2 = m + 0.5 * dt * k2;
        const MatXd N2 = stateNormJacobian(g_2, nx, quat_idx);
        const MatXd Jx2 = I + 0.5 * dt * dk2_dm;
        const MatXd Ju2 = 0.5 * dt * dk2_du;
        Mx_in = N2 * Jx2;
        Mu_in = N2 * Ju2;
        Hxx_in = matOverCube(N2, cubeScale(0.5 * dt, Kxx2));
        Hux_in = matOverCube(N2, cubeScale(0.5 * dt, Kux2));
        Huu_in = matOverCube(N2, cubeScale(0.5 * dt, Kuu2));
        addNormHessianTerm(Hxx_in, Hux_in, Huu_in,
                           stateNormHessian(g_2, nx, quat_idx), Jx2, Ju2, quat_idx);
    }
    VecXd x_2 = m + 0.5 * dt * k2;
    x_2.segment<4>(quat_idx).normalize();

    // Stage 3: k3 = f(x_2, u)
    MatXd A3(nx, nx), B3(nx, nu);
    VecXd k3(nx);
    auto fxx3 = makeXX(); auto fux3 = makeUX(); auto fuu3 = makeUU();
    dynamics_hess(t + 0.5 * dt, x_2, u, A3, B3, k3, fxx3, fux3, fuu3);
    auto K3 = composeF(A3, fxx3, fux3, fuu3, Mx_in, Mu_in, Hxx_in, Hux_in, Huu_in);
    auto& Kxx3 = std::get<0>(K3); auto& Kux3 = std::get<1>(K3); auto& Kuu3 = std::get<2>(K3);
    const MatXd dk3_dm = A3 * Mx_in;
    const MatXd dk3_du = A3 * Mu_in + B3;

    // Stage 4 input: x_3 = norm(m + dt·k3)
    {
        const VecXd g_3 = m + dt * k3;
        const MatXd N3 = stateNormJacobian(g_3, nx, quat_idx);
        const MatXd Jx3 = I + dt * dk3_dm;
        const MatXd Ju3 = dt * dk3_du;
        Mx_in = N3 * Jx3;
        Mu_in = N3 * Ju3;
        Hxx_in = matOverCube(N3, cubeScale(dt, Kxx3));
        Hux_in = matOverCube(N3, cubeScale(dt, Kux3));
        Huu_in = matOverCube(N3, cubeScale(dt, Kuu3));
        addNormHessianTerm(Hxx_in, Hux_in, Huu_in,
                           stateNormHessian(g_3, nx, quat_idx), Jx3, Ju3, quat_idx);
    }
    VecXd x_3 = m + dt * k3;
    x_3.segment<4>(quat_idx).normalize();

    // Stage 4: k4 = f(x_3, u)
    MatXd A4(nx, nx), B4(nx, nu);
    VecXd k4(nx);
    auto fxx4 = makeXX(); auto fux4 = makeUX(); auto fuu4 = makeUU();
    dynamics_hess(t + dt, x_3, u, A4, B4, k4, fxx4, fux4, fuu4);
    auto K4 = composeF(A4, fxx4, fux4, fuu4, Mx_in, Mu_in, Hxx_in, Hux_in, Huu_in);
    auto& Kxx4 = std::get<0>(K4); auto& Kux4 = std::get<1>(K4); auto& Kuu4 = std::get<2>(K4);
    const MatXd dk4_dm = A4 * Mx_in;
    const MatXd dk4_du = A4 * Mu_in + B4;

    // Combine into Φ(m) = norm(g_out), g_out = m + dt/6·(k1+2k2+2k3+k4).
    // The g_out-level Hessian is the weighted sum of the K_i; the output
    // normalization then contributes (a) its first-deriv chain via N_next and
    // (b) its own curvature via addNormHessianTerm at g_out.
    const double w = dt / 6.0;
    auto sum4 = [&](const std::vector<MatXd>& A,
                    const std::vector<MatXd>& B,
                    const std::vector<MatXd>& C,
                    const std::vector<MatXd>& D)
    {
        std::vector<MatXd> out;
        out.reserve(A.size());
        for (std::size_t i = 0; i < A.size(); ++i) {
            out.emplace_back(w * (A[i] + 2.0 * B[i] + 2.0 * C[i] + D[i]));
        }
        return out;
    };
    auto G_xx = sum4(Kxx1, Kxx2, Kxx3, Kxx4);  // ∂²g_out/∂m∂m
    auto G_ux = sum4(Kux1, Kux2, Kux3, Kux4);
    auto G_uu = sum4(Kuu1, Kuu2, Kuu3, Kuu4);

    const VecXd g_out = m + w * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    const MatXd N_next = stateNormJacobian(g_out, nx, quat_idx);

    // Φ_x = dΦ/dm = N_next·(I + (dt/6)Σ dkᵢ_dm)  (= dPhi_dm from rk4_jacobians).
    const MatXd Jx_out = I + w * (dk1_dm + 2.0 * dk2_dm + 2.0 * dk3_dm + dk4_dm);
    const MatXd Ju_out = w * (dk1_du + 2.0 * dk2_du + 2.0 * dk3_du + dk4_du);
    const MatXd Phi_x = N_next * Jx_out;

    // Φ_xx[l] = (N_next·G_xx)[l] + (Jx_outᵀ·d²N(g_out)·Jx_out)[l], etc.
    std::vector<MatXd> Phi_xx = matOverCube(N_next, G_xx);
    std::vector<MatXd> Phi_ux = matOverCube(N_next, G_ux);
    std::vector<MatXd> Phi_uu = matOverCube(N_next, G_uu);
    addNormHessianTerm(Phi_xx, Phi_ux, Phi_uu,
                       stateNormHessian(g_out, nx, quat_idx), Jx_out, Ju_out, quat_idx);

    // Chain the input normalization (base change m = norm(x), dm/dx = N0):
    //   F_xx[l] = N0ᵀ·Φ_xx[l]·N0 + Σ_a Φ_x[l,a]·d²N0[a]
    //   F_ux[l] = Φ_ux[l]·N0
    //   F_uu[l] = Φ_uu[l]
    F_xx.assign(static_cast<std::size_t>(nx), MatXd::Zero(nx, nx));
    F_ux.assign(static_cast<std::size_t>(nx), MatXd::Zero(nu, nx));
    F_uu.assign(static_cast<std::size_t>(nx), MatXd::Zero(nu, nu));
    for (int l = 0; l < nx; ++l) {
        const std::size_t ls = static_cast<std::size_t>(l);
        MatXd fxx = N0.transpose() * Phi_xx[ls] * N0;
        // + Σ_a Φ_x(l,a)·d²N0[a]   (only the four quat slices of d²N0 are nonzero)
        for (int a = 0; a < 4; ++a) {
            const double w_la = Phi_x(l, quat_idx + a);
            if (w_la != 0.0) {
                fxx.noalias() += w_la * d2N0[static_cast<std::size_t>(quat_idx + a)];
            }
        }
        F_xx[ls] = std::move(fxx);
        F_ux[ls] = Phi_ux[ls] * N0;
        F_uu[ls] = Phi_uu[ls];
    }
}
