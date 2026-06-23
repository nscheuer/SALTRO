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
 */

template <typename State, typename DerivFunc>
void rk4_step(
    DerivFunc&& f,
    const State& x,
    double t,
    double dt,
    State& x_out
) {
    State k1, k2, k3, k4, xtemp;

    f(t, x, k1);

    xtemp = x + 0.5 * dt * k1;
    f(t + 0.5 * dt, xtemp, k2);

    xtemp = x + 0.5 * dt * k2;
    f(t + 0.5 * dt, xtemp, k3);

    xtemp = x + dt * k3;
    f(t + dt, xtemp, k4);

    x_out = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
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
    Eigen::MatrixXd dk1_dx(nx, nx), dk2_dx(nx, nx), dk3_dx(nx, nx), dk4_dx(nx, nx);
    Eigen::MatrixXd dk1_du(nx, nu), dk2_du(nx, nu), dk3_du(nx, nu), dk4_du(nx, nu);
    Eigen::VectorXd x2(nx), x3(nx), x4(nx);
    Eigen::VectorXd k1(nx), k2(nx), k3(nx), k4(nx);

    // Normalization Jacobian at the initial state
    const Eigen::MatrixXd N0 = stateNormJacobian(x, nx, quat_idx);

    // ========================================================================
    // Stage 1: k1 = f(t, x_norm, u)
    // ========================================================================
    dynamics_jac(t, x, u, A_c1, B_c1, k1);
    // dk1/dx0_raw = A_c1 * N0  (chain rule through initial normalization)
    dk1_dx = A_c1 * N0;
    dk1_du = B_c1;

    // ========================================================================
    // Stage 2: k2 = f(t + dt/2, norm(x + dt/2 * k1), u)
    // ========================================================================
    x2 = x + 0.5 * dt * k1;
    // Normalize quaternion at intermediate state
    x2.segment<4>(quat_idx).normalize();
    const Eigen::MatrixXd N2 = stateNormJacobian(x + 0.5 * dt * k1, nx, quat_idx);
    dynamics_jac(t + 0.5 * dt, x2, u, A_c2, B_c2, k2);
    // dx2_raw/dx0_raw = I + 0.5*dt*dk1_dx, then norm: dx2/dx0_raw = N2 * (I + 0.5*dt*dk1_dx)
    // dk2/dx0_raw = A_c2 * N2 * (I + 0.5*dt*dk1_dx)
    dk2_dx = A_c2 * N2 * (I + 0.5 * dt * dk1_dx);
    dk2_du = A_c2 * N2 * (0.5 * dt * dk1_du) + B_c2;

    // ========================================================================
    // Stage 3: k3 = f(t + dt/2, norm(x + dt/2 * k2), u)
    // ========================================================================
    x3 = x + 0.5 * dt * k2;
    x3.segment<4>(quat_idx).normalize();
    const Eigen::MatrixXd N3 = stateNormJacobian(x + 0.5 * dt * k2, nx, quat_idx);
    dynamics_jac(t + 0.5 * dt, x3, u, A_c3, B_c3, k3);
    dk3_dx = A_c3 * N3 * (I + 0.5 * dt * dk2_dx);
    dk3_du = A_c3 * N3 * (0.5 * dt * dk2_du) + B_c3;

    // ========================================================================
    // Stage 4: k4 = f(t + dt, norm(x + dt * k3), u)
    // ========================================================================
    x4 = x + dt * k3;
    x4.segment<4>(quat_idx).normalize();
    const Eigen::MatrixXd N4 = stateNormJacobian(x + dt * k3, nx, quat_idx);
    dynamics_jac(t + dt, x4, u, A_c4, B_c4, k4);
    dk4_dx = A_c4 * N4 * (I + dt * dk3_dx);
    dk4_du = A_c4 * N4 * (dt * dk3_du) + B_c4;

    // ========================================================================
    // Assemble discrete Jacobians
    // ========================================================================
    // x_{k+1}_raw = x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    // Then normalize: x_{k+1} = norm(x_{k+1}_raw)
    Eigen::VectorXd x_next_raw = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    const Eigen::MatrixXd N_next = stateNormJacobian(x_next_raw, nx, quat_idx);

    Eigen::MatrixXd A_raw = N0 + (dt / 6.0) * (dk1_dx + 2.0 * dk2_dx + 2.0 * dk3_dx + dk4_dx);
    Eigen::MatrixXd B_raw = (dt / 6.0) * (dk1_du + 2.0 * dk2_du + 2.0 * dk3_du + dk4_du);

    // Final normalization at output
    A_discrete = N_next * A_raw;
    B_discrete = N_next * B_raw;
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
 * IMPORTANT — consistency with rk4_jacobians: this routine uses the SAME pure
 * RK4 chain rule as the non-normalizing `rk4_jacobians` above (no per-substage
 * quaternion renormalization). The reduced-state projection in the backward
 * pass (via the G matrices) handles the manifold structure, exactly as it does
 * for A_k/B_k. If `rk4_jacobians` is ever changed to normalize substeps (the
 * G6 / PR #41 work), this routine must be updated in lockstep so the first- and
 * second-order discrete models stay consistent.
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
    std::vector<Eigen::MatrixXd>& F_uu
) {
    using MatXd = Eigen::MatrixXd;
    using VecXd = Eigen::VectorXd;
    using saltro_rk4_detail::matTimesCube;
    using saltro_rk4_detail::cubeTimesMat;
    using saltro_rk4_detail::matTimesCubeT;
    using saltro_rk4_detail::matOverCube;
    using saltro_rk4_detail::cubeAdd;
    using saltro_rk4_detail::cubeScale;

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

    // Stage input 0 (= x): dx_0/dx = I, dx_0/du = 0, second derivatives zero.
    MatXd Mx_in = I;
    MatXd Mu_in = MatXd::Zero(nx, nu);
    auto Hxx_in = makeXX();
    auto Hux_in = makeUX();
    auto Huu_in = makeUU();

    // Stage 1: k1 = f(x, u)
    MatXd A1(nx, nx), B1(nx, nu);
    VecXd k1(nx);
    auto fxx1 = makeXX(); auto fux1 = makeUX(); auto fuu1 = makeUU();
    dynamics_hess(t, x, u, A1, B1, k1, fxx1, fux1, fuu1);
    auto K1 = composeF(A1, fxx1, fux1, fuu1, Mx_in, Mu_in, Hxx_in, Hux_in, Huu_in);
    auto& Kxx1 = std::get<0>(K1); auto& Kux1 = std::get<1>(K1); auto& Kuu1 = std::get<2>(K1);
    const MatXd dk1_dx = A1 * Mx_in;   // = A1
    const MatXd dk1_du = A1 * Mu_in + B1;  // = B1

    // Stage 2 input: x_1 = x + 0.5·dt·k1
    {
        Mx_in = I + 0.5 * dt * dk1_dx;
        Mu_in = 0.5 * dt * dk1_du;
        Hxx_in = cubeScale(0.5 * dt, Kxx1);
        Hux_in = cubeScale(0.5 * dt, Kux1);
        Huu_in = cubeScale(0.5 * dt, Kuu1);
    }
    const VecXd x_1 = x + 0.5 * dt * k1;

    // Stage 2: k2 = f(x_1, u)
    MatXd A2(nx, nx), B2(nx, nu);
    VecXd k2(nx);
    auto fxx2 = makeXX(); auto fux2 = makeUX(); auto fuu2 = makeUU();
    dynamics_hess(t + 0.5 * dt, x_1, u, A2, B2, k2, fxx2, fux2, fuu2);
    auto K2 = composeF(A2, fxx2, fux2, fuu2, Mx_in, Mu_in, Hxx_in, Hux_in, Huu_in);
    auto& Kxx2 = std::get<0>(K2); auto& Kux2 = std::get<1>(K2); auto& Kuu2 = std::get<2>(K2);
    const MatXd dk2_dx = A2 * Mx_in;
    const MatXd dk2_du = A2 * Mu_in + B2;

    // Stage 3 input: x_2 = x + 0.5·dt·k2
    {
        Mx_in = I + 0.5 * dt * dk2_dx;
        Mu_in = 0.5 * dt * dk2_du;
        Hxx_in = cubeScale(0.5 * dt, Kxx2);
        Hux_in = cubeScale(0.5 * dt, Kux2);
        Huu_in = cubeScale(0.5 * dt, Kuu2);
    }
    const VecXd x_2 = x + 0.5 * dt * k2;

    // Stage 3: k3 = f(x_2, u)
    MatXd A3(nx, nx), B3(nx, nu);
    VecXd k3(nx);
    auto fxx3 = makeXX(); auto fux3 = makeUX(); auto fuu3 = makeUU();
    dynamics_hess(t + 0.5 * dt, x_2, u, A3, B3, k3, fxx3, fux3, fuu3);
    auto K3 = composeF(A3, fxx3, fux3, fuu3, Mx_in, Mu_in, Hxx_in, Hux_in, Huu_in);
    auto& Kxx3 = std::get<0>(K3); auto& Kux3 = std::get<1>(K3); auto& Kuu3 = std::get<2>(K3);
    const MatXd dk3_dx = A3 * Mx_in;
    const MatXd dk3_du = A3 * Mu_in + B3;

    // Stage 4 input: x_3 = x + dt·k3
    {
        Mx_in = I + dt * dk3_dx;
        Mu_in = dt * dk3_du;
        Hxx_in = cubeScale(dt, Kxx3);
        Hux_in = cubeScale(dt, Kux3);
        Huu_in = cubeScale(dt, Kuu3);
    }
    const VecXd x_3 = x + dt * k3;

    // Stage 4: k4 = f(x_3, u)
    MatXd A4(nx, nx), B4(nx, nu);
    VecXd k4(nx);
    auto fxx4 = makeXX(); auto fux4 = makeUX(); auto fuu4 = makeUU();
    dynamics_hess(t + dt, x_3, u, A4, B4, k4, fxx4, fux4, fuu4);
    auto K4 = composeF(A4, fxx4, fux4, fuu4, Mx_in, Mu_in, Hxx_in, Hux_in, Huu_in);
    auto& Kxx4 = std::get<0>(K4); auto& Kux4 = std::get<1>(K4); auto& Kuu4 = std::get<2>(K4);

    // Combine: x_{k+1} = x + dt/6·(k1 + 2·k2 + 2·k3 + k4).
    // Linear in the k_i, so the discrete Hessian is the weighted sum of the K_i.
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
    F_xx = sum4(Kxx1, Kxx2, Kxx3, Kxx4);
    F_ux = sum4(Kux1, Kux2, Kux3, Kux4);
    F_uu = sum4(Kuu1, Kuu2, Kuu3, Kuu4);
}
