#pragma once

#include <utility>
#include <vector>
#include <Eigen/Dense>

#include <saltro/math/tensor_ops.h>

namespace saltro::math {

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

/**
 * @brief Compute the state normalization Hessian.
 *
 * Returns nx slices of (nx × nx) matrices. Slice `l` is `∂²(x_norm[l])/∂(x_raw)²`.
 * For non-quaternion components (l outside [quat_idx, quat_idx+3]), the slice
 * is identically zero (those state components pass through normalize unchanged).
 *
 * For quaternion components (l in [quat_idx, quat_idx+3]), the only nonzero
 * block is the 4×4 quaternion-quaternion sub-block. Closed form for q_norm =
 * q/‖q‖, with r = ‖q‖:
 *
 *     ∂²q_norm_a/∂q_b∂q_c
 *         = -δ_ab q_c / r³  -  δ_ac q_b / r³  -  δ_bc q_a / r³  +  3 q_a q_b q_c / r⁵
 *
 * (where a, b, c are local indices within the 4-element quaternion block).
 *
 * Required for accurate `rk4_hessians` composition through stage state
 * normalization. Without this, the discrete dynamics Hessian for quaternion
 * outputs is missing the normalization-curvature contribution and is
 * systematically wrong for q-q couplings (FD diverges by O(1) on entries
 * like ∂²q_x_{k+1}/∂q_0∂q_x = -1).
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
        const std::size_t out_idx = static_cast<std::size_t>(quat_idx + a);
        Eigen::Matrix4d block = Eigen::Matrix4d::Zero();
        for (int b = 0; b < 4; ++b) {
            for (int c = 0; c < 4; ++c) {
                double val = 3.0 * q(a) * q(b) * q(c) / r5;
                if (a == b) val -= q(c) / r3;
                if (a == c) val -= q(b) / r3;
                if (b == c) val -= q(a) / r3;
                block(b, c) = val;
            }
        }
        H[out_idx].block<4, 4>(quat_idx, quat_idx) = block;
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
    Eigen::MatrixXd dk1_dx(nx, nx), dk2_dx(nx, nx), dk3_dx(nx, nx), dk4_dx(nx, nx);
    Eigen::MatrixXd dk1_du(nx, nu), dk2_du(nx, nu), dk3_du(nx, nu), dk4_du(nx, nu);
    Eigen::VectorXd x2(nx), x3(nx), x4(nx);
    Eigen::VectorXd k1(nx), k2(nx), k3(nx), k4(nx);

    // Following the original ALTRO implementation exactly:
    // Each intermediate raw state x_ir = x_norm + coeff*dt*k_i is normalized
    // before evaluating dynamics. The chain rule tracks derivatives w.r.t.
    // the initial RAW state x0_raw through all normalization steps.
    //
    // Key: dxd_i/dx0_raw = A_ci * N_i * (dx_ir/dx0_raw)
    //      dx_ir/dx0_raw = N0 + coeff*dt*(dxd_{i-1}/dx0_raw)  [note N0, not I]

    // Normalize the initial state
    Eigen::VectorXd x_norm = x;
    x_norm.segment<4>(quat_idx).normalize();
    const Eigen::MatrixXd N0 = stateNormJacobian(x, nx, quat_idx);

    // Stage 1: k1 = f(t, x_norm, u)
    dynamics_jac(t, x_norm, u, A_c1, B_c1, k1);
    // dxd0/dx0_raw = A_c1 * N0
    dk1_dx = A_c1 * N0;
    dk1_du = B_c1;

    // Stage 2: x1r = x_norm + 0.5*dt*k1, x1 = norm(x1r)
    Eigen::VectorXd x1r = x_norm + 0.5 * dt * k1;
    x2 = x1r;
    x2.segment<4>(quat_idx).normalize();
    const Eigen::MatrixXd N1 = stateNormJacobian(x1r, nx, quat_idx);
    dynamics_jac(t + 0.5 * dt, x2, u, A_c2, B_c2, k2);
    // dx1r/dx0_raw = N0 + 0.5*dt*(dxd0/dx0_raw) = N0 + 0.5*dt*A_c1*N0
    Eigen::MatrixXd dx1r_dx0r = N0 + 0.5 * dt * dk1_dx;
    dk2_dx = A_c2 * N1 * dx1r_dx0r;
    dk2_du = A_c2 * N1 * (0.5 * dt * dk1_du) + B_c2;

    // Stage 3: x2r = x_norm + 0.5*dt*k2, x2 = norm(x2r)
    Eigen::VectorXd x2r = x_norm + 0.5 * dt * k2;
    x3 = x2r;
    x3.segment<4>(quat_idx).normalize();
    const Eigen::MatrixXd N2 = stateNormJacobian(x2r, nx, quat_idx);
    dynamics_jac(t + 0.5 * dt, x3, u, A_c3, B_c3, k3);
    // dx2r/dx0_raw = N0 + 0.5*dt*(dxd1/dx0_raw)
    Eigen::MatrixXd dx2r_dx0r = N0 + 0.5 * dt * dk2_dx;
    dk3_dx = A_c3 * N2 * dx2r_dx0r;
    dk3_du = A_c3 * N2 * (0.5 * dt * dk2_du) + B_c3;

    // Stage 4: x3r = x_norm + dt*k3, x3 = norm(x3r)
    Eigen::VectorXd x3r = x_norm + dt * k3;
    x4 = x3r;
    x4.segment<4>(quat_idx).normalize();
    const Eigen::MatrixXd N3 = stateNormJacobian(x3r, nx, quat_idx);
    dynamics_jac(t + dt, x4, u, A_c4, B_c4, k4);
    // dx3r/dx0_raw = N0 + dt*(dxd2/dx0_raw)
    Eigen::MatrixXd dx3r_dx0r = N0 + dt * dk3_dx;
    dk4_dx = A_c4 * N3 * dx3r_dx0r;
    dk4_du = A_c4 * N3 * (dt * dk3_du) + B_c4;

    // Assemble: x_{k+1}_raw = x_norm + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    Eigen::VectorXd x_next_raw = x_norm + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    const Eigen::MatrixXd N_next = stateNormJacobian(x_next_raw, nx, quat_idx);

    // dx_{k+1}_raw/dx0_raw = N0 + (dt/6)*(dk1_dx + 2*dk2_dx + 2*dk3_dx + dk4_dx)
    Eigen::MatrixXd A_raw = N0 + (dt / 6.0) * (dk1_dx + 2.0 * dk2_dx + 2.0 * dk3_dx + dk4_dx);
    Eigen::MatrixXd B_raw = (dt / 6.0) * (dk1_du + 2.0 * dk2_du + 2.0 * dk3_du + dk4_du);

    // Final normalization: dx_{k+1}/dx0_raw = N_next * A_raw
    A_discrete = N_next * A_raw;
    B_discrete = N_next * B_raw;
}

/**
 * @brief Compute discrete-time dynamics Hessians for an RK4 integrator step.
 *
 * Given a continuous-time dynamics callback that supplies both Jacobians
 * (A_c = ∂f/∂x, B_c = ∂f/∂u) and Hessians (f_xx = ∂²f/∂x², f_ux = ∂²f/∂u∂x,
 * f_uu = ∂²f/∂u²), this function composes them through all four RK4 stages
 * to produce the discrete-time Hessians of x_{k+1} with respect to x_k and u_k:
 *   F_xx[l] = ∂²x_{k+1}_l / ∂x_k∂x_k      (nx slices, each nx×nx)
 *   F_ux[l] = ∂²x_{k+1}_l / ∂u_k∂x_k      (nx slices, each nu×nx)
 *   F_uu[l] = ∂²x_{k+1}_l / ∂u_k∂u_k      (nx slices, each nu×nu)
 *
 * Composition includes the FULL state-normalization curvature at every
 * stage (input, four intermediate raw states, final raw state) via
 * `stateNormHessian`. The earlier v1 simplification that dropped these
 * terms made F_xx slices for quaternion outputs systematically wrong
 * (e.g. `∂²q_x_{k+1}/∂q_w∂q_x = -1` analytically zero in v1, FD gives -1).
 *
 * The callback signature matches the one used by `rk4_jacobians`, extended
 * with three Hessian output std::vectors filled with nx slices (each nx×nx
 * for f_xx, nu×nx for f_ux, nu×nu for f_uu).
 *
 * @tparam DynHessFunc callback type.
 * @param dynamics_hess Function that returns continuous-time Jacobians and
 *        Hessians at (t, x, u).
 * @param x         Current state (raw — need not be pre-normalized).
 * @param u         Current control.
 * @param t         Current time.
 * @param dt        Step size.
 * @param[out] F_xx discrete-time ∂²x_{k+1}/∂x² (nx slices, each nx×nx).
 * @param[out] F_ux discrete-time ∂²x_{k+1}/∂u∂x (nx slices, each nu×nx).
 * @param[out] F_uu discrete-time ∂²x_{k+1}/∂u² (nx slices, each nu×nu).
 * @param quat_idx Starting index of the quaternion in the state vector.
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
    using saltro::math::matTimesCube;
    using saltro::math::cubeTimesMat;
    using saltro::math::matTimesCubeT;
    using saltro::math::matOverCube;
    using saltro::math::cubeAdd;
    using saltro::math::cubeScale;

    const int nx = static_cast<int>(x.size());
    const int nu = static_cast<int>(u.size());

    // Pre-stage normalization of the input state.
    VecXd x_norm = x;
    x_norm.segment<4>(quat_idx).normalize();
    const MatXd N0 = stateNormJacobian(x, nx, quat_idx);
    const auto NH0 = stateNormHessian(x, nx, quat_idx);

    auto makeXX = [nx]() {
        return std::vector<MatXd>(static_cast<std::size_t>(nx), MatXd::Zero(nx, nx));
    };
    auto makeUX = [nx, nu]() {
        return std::vector<MatXd>(static_cast<std::size_t>(nx), MatXd::Zero(nu, nx));
    };
    auto makeUU = [nx, nu]() {
        return std::vector<MatXd>(static_cast<std::size_t>(nx), MatXd::Zero(nu, nu));
    };

    // Compose first/second derivatives of k_i = f(x_in, u) through stage input
    // (x_in, u), where x_in has Jacobians (Mx, Mu) and Hessians (H_xx, H_ux, H_uu)
    // w.r.t. (x_raw, u). Returns (Kxx, Kux, Kuu) = d²k_i/d(·)d(·).
    auto composeF = [&](const MatXd& A,
                        const std::vector<MatXd>& fxx,
                        const std::vector<MatXd>& fux,
                        const std::vector<MatXd>& fuu,
                        const MatXd& Mx, const MatXd& Mu,
                        const std::vector<MatXd>& H_xx,
                        const std::vector<MatXd>& H_ux,
                        const std::vector<MatXd>& H_uu)
    {
        // Kxx[l] = Mx^T · fxx[l] · Mx + (A · H_xx)[l]
        auto Kxx = cubeAdd(
            matTimesCube(Mx.transpose(), cubeTimesMat(fxx, Mx)),
            matOverCube(A, H_xx)
        );
        // Kux[l] = Mu^T · fxx[l] · Mx + fux[l] · Mx + (A · H_ux)[l]
        auto Kux = cubeAdd(
            cubeAdd(
                matTimesCube(Mu.transpose(), cubeTimesMat(fxx, Mx)),
                cubeTimesMat(fux, Mx)
            ),
            matOverCube(A, H_ux)
        );
        // Kuu[l] = Mu^T · fxx[l] · Mu + fux[l]·Mu + (fux[l]·Mu)^T + fuu[l] + (A·H_uu)[l]
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

    // Push (dx_r/d·, d²x_r/d·d·) through the per-stage normalization
    // x_new = norm(x_r). NH = stateNormHessian(x_r). Returns (N, dx_new/d·,
    // d²x_new/d·d·).
    auto normalizeStage = [&](const VecXd& xr,
                              const MatXd& dxrdx, const MatXd& dxrdu,
                              const std::vector<MatXd>& Hr_xx,
                              const std::vector<MatXd>& Hr_ux,
                              const std::vector<MatXd>& Hr_uu)
    {
        const MatXd N = stateNormJacobian(xr, nx, quat_idx);
        const auto NH = stateNormHessian(xr, nx, quat_idx);
        const MatXd dxdx = N * dxrdx;
        const MatXd dxdu = N * dxrdu;
        auto H_xx = cubeAdd(
            matTimesCube(dxrdx.transpose(), cubeTimesMat(NH, dxrdx)),
            matOverCube(N, Hr_xx)
        );
        auto H_ux = cubeAdd(
            matTimesCube(dxrdu.transpose(), cubeTimesMat(NH, dxrdx)),
            matOverCube(N, Hr_ux)
        );
        auto H_uu = cubeAdd(
            matTimesCube(dxrdu.transpose(), cubeTimesMat(NH, dxrdu)),
            matOverCube(N, Hr_uu)
        );
        return std::make_tuple(N, dxdx, dxdu,
                               std::move(H_xx), std::move(H_ux), std::move(H_uu));
    };

    // ------------------------------------------------------------------
    // Stage input 0 (= x_norm): dx_0/dx_raw = N0, dx_0/du = 0,
    //                           d²x_0/dx_raw² = NH0, d²x_0/du... = 0.
    // ------------------------------------------------------------------
    MatXd Mx_in = N0;
    MatXd Mu_in = MatXd::Zero(nx, nu);
    auto Hxx_in = NH0;
    auto Hux_in = makeUX();
    auto Huu_in = makeUU();

    // ------------------------------------------------------------------
    // Stage 1: k1 = f(x_norm, u)
    // ------------------------------------------------------------------
    MatXd A1(nx, nx), B1(nx, nu);
    VecXd k1(nx);
    auto fxx1 = makeXX(); auto fux1 = makeUX(); auto fuu1 = makeUU();
    dynamics_hess(t, x_norm, u, A1, B1, k1, fxx1, fux1, fuu1);
    auto K1 = composeF(A1, fxx1, fux1, fuu1, Mx_in, Mu_in, Hxx_in, Hux_in, Huu_in);
    auto& Kxx1 = std::get<0>(K1); auto& Kux1 = std::get<1>(K1); auto& Kuu1 = std::get<2>(K1);
    const MatXd dk1_dx = A1 * Mx_in;     // = A1 · N0
    const MatXd dk1_du = B1;              // Mu_in = 0

    // ------------------------------------------------------------------
    // Build stage-2 input: x_1r = x_norm + 0.5·dt·k1, x_1 = norm(x_1r).
    // ------------------------------------------------------------------
    const VecXd x_1r = x_norm + 0.5 * dt * k1;
    const MatXd dx1r_dx = N0 + 0.5 * dt * dk1_dx;
    const MatXd dx1r_du = 0.5 * dt * dk1_du;
    auto H1r_xx = cubeAdd(NH0, cubeScale(0.5 * dt, Kxx1));
    auto H1r_ux = cubeScale(0.5 * dt, Kux1);
    auto H1r_uu = cubeScale(0.5 * dt, Kuu1);
    VecXd x_1 = x_1r; x_1.segment<4>(quat_idx).normalize();
    {
        auto NS = normalizeStage(x_1r, dx1r_dx, dx1r_du, H1r_xx, H1r_ux, H1r_uu);
        Mx_in = std::get<1>(NS); Mu_in = std::get<2>(NS);
        Hxx_in = std::move(std::get<3>(NS));
        Hux_in = std::move(std::get<4>(NS));
        Huu_in = std::move(std::get<5>(NS));
    }

    // ------------------------------------------------------------------
    // Stage 2: k2 = f(x_1, u)
    // ------------------------------------------------------------------
    MatXd A2(nx, nx), B2(nx, nu);
    VecXd k2(nx);
    auto fxx2 = makeXX(); auto fux2 = makeUX(); auto fuu2 = makeUU();
    dynamics_hess(t + 0.5 * dt, x_1, u, A2, B2, k2, fxx2, fux2, fuu2);
    auto K2 = composeF(A2, fxx2, fux2, fuu2, Mx_in, Mu_in, Hxx_in, Hux_in, Huu_in);
    auto& Kxx2 = std::get<0>(K2); auto& Kux2 = std::get<1>(K2); auto& Kuu2 = std::get<2>(K2);
    const MatXd dk2_dx = A2 * Mx_in;
    const MatXd dk2_du = A2 * Mu_in + B2;

    // ------------------------------------------------------------------
    // Build stage-3 input: x_2r = x_norm + 0.5·dt·k2, x_2 = norm(x_2r).
    // ------------------------------------------------------------------
    const VecXd x_2r = x_norm + 0.5 * dt * k2;
    const MatXd dx2r_dx = N0 + 0.5 * dt * dk2_dx;
    const MatXd dx2r_du = 0.5 * dt * dk2_du;
    auto H2r_xx = cubeAdd(NH0, cubeScale(0.5 * dt, Kxx2));
    auto H2r_ux = cubeScale(0.5 * dt, Kux2);
    auto H2r_uu = cubeScale(0.5 * dt, Kuu2);
    VecXd x_2 = x_2r; x_2.segment<4>(quat_idx).normalize();
    {
        auto NS = normalizeStage(x_2r, dx2r_dx, dx2r_du, H2r_xx, H2r_ux, H2r_uu);
        Mx_in = std::get<1>(NS); Mu_in = std::get<2>(NS);
        Hxx_in = std::move(std::get<3>(NS));
        Hux_in = std::move(std::get<4>(NS));
        Huu_in = std::move(std::get<5>(NS));
    }

    // ------------------------------------------------------------------
    // Stage 3: k3 = f(x_2, u)
    // ------------------------------------------------------------------
    MatXd A3(nx, nx), B3(nx, nu);
    VecXd k3(nx);
    auto fxx3 = makeXX(); auto fux3 = makeUX(); auto fuu3 = makeUU();
    dynamics_hess(t + 0.5 * dt, x_2, u, A3, B3, k3, fxx3, fux3, fuu3);
    auto K3 = composeF(A3, fxx3, fux3, fuu3, Mx_in, Mu_in, Hxx_in, Hux_in, Huu_in);
    auto& Kxx3 = std::get<0>(K3); auto& Kux3 = std::get<1>(K3); auto& Kuu3 = std::get<2>(K3);
    const MatXd dk3_dx = A3 * Mx_in;
    const MatXd dk3_du = A3 * Mu_in + B3;

    // ------------------------------------------------------------------
    // Build stage-4 input: x_3r = x_norm + dt·k3, x_3 = norm(x_3r).
    // ------------------------------------------------------------------
    const VecXd x_3r = x_norm + dt * k3;
    const MatXd dx3r_dx = N0 + dt * dk3_dx;
    const MatXd dx3r_du = dt * dk3_du;
    auto H3r_xx = cubeAdd(NH0, cubeScale(dt, Kxx3));
    auto H3r_ux = cubeScale(dt, Kux3);
    auto H3r_uu = cubeScale(dt, Kuu3);
    VecXd x_3 = x_3r; x_3.segment<4>(quat_idx).normalize();
    {
        auto NS = normalizeStage(x_3r, dx3r_dx, dx3r_du, H3r_xx, H3r_ux, H3r_uu);
        Mx_in = std::get<1>(NS); Mu_in = std::get<2>(NS);
        Hxx_in = std::move(std::get<3>(NS));
        Hux_in = std::move(std::get<4>(NS));
        Huu_in = std::move(std::get<5>(NS));
    }

    // ------------------------------------------------------------------
    // Stage 4: k4 = f(x_3, u)
    // ------------------------------------------------------------------
    MatXd A4(nx, nx), B4(nx, nu);
    VecXd k4(nx);
    auto fxx4 = makeXX(); auto fux4 = makeUX(); auto fuu4 = makeUU();
    dynamics_hess(t + dt, x_3, u, A4, B4, k4, fxx4, fux4, fuu4);
    auto K4 = composeF(A4, fxx4, fux4, fuu4, Mx_in, Mu_in, Hxx_in, Hux_in, Huu_in);
    auto& Kxx4 = std::get<0>(K4); auto& Kux4 = std::get<1>(K4); auto& Kuu4 = std::get<2>(K4);
    const MatXd dk4_dx = A4 * Mx_in;
    const MatXd dk4_du = A4 * Mu_in + B4;

    // ------------------------------------------------------------------
    // Combine: x_next_raw = x_norm + dt/6 · (k1 + 2·k2 + 2·k3 + k4)
    //   d²x_next_raw/d(·)d(·) = NH0 (only on xx) + dt/6 · Σ w_i · K_i.
    // ------------------------------------------------------------------
    const double w = dt / 6.0;
    const VecXd x_next_raw = x_norm + w * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    const MatXd dxnext_raw_dx = N0 + w * (dk1_dx + 2.0 * dk2_dx + 2.0 * dk3_dx + dk4_dx);
    const MatXd dxnext_raw_du = w * (dk1_du + 2.0 * dk2_du + 2.0 * dk3_du + dk4_du);

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
    auto Hraw_xx = cubeAdd(NH0, sum4(Kxx1, Kxx2, Kxx3, Kxx4));
    auto Hraw_ux = sum4(Kux1, Kux2, Kux3, Kux4);
    auto Hraw_uu = sum4(Kuu1, Kuu2, Kuu3, Kuu4);

    // Final normalization: x_{k+1} = norm(x_next_raw).
    const MatXd N_next = stateNormJacobian(x_next_raw, nx, quat_idx);
    const auto NH_next = stateNormHessian(x_next_raw, nx, quat_idx);

    F_xx = cubeAdd(
        matTimesCube(dxnext_raw_dx.transpose(),
                     cubeTimesMat(NH_next, dxnext_raw_dx)),
        matOverCube(N_next, Hraw_xx)
    );
    F_ux = cubeAdd(
        matTimesCube(dxnext_raw_du.transpose(),
                     cubeTimesMat(NH_next, dxnext_raw_dx)),
        matOverCube(N_next, Hraw_ux)
    );
    F_uu = cubeAdd(
        matTimesCube(dxnext_raw_du.transpose(),
                     cubeTimesMat(NH_next, dxnext_raw_du)),
        matOverCube(N_next, Hraw_uu)
    );
}

}  // namespace saltro::math