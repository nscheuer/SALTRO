#pragma once

#include <utility>
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
template <typename DynamicsJacFunc>
void rk4_jacobians(
    DynamicsJacFunc&& dynamics_jac,
    const Eigen::Ref<const Eigen::VectorXd>& x,
    const Eigen::Ref<const Eigen::VectorXd>& u,
    double t,
    double dt,
    Eigen::Ref<Eigen::MatrixXd> A_discrete,
    Eigen::Ref<Eigen::MatrixXd> B_discrete
) {
    const int nx = static_cast<int>(x.size());
    const int nu = static_cast<int>(u.size());

    // Identity matrix (reused multiple times)
    const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(nx, nx);

    // Continuous-time Jacobians at each stage
    Eigen::MatrixXd A_c1(nx, nx), A_c2(nx, nx), A_c3(nx, nx), A_c4(nx, nx);
    Eigen::MatrixXd B_c1(nx, nu), B_c2(nx, nu), B_c3(nx, nu), B_c4(nx, nu);

    // Discrete sensitivities ∂k_i/∂x and ∂k_i/∂u
    Eigen::MatrixXd dk1_dx(nx, nx), dk2_dx(nx, nx), dk3_dx(nx, nx), dk4_dx(nx, nx);
    Eigen::MatrixXd dk1_du(nx, nu), dk2_du(nx, nu), dk3_du(nx, nu), dk4_du(nx, nu);

    // Intermediate states for RK4 stages
    Eigen::VectorXd x2(nx), x3(nx), x4(nx);
    Eigen::VectorXd k1(nx), k2(nx), k3(nx), k4(nx);

    // ========================================================================
    // Stage 1: k1 = f(t, x, u)
    // ========================================================================
    dynamics_jac(t, x, u, A_c1, B_c1, k1);
    dk1_dx = A_c1;
    dk1_du = B_c1;

    // ========================================================================
    // Stage 2: k2 = f(t + dt/2, x + dt/2 * k1, u)
    // ========================================================================
    x2 = x + 0.5 * dt * k1;
    dynamics_jac(t + 0.5 * dt, x2, u, A_c2, B_c2, k2);
    dk2_dx = A_c2 * (I + 0.5 * dt * dk1_dx);
    dk2_du = A_c2 * (0.5 * dt * dk1_du) + B_c2;

    // ========================================================================
    // Stage 3: k3 = f(t + dt/2, x + dt/2 * k2, u)
    // ========================================================================
    x3 = x + 0.5 * dt * k2;
    dynamics_jac(t + 0.5 * dt, x3, u, A_c3, B_c3, k3);
    dk3_dx = A_c3 * (I + 0.5 * dt * dk2_dx);
    dk3_du = A_c3 * (0.5 * dt * dk2_du) + B_c3;

    // ========================================================================
    // Stage 4: k4 = f(t + dt, x + dt * k3, u)
    // ========================================================================
    x4 = x + dt * k3;
    dynamics_jac(t + dt, x4, u, A_c4, B_c4, k4);
    dk4_dx = A_c4 * (I + dt * dk3_dx);
    dk4_du = A_c4 * (dt * dk3_du) + B_c4;

    // ========================================================================
    // Assemble discrete Jacobians
    // ========================================================================
    // A_discrete = I + (dt/6) * (∂k1/∂x + 2*∂k2/∂x + 2*∂k3/∂x + ∂k4/∂x)
    A_discrete = I + (dt / 6.0) * (dk1_dx + 2.0 * dk2_dx + 2.0 * dk3_dx + dk4_dx);

    // B_discrete = (dt/6) * (∂k1/∂u + 2*∂k2/∂u + 2*∂k3/∂u + ∂k4/∂u)
    B_discrete = (dt / 6.0) * (dk1_du + 2.0 * dk2_du + 2.0 * dk3_du + dk4_du);
}