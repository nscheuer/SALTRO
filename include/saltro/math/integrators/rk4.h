#pragma once

#include <utility>

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