#pragma once

#include <utility>

/**
 * @brief Perform a single step of second-order Runge-Kutta (RK2) integration.
 *
 * This function performs one time step of the explicit second-order Runge-Kutta
 * method (midpoint method) for solving ordinary differential equations of the form:
 * \f[
 * \frac{d\mathbf{x}}{dt} = \mathbf{f}(t, \mathbf{x})
 * \f]
 *
 * The RK2 method updates the state using two function evaluations:
 * \f[
 * \mathbf{k}_1 = \mathbf{f}(t, \mathbf{x}) \\
 * \mathbf{k}_2 = \mathbf{f}\left(t + \frac{\Delta t}{2}, \mathbf{x} + \frac{\Delta t}{2}\mathbf{k}_1\right) \\
 * \mathbf{x}_{\text{out}} = \mathbf{x} + \Delta t \, \mathbf{k}_2
 * \f]
 *
 * RK2 has a local truncation error of \f$O(\Delta t^3)\f$ and is second-order accurate.
 * It is a good compromise between accuracy and computational cost for many applications.
 *
 * @tparam State Type of the state vector (must support scalar multiplication and addition).
 * @tparam DerivFunc Type of the derivative function/functor.
 *
 * @param f Derivative function with signature: `void f(double t, const State& x, State& dxdt)`.
 *           Computes the derivative at time `t` and state `x`, storing the result in `dxdt`.
 * @param x Current state vector.
 * @param t Current time.
 * @param dt Time step size.
 * @param x_out Output state vector after one RK2 step.
 */

template <typename State, typename DerivFunc>
void rk2_step(
    DerivFunc&& f,
    const State& x,
    double t,
    double dt,
    State& x_out
) {
    State k1, k2, xtemp;

    f(t, x, k1);

    xtemp = x + 0.5 * dt * k1;
    f(t + 0.5 * dt, xtemp, k2);

    x_out = x + dt * k2;
}
