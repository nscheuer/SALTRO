#pragma once

namespace saltro::math {
/**
 * @brief Add two real numbers.
 *
 * Computes:
 * \f[
 * c = a + b
 * \f]
 *
 * Example equation with vectors:
 * \f[
 * \mathbf{x}_{k+1} = \mathbf{x}_k + \Delta \mathbf{x}
 * \f]
 *
 * @param a First value
 * @param b Second value
 * @return Sum \f$a+b\f$
 */
double add(double a, double b);
}