#pragma once

namespace saltro::constants::IGRF13 {

/**
 * @brief Maximum degree of spherical harmonic expansion (IGRF-13).
 * 
 * The IGRF-13 magnetic field model uses Gauss coefficients up to
 * degree and order \f$n = m = 13\f$, providing high accuracy for
 * global field representation across all altitudes.
 */
constexpr int NMAX = 13;

/**
 * @brief Gauss coefficients (cosine terms) for IGRF-13 model.
 * 
 * Tabulated Gauss coefficients \f$g_n^m\f$ representing the cosine
 * components of the spherical harmonic expansion of Earth's magnetic field.
 * Array is dimensioned as \f$(N_{\max}+1) \times (N_{\max}+1)\f$.
 */
extern const double G[NMAX+1][NMAX+1];

/**
 * @brief Gauss coefficients (sine terms) for IGRF-13 model.
 * 
 * Tabulated Gauss coefficients \f$h_n^m\f$ representing the sine
 * components of the spherical harmonic expansion. Array is dimensioned
 * as \f$(N_{\max}+1) \times (N_{\max}+1)\f$.
 */
extern const double H[NMAX+1][NMAX+1];

/**
 * @brief Earth reference radius for IGRF coefficients.
 * 
 * Defined as \f$6\,371\,200\f$ meters. Gauss coefficients are
 * normalized with respect to this radius.
 */
constexpr double EARTH_REFERENCE_RADIUS = 6371200.0;

}