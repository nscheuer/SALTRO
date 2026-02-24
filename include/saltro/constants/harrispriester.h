#pragma once

#include <array>
#include <cstddef>

namespace saltro::constants {

/**
 * @brief WGS84 ellipsoid semi-major axis (equatorial radius).
 * 
 * Defined as \f$6\,378\,137\f$ meters. Used for computing geodetic altitude
 * from Earth-centered coordinates.
 */
inline constexpr double WGS84_A_M = 6378137.0;

/**
 * @brief WGS84 ellipsoid flattening parameter.
 * 
 * Defined as \f$f = \frac{1}{298.257223563} \approx 3.357 \times 10^{-3}\f$.
 * Characterizes Earth's oblate spheroid shape.
 */
inline constexpr double WGS84_F = 1.0 / 298.257223563;

/**
 * @brief Solar direction lag angle for Harris-Priester model.
 * 
 * Defines the phase lag between the spacecraft's local solar noon and the
 * actual peak of the atmospheric bulge. Defined as \f$30°\f$ (≈ 0.524 radians).
 * Accounts for the diurnal bulge offset due to atmospheric winds.
 */
inline constexpr double HARRIS_PRIESTER_LAG_RAD = 0.5235987755982988730771072305465838;

/**
 * @brief Minimum cosine value threshold for Harris-Priester model.
 * 
     * Used to avoid division by very small numbers in the weighting function.
 * Set to \f$10^{-12}\f$ for numerical stability.
 */
inline constexpr double HARRIS_PRIESTER_MIN_COS = 1e-12;

/**
 * @brief Exponent for cosine weighting in Harris-Priester model.
 * 
 * Controls the steepness of the day-night density transition. An exponent of 4
 * provides a smooth transition between minimum and maximum densities.
 */
inline constexpr double HARRIS_PRIESTER_COS_EXPONENT = 4.0;

/**
 * @brief Single entry in the Harris-Priester atmospheric density table.
 * 
 * Each entry corresponds to a discrete altitude level and provides both
 * minimum (nightside) and maximum (dayside) density values.
 */
struct HarrisPriesterEntry {
    double alt_m;           ///< Altitude above Earth surface in meters
    double rho_min_kg_m3;   ///< Minimum (nightside) density in kg/m³
    double rho_max_kg_m3;   ///< Maximum (dayside) density in kg/m³
};

/**
 * @brief Harris-Priester atmospheric density model lookup table.
 * 
 * Tabulated density values at 50 discrete altitudes from 100 to 1000 km.
 * Each entry contains minimum (nightside) and maximum (dayside) density
 * values. Interpolation or table lookup is performed during density
 * computation to evaluate the model at arbitrary altitudes.
 * 
 * Reference: Harris, M. J., and W. Priester, 1962.
 */
inline constexpr std::array<HarrisPriesterEntry, 50> HARRIS_PRIESTER_TABLE = {{
    {  100000.0, 4.974e-07, 4.974e-07 },
    {  120000.0, 2.490e-08, 2.490e-08 },
    {  130000.0, 8.377e-09, 8.710e-09 },
    {  140000.0, 3.899e-09, 4.059e-09 },
    {  150000.0, 2.122e-09, 2.215e-09 },
    {  160000.0, 1.263e-09, 1.344e-09 },
    {  170000.0, 8.008e-10, 8.758e-10 },
    {  180000.0, 5.283e-10, 6.010e-10 },
    {  190000.0, 3.617e-10, 4.297e-10 },
    {  200000.0, 2.557e-10, 3.162e-10 },
    {  210000.0, 1.839e-10, 2.396e-10 },
    {  220000.0, 1.341e-10, 1.853e-10 },
    {  230000.0, 9.949e-11, 1.455e-10 },
    {  240000.0, 7.488e-11, 1.157e-10 },
    {  250000.0, 5.709e-11, 9.308e-11 },
    {  260000.0, 4.403e-11, 7.555e-11 },
    {  270000.0, 3.430e-11, 6.182e-11 },
    {  280000.0, 2.697e-11, 5.095e-11 },
    {  290000.0, 2.139e-11, 4.226e-11 },
    {  300000.0, 1.708e-11, 3.526e-11 },
    {  320000.0, 1.099e-11, 2.511e-11 },
    {  340000.0, 7.214e-12, 1.819e-11 },
    {  360000.0, 4.824e-12, 1.337e-11 },
    {  380000.0, 3.274e-12, 9.955e-12 },
    {  400000.0, 2.249e-12, 7.492e-12 },
    {  420000.0, 1.558e-12, 5.684e-12 },
    {  440000.0, 1.091e-12, 4.355e-12 },
    {  460000.0, 7.701e-13, 3.362e-12 },
    {  480000.0, 5.474e-13, 2.612e-12 },
    {  500000.0, 3.916e-13, 2.042e-12 },
    {  520000.0, 2.819e-13, 1.605e-12 },
    {  540000.0, 2.042e-13, 1.267e-12 },
    {  560000.0, 1.488e-13, 1.005e-12 },
    {  580000.0, 1.092e-13, 7.997e-13 },
    {  600000.0, 8.070e-14, 6.390e-13 },
    {  620000.0, 6.012e-14, 5.123e-13 },
    {  640000.0, 4.519e-14, 4.121e-13 },
    {  660000.0, 3.430e-14, 3.325e-13 },
    {  680000.0, 2.632e-14, 2.691e-13 },
    {  700000.0, 2.043e-14, 2.185e-13 },
    {  720000.0, 1.607e-14, 1.779e-13 },
    {  740000.0, 1.281e-14, 1.452e-13 },
    {  760000.0, 1.036e-14, 1.190e-13 },
    {  780000.0, 8.496e-15, 9.776e-14 },
    {  800000.0, 7.069e-15, 8.059e-14 },
    {  840000.0, 4.680e-15, 5.741e-14 },
    {  880000.0, 3.200e-15, 4.210e-14 },
    {  920000.0, 2.210e-15, 3.130e-14 },
    {  960000.0, 1.560e-15, 2.360e-14 },
    { 1000000.0, 1.150e-15, 1.810e-14 }
}};

}