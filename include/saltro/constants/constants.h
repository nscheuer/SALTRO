#pragma once
#include <cmath>

namespace saltro::constants {
    inline constexpr double DEG2RAD = M_PI / 180.0;
    inline constexpr double SEC_PER_JULIAN_CENTURY = 36525.0 * 86400.0;

    inline constexpr double AU_M = 149597870700.0;

    inline constexpr double C_LIGHT = 299792458.0; // m/s
    inline constexpr double SOLAR_CONSTANT = 1361.0; // W/m^2

    inline constexpr double MU_EARTH = 3.986004418e14; // km^3/s^2
    inline constexpr double R_EARTH = 6378136.3; // m
    inline constexpr double J2_EARTH = 1.08263e-3;

    inline constexpr double MU0_OVER_4PI = 1e-7; // N/A^2
    inline constexpr double M_EARTH = 7.94e22; // A*m^2
    inline constexpr double K = MU0_OVER_4PI * M_EARTH; // T*m^3
    inline constexpr double TILT_RAD = 11.0 * DEG2RAD; // radians
    inline constexpr double LON_RAD = 289.0 * DEG2RAD; // radians
}