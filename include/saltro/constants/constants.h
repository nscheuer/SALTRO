#pragma once
#include <cmath>

namespace saltro::constants {
    /**
     * @brief Conversion factor from degrees to radians.
     * 
     * Defined as \f$\frac{\pi}{180}\f$.
     */
    inline constexpr double DEG2RAD = M_PI / 180.0;
    
    /**
     * @brief Number of seconds in one Julian century.
     * 
     * Defined as \f$36525 \times 86400 = 3155760000\f$. Used for time conversions
     * in ephemeris computations.
     */
    inline constexpr double SEC_PER_JULIAN_CENTURY = 36525.0 * 86400.0;

    /**
     * @brief Astronomical Unit in meters.
     * 
     * The mean distance from Earth to Sun: \f$1.49597870700 \times 10^{11}\f$ meters,
     * used in solar ephemeris calculations.
     */
    inline constexpr double AU_M = 149597870700.0;

    /**
     * @brief Speed of light in vacuum.
     * 
     * Defined as exactly \f$299\,792\,458\f$ m/s.
     */
    inline constexpr double C_LIGHT = 299792458.0; // m/s
    
    /**
     * @brief Solar irradiance constant.
     * 
     * Mean solar irradiance at 1 AU, approximately \f$1361\f$ W/m².
     * Used for solar radiation pressure drag calculations.
     */
    inline constexpr double SOLAR_CONSTANT = 1361.0; // W/m^2
    
    /**
     * @brief Solar radius in meters.
     * 
     * Approximate mean radius of the Sun: \f$6.957 \times 10^8\f$ m.
     */
    inline constexpr double R_SUN = 6.957e8; // m

    /**
     * @brief Earth's gravitational parameter.
     * 
        * Defined as \f$\mu = 3.986\,004\,418 \times 10^{14}\f$ m³/s²,
     * where \f$\mu = GM\f$ and \f$G\f$ is the gravitational constant.
     * Used in Kepler orbit propagation.
     */
    inline constexpr double MU_EARTH = 3.986004418e14; // km^3/s^2
    
    /**
     * @brief Earth's mean equatorial radius.
     * 
     * Defined as \f$6\,378\,136.3\f$ meters.
     */
    inline constexpr double R_EARTH = 6378136.3; // m
    
    /**
     * @brief Earth's oblateness parameter (J2).
     * 
     * Dimensionless coefficient describing Earth's equatorial bulge:
     * \f[J_2 \approx 1.08263 \times 10^{-3}\f]
     * Controls the magnitude of the J2 perturbation in orbit propagation.
     */
    inline constexpr double J2_EARTH = 1.08263e-3;

    /**
     * @brief Vacuum permeability divided by \f$4\pi\f$.
     * 
     * Defined as \f$\mu_0/(4\pi) = 10^{-7}\f$ N/A².
     */
    inline constexpr double MU0_OVER_4PI = 1e-7; // N/A^2
    
    /**
     * @brief Earth's magnetic dipole moment magnitude.
     * 
     * Approximately \f$7.94 \times 10^{22}\f$ A·m².
     */
    inline constexpr double M_EARTH = 7.94e22; // A*m^2
    
    /**
     * @brief Product of permeability and magnetic moment.
     * 
     * Defined as \f$K = \frac{\mu_0}{4\pi} M_{\text{Earth}}\f$ in units of T·m³.
     * Used in dipole magnetic field calculations.
     */
    inline constexpr double K = MU0_OVER_4PI * M_EARTH; // T*m^3
    
    /**
     * @brief Tilt angle of Earth's magnetic dipole.
     * 
     * Defined as \f$11°\f$ in radians. The dipole is tilted relative to
     * the geographic pole in the dipole model.
     */
    inline constexpr double TILT_RAD = 11.0 * DEG2RAD; // radians
    
    /**
     * @brief Longitude of the magnetic dipole axis.
     * 
     * Defined as \f$289°\f$ in radians (measured from prime meridian).
     * Used to locate the magnetic dipole relative to Earth's geographic reference frame.
     */
    inline constexpr double LON_RAD = 289.0 * DEG2RAD; // radians
}