#pragma once

namespace saltro::constants::NRELSPA {

    /**
     * @brief Heliocentric longitude tables (L0-L5).
     * 
     * Tabulated Fourier series coefficients for computing the heliocentric
     * ecliptic longitude of the Earth-Sun line. Six tables correspond to
     * increasingly higher precision terms in the NREL Solar Position Algorithm.
     * Each table row contains three coefficients: amplitude, angle frequency,
     * and phase offset.
     */
    extern const double L0_TABLE[][3]; extern const int L0_COUNT;
    extern const double L1_TABLE[][3]; extern const int L1_COUNT;
    extern const double L2_TABLE[][3]; extern const int L2_COUNT;
    extern const double L3_TABLE[][3]; extern const int L3_COUNT;
    extern const double L4_TABLE[][3]; extern const int L4_COUNT;
    extern const double L5_TABLE[][3]; extern const int L5_COUNT;

    /**
     * @brief Heliocentric latitude tables (B0-B1).
     * 
     * Tabulated Fourier coefficients for heliocentric ecliptic latitude.
     * Two tables provide increasing precision in latitude calculations.
     */
    extern const double B0_TABLE[][3]; extern const int B0_COUNT;
    extern const double B1_TABLE[][3]; extern const int B1_COUNT;

    /**
     * @brief Heliocentric radius tables (R0-R4).
     * 
     * Tabulated Fourier coefficients for Earth-Sun distance (in AU).
     * Five tables provide increasing precision in radius vector calculations.
     */
    extern const double R0_TABLE[][3]; extern const int R0_COUNT;
    extern const double R1_TABLE[][3]; extern const int R1_COUNT;
    extern const double R2_TABLE[][3]; extern const int R2_COUNT;
    extern const double R3_TABLE[][3]; extern const int R3_COUNT;
    extern const double R4_TABLE[][3]; extern const int R4_COUNT;

    /**
     * @brief Nutation coefficients table (IAU 1980 standard).
     * 
     * Tabulated coefficients for computing nutation corrections to Earth's
     * precession. Contains 63 terms as per the IAU 1980 nutation model.
     * Each row contains 9 coefficients for harmonic and polynomial evaluation.
     */
    extern const double Y_TABLE[][9];
    extern const int Y_COUNT;

} // namespace saltro::constants::NRELSPA