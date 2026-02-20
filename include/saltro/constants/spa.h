#pragma once

namespace saltro::constants::NRELSPA {

    // Heliocentric Tables (L, B, R)
    // L0-L5: Longitude, B0-B1: Latitude, R0-R4: Radius
    extern const double L0_TABLE[][3]; extern const int L0_COUNT;
    extern const double L1_TABLE[][3]; extern const int L1_COUNT;
    extern const double L2_TABLE[][3]; extern const int L2_COUNT;
    extern const double L3_TABLE[][3]; extern const int L3_COUNT;
    extern const double L4_TABLE[][3]; extern const int L4_COUNT;
    extern const double L5_TABLE[][3]; extern const int L5_COUNT;

    extern const double B0_TABLE[][3]; extern const int B0_COUNT;
    extern const double B1_TABLE[][3]; extern const int B1_COUNT;

    extern const double R0_TABLE[][3]; extern const int R0_COUNT;
    extern const double R1_TABLE[][3]; extern const int R1_COUNT;
    extern const double R2_TABLE[][3]; extern const int R2_COUNT;
    extern const double R3_TABLE[][3]; extern const int R3_COUNT;
    extern const double R4_TABLE[][3]; extern const int R4_COUNT;

    // Nutation Table (63 terms as per IAU 1980)
    extern const double Y_TABLE[][9];
    extern const int Y_COUNT;

} // namespace saltro::constants::NRELSPA