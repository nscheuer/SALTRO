#pragma once
#include <array>

namespace saltro::constants {

constexpr int IGRF8_NMAX = 8;

extern const double IGRF8_G[IGRF8_NMAX+1][IGRF8_NMAX+1];
extern const double IGRF8_H[IGRF8_NMAX+1][IGRF8_NMAX+1];

constexpr double IGRF_EARTH_REFERENCE_RADIUS = 6371200.0;

}