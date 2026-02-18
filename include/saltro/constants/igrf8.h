#pragma once
#include <array>

namespace saltro::constants::IGRF8 {

constexpr int NMAX = 8;

extern const double G[NMAX+1][NMAX+1];
extern const double H[NMAX+1][NMAX+1];

constexpr double EARTH_REFERENCE_RADIUS = 6371200.0;

}