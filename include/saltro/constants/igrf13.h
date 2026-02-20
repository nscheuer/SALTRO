#pragma once

namespace saltro::constants::IGRF13 {

constexpr int NMAX = 13;

extern const double G[NMAX+1][NMAX+1];
extern const double H[NMAX+1][NMAX+1];

constexpr double EARTH_REFERENCE_RADIUS = 6371200.0;

}