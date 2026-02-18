#pragma once

namespace saltro::constants {

constexpr int IGRF13_NMAX = 13;

extern const double IGRF13_G[IGRF13_NMAX+1][IGRF13_NMAX+1];
extern const double IGRF13_H[IGRF13_NMAX+1][IGRF13_NMAX+1];

constexpr double IGRF_EARTH_REFERENCE_RADIUS = 6371200.0;

}