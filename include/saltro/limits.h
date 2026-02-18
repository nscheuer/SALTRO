#pragma once

namespace saltro::limits {
    inline constexpr int MAX_LENGTH_TRAJ = 5000;

    inline constexpr int KEPLER_MAX_ITERS = 8;
    inline constexpr double KEPLER_TOLERANCE = 1e-10;

    inline constexpr int MAX_NUM_PASSES = 3;
    inline constexpr int MAX_NUM_MTQ    = 4;
    inline constexpr int MAX_NUM_RW     = 4;
}