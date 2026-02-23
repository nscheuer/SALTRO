#pragma once

namespace saltro::limits {
    inline constexpr int MAX_LENGTH_TRAJ = 5000;

    inline constexpr int KEPLER_MAX_ITERS = 8;
    inline constexpr double KEPLER_TOLERANCE = 1e-10;

    inline constexpr int MAX_NUM_PASSES = 3;
    inline constexpr int MAX_NUM_MTQ    = 4;
    inline constexpr int MAX_NUM_RW     = 4;
    inline constexpr int MAX_NUM_GEOMETRY_FACES = 20;

    // Derived maximum dimensions for statically-allocated vectors/matrices.
    // These ensure no heap allocation occurs in the Satellite class or configs.
    inline constexpr int MAX_STATE_DIM         = 7 + MAX_NUM_RW;                                  // 11
    inline constexpr int MAX_REDUCED_STATE_DIM = 6 + MAX_NUM_RW;                                  // 10
    inline constexpr int MAX_CTRL_DIM          = MAX_NUM_MTQ + MAX_NUM_RW;                         //  8
    inline constexpr int MAX_CONSTRAINT_DIM    = 1 + 1 + 2 * MAX_NUM_MTQ + 5 * MAX_NUM_RW;        // 30
    //                                           AV  sun  MTQ bounds       RW (torq+mom+stiction)
}