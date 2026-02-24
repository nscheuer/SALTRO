#pragma once

namespace saltro::limits {
    /**
     * @brief Maximum length of a discretized trajectory.
     * 
     * Defines the compile-time maximum number of samples that can be stored
     * in trajectory arrays. Used for pre-allocation in fixed-size containers.
     */
    inline constexpr int MAX_LENGTH_TRAJ = 5000;

    /**
     * @brief Maximum number of iterations for Kepler equation solver.
     * 
     * Controls the iteration limit in Newton-Raphson or similar root-finding
     * algorithms when solving the Kepler equation for time of flight problems.
     */
    inline constexpr int KEPLER_MAX_ITERS = 8;
    
    /**
     * @brief Convergence tolerance for Kepler equation solver.
     * 
     * Defines the relative error threshold for terminating iterations in
     * the Kepler equation solver. A value of \f$10^{-10}\f$ ensures double
     * precision accuracy.
     */
    inline constexpr double KEPLER_TOLERANCE = 1e-10;

    /**
     * @brief Maximum number of outer optimization loop passes.
     * 
     * Constrains the outer iterations in trajectory optimization algorithms
     * such as ALTRO.
     */
    inline constexpr int MAX_NUM_PASSES = 3;
    
    /**
     * @brief Maximum number of magnetorquers (MTQ) on the satellite.
     * 
     * Defines the upper limit for magnetic torque coils that can be attached
     * to the satellite. Used for pre-allocation of control authority vectors.
     */
    inline constexpr int MAX_NUM_MTQ    = 4;
    
    /**
     * @brief Maximum number of reaction wheels (RW) on the satellite.
     * 
     * Defines the upper limit for momentum storage wheels on the satellite.
     * Used for pre-allocation of angular momentum state vectors.
     */
    inline constexpr int MAX_NUM_RW     = 4;
    
    /**
     * @brief Maximum number of geometric faces for disturbance computation.
     * 
     * Defines the upper limit for surface elements that can be modeled in
     * aerodynamic drag and solar radiation pressure calculations.
     */
    inline constexpr int MAX_NUM_GEOMETRY_FACES = 20;

    /**
     * @brief Derived maximum state dimension.
     * 
     * Equals \f$7 + \text{MAX\_NUM\_RW}\f$, where 7 accounts for attitude (quaternion)
     * and angular velocity (3D), plus one slot per reaction wheel for momentum storage.
     * Guarantees no heap allocation in fixed-size state containers.
     */
    inline constexpr int MAX_STATE_DIM         = 7 + MAX_NUM_RW;                                  // 11
    
    /**
     * @brief Derived maximum reduced state dimension.
     * 
     * Equals \f$6 + \text{MAX\_NUM\_RW}\f$, representing the minimal state needed
     * for optimization (attitude error + angular velocity + RW momentum).
     */
    inline constexpr int MAX_REDUCED_STATE_DIM = 6 + MAX_NUM_RW;                                  // 10
    
    /**
     * @brief Derived maximum control input dimension.
     * 
     * Equals \f$\text{MAX\_NUM\_MTQ} + \text{MAX\_NUM\_RW}\f$, the total number
     * of control channels (magnetorquer dipoles + reaction wheel torques).
     */
    inline constexpr int MAX_CTRL_DIM          = MAX_NUM_MTQ + MAX_NUM_RW;                         //  8
    
    /**
     * @brief Derived maximum constraint dimension.
     * 
     * Equals \f$1 + 1 + 2 \cdot \text{MAX\_NUM\_MTQ} + 5 \cdot \text{MAX\_NUM\_RW}\f$.
     * Accounts for: angular velocity bound (1), sun pointing constraint (1),
     * MTQ dipole bounds (2 per MTQ), and RW constraints: torque limit, momentum limits,
     * and friction (5 per RW).
     */
    inline constexpr int MAX_CONSTRAINT_DIM    = 1 + 1 + 2 * MAX_NUM_MTQ + 5 * MAX_NUM_RW;        // 30
}