/**
 * @file warm_start.h
 * @brief Initial trajectory generation for iLQR warm-starting.
 */
#pragma once

#include <Eigen/Dense>

#include <saltro/limits.h>
#include <saltro/pybind/satellite.h>

namespace saltro::optimizer {

/**
 * @brief Generate initial trajectory for warm-starting iLQR optimization.
 *
 * Produces a kinematically feasible initial guess for the state and control trajectories
 * by simulating the satellite under a reference controller (e.g., B-dot, zero control,
 * or excitation controller) specified in the settings.
 *
 * The warm-start trajectory is computed by:
 * 1. Selecting the reference controller from `settings.init_traj.initcontroller`
 * 2. Forward-propagating the dynamics from x0 using the controller:
 *    \f[
 *    \mathbf{x}_{k+1} = \mathbf{f}(\mathbf{x}_k, \mathbf{u}_k^{\text{ref}}, t_k)
 *    \f]
 *    where \f$\mathbf{u}_k^{\text{ref}}\f$ is computed by the reference controller
 * 3. Storing the resulting state and control sequences in X and U
 *
 * A good warm-start is crucial for iLQR convergence, as it:
 * - Initializes the trajectory near the feasible constraint manifold
 * - Reduces initial cost to reasonable values
 * - Provides a starting point within the basin of attraction of the optimizer
 *
 * Typical controllers used:
 * - **B-dot controller**: Detumbles the satellite by opposing magnetic field rate
 * - **Zero controller**: Open-loop propagation with zero actuation
 * - **Excitation controller**: Adds persistent excitation for parameter estimation
 *
 * @param settings Planner settings containing initial trajectory configuration
 *                 (controller type, gains, etc.)
 * @param satellite Satellite model with dynamics and reference controllers
 * @param x0 Initial attitude state (7+nRW × 1): [ω(3), q(4), h_rw(nRW)]
 * @param jtime Mission time vector in Julian centuries (N × 1)
 * @param q_goal Goal quaternion trajectory (4 × N) for reference controller
 * @param boresight Boresight constraint trajectory (3 × N), unit vectors
 * @param N Horizon length (number of timesteps)
 * @param R Position trajectory in ECI frame (3 × MAX_LENGTH_TRAJ), meters
 * @param V Velocity trajectory in ECI frame (3 × MAX_LENGTH_TRAJ), m/s
 * @param B Magnetic field trajectory in ECI frame (3 × MAX_LENGTH_TRAJ), Tesla
 * @param S Sun direction trajectory in ECI frame (3 × MAX_LENGTH_TRAJ), unit vectors
 * @param rho Atmospheric density trajectory (1 × MAX_LENGTH_TRAJ), kg/m³
 * @param X Output: initial state trajectory (N × state_dim)
 * @param U Output: initial control trajectory (N-1 × input_dim)
 * @return true if warm-start succeeded, false if propagation failed
 */
bool warm_start(
    const PlannerSettings& settings,
    const Satellite& satellite,
    const Satellite::VecX& x0,
    const Eigen::Ref<const Eigen::VectorXd>& jtime,
    const Eigen::Ref<const Eigen::MatrixXd>& q_goal,
    const Eigen::Ref<const Eigen::MatrixXd>& boresight,
    int N,
    const Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& R,
    const Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& V,
    const Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& B,
    const Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& S,
    const Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ>& rho,
    Eigen::Ref<Eigen::MatrixXd> X,
    Eigen::Ref<Eigen::MatrixXd> U
);

}
