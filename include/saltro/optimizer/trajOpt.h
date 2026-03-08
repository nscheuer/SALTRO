/**
 * @file trajOpt.h
 * @brief Top-level trajectory optimization interface combining warm-start and iLQR.
 */
#pragma once

#include <vector>
#include <Eigen/Dense>

#include <saltro/limits.h>
#include <saltro/pybind/plannersettings.h>
#include <saltro/pybind/satellite.h>

namespace saltro::optimizer {

/**
 * @brief Solve satellite attitude trajectory optimization problem.
 *
 * High-level interface that orchestrates the complete trajectory optimization process:
 * 1. Warm-start: Generate initial trajectory using reference controller or open-loop propagation
 * 2. iLQR: Iteratively refine trajectory using differential dynamic programming
 *
 * The optimization minimizes a cost function that typically includes:
 * - Attitude tracking error (quaternion distance to q_goal)
 * - Angular velocity regulation
 * - Control effort (torques, momentum management)
 * - Constraint violations (augmented Lagrangian penalties for path constraints)
 *
 * The problem is formulated as:
 * \f[
 * \min_{\mathbf{u}_0, \ldots, \mathbf{u}_{N-1}} \, \ell_N(\mathbf{x}_N) + 
 * \sum_{k=0}^{N-1} \ell_k(\mathbf{x}_k, \mathbf{u}_k)
 * \f]
 * subject to:
 * \f[
 * \mathbf{x}_{k+1} = \mathbf{f}(\mathbf{x}_k, \mathbf{u}_k, t_k), \quad k = 0, \ldots, N-1
 * \f]
 * where \f$\mathbf{x} = [\boldsymbol{\omega}, \mathbf{q}, \mathbf{h}_{\text{rw}}]\f$ is the
 * attitude state and \f$\mathbf{u}\f$ contains actuator commands (MTQ dipoles, RW torques).
 *
 * @param settings Planner configuration containing cost weights, constraints, AL parameters,
 *                 iLQR settings, number of passes, and timestep
 * @param satellite Satellite model with inertia, actuators, and dynamics/cost functions
 * @param x0 Initial attitude state (7+nRW × 1): [ω(3), q(4), h_rw(nRW)]
 * @param r0 Initial position in ECI frame (3 × 1), meters
 * @param v0 Initial velocity in ECI frame (3 × 1), m/s
 * @param jtime Mission time vector in Julian centuries (N × 1)
 * @param q_goal Goal quaternion trajectory (4 × N) defining desired attitude over time
 * @param boresight Boresight constraint trajectory (3 × N), unit vectors in body frame
 * @param X Output: optimized state trajectory (N × state_dim)
 * @param U Output: optimized control trajectory (N-1 × input_dim)
 * @param K Output: final feedback gain trajectory (N-1 × input_dim × reduced_state_dim)
 * @param state_dim State dimension (7 + nRW)
 * @param input_dim Control dimension (3*nMTQ + nRW + 3 for magic torque)
 * @param N Output: horizon length (updated if trajectory is extended/truncated)
 * @return true if optimization succeeded (converged or reached max iterations),
 *         false if validation failed or numerical error occurred
 */
bool trajOpt(
	const PlannerSettings& settings,
	const Satellite& satellite,
	const Satellite::VecX& x0,
	const Eigen::Vector3d& r0,
	const Eigen::Vector3d& v0,
	const Eigen::Ref<const Eigen::VectorXd>& jtime,
	const Eigen::Ref<const Eigen::MatrixXd>& q_goal,
	const Eigen::Ref<const Eigen::MatrixXd>& boresight,

	Eigen::Ref<Eigen::MatrixXd> X,
	Eigen::Ref<Eigen::MatrixXd> U,
	Eigen::Ref<Eigen::MatrixXd> K,

	int state_dim,
	int input_dim,
	int& N
);

}
