#pragma once

#include <Eigen/Dense>
#include <saltro/limits.h>

namespace saltro::orbits {

/**
 * @brief Propagate an orbit under the influence of the J2 perturbation using RK4 integration.
 *
 * This function integrates the equations of motion for a satellite orbiting Earth,
 * including the dominant perturbation due to Earth's oblateness (J2 effect).
 *
 * The dynamics combine the two-body gravitational acceleration with the J2 perturbation:
 * \f[
 * \frac{d^2\mathbf{r}}{dt^2} = -\frac{\mu}{r^3}\mathbf{r} + \mathbf{a}_{J2}
 * \f]
 *
 * where \f$\mathbf{a}_{J2}\f$ is the acceleration due to the quadrupole moment:
 * \f[
 * \mathbf{a}_{J2} = \frac{3}{2}J_2\mu\frac{R_e^2}{r^5}
 * \left[
 * x\left(5\frac{z^2}{r^2} - 1\right)\hat{\mathbf{i}} + 
 * y\left(5\frac{z^2}{r^2} - 1\right)\hat{\mathbf{j}} + 
 * z\left(5\frac{z^2}{r^2} - 3\right)\hat{\mathbf{k}}
 * \right]
 * \f]
 *
 * The integration uses fourth-order Runge-Kutta (RK4) with deterministic adaptive
 * substepping. The time step is automatically divided into smaller substeps if needed
 * to maintain accuracy, with a configurable maximum substep size.
 *
 * The J2 perturbation causes measurable orbital effects:
 * - **Nodal regression**: The right ascension of ascending node (RAAN) decreases.
 * - **Apsidal precession**: The argument of perigee changes.
 * - **Inclination conservation**: Inclination remains approximately constant.
 *
 * @param r0 Initial position vector (3D, in meters).
 * @param v0 Initial velocity vector (3D, in m/s).
 * @param jtime Row vector of Julian centuries at which to compute the trajectory.
 *              The first element is used as the reference epoch \f$t_0\f$.
 *              All subsequent times are propagated from the first element.
 * @param jtime_length Number of time points to propagate (length of jtime and output arrays).
 * @param R Output matrix of position vectors (3 × jtime_length).
 *          Each column \f$R_i\f$ contains the position at time \f$t_i\f$.
 * @param V Output matrix of velocity vectors (3 × jtime_length).
 *          Each column \f$V_i\f$ contains the velocity at time \f$t_i\f$.
 *
 * @return True if integration completed successfully, false otherwise.
 *
 * @note
 * - Input time (`jtime`) must be in **Julian centuries**, not days. Multiply Julian
 *   days by \f$1/36525\f$ to convert.
 * - The initial state (r0, v0) is placed at the first output at index 0.
 * - All subsequent states are propagated from the first time point.
 * - The arrays R and V must have size at least (3 × MAX_LENGTH_TRAJ).
 */

bool compute_orbit_J2_RK4(
    const Eigen::Vector3d& r0,
    const Eigen::Vector3d& v0,
    const Eigen::Matrix<double,1,saltro::limits::MAX_LENGTH_TRAJ>& jtime,
    const int jtime_length,

    Eigen::Matrix<double,3,saltro::limits::MAX_LENGTH_TRAJ>& R,
    Eigen::Matrix<double,3,saltro::limits::MAX_LENGTH_TRAJ>& V
);

}