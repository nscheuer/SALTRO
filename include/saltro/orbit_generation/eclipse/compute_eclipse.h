#pragma once

#include <Eigen/Dense>
#include <saltro/limits.h>

namespace saltro::orbits {

/**
 * @brief Compute Earth eclipse shadow along a trajectory using a selected model.
 *
 * Dispatches eclipse computation to one of the available eclipse shadow
 * models based on the integer selector \p eclipse_model. When a satellite
 * is in Earth's shadow (eclipse), the Sun vector is zeroed out.
 *
 * For each trajectory sample \f$k\f$, the eclipse state is evaluated and
 * the Sun vector is set to zero if eclipsed:
 * \f[
 * \mathbf{S}_k = \begin{cases}
 * \mathbf{0} & \text{if satellite is in eclipse} \\
 * \mathbf{S}_k & \text{otherwise}
 * \end{cases}
 * \f]
 *
 * Supported models:
 * - \f$0\f$ → Cylindrical shadow model
 * - \f$1\f$ → Analytical penumbra/umbra shadow cone model
 *
 * @param R Spacecraft position vectors (meters). Each column corresponds
 *          to one trajectory sample.
 * @param jtime Julian time values (unused for current models, included for
 *              future expansion).
 * @param jtime_length Number of valid trajectory samples.
 * @param eclipse_model Integer identifier selecting the eclipse model:
 *        0 = cylindrical shadow.
 *        1 = analytical penumbra/umbra shadow cone.
 * @param S Spacecraft-to-Sun vectors (meters), column-wise. Modified in-place;
 *          set to zero where spacecraft is in eclipse.
 *
 * @return True if the selected model computation succeeds for all samples,
 *         false if the model identifier is invalid or a computation fails.
 */
bool compute_eclipse(
    const Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& R,
    const Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ>& jtime,
    const int jtime_length,
    const int eclipse_model,

    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& S
);

}
