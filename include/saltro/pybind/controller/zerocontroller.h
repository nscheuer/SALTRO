/**
 * @file zerocontroller.h
 * @brief Open-loop controller that outputs zero actuation.
 */
#pragma once

#include <saltro/pybind/controller/controller.h>

namespace saltro::controller {

/**
 * @brief Zero controller for open-loop trajectory validation.
 *
 * Implements a trivial control law that always outputs zero actuation, allowing
 * the satellite to evolve under:
 * - Passive dynamics (gravity gradient, wheel momentum exchange)
 * - Environmental disturbances (aerodynamic drag, SRP, magnetic residual dipole)
 * - No active control torques
 *
 * This controller is primarily used for:
 * - Validating reference trajectory tracking without feedback
 * - Benchmarking passive stability
 * - Testing disturbance models in isolation
 * - Debugging dynamics propagation without control coupling
 *
 * The control output is always:
 * \f[
 * \mathbf{u} = \mathbf{0} \in \mathbb{R}^{n_u}
 * \f]
 * where \f$n_u\f$ is the total actuator dimension (MTQs + RWs + magic torque).
 */
class ZeroController final : public Controller {
public:
    /**
     * @brief Construct a zero controller.
     *
     * @param satellite Satellite model (used to determine control vector dimension)
     */
    explicit ZeroController(const Satellite& satellite);

    /**
     * @brief Return zero control input.
     *
     * Ignores all inputs and returns a zero control vector of appropriate dimension.
     *
     * @param x Attitude state (unused)
     * @param B_eci Magnetic field (unused)
     * @param q_goal Goal quaternion (unused)
     * @param boresight_body Boresight constraint (unused)
     * @return Zero control vector (nu × 1)
     */
    Satellite::VecX find_u(
        const Satellite::VecX& x,
        const Eigen::Vector3d& B_eci,
        const Eigen::Vector4d& q_goal,
        const Eigen::Vector3d& boresight_body
    ) const override;

protected:
    /**
     * @brief No-op gain tuning (zero controller has no gains).
     *
     * This method does nothing since the zero controller has no tunable parameters.
     */
    void autoTuneGains() override;
};

}
