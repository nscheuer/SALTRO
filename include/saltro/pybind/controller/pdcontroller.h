/**
 * @file pdcontroller.h
 * @brief Actuator-agnostic quaternion PD controller.
 */
#pragma once

#include <saltro/pybind/controller/controller.h>

namespace saltro::controller {

/**
 * @brief Actuator-agnostic quaternion PD controller.
 *
 * Computes a desired body-frame torque via quaternion or vector PD plus
 * angular-rate damping, then allocates the torque across available actuators
 * by least squares.
 *
 * **Quaternion goal** (\f$q_{\text{goal}}\f$ a unit quaternion):
 * \f[
 *   \boldsymbol{\tau}_{\text{des}} = -k_p\,\mathbf{q}_{\text{err,vec}} - k_d\,\boldsymbol{\omega}
 * \f]
 *
 * **Vector goal** (\f$q_{\text{goal}}=[\mathrm{NaN}, \hat{r}_{\text{eci}}]\f$,
 * matching the Satellite stageCost sentinel):
 * \f[
 *   \boldsymbol{\tau}_{\text{des}} =
 *       +k_p\,\bigl(\mathbf{bs}_{\text{body}} \times R(q)^\top\hat{r}_{\text{eci}}\bigr)
 *       - k_d\,\boldsymbol{\omega}
 * \f]
 * The cross product has magnitude \f$\sin\theta_{\text{err}}\f$ and points in the
 * body-frame direction that rotates \f$\mathbf{bs}\f$ toward
 * \f$R^\top\hat{r}\f$.
 *
 * Allocation solves
 * \f[
 *   \min_{\mathbf{u}} \;\|J\mathbf{u} - \boldsymbol{\tau}_{\text{des}}\|^2
 *                   + \mathbf{u}^\top W \mathbf{u}
 * \f]
 * where \f$J = \partial\boldsymbol{\tau}_{\text{actuator}}/\partial\mathbf{u}\f$
 * is the numerical Jacobian of Satellite::actuatorTorque, and \f$W\f$ weights
 * actuators by their inverse-squared authority (column norm of \f$J\f$).  This
 * dispatches through the satellite's actuator mixing and is therefore agnostic
 * to actuator topology — any MTQ/RW combination, or future actuator types.
 *
 * The result is clamped to actuator limits by **uniform scale-to-max**: if any
 * channel would exceed its limit, all channels scale by the same factor so the
 * torque direction is preserved.  Independent per-channel clipping distorts
 * the direction and can send \f$\boldsymbol{\tau}_{\text{actual}}\f$ further
 * from \f$\boldsymbol{\tau}_{\text{des}}\f$ than no allocation at all.
 *
 * ## Gain selection
 *
 * autoTuneGains sets a conservative second-order default
 * (\f$\omega_n\approx 0.1\,\text{rad/s},\,\zeta=0.7\f$).  Spike removal
 * overrides via setGains with more aggressive values.
 */
class PDController final : public Controller {
public:
    /**
     * @brief Construct a PD controller with auto-tuned gains.
     * @param satellite Satellite model (inertia, actuators)
     */
    explicit PDController(const Satellite& satellite);

    /**
     * @brief Compute control input via quaternion PD + least-squares allocation.
     *
     * @param x Attitude state (7+nRW): [ω(3), q(4), h_rw(nRW)]
     * @param B_eci Magnetic field in ECI frame (T)
     * @param q_goal Either a unit quaternion target (scalar-first) or
     *   ``[NaN, r̂_eci]`` for a vector-pointing target.
     * @param boresight_body Boresight direction in body frame. Used by
     *   the vector-goal branch; ignored when q_goal is a quaternion.
     * @return Control vector (nu × 1): [m_mtq, τ_rw, …]
     */
    Satellite::VecX find_u(
        const Satellite::VecX& x,
        const Eigen::Vector3d& B_eci,
        const Eigen::Vector4d& q_goal,
        const Eigen::Vector3d& boresight_body
    ) const override;

    /// Override auto-tuned gains (used by spike removal).
    void setGains(double kp_q, double kd_w);
    /// Set RW preference: 0 = MTQ-only, 1 = equal weight.
    void setRWScale(double rw_scale);

    double kp_q() const { return kp_q_; }
    double kd_w() const { return kd_w_; }
    double rwScale() const { return rw_scale_; }

protected:
    /**
     * @brief Auto-tune to a conservative second-order response.
     *
     * \f$\omega_n = 0.1\,\text{rad/s}\f$, \f$\zeta = 0.7\f$, based on the
     * mean inertia tensor eigenvalue.
     */
    void autoTuneGains() override;

private:
    double kp_q_ = 0.0;
    double kd_w_ = 0.0;
    double rw_scale_ = 1.0;
};

}
