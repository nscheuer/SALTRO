#pragma once

#include <saltro/pybind/actuators/actuator.h>
#include <Eigen/Dense>

/**
 * @brief "Magic" (direct body-torque) actuator class.
 *
 * Represents an idealised actuator that applies a body-frame torque
 * directly along a fixed body axis, with no environmental dependence
 * and no momentum-storage state. The torque model is
 * \f[
 *     \boldsymbol{\tau} = u \cdot \mathbf{a},
 * \f]
 * where \f$\mathbf{a}\f$ is the unit body axis and \f$u\f$ is the
 * commanded torque magnitude.
 *
 * Unlike MTQs (whose torque depends on the geomagnetic field through
 * \f$\boldsymbol{\tau} = -\mathbf{B}_b \times \mathbf{a} \cdot u\f$) and
 * RWs (which carry their own angular-momentum state and exchange
 * momentum via Newton's third law), magic actuators are dynamically
 * trivial:
 *
 * - No state dependence (\f$\partial \boldsymbol{\tau}/\partial \mathbf{x} = 0\f$).
 * - No environment dependence (no B-field, sun, or orbit needed).
 * - Constant Jacobian (\f$\partial \boldsymbol{\tau}/\partial u = \mathbf{a}\f$).
 *
 * Use cases: modelling thrusters with a fixed thrust direction, or as a
 * "clean" body-torque commander in tests where the MTQ rank-deficiency
 * and RW back-reaction would otherwise obscure the property under test.
 *
 * @see Actuator for the base class interface.
 */
class Magic final : public Actuator {
public:
    using Vec3 = Actuator::Vec3;
    using BaseState = Actuator::BaseState;

    using Mat13 = Actuator::Mat13;
    using Mat73 = Actuator::Mat73;

    using T113 = Actuator::T113;
    using T173 = Actuator::T173;
    using T773 = Actuator::T773;

    using Actuator::torque;
    using Actuator::dtorq_du;
    using Actuator::dtorq_dbasestate;
    using Actuator::ddtorq_dudu;
    using Actuator::ddtorq_dudbasestate;
    using Actuator::ddtorq_dbasestatedbasestate;

    /**
     * @brief Construct a magic actuator with specified parameters.
     *
     * @param axis Unit vector specifying the torque direction in body frame.
     * @param max_torque Maximum torque magnitude (N*m).
     */
    Magic(const Vec3 &axis, double max_torque);

    /**
     * @brief Compute the body-frame torque produced by this magic actuator.
     *
     * Returns \f$u \cdot \mathbf{a}\f$ -- the torque is exactly linear in
     * the control input, with no state or environment dependence.
     *
     * @param u Control input (torque magnitude, N*m).
     * @param x Base state (unused; included for interface consistency).
     * @return 3D torque vector in body frame (N*m).
     */
    Vec3 torque(double u, const BaseState& x) const override;

    /**
     * @brief Jacobian of torque with respect to the control input.
     *
     * Constant for a magic actuator:
     * \f$\partial \boldsymbol{\tau} / \partial u = \mathbf{a}^\top\f$
     * (returned as a 1x3 row).
     *
     * @param u Control input (unused).
     * @param x Base state (unused).
     * @return 1x3 Jacobian.
     */
    Mat13 dtorq_du(double u, const BaseState& x) const override;

    /**
     * @brief Jacobian of torque with respect to the base state.
     *
     * Zero -- the magic torque is independent of \f$\boldsymbol{\omega}\f$ and \f$\mathbf{q}\f$.
     *
     * @param u Control input (unused).
     * @param x Base state (unused).
     * @return 7x3 zero matrix.
     */
    Mat73 dtorq_dbasestate(double u, const BaseState& x) const override;

    /**
     * @brief Second derivative of torque with respect to the control input.
     *
     * Zero -- torque is affine in \f$u\f$.
     */
    T113 ddtorq_dudu(double u, const BaseState& x) const override;

    /**
     * @brief Mixed second derivative \f$\partial^2 \boldsymbol{\tau}/(\partial u \, \partial \mathbf{x})\f$.
     *
     * Zero -- torque doesn't depend on the base state.
     */
    T173 ddtorq_dudbasestate(double u, const BaseState& x) const override;

    /**
     * @brief Second derivative of torque with respect to the base state.
     *
     * Zero -- torque doesn't depend on the base state.
     */
    T773 ddtorq_dbasestatedbasestate(double u, const BaseState& x) const override;
};
