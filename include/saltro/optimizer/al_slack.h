#pragma once

#include <algorithm>
#include <cmath>

#include <saltro/pybind/plannersettings.h>

namespace saltro::optimizer {

/**
 * @brief Analytic slack elimination for state-constraint AL relaxation.
 *
 * With AugLagConfig::use_state_slack on, each STATE constraint c(x) ≤ 0 is
 * relaxed to c(x) − s ≤ 0, s ≥ 0, with slack cost ρ·s + ½σ·s² added to the
 * merit. Because the slack enters only the AL terms, it is minimized in
 * closed form per constraint instead of being added to the iLQR decision
 * space, and all four AL sites (BP stage block, BP terminal seed, FP merit,
 * alilqr λ update) evaluate the SAME softened penalty by replacing c with
 * c̃ = c − s*. The helpers below are the single source of truth for that
 * math; every site must go through them so the quadratic model, the merit,
 * and the dual update stay mutually consistent (the same contract the
 * active-set semantics already obey).
 *
 * Envelope theorem: s* is the argmin of the slack-augmented AL term, so the
 * gradient of the minimized term w.r.t. x is the usual AL gradient evaluated
 * at c̃ (the chain term through ds*(x) vanishes). For the Gauss-Newton curvature, the exact
 * second derivative of the softened penalty is μσ/(μ+σ) while the slack is
 * interior (μ when it is not), which alSlackWeights returns as mu_eff.
 */

/// True for constraint families that depend on the STATE (and therefore can
/// only be satisfied through trajectory shape, not by clipping a control):
/// these are the families that receive slack variables.
inline bool isStateSlackFamily(const int family)
{
    return family == static_cast<int>(ConstraintFamily::AngularVelocity)
        || family == static_cast<int>(ConstraintFamily::SunAvoidance)
        || family == static_cast<int>(ConstraintFamily::RWMomentum);
}

/// Closed-form minimizer of  ρ·s + ½σ·s² + λ(c−s) + ½μ(c−s)²  over s ≥ 0.
/// (The active-set μ gate is handled by the caller through the shifted
/// constraint; for λ > 0 the penalty is active regardless of sign, and for
/// λ = 0 the formula reduces to the correct one-sided result.)
inline double optimalSlack(
    const double c,
    const double lambda,
    const double mu,
    const double slack_rho,
    const double slack_sigma
)
{
    const double denom = mu + slack_sigma;
    if (!(denom > 0.0) || !std::isfinite(denom)) {
        return 0.0;
    }
    return std::max(0.0, (lambda + mu * c - slack_rho) / denom);
}

/// Merit (objective) contribution of one constraint entry, slack included:
///   ρ·s* + ½σ·s*² + λ·c̃ + [c̃ > 0 or λ > 0]·½μ·c̃²,  c̃ = c − s*.
/// With slack_on == false this is exactly the existing active-set AL term.
inline double alSlackMeritTerm(
    const double c,
    const double lambda,
    const double mu,
    const bool slack_on,
    const double slack_rho,
    const double slack_sigma
)
{
    double c_eff = c;
    double slack_cost = 0.0;
    if (slack_on) {
        const double s = optimalSlack(c, lambda, mu, slack_rho, slack_sigma);
        c_eff = c - s;
        slack_cost = slack_rho * s + 0.5 * slack_sigma * s * s;
    }
    double term = slack_cost + lambda * c_eff;
    if (c_eff > 0.0 || lambda > 0.0) {
        term += 0.5 * mu * c_eff * c_eff;
    }
    return term;
}

/// Linearization weights of the (softened) AL term for the backward pass:
///   w      — gradient coefficient on c_x (w = λ + active·μ·c̃; saturates at
///            ρ + σ·s* while the slack is interior),
///   mu_eff — Gauss-Newton curvature coefficient on c_x c_xᵀ (μ when the
///            term is active and the slack is at its bound, μσ/(μ+σ) while
///            the slack is interior — 0 for a pure linear price σ = 0).
inline void alSlackWeights(
    const double c,
    const double lambda,
    const double mu,
    const bool slack_on,
    const double slack_rho,
    const double slack_sigma,
    double& w,
    double& mu_eff
)
{
    double c_eff = c;
    double s = 0.0;
    if (slack_on) {
        s = optimalSlack(c, lambda, mu, slack_rho, slack_sigma);
        c_eff = c - s;
    }

    const bool active = (c_eff > 0.0 || lambda > 0.0);
    w = lambda + (active ? mu * c_eff : 0.0);
    if (!active) {
        mu_eff = 0.0;
    } else if (s > 0.0) {
        const double denom = mu + slack_sigma;
        mu_eff = (denom > 0.0) ? (mu * slack_sigma / denom) : 0.0;
    } else {
        mu_eff = mu;
    }
}

} // namespace saltro::optimizer
