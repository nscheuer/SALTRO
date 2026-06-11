#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>

#include <saltro/optimizer/al_slack.h>

using namespace saltro;
using namespace saltro::optimizer;
using Catch::Approx;

namespace {

// Baseline (no-slack) active-set AL merit term, written out independently of
// al_slack.h so the slack-off path is checked against the historical formula.
double baselineMeritTerm(const double c, const double lambda, const double mu)
{
	double term = lambda * c;
	if (c > 0.0 || lambda > 0.0) {
		term += 0.5 * mu * c * c;
	}
	return term;
}

} // namespace

TEST_CASE("isStateSlackFamily covers exactly the state families", "[al_slack]") {
	REQUIRE(isStateSlackFamily(static_cast<int>(ConstraintFamily::AngularVelocity)));
	REQUIRE(isStateSlackFamily(static_cast<int>(ConstraintFamily::SunAvoidance)));
	REQUIRE(isStateSlackFamily(static_cast<int>(ConstraintFamily::RWMomentum)));

	REQUIRE_FALSE(isStateSlackFamily(static_cast<int>(ConstraintFamily::MTQSaturation)));
	REQUIRE_FALSE(isStateSlackFamily(static_cast<int>(ConstraintFamily::RWTorqueSat)));
	REQUIRE_FALSE(isStateSlackFamily(static_cast<int>(ConstraintFamily::RWStiction)));
	REQUIRE_FALSE(isStateSlackFamily(static_cast<int>(ConstraintFamily::MagicTorqueSat)));
	REQUIRE_FALSE(isStateSlackFamily(-1));
}

TEST_CASE("optimalSlack closed form", "[al_slack]") {
	// Below the price threshold the slack stays at its bound.
	REQUIRE(optimalSlack(0.5, 0.0, 1.0, 10.0, 0.0) == 0.0);   // mu*c = 0.5 < rho
	REQUIRE(optimalSlack(-1.0, 0.0, 100.0, 10.0, 0.0) == 0.0); // satisfied constraint
	REQUIRE(optimalSlack(0.0, 5.0, 1.0, 10.0, 0.0) == 0.0);    // lambda < rho

	// Above it, s* = (lambda + mu*c - rho)/(mu + sigma).
	REQUIRE(optimalSlack(2.0, 0.0, 10.0, 10.0, 0.0) == Approx(1.0));
	REQUIRE(optimalSlack(2.0, 5.0, 10.0, 10.0, 0.0) == Approx(1.5));
	REQUIRE(optimalSlack(2.0, 0.0, 10.0, 10.0, 10.0) == Approx(0.5));

	// Degenerate penalty: no slack rather than a division blowup.
	REQUIRE(optimalSlack(2.0, 0.0, 0.0, 10.0, 0.0) == 0.0);

	// Never negative.
	REQUIRE(optimalSlack(-100.0, 0.0, 1e8, 1.0, 0.0) == 0.0);
}

TEST_CASE("alSlackMeritTerm with slack off equals the baseline AL term", "[al_slack]") {
	const double cases[][3] = {
		{0.5, 0.0, 2.0}, {-0.5, 0.0, 2.0}, {-0.5, 3.0, 2.0},
		{1.7, 0.3, 25.0}, {0.0, 0.0, 1.0},
	};
	for (const auto& t : cases) {
		REQUIRE(alSlackMeritTerm(t[0], t[1], t[2], false, 10.0, 0.0)
		        == Approx(baselineMeritTerm(t[0], t[1], t[2])).margin(1e-15));
	}
}

TEST_CASE("alSlackMeritTerm: slack never increases the merit term", "[al_slack]") {
	// s = 0 is always feasible for the inner minimization, so the slacked
	// term is <= the unslacked term for any (c, lambda, mu).
	for (const double c : {-1.0, 0.0, 0.3, 1.0, 5.0}) {
		for (const double lam : {0.0, 1.0, 20.0}) {
			for (const double mu : {0.1, 10.0, 1e4}) {
				const double off = alSlackMeritTerm(c, lam, mu, false, 10.0, 0.0);
				const double on = alSlackMeritTerm(c, lam, mu, true, 10.0, 0.0);
				REQUIRE(on <= off + 1e-12);
			}
		}
	}
}

TEST_CASE("alSlackWeights gradient matches finite-difference of the merit (envelope theorem)",
          "[al_slack]") {
	// The BP gradient weight and the FP merit must describe the same softened
	// penalty; otherwise the line search rejects the BP's descent directions.
	const double rho = 10.0;
	for (const double sigma : {0.0, 5.0}) {
		for (const double c : {-0.5, 0.2, 0.9, 1.5, 4.0}) {
			for (const double lam : {0.0, 2.0, 15.0}) {
				const double mu = 20.0;
				double w = 0.0;
				double mu_eff = 0.0;
				alSlackWeights(c, lam, mu, true, rho, sigma, w, mu_eff);

				const double h = 1e-7;
				const double fd =
					(alSlackMeritTerm(c + h, lam, mu, true, rho, sigma)
					 - alSlackMeritTerm(c - h, lam, mu, true, rho, sigma)) / (2.0 * h);

				REQUIRE(w == Approx(fd).margin(1e-5));
				REQUIRE(mu_eff >= 0.0);
			}
		}
	}
}

TEST_CASE("alSlackWeights caps the constraint force at the slack price", "[al_slack]") {
	// Without slack the gradient weight grows as mu*c; with slack it
	// saturates at rho + sigma*s*. This cap is the whole point: a hard case
	// with a large transient violation no longer injects mu-scaled gradients
	// into the backward pass.
	const double rho = 10.0;
	const double mu = 1e6;

	double w = 0.0;
	double mu_eff = 0.0;
	alSlackWeights(5.0, 0.0, mu, true, rho, 0.0, w, mu_eff);
	REQUIRE(w == Approx(rho).margin(1e-9));
	REQUIRE(mu_eff == 0.0);  // pure linear price: no curvature while interior

	// sigma > 0: force grows gently and curvature is mu*sigma/(mu+sigma).
	const double sigma = 4.0;
	alSlackWeights(5.0, 0.0, mu, true, rho, sigma, w, mu_eff);
	const double s = optimalSlack(5.0, 0.0, mu, rho, sigma);
	REQUIRE(w == Approx(rho + sigma * s).margin(1e-6));
	REQUIRE(mu_eff == Approx(mu * sigma / (mu + sigma)).margin(1e-9));

	// Slack at bound (not economical): exact baseline weights.
	alSlackWeights(5.0, 0.0, 1.0, true, rho, 0.0, w, mu_eff);  // mu*c = 5 < rho
	REQUIRE(w == Approx(5.0));
	REQUIRE(mu_eff == Approx(1.0));
}

TEST_CASE("alSlackMeritTerm is continuous at the slack activation boundary", "[al_slack]") {
	// At lambda + mu*c = rho the slack switches on; the softened penalty must
	// join the quadratic branch with no jump (C^1: values and slopes agree).
	const double mu = 8.0;
	const double rho = 10.0;
	const double lam = 3.0;
	const double c_star = (rho - lam) / mu;

	const double eps = 1e-9;
	const double below = alSlackMeritTerm(c_star - eps, lam, mu, true, rho, 0.0);
	const double above = alSlackMeritTerm(c_star + eps, lam, mu, true, rho, 0.0);
	REQUIRE(below == Approx(above).margin(1e-6));
}
