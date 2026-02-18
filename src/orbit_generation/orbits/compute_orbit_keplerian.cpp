#include <saltro/orbit_generation/orbits/compute_orbit_keplerian.h>
#include <saltro/constants/constants.h>
#include <cmath>

namespace saltro::orbits {

static inline void stumpff(double z, double&C, double& S){
    if (std::abs(z) < 1e-8) {
        const double z2 = z*z;
        C = 0.5 - z/24.0 + z2/720.0;
        S = 1.0/6.0 - z/120.0 + z2/5040.0;
        return;
    }

    if (z > 0.0) {
        double s = std::sqrt(z);
        C = (1.0 - std::cos(s)) / z;
        S = (s - std::sin(s)) / (s*s*s);
    } else {
        double s = std::sqrt(-z);
        C = (std::cosh(s) - 1.0) / (-z);
        S = (std::sinh(s) - s) / (s*s*s);
    }
}


bool compute_orbit_keplerian(
    const Eigen::Vector3d& r0,
    const Eigen::Vector3d& v0,
    const Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ>& jtime,
    const int jtime_length,

    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& R,
    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& V
) {
    const double mu = saltro::constants::MU_EARTH;
    const double sqrt_mu = std::sqrt(mu);

    const double r0mag = r0.norm();
    const double v0sq = v0.squaredNorm();
    const double alpha = 2.0/r0mag - v0sq/mu;
    const double vr0 = r0.dot(v0) / r0mag;

    const double t0 = jtime(0);

    for (int k = 0; k < jtime_length; ++k) {
        const double dt = (jtime(k) - t0) * saltro::constants::SEC_PER_JULIAN_CENTURY;

        if (dt == 0.0) {
            R.col(k) = r0;
            V.col(k) = v0;
            continue;
        }

        double chi = sqrt_mu * std::abs(alpha) * dt;
        if (alpha == 0.0) chi = sqrt_mu * dt / r0mag;

        for (int i = 0; i < saltro::limits::KEPLER_MAX_ITERS; ++i) {
            double z = alpha * chi * chi;
            double C, S;
            stumpff(z, C, S);

            double F = r0mag*vr0/sqrt_mu * chi*chi*C + (1.0 - alpha*r0mag)*chi*chi*chi*S + r0mag*chi - sqrt_mu*dt;
            double dF = r0mag*vr0/sqrt_mu * chi*(1.0 - z*S) + (1.0 - alpha*r0mag)*chi*chi*C + r0mag;

            double delta = F/dF;
            chi -= delta;

            if (std::abs(delta) < saltro::limits::KEPLER_TOLERANCE) break;
        }
        double z = alpha * chi * chi;
        double C, S;
        stumpff(z, C, S);

        double f = 1.0 - chi*chi/r0mag*C;
        double g = dt - chi*chi*chi/sqrt_mu*S;

        Eigen::Vector3d r = f*r0 + g*v0;
        double rmag = r.norm();

        double fdot = sqrt_mu/(rmag*r0mag) * (z*S - 1.0) * chi;
        double gdot = 1.0 - chi*chi/rmag * C;

        Eigen::Vector3d v = fdot*r0 + gdot*v0;

        R.col(k) = r;
        V.col(k) = v;
    }
    return true;
}

}