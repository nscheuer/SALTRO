#include <saltro/orbit_generation/orbits/compute_orbit_J2_RK4.h>
#include <saltro/math/integrators/rk4.h>
#include <saltro/constants/constants.h>
#include <cmath>

namespace saltro::orbits {

using State6 = Eigen::Matrix<double,6,1>;

static inline void j2_dynamics(
    double,
    const State6& x,
    State6& dxdt)
{
    constexpr double mu  = saltro::constants::MU_EARTH;
    constexpr double Re  = saltro::constants::R_EARTH;
    constexpr double J2  = saltro::constants::J2_EARTH;

    const Eigen::Vector3d r = x.head<3>();
    const Eigen::Vector3d v = x.tail<3>();

    const double r2 = r.squaredNorm();
    const double r1 = std::sqrt(r2);
    const double r3 = r2 * r1;
    const double r5 = r3 * r2;

    // Two-body
    Eigen::Vector3d a = (-mu / r3) * r;

    // J2
    const double z2 = r.z() * r.z();
    const double f = 1.5 * J2 * mu * (Re*Re) / r5;
    const double g = 5.0 * z2 / r2;

    a.x() += f * r.x() * (g - 1.0);
    a.y() += f * r.y() * (g - 1.0);
    a.z() += f * r.z() * (g - 3.0);

    dxdt.head<3>() = v;
    dxdt.tail<3>() = a;
}


bool compute_orbit_J2_RK4(
    const Eigen::Vector3d& r0,
    const Eigen::Vector3d& v0,
    const Eigen::Matrix<double,1,saltro::limits::MAX_LENGTH_TRAJ>& jtime,
    const int jtime_length,

    Eigen::Matrix<double,3,saltro::limits::MAX_LENGTH_TRAJ>& R,
    Eigen::Matrix<double,3,saltro::limits::MAX_LENGTH_TRAJ>& V
) {
    constexpr double SEC_PER_CENT = saltro::constants::SEC_PER_JULIAN_CENTURY;

    // ---- internal step control ----
    constexpr double H_MAX = 10.0;   // seconds (tunable)

    State6 x;
    x.head<3>() = r0;
    x.tail<3>() = v0;

    R.col(0) = r0;
    V.col(0) = v0;

    const double t0 = jtime(0);

    for (int k = 1; k < jtime_length; ++k)
    {
        const double dt_total =
            (jtime(k) - t0) * SEC_PER_CENT;

        double dt = dt_total;

        // propagate from previous sample
        const double t_prev =
            (jtime(k-1) - t0) * SEC_PER_CENT;

        double dt_step = dt - t_prev;

        if (dt_step == 0.0) {
            R.col(k) = x.head<3>();
            V.col(k) = x.tail<3>();
            continue;
        }

        // ---- deterministic substepping ----
        int n = static_cast<int>(std::ceil(std::abs(dt_step) / H_MAX));
        if (n < 1) n = 1;

        const double h = dt_step / static_cast<double>(n);

        double t = t_prev;

        for (int i = 0; i < n; ++i)
        {
            State6 x_new;
            saltro::math::rk4_step(j2_dynamics, x, t, h, x_new);
            x = x_new;
            t += h;
        }

        R.col(k) = x.head<3>();
        V.col(k) = x.tail<3>();
    }

    return true;
}

}