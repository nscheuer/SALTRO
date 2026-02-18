#include <saltro/orbit_generation/magnetics/compute_magnetic_igrf13.h>
#include <saltro/constants/igrf13.h>
#include <saltro/constants/constants.h>
#include <saltro/math/frames.h>
#include <cmath>

namespace saltro::orbit {
    
static inline void legendre(
    int nmax,
    double theta,
    double P[14][14],
    double dP[14][14]
){
    const double ct = std::cos(theta);
    const double st = std::sin(theta);

    P[0][0] = 1.0;
    dP[0][0] = 0.0;

    for(int n = 1; n <= nmax; ++n){
        P[n][n] = P[n-1][n-1] * st;
        dP[n][n] = P[n-1][n-1] * ct + dP[n-1][n-1] * st;

        P[n][n-1] = (2*n-1) * ct * P[n-1][n-1];
        dP[n][n-1] = (2*n-1)*(ct*dP[n-1][n-1] - st*P[n-1][n-1]);

        for(int m=0; m<=n-2; ++m){
            P[n][m] =
                ((2*n-1)*ct*P[n-1][m] - (n+m-1)*P[n-2][m])/(n-m);

            dP[n][m] =
                ((2*n-1)*(ct*dP[n-1][m] - st*P[n-1][m])
                - (n+m-1)*dP[n-2][m])/(n-m);
        }
    }
}

bool compute_magnetic_igrf13(
    const Eigen::Matrix<double,3,saltro::limits::MAX_LENGTH_TRAJ>& R,
    const Eigen::Matrix<double,1,saltro::limits::MAX_LENGTH_TRAJ>& jtime,
    const int jtime_length,

    Eigen::Matrix<double,3,saltro::limits::MAX_LENGTH_TRAJ>& B
){
    constexpr int NMAX = saltro::constants::IGRF13::NMAX;
    const double a = saltro::constants::IGRF13::EARTH_REFERENCE_RADIUS;

    for(int k=0; k<jtime_length; ++k){

        const Eigen::Vector3d r_eci = R.col(k);
        const double rmag = r_eci.norm();

        if(rmag <= 0.0){
            B.col(k).setZero();
            continue;
        }

        // ---- ECI → ECEF ----
        const Eigen::Matrix3d C_ecef_eci =
            saltro::math::eci_to_ecef_dcm(jtime(k));

        const Eigen::Vector3d r_ecef = C_ecef_eci * r_eci;

        const double x = r_ecef.x();
        const double y = r_ecef.y();
        const double z = r_ecef.z();

        const double r = std::sqrt(x*x + y*y + z*z);

        if(r <= 0.0){
            B.col(k).setZero();
            continue;
        }

        const double theta = std::acos(z/r);
        const double phi   = std::atan2(y,x);

        double P[14][14]{};
        double dP[14][14]{};

        legendre(NMAX, theta, P, dP);

        double Br = 0.0;
        double Bt = 0.0;
        double Bp = 0.0;

        for(int n=1; n<=NMAX; ++n){

            const double rn = std::pow(a/r, n+2);

            for(int m=0; m<=n; ++m){

                const double g = saltro::constants::IGRF13::G[n][m];
                const double h = saltro::constants::IGRF13::H[n][m];

                const double cos_m = std::cos(m*phi);
                const double sin_m = std::sin(m*phi);

                const double tmp = g*cos_m + h*sin_m;

                Br += rn*(n+1)*tmp*P[n][m];
                Bt -= rn*tmp*dP[n][m];

                if(m>0){
                    Bp += rn*m*(g*sin_m - h*cos_m)*P[n][m];
                }
            }
        }

        const double st = std::sin(theta);
        if(std::abs(st) > 1e-12)
            Bp /= st;
        else
            Bp = 0.0;

        const double ct = std::cos(theta);
        const double sp = std::sin(phi);
        const double cp = std::cos(phi);

        double Bx =
            st*cp*Br + ct*cp*Bt - sp*Bp;

        double By =
            st*sp*Br + ct*sp*Bt + cp*Bp;

        double Bz =
            ct*Br - st*Bt;

        Eigen::Vector3d B_ecef(Bx*1e-9, By*1e-9, Bz*1e-9);

        const Eigen::Matrix3d C_eci_ecef =
            saltro::math::ecef_to_eci_dcm(jtime(k));

        B.col(k) = C_eci_ecef * B_ecef;
    }

    return true;
}

}
