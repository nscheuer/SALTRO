#include <iostream>
#include <Eigen/Dense>

#include <saltro/orbit_generation/generate_orbit.h>
#include <saltro/limits.h>   // for MAX_LENGTH_TRAJ

int main() {
    using namespace saltro;

    // Initial state (example values)
    Eigen::Vector3d r0(7000e3, 0.0, 0.0);   // meters
    Eigen::Vector3d v0(0.0, 7.5e3, 0.0);    // m/s

    // Time array
    Eigen::Matrix<double, 1, limits::MAX_LENGTH_TRAJ> jtime;
    int jtime_length = 10;

    for (int i = 0; i < jtime_length; ++i) {
        jtime(i) = i * 60.0; // 60-second step
    }

    // Output arrays
    Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> R, V, B, S;
    Eigen::Matrix<double, 1, limits::MAX_LENGTH_TRAJ> rho;

    // Run with model 0 for everything
    bool ok = orbits::generate_orbit(
        r0, v0,
        jtime, jtime_length,
        0,  // orbit_model
        0,  // magnetic_model
        0,  // sun_model
        0,  // eclipse_model
        0,  // density_model
        R, V, B, S, rho
    );

    if (!ok) {
        std::cerr << "generate_orbit failed\n";
        return 1;
    }

    auto print_series = [&](const char* name,
                            const auto& M,
                            int rows) {
        std::cout << name << ":\n";
        std::cout << "  initial: ";
        for (int r = 0; r < rows; ++r) {
            std::cout << M(r, 0) << " ";
        }
        std::cout << "\n  at jtime_length-1: ";
        for (int r = 0; r < rows; ++r) {
            std::cout << M(r, jtime_length - 1) << " ";
        }
        std::cout << "\n\n";
    };

    print_series("R", R, 3);
    print_series("V", V, 3);
    print_series("B", B, 3);
    print_series("S", S, 3);

    std::cout << "rho:\n";
    std::cout << "  initial: " << rho(0) << "\n";
    std::cout << "  at jtime_length-1: " << rho(jtime_length - 1) << "\n";

    return 0;
}