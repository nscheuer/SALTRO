#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>

#include <Eigen/Dense>

#include <saltro/limits.h>
#include <saltro/optimizer/warm_start.h>
#include <saltro/pybind/satellite.h>

using namespace saltro;

namespace {

constexpr double PI = 3.14159265358979323846;
constexpr double DEG2RAD = PI / 180.0;
constexpr double SEC_PER_CENTURY = 36525.0 * 86400.0;

constexpr int NUM_SAMPLES = 5;
constexpr double DT_SECONDS = 10.0;
constexpr double SIM_SECONDS = 1000.0;
constexpr int N = static_cast<int>(SIM_SECONDS / DT_SECONDS) + 1;
constexpr double TUMBLE_STOP_THRESHOLD = 0.5 * DEG2RAD;
constexpr unsigned SEED = 20260302u;

Eigen::Vector3d randomUnitVector(std::mt19937& rng) {
    std::normal_distribution<double> normal(0.0, 1.0);
    Eigen::Vector3d v(normal(rng), normal(rng), normal(rng));
    const double n = v.norm();
    if (n < 1e-12) {
        return Eigen::Vector3d::UnitX();
    }
    return v / n;
}

Eigen::Matrix3d randomInertia(std::mt19937& rng) {
    std::uniform_real_distribution<double> mass_dist(4.0, 60.0);
    std::uniform_real_distribution<double> size_dist(0.10, 0.55);

    const double mass = mass_dist(rng);
    const double lx = size_dist(rng);
    const double ly = size_dist(rng);
    const double lz = size_dist(rng);

    Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
    J(0, 0) = (mass / 12.0) * (ly * ly + lz * lz);
    J(1, 1) = (mass / 12.0) * (lx * lx + lz * lz);
    J(2, 2) = (mass / 12.0) * (lx * lx + ly * ly);
    return J;
}

void makeRandomSatelliteCase(
    std::mt19937& rng,
    PlannerSettings& settings,
    Eigen::Matrix3d& J,
    std::unique_ptr<Satellite>& satellite
) {
    settings = PlannerSettings();
    settings.init_traj.initcontroller = 2;

    settings.disturbances.plan_for_aero = false;
    settings.disturbances.plan_for_gg = false;
    settings.disturbances.plan_for_srp = false;
    settings.disturbances.plan_for_prop = false;
    settings.disturbances.plan_for_gendist = false;
    settings.disturbances.plan_for_resdipole = false;

    settings.num_passes = 1;
    settings.passes[0].dt = DT_SECONDS;

    J = randomInertia(rng);
    satellite = std::make_unique<Satellite>(J, settings);

    const double Javg = std::max(1e-6, J.trace() / 3.0);

    std::uniform_int_distribution<int> n_mtq_dist(2, 3);
    std::uniform_int_distribution<int> n_rw_dist(1, 3);
    std::uniform_real_distribution<double> scale_dist(0.7, 1.4);

    const int n_mtq = n_mtq_dist(rng);
    const int n_rw = n_rw_dist(rng);

    const double mtq_base = std::clamp(0.8 * std::sqrt(Javg), 0.03, 0.35);
    const double rw_torque_base = std::clamp(0.006 * Javg, 8e-5, 8e-3);
    const double rw_hmax_base = std::clamp(0.35 * Javg, 0.005, 0.12);

    for (int i = 0; i < n_mtq; ++i) {
        const Eigen::Vector3d axis = randomUnitVector(rng);
        const double max_dipole = std::clamp(mtq_base * scale_dist(rng), 0.02, 0.40);
        satellite->addMTQ(axis, max_dipole);
    }

    for (int i = 0; i < n_rw; ++i) {
        const Eigen::Vector3d axis = randomUnitVector(rng);
        const double max_torque = std::clamp(rw_torque_base * scale_dist(rng), 5e-5, 0.010);
        const double rw_J = std::clamp(0.015 * Javg * scale_dist(rng), 5e-6, 2e-3);
        const double h_max = std::clamp(rw_hmax_base * scale_dist(rng), 0.004, 0.16);
        satellite->addRW(axis, max_torque, rw_J, 0.0, h_max);
    }
}

Satellite::VecX randomInitialState(const Satellite& satellite, std::mt19937& rng) {
    Satellite::VecX x0 = Satellite::VecX::Zero(satellite.stateDim());

    std::uniform_real_distribution<double> w_mag_dist(4.0 * DEG2RAD, 15.0 * DEG2RAD);
    const double w_mag = w_mag_dist(rng);
    x0.segment<3>(Satellite::AV_INDEX) = w_mag * randomUnitVector(rng);

    std::uniform_real_distribution<double> angle_dist(-PI, PI);
    const double angle = angle_dist(rng);
    const Eigen::Vector3d axis = randomUnitVector(rng);

    Eigen::Vector4d q;
    q(0) = std::cos(0.5 * angle);
    q.segment<3>(1) = axis * std::sin(0.5 * angle);
    q.normalize();
    x0.segment<4>(Satellite::QUAT_INDEX) = q;

    return x0;
}

void makeConstantEnvironment(
    Eigen::VectorXd& jtime,
    Eigen::MatrixXd& q_goal,
    Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ>& R,
    Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ>& V,
    Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ>& B,
    Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ>& S,
    Eigen::Matrix<double, 1, limits::MAX_LENGTH_TRAJ>& rho
) {
    jtime = Eigen::VectorXd::Zero(N);
    q_goal = Eigen::MatrixXd::Zero(4, N);

    R.setZero();
    V.setZero();
    B.setZero();
    S.setZero();
    rho.setZero();

    const Eigen::Vector3d B_const(2.2e-5, -1.6e-5, 3.1e-5);

    for (int k = 0; k < N; ++k) {
        const double t_sec = static_cast<double>(k) * DT_SECONDS;
        jtime(k) = 0.25 + t_sec / SEC_PER_CENTURY;
        q_goal(0, k) = 1.0;

        B.col(k) = B_const;
    }
}

} // namespace

int main() {
    std::mt19937 rng(SEED);

    Eigen::VectorXd jtime;
    Eigen::MatrixXd q_goal;
    Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> R;
    Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> V;
    Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> B;
    Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> S;
    Eigen::Matrix<double, 1, limits::MAX_LENGTH_TRAJ> rho;

    makeConstantEnvironment(jtime, q_goal, R, V, B, S, rho);

    int n_pass = 0;

    for (int sample = 0; sample < NUM_SAMPLES; ++sample) {
        PlannerSettings settings;
        Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
        std::unique_ptr<Satellite> satellite;
        makeRandomSatelliteCase(rng, settings, J, satellite);

        const Satellite::VecX x0 = randomInitialState(*satellite, rng);

        Eigen::MatrixXd X = Eigen::MatrixXd::Zero(satellite->stateDim(), N);
        Eigen::MatrixXd U = Eigen::MatrixXd::Zero(satellite->controlDim(), N);

        bool ok = false;
        try {
            ok = optimizer::warm_start(
                settings,
                *satellite,
                x0,
                jtime,
                q_goal,
                N,
                R,
                V,
                B,
                S,
                rho,
                X,
                U
            );
        } catch (const std::exception& exc) {
            std::cout
                << "[sample " << std::setw(3) << std::setfill('0') << sample << "] "
                << "warm_start exception: " << exc.what() << " | "
                << "nMTQ=" << satellite->numMTQ() << " nRW=" << satellite->numRW() << " DT=" << DT_SECONDS
                << '\n';
            continue;
        }

        if (!ok) {
            std::cout
                << "[sample " << std::setw(3) << std::setfill('0') << sample << "] "
                << "warm_start failed\n";
            continue;
        }

        const double initial_w = x0.segment<3>(Satellite::AV_INDEX).norm();
        const double final_w = X.col(N - 1).segment<3>(Satellite::AV_INDEX).norm();
        const bool passed = std::isfinite(final_w) && (final_w <= TUMBLE_STOP_THRESHOLD) && (final_w < initial_w);
        n_pass += static_cast<int>(passed);

        std::cout
            << "[sample " << std::setw(3) << std::setfill('0') << sample << "] "
            << (passed ? "PASS" : "FAIL") << ' '
            << "nMTQ=" << satellite->numMTQ() << " nRW=" << satellite->numRW() << ' '
            << "|w0|=" << std::fixed << std::setprecision(6) << initial_w << ' '
            << "|wf|=" << std::fixed << std::setprecision(6) << final_w << ' '
            << "Jdiag=[" << J(0, 0) << ", " << J(1, 1) << ", " << J(2, 2) << "]\n";
    }

    std::cout << "\nMonte Carlo summary: " << n_pass << '/' << NUM_SAMPLES << " passed\n";

    return 0;
}
