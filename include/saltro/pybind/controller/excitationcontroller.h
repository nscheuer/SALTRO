#pragma once

#include <saltro/pybind/controller/controller.h>

namespace saltro::controller {

class ExcitationController final : public Controller {
public:
    explicit ExcitationController(const Satellite& satellite);

    Satellite::VecX find_u(
        const Satellite::VecX& x,
        const Eigen::Vector3d& B_eci,
        const Eigen::Vector4d& q_goal
    ) const override;

protected:
    void autoTuneGains() override;

private:
    double rw_excitation_fraction_ = 0.04;
    double mtq_excitation_fraction_ = 0.04;
    double rw_command_fraction_limit_ = 0.10;
    double mtq_command_fraction_limit_ = 0.15;
    double rw_rate_damping_gain_ = 0.0;
    double mtq_rate_damping_gain_ = 0.0;
    double omega_ref_rad_s_ = 5.0 * 3.14159265358979323846 / 180.0;
};

}
