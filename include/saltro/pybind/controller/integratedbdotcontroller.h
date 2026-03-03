#pragma once

#include <vector>

#include <saltro/pybind/controller/controller.h>

namespace saltro::controller {

class IntegratedBdotController final : public Controller {
public:
    explicit IntegratedBdotController(const Satellite& satellite);

    Satellite::VecX find_u(
        const Satellite::VecX& x,
        const Eigen::Vector3d& B_eci,
        const Eigen::Vector4d& q_goal,
        const Eigen::Vector3d& boresight_body
    ) const override;

protected:
    void autoTuneGains() override;

private:
    double max_rate_ref_ = 5.0 * 3.14159265358979323846 / 180.0;

    std::vector<double> mtq_kp_;
    std::vector<double> mtq_kd_;
    std::vector<double> rw_kp_;
    std::vector<double> rw_kd_;

    mutable std::vector<double> mtq_prev_bdot_;
    mutable std::vector<double> rw_prev_w_;
};

}
