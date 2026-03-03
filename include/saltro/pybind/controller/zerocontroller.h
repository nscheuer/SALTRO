#pragma once

#include <saltro/pybind/controller/controller.h>

namespace saltro::controller {

class ZeroController final : public Controller {
public:
    explicit ZeroController(const Satellite& satellite);

    Satellite::VecX find_u(
        const Satellite::VecX& x,
        const Eigen::Vector3d& B_eci,
        const Eigen::Vector4d& q_goal,
        const Eigen::Vector3d& boresight_body
    ) const override;

protected:
    void autoTuneGains() override;
};

}
