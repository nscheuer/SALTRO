#include <saltro/pybind/controller/zerocontroller.h>

namespace saltro::controller {

ZeroController::ZeroController(const Satellite& satellite)
    : Controller(satellite) {
    autoTuneGains();
}

Satellite::VecX ZeroController::find_u(
    const Satellite::VecX& x,
    const Eigen::Vector3d& B_eci,
    const Eigen::Vector4d& q_goal
) const {
    (void)x;
    (void)B_eci;
    (void)q_goal;

    Satellite::VecX u(satellite_.controlDim());
    u.setZero();
    return u;
}

void ZeroController::autoTuneGains() {
}

}
