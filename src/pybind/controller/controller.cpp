#include <saltro/pybind/controller/controller.h>

#include <algorithm>
#include <cmath>

namespace saltro::controller {

Controller::Controller(const Satellite& satellite)
    : satellite_(satellite) {}

void Controller::set_dt(double dt) {
    if (dt_initialized_) {
        return;
    }

    if (!std::isfinite(dt) || dt <= 0.0) {
        return;
    }

    constexpr double DT_MIN = 1e-4;
    constexpr double DT_MAX = 1e4;
    dt_seconds_ = std::clamp(dt, DT_MIN, DT_MAX);

    autoTuneGains();
    dt_initialized_ = true;
}

double Controller::dt_seconds() const {
    return dt_seconds_;
}

}
