#include <saltro/pybind/controller/controller.h>

namespace saltro::controller {

Controller::Controller(const Satellite& satellite)
    : satellite_(satellite) {}

}
