#pragma once

#include <saltro/pybind/satellite.h>
#include <string>

namespace saltro::optimizer::validation {

bool validateSatellite(const Satellite& satellite, std::string& error_msg);

}
