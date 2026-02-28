#pragma once

#include <saltro/pybind/plannersettings.h>
#include <string>

namespace saltro::optimizer::validation {

bool validatePlannerSettings(const PlannerSettings& settings, std::string& error_msg);

}
