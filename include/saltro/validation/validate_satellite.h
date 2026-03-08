#pragma once

#include <saltro/pybind/satellite.h>
#include <string>

namespace saltro::validation {

/**
 * @brief Validate satellite configuration for trajectory optimization.
 * 
 * Performs comprehensive checks on:
 * - Inertia matrix properties (positive definiteness, finiteness, determinant)
 * - Actuator configurations (MTQs and RWs)
 * - Actuator counts and limits
 * - Geometry configuration (if set)
 * - Associated planner settings
 * 
 * @param satellite Satellite model to validate.
 * @param error_msg Reference to string that receives error message on failure.
 * @return True if satellite configuration is valid, false otherwise.
 */
bool validateSatellite(const Satellite& satellite, std::string& error_msg);

}
