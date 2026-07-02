#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <Eigen/Dense>
#include <saltro/validation/validate_plannersettings.h>
#include <saltro/pybind/plannersettings.h>
#include <cmath>
#include <string>
#include <limits>

// ============================================================================
// Helper Functions
// ============================================================================

static PlannerSettings validSettings() {
    PlannerSettings settings;
    settings.num_passes = 1;
    
    // Set up valid constraint configuration
    settings.constraints.control_limit_scale = 0.75;
    settings.constraints.u_max = Eigen::VectorXd::Constant(3, 1.0);
    settings.constraints.wmax = 0.3;
    settings.constraints.sun_limit_angle = 0.35;
    
    // Set up valid disturbance configuration
    settings.disturbances.coeff_N = 3;
    
    // Set up valid init_traj configuration
    settings.init_traj.initcontroller = 0;
    
    // Set up one valid pass
    PassConfig& pass = settings.passes[0];
    pass.dt = 1.0;
    
    // Valid cost config (use defaults)
    pass.cost = CostConfig();
    
    // Valid auglag config (use defaults)
    pass.auglag = AugLagConfig();
    
    // Valid ilqr config (use defaults)
    pass.ilqr = ILQRConfig();
    
    // Valid reg config (use defaults)
    pass.reg = RegularizationConfig();
    
    // Valid line search config (use defaults)
    pass.linesearch = LineSearchConfig();
    
    return settings;
}

// ============================================================================
// Basic Validation Tests
// ============================================================================

TEST_CASE("Valid PlannerSettings passes validation", "[plannersettings][validation]") {
    PlannerSettings settings = validSettings();
    std::string error_msg;
    
    REQUIRE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg.empty());
}

TEST_CASE("Valid PlannerSettings with zero passes", "[plannersettings][validation]") {
    PlannerSettings settings = validSettings();
    settings.num_passes = 0;
    std::string error_msg;
    
    REQUIRE(saltro::validation::validatePlannerSettings(settings, error_msg));
}

TEST_CASE("Valid PlannerSettings with multiple passes", "[plannersettings][validation]") {
    PlannerSettings settings = validSettings();
    settings.num_passes = 2;
    settings.passes[0] = settings.passes[0];  // Copy first pass config
    settings.passes[1] = settings.passes[0];  // Same config for second pass
    std::string error_msg;
    
    REQUIRE(saltro::validation::validatePlannerSettings(settings, error_msg));
}

// ============================================================================
// num_passes Validation Tests
// ============================================================================

TEST_CASE("Invalid num_passes - negative", "[plannersettings][validation][num_passes]") {
    PlannerSettings settings = validSettings();
    settings.num_passes = -1;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "num_passes out of range");
}

TEST_CASE("Invalid num_passes - exceeds maximum", "[plannersettings][validation][num_passes]") {
    PlannerSettings settings = validSettings();
    settings.num_passes = MAX_OUTER_PASSES + 1;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "num_passes out of range");
}

// ============================================================================
// Constraint Configuration Validation Tests
// ============================================================================

TEST_CASE("Invalid control_limit_scale - negative", "[plannersettings][validation][constraints]") {
    PlannerSettings settings = validSettings();
    settings.constraints.control_limit_scale = -0.1;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "control_limit_scale invalid");
}

TEST_CASE("Invalid control_limit_scale - NaN", "[plannersettings][validation][constraints]") {
    PlannerSettings settings = validSettings();
    settings.constraints.control_limit_scale = std::numeric_limits<double>::quiet_NaN();
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "control_limit_scale invalid");
}

TEST_CASE("Invalid control_limit_scale - infinity", "[plannersettings][validation][constraints]") {
    PlannerSettings settings = validSettings();
    settings.constraints.control_limit_scale = std::numeric_limits<double>::infinity();
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "control_limit_scale invalid");
}

TEST_CASE("Invalid wmax - negative", "[plannersettings][validation][constraints]") {
    PlannerSettings settings = validSettings();
    settings.constraints.wmax = -0.1;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "wmax invalid");
}

TEST_CASE("Invalid wmax - zero", "[plannersettings][validation][constraints]") {
    PlannerSettings settings = validSettings();
    settings.constraints.wmax = 0.0;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "wmax invalid");
}

TEST_CASE("Invalid wmax - NaN", "[plannersettings][validation][constraints]") {
    PlannerSettings settings = validSettings();
    settings.constraints.wmax = std::numeric_limits<double>::quiet_NaN();
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "wmax invalid");
}

TEST_CASE("Invalid sun_limit_angle - negative", "[plannersettings][validation][constraints]") {
    PlannerSettings settings = validSettings();
    settings.constraints.sun_limit_angle = -0.1;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "sun_limit_angle invalid");
}

TEST_CASE("Invalid sun_limit_angle - exceeds pi", "[plannersettings][validation][constraints]") {
    PlannerSettings settings = validSettings();
    settings.constraints.sun_limit_angle = M_PI + 0.1;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "sun_limit_angle invalid");
}

TEST_CASE("Invalid sun_limit_angle - NaN", "[plannersettings][validation][constraints]") {
    PlannerSettings settings = validSettings();
    settings.constraints.sun_limit_angle = std::numeric_limits<double>::quiet_NaN();
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "sun_limit_angle invalid");
}

TEST_CASE("Invalid u_max - empty vector", "[plannersettings][validation][constraints]") {
    PlannerSettings settings = validSettings();
    settings.constraints.u_max = Eigen::VectorXd();
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "u_max is empty");
}

TEST_CASE("Invalid u_max - contains negative values", "[plannersettings][validation][constraints]") {
    PlannerSettings settings = validSettings();
    settings.constraints.u_max = Eigen::VectorXd::Constant(3, 1.0);
    settings.constraints.u_max(1) = -0.5;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "u_max contains invalid values");
}

TEST_CASE("Invalid u_max - contains NaN", "[plannersettings][validation][constraints]") {
    PlannerSettings settings = validSettings();
    settings.constraints.u_max = Eigen::VectorXd::Constant(3, 1.0);
    settings.constraints.u_max(0) = std::numeric_limits<double>::quiet_NaN();
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "u_max contains invalid values");
}

// ============================================================================
// Disturbance Configuration Validation Tests
// ============================================================================

TEST_CASE("Invalid disturbances.coeff_N - negative", "[plannersettings][validation][disturbances]") {
    PlannerSettings settings = validSettings();
    settings.disturbances.coeff_N = -1;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "disturbances.coeff_N invalid");
}

// ============================================================================
// Init Trajectory Configuration Validation Tests
// ============================================================================

TEST_CASE("Invalid init_traj.initcontroller - negative", "[plannersettings][validation][init_traj]") {
    PlannerSettings settings = validSettings();
    settings.init_traj.initcontroller = -1;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "init_traj.initcontroller invalid");
}

// ============================================================================
// Pass dt Validation Tests
// ============================================================================

TEST_CASE("Invalid pass dt - negative", "[plannersettings][validation][pass][dt]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].dt = -0.1;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "pass dt invalid");
}

TEST_CASE("Invalid pass dt - zero", "[plannersettings][validation][pass][dt]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].dt = 0.0;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "pass dt invalid");
}

TEST_CASE("Invalid pass dt - NaN", "[plannersettings][validation][pass][dt]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].dt = std::numeric_limits<double>::quiet_NaN();
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "pass dt invalid");
}

// ============================================================================
// Cost Configuration Validation Tests
// ============================================================================

TEST_CASE("Invalid cost.angle - negative", "[plannersettings][validation][cost]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].cost.angle = -1.0;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "cost.angle invalid");
}

TEST_CASE("Invalid cost.angle - NaN", "[plannersettings][validation][cost]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].cost.angle = std::numeric_limits<double>::quiet_NaN();
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "cost.angle invalid");
}

TEST_CASE("Invalid cost.ang_vel - negative", "[plannersettings][validation][cost]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].cost.ang_vel = -1.0;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "cost.ang_vel invalid");
}

TEST_CASE("Invalid cost.ang_vel_mag - negative", "[plannersettings][validation][cost]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].cost.ang_vel_mag = -1.0;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "cost.ang_vel_mag invalid");
}

TEST_CASE("Invalid cost.ang_vel_err_dir - negative", "[plannersettings][validation][cost]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].cost.ang_vel_err_dir = -1.0;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "cost.ang_vel_err_dir invalid");
}

TEST_CASE("Invalid cost.control_mult - negative", "[plannersettings][validation][cost]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].cost.control_mult = -1.0;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "cost.control_mult invalid");
}

TEST_CASE("Invalid cost.mtq_control_weight - negative", "[plannersettings][validation][cost]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].cost.mtq_control_weight = -1.0;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "cost.mtq_control_weight invalid");
}

TEST_CASE("Invalid cost.rw_control_weight - negative", "[plannersettings][validation][cost]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].cost.rw_control_weight = -1.0;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "cost.rw_control_weight invalid");
}

TEST_CASE("Invalid cost.magic_control_weight - negative", "[plannersettings][validation][cost]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].cost.magic_control_weight = -1.0;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "cost.magic_control_weight invalid");
}

TEST_CASE("Invalid cost.rw_AM_weight - negative", "[plannersettings][validation][cost]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].cost.rw_AM_weight = -1.0;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "cost.rw_AM_weight invalid");
}

TEST_CASE("Invalid cost.rw_stic_weight - negative", "[plannersettings][validation][cost]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].cost.rw_stic_weight = -1.0;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "cost.rw_stic_weight invalid");
}

TEST_CASE("Invalid cost.RWh_max_mult - negative", "[plannersettings][validation][cost]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].cost.RWh_max_mult = -0.1;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "cost.RWh_max_mult invalid");
}

TEST_CASE("Invalid cost.RWh_max_mult - exceeds 1.0", "[plannersettings][validation][cost]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].cost.RWh_max_mult = 1.1;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "cost.RWh_max_mult invalid");
}

TEST_CASE("Invalid cost.RWh_stiction_mult - negative", "[plannersettings][validation][cost]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].cost.RWh_stiction_mult = -0.1;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "cost.RWh_stiction_mult invalid");
}

TEST_CASE("Invalid cost.RWh_stiction_mult - exceeds 1.0", "[plannersettings][validation][cost]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].cost.RWh_stiction_mult = 1.1;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "cost.RWh_stiction_mult invalid");
}

TEST_CASE("Invalid cost.RWh_ok_mult - negative", "[plannersettings][validation][cost]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].cost.RWh_ok_mult = -0.1;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "cost.RWh_ok_mult invalid");
}

TEST_CASE("Invalid cost.RWh_ok_mult - exceeds 1.0", "[plannersettings][validation][cost]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].cost.RWh_ok_mult = 1.1;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "cost.RWh_ok_mult invalid");
}

TEST_CASE("Invalid cost.angle_N - negative", "[plannersettings][validation][cost]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].cost.angle_N = -1.0;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "cost.angle_N invalid");
}

TEST_CASE("Invalid cost.ang_vel_N - negative", "[plannersettings][validation][cost]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].cost.ang_vel_N = -1.0;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "cost.ang_vel_N invalid");
}

TEST_CASE("Invalid cost.ang_vel_mag_N - negative", "[plannersettings][validation][cost]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].cost.ang_vel_mag_N = -1.0;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "cost.ang_vel_mag_N invalid");
}

TEST_CASE("Invalid cost.ang_vel_err_dir_N - negative", "[plannersettings][validation][cost]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].cost.ang_vel_err_dir_N = -1.0;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "cost.ang_vel_err_dir_N invalid");
}

static const std::string kAfcInvalidMsg =
    "cost.ang_cost_func_type invalid (implemented set {0,1,3,5})";
static const std::string kHuberInvalidMsg =
    "cost.ang_cost_huber_delta invalid (must be finite and > 0)";

TEST_CASE("Invalid cost.ang_cost_func_type - negative", "[plannersettings][validation][cost]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].cost.ang_cost_func_type = -1;
    std::string error_msg;

    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == kAfcInvalidMsg);
}

TEST_CASE("Invalid cost.ang_cost_func_type - removed type 2", "[plannersettings][validation][cost]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].cost.ang_cost_func_type = 2;  // removed (raw acos, concave + singular)
    std::string error_msg;

    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == kAfcInvalidMsg);
}

TEST_CASE("Invalid cost.ang_cost_func_type - removed type 4", "[plannersettings][validation][cost]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].cost.ang_cost_func_type = 4;  // removed (was (1-d)^2)
    std::string error_msg;

    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == kAfcInvalidMsg);
}

TEST_CASE("Invalid cost.ang_cost_func_type - above implemented set", "[plannersettings][validation][cost]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].cost.ang_cost_func_type = 6;  // never implemented
    std::string error_msg;

    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == kAfcInvalidMsg);
}

TEST_CASE("Valid cost.ang_cost_func_type - implemented set {0,1,3,5}", "[plannersettings][validation][cost]") {
    for (int type : {0, 1, 3, 5}) {
        PlannerSettings settings = validSettings();
        settings.passes[0].cost.ang_cost_func_type = type;
        std::string error_msg;

        REQUIRE(saltro::validation::validatePlannerSettings(settings, error_msg));
    }
}

TEST_CASE("Valid cost.ang_cost_huber_delta - positive finite values", "[plannersettings][validation][cost]") {
    for (double delta : {1e-3, 0.1, 0.35, 1.0, 10.0}) {
        PlannerSettings settings = validSettings();
        settings.passes[0].cost.ang_cost_func_type = 5;
        settings.passes[0].cost.ang_cost_huber_delta = delta;
        std::string error_msg;

        REQUIRE(saltro::validation::validatePlannerSettings(settings, error_msg));
    }
}

TEST_CASE("Invalid cost.ang_cost_huber_delta - non-positive", "[plannersettings][validation][cost]") {
    // Rejected regardless of the selected shape — in particular type 5 with
    // non-positive delta is always refused.
    for (double delta : {0.0, -0.35}) {
        for (int type : {5, 3}) {
            PlannerSettings settings = validSettings();
            settings.passes[0].cost.ang_cost_func_type = type;
            settings.passes[0].cost.ang_cost_huber_delta = delta;
            std::string error_msg;

            REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
            REQUIRE(error_msg == kHuberInvalidMsg);
        }
    }
}

TEST_CASE("Invalid cost.ang_cost_huber_delta - non-finite", "[plannersettings][validation][cost]") {
    for (double delta : {std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::infinity()}) {
        PlannerSettings settings = validSettings();
        settings.passes[0].cost.ang_cost_func_type = 5;
        settings.passes[0].cost.ang_cost_huber_delta = delta;
        std::string error_msg;

        REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
        REQUIRE(error_msg == kHuberInvalidMsg);
    }
}

// ============================================================================
// Augmented Lagrangian Configuration Validation Tests
// ============================================================================

TEST_CASE("Invalid auglag.max_outer_iters - negative", "[plannersettings][validation][auglag]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].auglag.max_outer_iters = -1;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "auglag.max_outer_iters invalid");
}

TEST_CASE("Invalid auglag.lag_mult_init - negative", "[plannersettings][validation][auglag]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].auglag.lag_mult_init = -0.1;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "auglag.lag_mult_init invalid");
}

TEST_CASE("Invalid auglag.lag_mult_init - NaN", "[plannersettings][validation][auglag]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].auglag.lag_mult_init = std::numeric_limits<double>::quiet_NaN();
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "auglag.lag_mult_init invalid");
}

TEST_CASE("Invalid auglag.lag_mult_max - negative", "[plannersettings][validation][auglag]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].auglag.lag_mult_max = -0.1;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "auglag.lag_mult_max invalid");
}

TEST_CASE("Invalid auglag.lag_mult_init exceeds lag_mult_max", "[plannersettings][validation][auglag]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].auglag.lag_mult_init = 100.0;
    settings.passes[0].auglag.lag_mult_max = 50.0;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "auglag.lag_mult_init must be <= lag_mult_max");
}

TEST_CASE("Invalid auglag.penalty_init - negative", "[plannersettings][validation][auglag]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].auglag.penalty_init = -0.1;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "auglag.penalty_init invalid");
}

TEST_CASE("Invalid auglag.penalty_init - zero", "[plannersettings][validation][auglag]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].auglag.penalty_init = 0.0;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "auglag.penalty_init invalid");
}

TEST_CASE("Invalid auglag.penalty_max - negative", "[plannersettings][validation][auglag]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].auglag.penalty_max = -0.1;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "auglag.penalty_max invalid");
}

TEST_CASE("Invalid auglag.penalty_max - zero", "[plannersettings][validation][auglag]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].auglag.penalty_max = 0.0;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "auglag.penalty_max invalid");
}

TEST_CASE("Invalid auglag.penalty_init exceeds penalty_max", "[plannersettings][validation][auglag]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].auglag.penalty_init = 100.0;
    settings.passes[0].auglag.penalty_max = 50.0;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "auglag.penalty_init must be <= penalty_max");
}

TEST_CASE("Invalid auglag.penalty_scale - zero", "[plannersettings][validation][auglag]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].auglag.penalty_scale = 0.0;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "auglag.penalty_scale invalid");
}

TEST_CASE("Invalid auglag.penalty_scale - one", "[plannersettings][validation][auglag]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].auglag.penalty_scale = 1.0;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "auglag.penalty_scale invalid");
}

TEST_CASE("Invalid auglag.constraint_tol - negative", "[plannersettings][validation][auglag]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].auglag.constraint_tol = -0.1;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "auglag.constraint_tol invalid");
}

TEST_CASE("Invalid auglag.constraint_tol - zero", "[plannersettings][validation][auglag]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].auglag.constraint_tol = 0.0;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "auglag.constraint_tol invalid");
}

TEST_CASE("Invalid auglag.total_cost_tol - negative", "[plannersettings][validation][auglag]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].auglag.total_cost_tol = -0.1;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "auglag.total_cost_tol invalid");
}

TEST_CASE("Invalid auglag.total_cost_tol - zero", "[plannersettings][validation][auglag]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].auglag.total_cost_tol = 0.0;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "auglag.total_cost_tol invalid");
}

// ============================================================================
// iLQR Configuration Validation Tests
// ============================================================================

TEST_CASE("Invalid ilqr.max_iters - negative", "[plannersettings][validation][ilqr]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].ilqr.max_iters = -1;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "ilqr.max_iters invalid");
}

TEST_CASE("Invalid ilqr.grad_tol - negative", "[plannersettings][validation][ilqr]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].ilqr.grad_tol = -0.1;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "ilqr.grad_tol invalid");
}

TEST_CASE("Invalid ilqr.grad_tol - zero", "[plannersettings][validation][ilqr]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].ilqr.grad_tol = 0.0;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "ilqr.grad_tol invalid");
}

TEST_CASE("Invalid ilqr.cost_tol - negative", "[plannersettings][validation][ilqr]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].ilqr.cost_tol = -0.1;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "ilqr.cost_tol invalid");
}

TEST_CASE("Invalid ilqr.z_count_lim - negative", "[plannersettings][validation][ilqr]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].ilqr.z_count_lim = -1;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "ilqr.z_count_lim invalid");
}

TEST_CASE("Invalid ilqr.max_cost - negative", "[plannersettings][validation][ilqr]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].ilqr.max_cost = -1.0;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "ilqr.max_cost invalid");
}

TEST_CASE("Invalid ilqr.max_cost - zero", "[plannersettings][validation][ilqr]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].ilqr.max_cost = 0.0;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "ilqr.max_cost invalid");
}

TEST_CASE("Invalid ilqr.state_bound - negative", "[plannersettings][validation][ilqr]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].ilqr.state_bound = -1.0;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "ilqr.state_bound invalid");
}

TEST_CASE("Invalid ilqr.state_bound - zero", "[plannersettings][validation][ilqr]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].ilqr.state_bound = 0.0;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "ilqr.state_bound invalid");
}

// ============================================================================
// Regularization Configuration Validation Tests
// ============================================================================

TEST_CASE("Invalid reg.reg_init - negative", "[plannersettings][validation][reg]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].reg.reg_init = -0.1;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "reg.reg_init invalid");
}

TEST_CASE("Invalid reg.reg_min - negative", "[plannersettings][validation][reg]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].reg.reg_min = -0.1;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "reg.reg_min invalid");
}

TEST_CASE("Invalid reg.reg_max - negative", "[plannersettings][validation][reg]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].reg.reg_max = -0.1;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "reg.reg_max invalid");
}

TEST_CASE("Invalid reg.reg_max - zero", "[plannersettings][validation][reg]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].reg.reg_max = 0.0;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "reg.reg_max invalid");
}

TEST_CASE("Invalid reg.reg_min exceeds reg_init", "[plannersettings][validation][reg]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].reg.reg_min = 10.0;
    settings.passes[0].reg.reg_init = 5.0;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "reg.reg_min must be <= reg_init");
}

TEST_CASE("Invalid reg.reg_init exceeds reg_max", "[plannersettings][validation][reg]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].reg.reg_init = 100.0;
    settings.passes[0].reg.reg_max = 50.0;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "reg.reg_init must be <= reg_max");
}

TEST_CASE("Invalid reg.reg_scale - zero", "[plannersettings][validation][reg]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].reg.reg_scale = 0.0;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "reg.reg_scale invalid");
}

TEST_CASE("Invalid reg.reg_scale - one", "[plannersettings][validation][reg]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].reg.reg_scale = 1.0;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "reg.reg_scale invalid");
}

TEST_CASE("Invalid reg.reg_bump - negative", "[plannersettings][validation][reg]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].reg.reg_bump = -0.1;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "reg.reg_bump invalid");
}

TEST_CASE("Invalid reg.reg_bump - zero", "[plannersettings][validation][reg]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].reg.reg_bump = 0.0;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "reg.reg_bump invalid");
}

TEST_CASE("Invalid reg.reg_min_cond - negative", "[plannersettings][validation][reg]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].reg.reg_min_cond = -1;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "reg.reg_min_cond invalid");
}

TEST_CASE("Invalid reg.rand_add_ratio - negative", "[plannersettings][validation][reg]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].reg.rand_add_ratio = -0.1;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "reg.rand_add_ratio invalid");
}

// ============================================================================
// Line Search Configuration Validation Tests
// ============================================================================

TEST_CASE("Invalid linesearch.max_iters - negative", "[plannersettings][validation][linesearch]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].linesearch.max_iters = -1;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "linesearch.max_iters invalid");
}

TEST_CASE("Invalid linesearch.beta1 - negative", "[plannersettings][validation][linesearch]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].linesearch.beta1 = -0.1;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "linesearch.beta1 invalid");
}

TEST_CASE("Invalid linesearch.beta1 - exceeds 1.0", "[plannersettings][validation][linesearch]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].linesearch.beta1 = 1.1;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "linesearch.beta1 invalid");
}

TEST_CASE("Invalid linesearch.beta2 - negative", "[plannersettings][validation][linesearch]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].linesearch.beta2 = -0.1;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "linesearch.beta2 invalid");
}

TEST_CASE("Invalid linesearch.beta2 - zero", "[plannersettings][validation][linesearch]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].linesearch.beta2 = 0.0;
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "linesearch.beta2 invalid");
}

// ============================================================================
// Edge Cases and Multiple Pass Tests
// ============================================================================

TEST_CASE("Multiple passes - first pass invalid", "[plannersettings][validation][multiple_passes]") {
    PlannerSettings settings = validSettings();
    settings.num_passes = 2;
    settings.passes[1] = settings.passes[0];
    settings.passes[0].dt = -1.0;  // Make first pass invalid
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "pass dt invalid");
}

TEST_CASE("Multiple passes - second pass invalid", "[plannersettings][validation][multiple_passes]") {
    PlannerSettings settings = validSettings();
    settings.num_passes = 2;
    settings.passes[1] = settings.passes[0];
    settings.passes[1].cost.angle = -1.0;  // Make second pass invalid
    std::string error_msg;
    
    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "cost.angle invalid");
}

TEST_CASE("Valid boundary values - zero penalty scale just above 1", "[plannersettings][validation][boundary]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].auglag.penalty_scale = 1.00001;
    std::string error_msg;
    
    REQUIRE(saltro::validation::validatePlannerSettings(settings, error_msg));
}

TEST_CASE("Valid boundary values - reg_scale just above 1", "[plannersettings][validation][boundary]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].reg.reg_scale = 1.00001;
    std::string error_msg;
    
    REQUIRE(saltro::validation::validatePlannerSettings(settings, error_msg));
}

TEST_CASE("Valid boundary values - sun_limit_angle at pi", "[plannersettings][validation][boundary]") {
    PlannerSettings settings = validSettings();
    settings.constraints.sun_limit_angle = M_PI;
    std::string error_msg;
    
    REQUIRE(saltro::validation::validatePlannerSettings(settings, error_msg));
}

TEST_CASE("Valid boundary values - zero cost weights", "[plannersettings][validation][boundary]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].cost.angle = 0.0;
    settings.passes[0].cost.ang_vel = 0.0;
    std::string error_msg;

    REQUIRE(saltro::validation::validatePlannerSettings(settings, error_msg));
}

// ============================================================================
// DDP second-order term knobs (G12/G13)
// ============================================================================

TEST_CASE("Regularization DDP knobs default OFF", "[plannersettings][validation][ddp]") {
    RegularizationConfig reg;
    REQUIRE_FALSE(reg.use_dynamics_hess);
    REQUIRE_FALSE(reg.use_constraint_hess);
    REQUIRE_FALSE(reg.psd_clip_quu_ddp);
}

TEST_CASE("psd_clip_quu_ddp bool accepted by validation", "[plannersettings][validation][ddp]") {
    for (bool val : {true, false}) {
        PlannerSettings settings = validSettings();
        settings.passes[0].reg.psd_clip_quu_ddp = val;
        settings.passes[0].reg.use_dynamics_hess = val;
        settings.passes[0].reg.use_constraint_hess = val;
        std::string error_msg;
        REQUIRE(saltro::validation::validatePlannerSettings(settings, error_msg));
        REQUIRE(error_msg.empty());
        REQUIRE(settings.passes[0].reg.psd_clip_quu_ddp == val);
    }
}

// ============================================================================
// PD warm-start goal-rate feedforward validation (G16)
// ============================================================================

TEST_CASE("Default pd_goal_rate_ff_enabled is off and valid", "[plannersettings][validation][pdff]") {
    PlannerSettings settings = validSettings();
    REQUIRE_FALSE(settings.init_traj.pd_goal_rate_ff_enabled);
    std::string error_msg;
    REQUIRE(saltro::validation::validatePlannerSettings(settings, error_msg));
}

TEST_CASE("Enabled PD goal-rate feedforward passes validation", "[plannersettings][validation][pdff]") {
    PlannerSettings settings = validSettings();
    settings.init_traj.initcontroller = 3;
    settings.init_traj.pd_goal_rate_ff_enabled = true;
    std::string error_msg;
    REQUIRE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(settings.init_traj.pd_goal_rate_ff_enabled);
}

TEST_CASE("Invalid cost.gn_curvature_max - negative", "[plannersettings][validation][cost]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].cost.gn_curvature_max = -1.0;
    std::string error_msg;

    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "cost.gn_curvature_max invalid");
}

TEST_CASE("Invalid cost.gn_curvature_max - NaN", "[plannersettings][validation][cost]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].cost.gn_curvature_max = std::numeric_limits<double>::quiet_NaN();
    std::string error_msg;

    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "cost.gn_curvature_max invalid");
}

TEST_CASE("Invalid cost.gn_curvature_max - infinity", "[plannersettings][validation][cost]") {
    PlannerSettings settings = validSettings();
    settings.passes[0].cost.gn_curvature_max = std::numeric_limits<double>::infinity();
    std::string error_msg;

    REQUIRE_FALSE(saltro::validation::validatePlannerSettings(settings, error_msg));
    REQUIRE(error_msg == "cost.gn_curvature_max invalid");
}

TEST_CASE("Valid cost.gn_curvature_max - zero disables and positive allowed", "[plannersettings][validation][cost]") {
    for (double cap : {0.0, 2.0, 10.0, 50.0}) {
        PlannerSettings settings = validSettings();
        settings.passes[0].cost.gn_curvature_max = cap;
        std::string error_msg;

        REQUIRE(saltro::validation::validatePlannerSettings(settings, error_msg));
    }
}
