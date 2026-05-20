"""
Comprehensive validation tests for PlannerSettings - mirrors test_validate_plannersettings.cpp
Tests all validation rules for PlannerSettings configuration.
"""

import pytest
import numpy as np
import saltro_py
import math


# ============================================================================
# Helper Functions
# ============================================================================

def valid_settings():
    """Create a valid PlannerSettings configuration"""
    settings = saltro_py.PlannerSettings()
    settings.num_passes = 1
    
    # Set up valid constraint configuration
    settings.constraints.control_limit_scale = 0.75
    settings.constraints.u_max = np.array([1.0, 1.0, 1.0])
    settings.constraints.wmax = 0.3
    settings.constraints.sun_limit_angle = 0.35
    
    # Set up valid disturbance configuration
    settings.disturbances.coeff_N = 3
    
    # Set up valid init_traj configuration
    settings.init_traj.initcontroller = 0
    
    # Set up one valid pass (use defaults which are valid)
    settings.passes[0].dt = 1.0
    settings.passes[0].cost = saltro_py.CostConfig()
    settings.passes[0].auglag = saltro_py.AugLagConfig()
    settings.passes[0].ilqr = saltro_py.ILQRConfig()
    settings.passes[0].reg = saltro_py.RegularizationConfig()
    # Default RegularizationConfig has reg_init=0 and reg_min=1e-8, which
    # fails the validator's `reg_min <= reg_init` check.  Bump reg_init so
    # validSettings() is actually valid.  See the C++ mirror of this fix
    # at tests/unit/validation/test_validate_plannersettings.cpp.
    settings.passes[0].reg.reg_init = settings.passes[0].reg.reg_min
    settings.passes[0].linesearch = saltro_py.LineSearchConfig()
    
    return settings


# ============================================================================
# Basic Validation Tests
# ============================================================================

def test_valid_plannersettings_passes_validation():
    """Valid PlannerSettings should pass validation"""
    settings = valid_settings()
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert ok
    assert error_msg == ""


def test_valid_plannersettings_with_zero_passes():
    """Valid PlannerSettings with zero passes should pass"""
    settings = valid_settings()
    settings.num_passes = 0
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert ok


def test_valid_plannersettings_with_multiple_passes():
    """Valid PlannerSettings with multiple passes should pass"""
    settings = valid_settings()
    settings.num_passes = 2
    # `settings.passes[1] = settings.passes[0]` runs without error in pybind
    # but doesn't actually deep-copy the nested reg/cost/auglag/etc. sub-
    # configs — they revert to the bound `PassConfig` defaults.  Apply the
    # same reg-init bump as the helper does for passes[0].
    settings.passes[1] = settings.passes[0]
    settings.passes[1].reg.reg_init = settings.passes[1].reg.reg_min
    ok, error_msg = saltro_py.validatePlannerSettings(settings)

    assert ok, f"multi-pass should be valid; got: {error_msg}"


# ============================================================================
# num_passes Validation Tests
# ============================================================================

def test_invalid_num_passes_negative():
    """Negative num_passes should fail validation"""
    settings = valid_settings()
    settings.num_passes = -1
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "num_passes out of range"


def test_invalid_num_passes_exceeds_maximum():
    """num_passes exceeding MAX_OUTER_PASSES should fail"""
    settings = valid_settings()
    # MAX_OUTER_PASSES is 2 in the C++ code
    settings.num_passes = 3
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "num_passes out of range"


# ============================================================================
# Constraint Configuration Validation Tests
# ============================================================================

def test_invalid_control_limit_scale_negative():
    """Negative control_limit_scale should fail"""
    settings = valid_settings()
    settings.constraints.control_limit_scale = -0.1
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "control_limit_scale invalid"


def test_invalid_control_limit_scale_nan():
    """NaN control_limit_scale should fail"""
    settings = valid_settings()
    settings.constraints.control_limit_scale = float('nan')
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "control_limit_scale invalid"


def test_invalid_control_limit_scale_infinity():
    """Infinite control_limit_scale should fail"""
    settings = valid_settings()
    settings.constraints.control_limit_scale = float('inf')
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "control_limit_scale invalid"


def test_invalid_wmax_negative():
    """Negative wmax should fail"""
    settings = valid_settings()
    settings.constraints.wmax = -0.1
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "wmax invalid"


def test_invalid_wmax_zero():
    """Zero wmax should fail"""
    settings = valid_settings()
    settings.constraints.wmax = 0.0
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "wmax invalid"


def test_invalid_wmax_nan():
    """NaN wmax should fail"""
    settings = valid_settings()
    settings.constraints.wmax = float('nan')
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "wmax invalid"


def test_invalid_sun_limit_angle_negative():
    """Negative sun_limit_angle should fail"""
    settings = valid_settings()
    settings.constraints.sun_limit_angle = -0.1
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "sun_limit_angle invalid"


def test_invalid_sun_limit_angle_exceeds_pi():
    """sun_limit_angle exceeding pi should fail"""
    settings = valid_settings()
    settings.constraints.sun_limit_angle = math.pi + 0.1
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "sun_limit_angle invalid"


def test_invalid_sun_limit_angle_nan():
    """NaN sun_limit_angle should fail"""
    settings = valid_settings()
    settings.constraints.sun_limit_angle = float('nan')
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "sun_limit_angle invalid"


def test_invalid_u_max_empty_vector():
    """Empty u_max vector should fail"""
    settings = valid_settings()
    settings.constraints.u_max = np.array([])
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "u_max is empty"


def test_invalid_u_max_contains_negative():
    """u_max containing negative values should fail"""
    settings = valid_settings()
    settings.constraints.u_max = np.array([1.0, -0.5, 1.0])
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "u_max contains invalid values"


def test_invalid_u_max_contains_nan():
    """u_max containing NaN should fail"""
    settings = valid_settings()
    settings.constraints.u_max = np.array([float('nan'), 1.0, 1.0])
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "u_max contains invalid values"


# ============================================================================
# Disturbance Configuration Validation Tests
# ============================================================================

def test_invalid_disturbances_coeff_n_negative():
    """Negative disturbances.coeff_N should fail"""
    settings = valid_settings()
    settings.disturbances.coeff_N = -1
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "disturbances.coeff_N invalid"


# ============================================================================
# Init Trajectory Configuration Validation Tests
# ============================================================================

def test_invalid_init_traj_initcontroller_negative():
    """Negative init_traj.initcontroller should fail"""
    settings = valid_settings()
    settings.init_traj.initcontroller = -1
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "init_traj.initcontroller invalid"


# ============================================================================
# Pass dt Validation Tests
# ============================================================================

def test_invalid_pass_dt_negative():
    """Negative pass dt should fail"""
    settings = valid_settings()
    settings.passes[0].dt = -0.1
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "pass dt invalid"


def test_invalid_pass_dt_zero():
    """Zero pass dt should fail"""
    settings = valid_settings()
    settings.passes[0].dt = 0.0
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "pass dt invalid"


def test_invalid_pass_dt_nan():
    """NaN pass dt should fail"""
    settings = valid_settings()
    settings.passes[0].dt = float('nan')
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "pass dt invalid"


# ============================================================================
# Cost Configuration Validation Tests
# ============================================================================

def test_invalid_cost_angle_negative():
    """Negative cost.angle should fail"""
    settings = valid_settings()
    settings.passes[0].cost.angle = -1.0
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "cost.angle invalid"


def test_invalid_cost_angle_nan():
    """NaN cost.angle should fail"""
    settings = valid_settings()
    settings.passes[0].cost.angle = float('nan')
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "cost.angle invalid"


def test_invalid_cost_ang_vel_negative():
    """Negative cost.ang_vel should fail"""
    settings = valid_settings()
    settings.passes[0].cost.ang_vel = -1.0
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "cost.ang_vel invalid"


def test_invalid_cost_ang_vel_mag_negative():
    """Negative cost.ang_vel_mag should fail"""
    settings = valid_settings()
    settings.passes[0].cost.ang_vel_mag = -1.0
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "cost.ang_vel_mag invalid"


def test_invalid_cost_ang_vel_err_dir_negative():
    """Negative cost.ang_vel_err_dir should fail"""
    settings = valid_settings()
    settings.passes[0].cost.ang_vel_err_dir = -1.0
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "cost.ang_vel_err_dir invalid"


def test_invalid_cost_control_mult_negative():
    """Negative cost.control_mult should fail"""
    settings = valid_settings()
    settings.passes[0].cost.control_mult = -1.0
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "cost.control_mult invalid"


def test_invalid_cost_mtq_control_weight_negative():
    """Negative cost.mtq_control_weight should fail"""
    settings = valid_settings()
    settings.passes[0].cost.mtq_control_weight = -1.0
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "cost.mtq_control_weight invalid"


def test_invalid_cost_rw_control_weight_negative():
    """Negative cost.rw_control_weight should fail"""
    settings = valid_settings()
    settings.passes[0].cost.rw_control_weight = -1.0
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "cost.rw_control_weight invalid"


def test_invalid_cost_magic_control_weight_negative():
    """Negative cost.magic_control_weight should fail"""
    settings = valid_settings()
    settings.passes[0].cost.magic_control_weight = -1.0
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "cost.magic_control_weight invalid"


def test_invalid_cost_rw_am_weight_negative():
    """Negative cost.rw_AM_weight should fail"""
    settings = valid_settings()
    settings.passes[0].cost.rw_AM_weight = -1.0
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "cost.rw_AM_weight invalid"


def test_invalid_cost_rw_stic_weight_negative():
    """Negative cost.rw_stic_weight should fail"""
    settings = valid_settings()
    settings.passes[0].cost.rw_stic_weight = -1.0
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "cost.rw_stic_weight invalid"


def test_invalid_cost_rwh_max_mult_negative():
    """Negative cost.RWh_max_mult should fail"""
    settings = valid_settings()
    settings.passes[0].cost.RWh_max_mult = -0.1
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "cost.RWh_max_mult invalid"


def test_invalid_cost_rwh_max_mult_exceeds_one():
    """cost.RWh_max_mult exceeding 1.0 should fail"""
    settings = valid_settings()
    settings.passes[0].cost.RWh_max_mult = 1.1
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "cost.RWh_max_mult invalid"


def test_invalid_cost_rwh_stiction_mult_negative():
    """Negative cost.RWh_stiction_mult should fail"""
    settings = valid_settings()
    settings.passes[0].cost.RWh_stiction_mult = -0.1
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "cost.RWh_stiction_mult invalid"


def test_invalid_cost_rwh_stiction_mult_exceeds_one():
    """cost.RWh_stiction_mult exceeding 1.0 should fail"""
    settings = valid_settings()
    settings.passes[0].cost.RWh_stiction_mult = 1.1
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "cost.RWh_stiction_mult invalid"


def test_invalid_cost_rwh_ok_mult_negative():
    """Negative cost.RWh_ok_mult should fail"""
    settings = valid_settings()
    settings.passes[0].cost.RWh_ok_mult = -0.1
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "cost.RWh_ok_mult invalid"


def test_invalid_cost_rwh_ok_mult_exceeds_one():
    """cost.RWh_ok_mult exceeding 1.0 should fail"""
    settings = valid_settings()
    settings.passes[0].cost.RWh_ok_mult = 1.1
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "cost.RWh_ok_mult invalid"


def test_invalid_cost_angle_n_negative():
    """Negative cost.angle_N should fail"""
    settings = valid_settings()
    settings.passes[0].cost.angle_N = -1.0
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "cost.angle_N invalid"


def test_invalid_cost_ang_vel_n_negative():
    """Negative cost.ang_vel_N should fail"""
    settings = valid_settings()
    settings.passes[0].cost.ang_vel_N = -1.0
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "cost.ang_vel_N invalid"


def test_invalid_cost_ang_vel_mag_n_negative():
    """Negative cost.ang_vel_mag_N should fail"""
    settings = valid_settings()
    settings.passes[0].cost.ang_vel_mag_N = -1.0
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "cost.ang_vel_mag_N invalid"


def test_invalid_cost_ang_vel_err_dir_n_negative():
    """Negative cost.ang_vel_err_dir_N should fail"""
    settings = valid_settings()
    settings.passes[0].cost.ang_vel_err_dir_N = -1.0
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "cost.ang_vel_err_dir_N invalid"


def test_invalid_cost_ang_cost_func_type_negative():
    """Negative cost.ang_cost_func_type should fail"""
    settings = valid_settings()
    settings.passes[0].cost.ang_cost_func_type = -1
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "cost.ang_cost_func_type invalid"


# ============================================================================
# Augmented Lagrangian Configuration Validation Tests
# ============================================================================

def test_invalid_auglag_max_outer_iters_negative():
    """Negative auglag.max_outer_iters should fail"""
    settings = valid_settings()
    settings.passes[0].auglag.max_outer_iters = -1
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "auglag.max_outer_iters invalid"


def test_invalid_auglag_lag_mult_init_negative():
    """Negative auglag.lag_mult_init should fail"""
    settings = valid_settings()
    settings.passes[0].auglag.lag_mult_init = -0.1
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "auglag.lag_mult_init invalid"


def test_invalid_auglag_lag_mult_init_nan():
    """NaN auglag.lag_mult_init should fail"""
    settings = valid_settings()
    settings.passes[0].auglag.lag_mult_init = float('nan')
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "auglag.lag_mult_init invalid"


def test_invalid_auglag_lag_mult_max_negative():
    """Negative auglag.lag_mult_max should fail"""
    settings = valid_settings()
    settings.passes[0].auglag.lag_mult_max = -0.1
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "auglag.lag_mult_max invalid"


def test_invalid_auglag_lag_mult_init_exceeds_max():
    """auglag.lag_mult_init exceeding lag_mult_max should fail"""
    settings = valid_settings()
    settings.passes[0].auglag.lag_mult_init = 100.0
    settings.passes[0].auglag.lag_mult_max = 50.0
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "auglag.lag_mult_init must be <= lag_mult_max"


def test_invalid_auglag_penalty_init_negative():
    """Negative auglag.penalty_init should fail"""
    settings = valid_settings()
    settings.passes[0].auglag.penalty_init = -0.1
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "auglag.penalty_init invalid"


def test_invalid_auglag_penalty_init_zero():
    """Zero auglag.penalty_init should fail"""
    settings = valid_settings()
    settings.passes[0].auglag.penalty_init = 0.0
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "auglag.penalty_init invalid"


def test_invalid_auglag_penalty_max_negative():
    """Negative auglag.penalty_max should fail"""
    settings = valid_settings()
    settings.passes[0].auglag.penalty_max = -0.1
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "auglag.penalty_max invalid"


def test_invalid_auglag_penalty_max_zero():
    """Zero auglag.penalty_max should fail"""
    settings = valid_settings()
    settings.passes[0].auglag.penalty_max = 0.0
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "auglag.penalty_max invalid"


def test_invalid_auglag_penalty_init_exceeds_max():
    """auglag.penalty_init exceeding penalty_max should fail"""
    settings = valid_settings()
    settings.passes[0].auglag.penalty_init = 100.0
    settings.passes[0].auglag.penalty_max = 50.0
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "auglag.penalty_init must be <= penalty_max"


def test_invalid_auglag_penalty_scale_zero():
    """Zero auglag.penalty_scale should fail"""
    settings = valid_settings()
    settings.passes[0].auglag.penalty_scale = 0.0
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "auglag.penalty_scale invalid"


def test_invalid_auglag_penalty_scale_one():
    """auglag.penalty_scale equal to 1.0 should fail"""
    settings = valid_settings()
    settings.passes[0].auglag.penalty_scale = 1.0
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "auglag.penalty_scale invalid"


def test_invalid_auglag_constraint_tol_negative():
    """Negative auglag.constraint_tol should fail"""
    settings = valid_settings()
    settings.passes[0].auglag.constraint_tol = -0.1
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "auglag.constraint_tol invalid"


def test_invalid_auglag_constraint_tol_zero():
    """Zero auglag.constraint_tol should fail"""
    settings = valid_settings()
    settings.passes[0].auglag.constraint_tol = 0.0
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "auglag.constraint_tol invalid"


def test_invalid_auglag_total_cost_tol_negative():
    """Negative auglag.total_cost_tol should fail"""
    settings = valid_settings()
    settings.passes[0].auglag.total_cost_tol = -0.1
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "auglag.total_cost_tol invalid"


def test_invalid_auglag_total_cost_tol_zero():
    """Zero auglag.total_cost_tol should fail"""
    settings = valid_settings()
    settings.passes[0].auglag.total_cost_tol = 0.0
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "auglag.total_cost_tol invalid"


# ============================================================================
# iLQR Configuration Validation Tests
# ============================================================================

def test_invalid_ilqr_max_iters_negative():
    """Negative ilqr.max_iters should fail"""
    settings = valid_settings()
    settings.passes[0].ilqr.max_iters = -1
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "ilqr.max_iters invalid"


def test_invalid_ilqr_grad_tol_negative():
    """Negative ilqr.grad_tol should fail"""
    settings = valid_settings()
    settings.passes[0].ilqr.grad_tol = -0.1
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "ilqr.grad_tol invalid"


def test_valid_ilqr_grad_tol_zero():
    """Zero ilqr.grad_tol should pass (disables gradient convergence check)"""
    settings = valid_settings()
    settings.passes[0].ilqr.grad_tol = 0.0
    ok, error_msg = saltro_py.validatePlannerSettings(settings)

    assert ok, f"grad_tol=0 should be valid (disables check), got: {error_msg}"


def test_invalid_ilqr_cost_tol_negative():
    """Negative ilqr.cost_tol should fail"""
    settings = valid_settings()
    settings.passes[0].ilqr.cost_tol = -0.1
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "ilqr.cost_tol invalid"


def test_invalid_ilqr_z_count_lim_negative():
    """Negative ilqr.z_count_lim should fail"""
    settings = valid_settings()
    settings.passes[0].ilqr.z_count_lim = -1
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "ilqr.z_count_lim invalid"


def test_invalid_ilqr_max_cost_negative():
    """Negative ilqr.max_cost should fail"""
    settings = valid_settings()
    settings.passes[0].ilqr.max_cost = -1.0
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "ilqr.max_cost invalid"


def test_invalid_ilqr_max_cost_zero():
    """Zero ilqr.max_cost should fail"""
    settings = valid_settings()
    settings.passes[0].ilqr.max_cost = 0.0
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "ilqr.max_cost invalid"


def test_invalid_ilqr_state_bound_negative():
    """Negative ilqr.state_bound should fail"""
    settings = valid_settings()
    settings.passes[0].ilqr.state_bound = -1.0
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "ilqr.state_bound invalid"


def test_invalid_ilqr_state_bound_zero():
    """Zero ilqr.state_bound should fail"""
    settings = valid_settings()
    settings.passes[0].ilqr.state_bound = 0.0
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "ilqr.state_bound invalid"


# ============================================================================
# Regularization Configuration Validation Tests
# ============================================================================

def test_invalid_reg_reg_init_negative():
    """Negative reg.reg_init should fail"""
    settings = valid_settings()
    settings.passes[0].reg.reg_init = -0.1
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "reg.reg_init invalid"


def test_invalid_reg_reg_min_negative():
    """Negative reg.reg_min should fail"""
    settings = valid_settings()
    settings.passes[0].reg.reg_min = -0.1
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "reg.reg_min invalid"


def test_invalid_reg_reg_max_negative():
    """Negative reg.reg_max should fail"""
    settings = valid_settings()
    settings.passes[0].reg.reg_max = -0.1
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "reg.reg_max invalid"


def test_invalid_reg_reg_max_zero():
    """Zero reg.reg_max should fail"""
    settings = valid_settings()
    settings.passes[0].reg.reg_max = 0.0
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "reg.reg_max invalid"


def test_invalid_reg_reg_min_exceeds_init():
    """reg.reg_min exceeding reg_init should fail"""
    settings = valid_settings()
    settings.passes[0].reg.reg_min = 10.0
    settings.passes[0].reg.reg_init = 5.0
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "reg.reg_min must be <= reg_init"


def test_invalid_reg_reg_init_exceeds_max():
    """reg.reg_init exceeding reg_max should fail"""
    settings = valid_settings()
    settings.passes[0].reg.reg_init = 100.0
    settings.passes[0].reg.reg_max = 50.0
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "reg.reg_init must be <= reg_max"


def test_invalid_reg_reg_scale_zero():
    """Zero reg.reg_scale should fail"""
    settings = valid_settings()
    settings.passes[0].reg.reg_scale = 0.0
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "reg.reg_scale invalid"


def test_invalid_reg_reg_scale_one():
    """reg.reg_scale equal to 1.0 should fail"""
    settings = valid_settings()
    settings.passes[0].reg.reg_scale = 1.0
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "reg.reg_scale invalid"


def test_invalid_reg_reg_bump_negative():
    """Negative reg.reg_bump should fail"""
    settings = valid_settings()
    settings.passes[0].reg.reg_bump = -0.1
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "reg.reg_bump invalid"


def test_invalid_reg_reg_bump_zero():
    """Zero reg.reg_bump should fail"""
    settings = valid_settings()
    settings.passes[0].reg.reg_bump = 0.0
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "reg.reg_bump invalid"


def test_invalid_reg_reg_min_cond_negative():
    """Negative reg.reg_min_cond should fail"""
    settings = valid_settings()
    settings.passes[0].reg.reg_min_cond = -1
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "reg.reg_min_cond invalid"


def test_invalid_reg_rand_add_ratio_negative():
    """Negative reg.rand_add_ratio should fail"""
    settings = valid_settings()
    settings.passes[0].reg.rand_add_ratio = -0.1
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "reg.rand_add_ratio invalid"


# ============================================================================
# Line Search Configuration Validation Tests
# ============================================================================

def test_invalid_linesearch_max_iters_negative():
    """Negative linesearch.max_iters should fail"""
    settings = valid_settings()
    settings.passes[0].linesearch.max_iters = -1
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "linesearch.max_iters invalid"


def test_invalid_linesearch_beta1_negative():
    """Negative linesearch.beta1 should fail"""
    settings = valid_settings()
    settings.passes[0].linesearch.beta1 = -0.1
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "linesearch.beta1 invalid"


def test_invalid_linesearch_beta1_exceeds_one():
    """linesearch.beta1 exceeding 1.0 should fail"""
    settings = valid_settings()
    settings.passes[0].linesearch.beta1 = 1.1
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "linesearch.beta1 invalid"


def test_invalid_linesearch_beta2_negative():
    """Negative linesearch.beta2 should fail"""
    settings = valid_settings()
    settings.passes[0].linesearch.beta2 = -0.1
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "linesearch.beta2 invalid"


def test_invalid_linesearch_beta2_zero():
    """Zero linesearch.beta2 should fail"""
    settings = valid_settings()
    settings.passes[0].linesearch.beta2 = 0.0
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "linesearch.beta2 invalid"


# ============================================================================
# Edge Cases and Multiple Pass Tests
# ============================================================================

def test_multiple_passes_first_pass_invalid():
    """Multiple passes with first pass invalid should fail"""
    settings = valid_settings()
    settings.num_passes = 2
    settings.passes[1] = settings.passes[0]
    settings.passes[0].dt = -1.0  # Make first pass invalid
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "pass dt invalid"


def test_multiple_passes_second_pass_invalid():
    """Multiple passes with second pass invalid should fail"""
    settings = valid_settings()
    settings.num_passes = 2
    settings.passes[1] = settings.passes[0]
    settings.passes[1].cost.angle = -1.0  # Make second pass invalid
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert not ok
    assert error_msg == "cost.angle invalid"


def test_valid_boundary_penalty_scale_just_above_one():
    """Valid boundary value: penalty_scale just above 1.0"""
    settings = valid_settings()
    settings.passes[0].auglag.penalty_scale = 1.00001
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert ok


def test_valid_boundary_reg_scale_just_above_one():
    """Valid boundary value: reg_scale just above 1.0"""
    settings = valid_settings()
    settings.passes[0].reg.reg_scale = 1.00001
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert ok


def test_valid_boundary_sun_limit_angle_at_pi():
    """Valid boundary value: sun_limit_angle at pi"""
    settings = valid_settings()
    settings.constraints.sun_limit_angle = math.pi
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert ok


def test_valid_boundary_zero_cost_weights():
    """Valid boundary value: zero cost weights"""
    settings = valid_settings()
    settings.passes[0].cost.angle = 0.0
    settings.passes[0].cost.ang_vel = 0.0
    ok, error_msg = saltro_py.validatePlannerSettings(settings)
    
    assert ok
