#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <saltro/pybind/plannersettings.h>

namespace py = pybind11;

void bind_plannersettings(py::module_& m) {
    py::class_<InitTrajConfig>(m, "InitTrajConfig")
        .def(py::init<>())
        .def_readwrite("initcontroller", &InitTrajConfig::initcontroller);

    py::class_<DisturbanceConfig>(m, "DisturbanceConfig")
        .def(py::init<>())
        .def_readwrite("plan_for_aero", &DisturbanceConfig::plan_for_aero)
        .def_readwrite("plan_for_prop", &DisturbanceConfig::plan_for_prop)
        .def_readwrite("plan_for_srp", &DisturbanceConfig::plan_for_srp)
        .def_readwrite("plan_for_gg", &DisturbanceConfig::plan_for_gg)
        .def_readwrite("plan_for_gendist", &DisturbanceConfig::plan_for_gendist)
        .def_readwrite("plan_for_resdipole", &DisturbanceConfig::plan_for_resdipole)
        .def_readwrite("srp_coeff", &DisturbanceConfig::srp_coeff)
        .def_readwrite("drag_coeff", &DisturbanceConfig::drag_coeff)
        .def_readwrite("coeff_N", &DisturbanceConfig::coeff_N)
        .def_readwrite("res_dipole", &DisturbanceConfig::res_dipole)
        .def_readwrite("prop_torque", &DisturbanceConfig::prop_torque)
        .def_readwrite("gendist_torque", &DisturbanceConfig::gendist_torque)
        .def_readwrite("J_est", &DisturbanceConfig::J_est);

    py::class_<ConstraintConfig>(m, "ConstraintConfig")
        .def(py::init<>())
        .def_readwrite("control_limit_scale", &ConstraintConfig::control_limit_scale)
        .def_readwrite("u_max", &ConstraintConfig::u_max)
        .def_readwrite("wmax", &ConstraintConfig::wmax)
        .def_readwrite("sun_limit_angle", &ConstraintConfig::sun_limit_angle);

    py::class_<CostConfig>(m, "CostConfig")
        .def(py::init<>())
        .def_readwrite("angle", &CostConfig::angle)
        .def_readwrite("ang_vel", &CostConfig::ang_vel)
        .def_readwrite("ang_vel_mag", &CostConfig::ang_vel_mag)
        .def_readwrite("ang_vel_err_dir", &CostConfig::ang_vel_err_dir)
        .def_readwrite("ang_vel_roll_ratio", &CostConfig::ang_vel_roll_ratio)
        .def_readwrite("ang_vel_err_dir_ratio", &CostConfig::ang_vel_err_dir_ratio)
        .def_readwrite("control_mult", &CostConfig::control_mult)
        .def_readwrite("mtq_control_weight", &CostConfig::mtq_control_weight)
        .def_readwrite("rw_control_weight", &CostConfig::rw_control_weight)
        .def_readwrite("magic_control_weight", &CostConfig::magic_control_weight)
        .def_readwrite("rw_AM_weight", &CostConfig::rw_AM_weight)
        .def_readwrite("rw_stic_weight", &CostConfig::rw_stic_weight)
        .def_readwrite("RWh_max_mult", &CostConfig::RWh_max_mult)
        .def_readwrite("RWh_stiction_mult", &CostConfig::RWh_stiction_mult)
        .def_readwrite("RWh_ok_mult", &CostConfig::RWh_ok_mult)
        .def_readwrite("angle_N", &CostConfig::angle_N)
        .def_readwrite("ang_vel_N", &CostConfig::ang_vel_N)
        .def_readwrite("ang_vel_mag_N", &CostConfig::ang_vel_mag_N)
        .def_readwrite("ang_vel_err_dir_N", &CostConfig::ang_vel_err_dir_N)
        .def_readwrite("ang_cost_func_type", &CostConfig::ang_cost_func_type)
        .def_readwrite("use_cost_hess", &CostConfig::use_cost_hess)
        .def_readwrite("cost_hess_gauss_newton", &CostConfig::cost_hess_gauss_newton)
        .def("setTerminalEmphasis", &CostConfig::setTerminalEmphasis,
             py::arg("k") = 100.0,
             "Scale all terminal weights by k, preserving stage ratios. "
             "k=1 matches stage; k=100 is strong terminal emphasis. "
             "Always prefer this over editing individual terminal fields.");

    py::class_<AugLagConfig>(m, "AugLagConfig")
        .def(py::init<>())
        .def_readwrite("max_outer_iters", &AugLagConfig::max_outer_iters)
        .def_readwrite("min_outer_iters", &AugLagConfig::min_outer_iters)
        .def_readwrite("constraint_tol_strict", &AugLagConfig::constraint_tol_strict)
        .def_readwrite("lag_mult_init", &AugLagConfig::lag_mult_init)
        .def_readwrite("lag_mult_max", &AugLagConfig::lag_mult_max)
        .def_readwrite("penalty_init", &AugLagConfig::penalty_init)
        .def_readwrite("penalty_max", &AugLagConfig::penalty_max)
        .def_readwrite("penalty_scale", &AugLagConfig::penalty_scale)
        .def_readwrite("constraint_tol", &AugLagConfig::constraint_tol)
        .def_readwrite("total_cost_tol", &AugLagConfig::total_cost_tol);

    py::class_<ILQRConfig>(m, "ILQRConfig")
        .def(py::init<>())
        .def_readwrite("max_iters", &ILQRConfig::max_iters)
        .def_readwrite("grad_tol", &ILQRConfig::grad_tol)
        .def_readwrite("cost_tol", &ILQRConfig::cost_tol)
        .def_readwrite("ilqr_cost_tol", &ILQRConfig::ilqr_cost_tol)
        .def_readwrite("z_count_lim", &ILQRConfig::z_count_lim)
        .def_readwrite("ls_attempts_lim", &ILQRConfig::ls_attempts_lim)
        .def_readwrite("max_cost", &ILQRConfig::max_cost)
        .def_readwrite("state_bound", &ILQRConfig::state_bound)
        .def_readwrite("ls_strict_decrease", &ILQRConfig::ls_strict_decrease)
        .def_readwrite("conjunctive_convergence", &ILQRConfig::conjunctive_convergence)
        .def_readwrite("persistent_reg", &ILQRConfig::persistent_reg);

    py::class_<RegularizationConfig>(m, "RegularizationConfig")
        .def(py::init<>())
        .def_readwrite("reg_init", &RegularizationConfig::reg_init)
        .def_readwrite("reg_min", &RegularizationConfig::reg_min)
        .def_readwrite("reg_max", &RegularizationConfig::reg_max)
        .def_readwrite("reg_scale", &RegularizationConfig::reg_scale)
        .def_readwrite("reg_bump", &RegularizationConfig::reg_bump)
        .def_readwrite("reg_min_cond", &RegularizationConfig::reg_min_cond)
        .def_readwrite("rand_add_ratio", &RegularizationConfig::rand_add_ratio)
        .def_readwrite("use_dynamics_hess", &RegularizationConfig::use_dynamics_hess)
        .def_readwrite("use_constraint_hess", &RegularizationConfig::use_constraint_hess)
        .def_readwrite("psd_clip_quu_ddp", &RegularizationConfig::psd_clip_quu_ddp)
        .def_readwrite("use_eigen_modification", &RegularizationConfig::use_eigen_modification)
        .def_readwrite("eigen_reg_use_abs", &RegularizationConfig::eigen_reg_use_abs)
        .def_readwrite("eigen_reg_use_relative_floor", &RegularizationConfig::eigen_reg_use_relative_floor)
        .def_readwrite("eigen_reg_condition_cap", &RegularizationConfig::eigen_reg_condition_cap)
        .def_readwrite("eigen_reg_mimic_uniform_trigger", &RegularizationConfig::eigen_reg_mimic_uniform_trigger)
        .def_readwrite("eigen_reg_add_mode", &RegularizationConfig::eigen_reg_add_mode);

    py::class_<LineSearchConfig>(m, "LineSearchConfig")
        .def(py::init<>())
        .def_readwrite("max_iters", &LineSearchConfig::max_iters)
        .def_readwrite("beta1", &LineSearchConfig::beta1)
        .def_readwrite("beta2", &LineSearchConfig::beta2);

    py::class_<SpikeRemovalConfig>(m, "SpikeRemovalConfig")
        .def(py::init<>())
        .def_readwrite("enabled", &SpikeRemovalConfig::enabled)
        .def_readwrite("start_at_iter", &SpikeRemovalConfig::start_at_iter)
        .def_readwrite("max_intervention_iters", &SpikeRemovalConfig::max_intervention_iters)
        .def_readwrite("blend_len", &SpikeRemovalConfig::blend_len)
        .def_readwrite("goal_switch_buffer", &SpikeRemovalConfig::goal_switch_buffer)
        .def_readwrite("min_consecutive", &SpikeRemovalConfig::min_consecutive)
        .def_readwrite("exit_fudge", &SpikeRemovalConfig::exit_fudge)
        .def_readwrite("min_prior_decrease_knots", &SpikeRemovalConfig::min_prior_decrease_knots)
        .def_readwrite("min_spike_ratio", &SpikeRemovalConfig::min_spike_ratio)
        .def_readwrite("max_spike_knots", &SpikeRemovalConfig::max_spike_knots)
        .def_readwrite("kp_q", &SpikeRemovalConfig::kp_q)
        .def_readwrite("kd_w", &SpikeRemovalConfig::kd_w)
        .def_readwrite("rw_scale", &SpikeRemovalConfig::rw_scale)
        .def_readwrite("omega_max", &SpikeRemovalConfig::omega_max)
        .def_readwrite("verbose", &SpikeRemovalConfig::verbose);

    py::class_<PassConfig>(m, "PassConfig")
        .def(py::init<>())
        .def_readwrite("cost", &PassConfig::cost)
        .def_readwrite("auglag", &PassConfig::auglag)
        .def_readwrite("ilqr", &PassConfig::ilqr)
        .def_readwrite("reg", &PassConfig::reg)
        .def_readwrite("linesearch", &PassConfig::linesearch)
        .def_readwrite("spike_removal", &PassConfig::spike_removal)
        .def_readwrite("dt", &PassConfig::dt);

    py::class_<TVLQRSettings>(m, "TVLQRSettings")
        .def(py::init<>())
        .def_readwrite("dt_tvlqr", &TVLQRSettings::dt_tvlqr)
        .def_readwrite("tvlqr_len", &TVLQRSettings::tvlqr_len)
        .def_readwrite("tvlqr_overlap", &TVLQRSettings::tvlqr_overlap);

    py::class_<PlannerSettings>(m, "PlannerSettings")
        .def(py::init<>())
        .def_readwrite("constraints", &PlannerSettings::constraints)
        .def_readwrite("disturbances", &PlannerSettings::disturbances)
        .def_readwrite("init_traj", &PlannerSettings::init_traj)
        .def_readwrite("tvlqr", &PlannerSettings::tvlqr)
        .def_readwrite("num_passes", &PlannerSettings::num_passes)
        .def_readwrite("passes", &PlannerSettings::passes);
}
