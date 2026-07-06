#include <saltro/validation/validate_plannersettings.h>
#include <cmath>

namespace saltro::validation {

bool validatePlannerSettings(const PlannerSettings& settings, std::string& error_msg) {
    if (settings.num_passes < 0 || settings.num_passes > MAX_OUTER_PASSES) {
        error_msg = "num_passes out of range";
        return false;
    }

    if (settings.constraints.control_limit_scale < 0.0 || !std::isfinite(settings.constraints.control_limit_scale)) {
        error_msg = "control_limit_scale invalid";
        return false;
    }

    if (settings.constraints.wmax <= 0.0 || !std::isfinite(settings.constraints.wmax)) {
        error_msg = "wmax invalid";
        return false;
    }

    if (settings.constraints.sun_limit_angle < 0.0 || settings.constraints.sun_limit_angle > M_PI || !std::isfinite(settings.constraints.sun_limit_angle)) {
        error_msg = "sun_limit_angle invalid";
        return false;
    }

    // Validate u_max vector
    if (settings.constraints.u_max.size() == 0) {
        error_msg = "u_max is empty";
        return false;
    }

    for (int i = 0; i < settings.constraints.u_max.size(); ++i) {
        if (!std::isfinite(settings.constraints.u_max(i)) || settings.constraints.u_max(i) < 0.0) {
            error_msg = "u_max contains invalid values";
            return false;
        }
    }

    // Validate disturbance configuration
    if (settings.disturbances.coeff_N < 0) {
        error_msg = "disturbances.coeff_N invalid";
        return false;
    }

    // Validate init_traj configuration
    if (settings.init_traj.initcontroller < 0) {
        error_msg = "init_traj.initcontroller invalid";
        return false;
    }

    // Validate TVLQR gain-generation settings
    if (!std::isfinite(settings.tvlqr.dt_tvlqr) || settings.tvlqr.dt_tvlqr < 0.0) {
        error_msg = "dt_tvlqr invalid";
        return false;
    }

    if (!std::isfinite(settings.tvlqr.tvlqr_len) || settings.tvlqr.tvlqr_len <= 0.0) {
        error_msg = "tvlqr_len invalid";
        return false;
    }

    if (!std::isfinite(settings.tvlqr.tvlqr_overlap) || settings.tvlqr.tvlqr_overlap < 0.0) {
        error_msg = "tvlqr_overlap invalid";
        return false;
    }

    // Validate passes if num_passes > 0
    if (settings.num_passes > 0) {
        for (int p = 0; p < settings.num_passes; ++p) {
            const auto& pass = settings.passes[p];

            // Validate dt
            if (pass.dt <= 0.0 || !std::isfinite(pass.dt)) {
                error_msg = "pass dt invalid";
                return false;
            }

            // Validate cost configuration
            if (pass.cost.angle < 0.0 || !std::isfinite(pass.cost.angle)) {
                error_msg = "cost.angle invalid";
                return false;
            }

            if (pass.cost.ang_vel < 0.0 || !std::isfinite(pass.cost.ang_vel)) {
                error_msg = "cost.ang_vel invalid";
                return false;
            }

            if (pass.cost.ang_vel_mag < 0.0 || !std::isfinite(pass.cost.ang_vel_mag)) {
                error_msg = "cost.ang_vel_mag invalid";
                return false;
            }

            if (pass.cost.ang_vel_err_dir < 0.0 || !std::isfinite(pass.cost.ang_vel_err_dir)) {
                error_msg = "cost.ang_vel_err_dir invalid";
                return false;
            }

            if (pass.cost.control_mult < 0.0 || !std::isfinite(pass.cost.control_mult)) {
                error_msg = "cost.control_mult invalid";
                return false;
            }

            if (pass.cost.mtq_control_weight < 0.0 || !std::isfinite(pass.cost.mtq_control_weight)) {
                error_msg = "cost.mtq_control_weight invalid";
                return false;
            }

            if (pass.cost.rw_control_weight < 0.0 || !std::isfinite(pass.cost.rw_control_weight)) {
                error_msg = "cost.rw_control_weight invalid";
                return false;
            }

            if (pass.cost.magic_control_weight < 0.0 || !std::isfinite(pass.cost.magic_control_weight)) {
                error_msg = "cost.magic_control_weight invalid";
                return false;
            }

            if (pass.cost.rw_AM_weight < 0.0 || !std::isfinite(pass.cost.rw_AM_weight)) {
                error_msg = "cost.rw_AM_weight invalid";
                return false;
            }

            if (pass.cost.rw_stic_weight < 0.0 || !std::isfinite(pass.cost.rw_stic_weight)) {
                error_msg = "cost.rw_stic_weight invalid";
                return false;
            }

            if (pass.cost.RWh_desat_mult < 0.0 || !std::isfinite(pass.cost.RWh_desat_mult)) {
                error_msg = "cost.RWh_desat_mult invalid";
                return false;
            }

            if (pass.cost.RWh_stiction_mult < 0.0 || pass.cost.RWh_stiction_mult > 1.0 || !std::isfinite(pass.cost.RWh_stiction_mult)) {
                error_msg = "cost.RWh_stiction_mult invalid";
                return false;
            }

            if (pass.cost.RWh_knee_frac < 0.0 || pass.cost.RWh_knee_frac > 1.0 || !std::isfinite(pass.cost.RWh_knee_frac)) {
                error_msg = "cost.RWh_knee_frac invalid";
                return false;
            }

            if (pass.cost.angle_N < 0.0 || !std::isfinite(pass.cost.angle_N)) {
                error_msg = "cost.angle_N invalid";
                return false;
            }

            if (pass.cost.ang_vel_N < 0.0 || !std::isfinite(pass.cost.ang_vel_N)) {
                error_msg = "cost.ang_vel_N invalid";
                return false;
            }

            if (pass.cost.ang_vel_mag_N < 0.0 || !std::isfinite(pass.cost.ang_vel_mag_N)) {
                error_msg = "cost.ang_vel_mag_N invalid";
                return false;
            }

            if (pass.cost.ang_vel_err_dir_N < 0.0 || !std::isfinite(pass.cost.ang_vel_err_dir_N)) {
                error_msg = "cost.ang_vel_err_dir_N invalid";
                return false;
            }

            // Implemented set is {0,1,3,5}; anything else (including the
            // removed types 2 (raw acos, concave + singular at both poles)
            // and 4 ((1-d)², = type 1 with a doubled weight) — retired ids
            // are NOT reused) is rejected here so it can't reach the cost
            // switches' throwing default. Migrate type 2 -> type 3
            // (Taylor-protected acos²) or type 0 (linear-in-d).
            {
                const int afc = pass.cost.ang_cost_func_type;
                if (afc != 0 && afc != 1 && afc != 3 && afc != 5) {
                    error_msg = "cost.ang_cost_func_type invalid (implemented set {0,1,3,5})";
                    return false;
                }
                // Pseudo-Huber crossover angle (rad): must be finite and > 0.
                // Rejected unconditionally (not just for type 5) so a bad
                // value can't lurk behind a later shape switch; in particular
                // type 5 with non-positive delta is always refused.
                const double hd = pass.cost.ang_cost_huber_delta;
                if (!(hd > 0.0) || !std::isfinite(hd)) {
                    error_msg = "cost.ang_cost_huber_delta invalid (must be finite and > 0)";
                    return false;
                }
            }

            // Opt-in GN curvature cap: 0.0 disables it. Must be nonnegative
            // and finite (it multiplies the angle weight to bound the rank-1
            // GN Hessian eigenvalue).
            if (pass.cost.gn_curvature_max < 0.0 || !std::isfinite(pass.cost.gn_curvature_max)) {
                error_msg = "cost.gn_curvature_max invalid";
                return false;
            }

            // Validate augmented Lagrangian configuration
            if (pass.auglag.max_outer_iters < 0) {
                error_msg = "auglag.max_outer_iters invalid";
                return false;
            }

            if (pass.auglag.lag_mult_init < 0.0 || !std::isfinite(pass.auglag.lag_mult_init)) {
                error_msg = "auglag.lag_mult_init invalid";
                return false;
            }

            if (pass.auglag.lag_mult_max < 0.0 || !std::isfinite(pass.auglag.lag_mult_max)) {
                error_msg = "auglag.lag_mult_max invalid";
                return false;
            }

            if (pass.auglag.lag_mult_init > pass.auglag.lag_mult_max) {
                error_msg = "auglag.lag_mult_init must be <= lag_mult_max";
                return false;
            }

            if (pass.auglag.penalty_init <= 0.0 || !std::isfinite(pass.auglag.penalty_init)) {
                error_msg = "auglag.penalty_init invalid";
                return false;
            }

            if (pass.auglag.penalty_max <= 0.0 || !std::isfinite(pass.auglag.penalty_max)) {
                error_msg = "auglag.penalty_max invalid";
                return false;
            }

            if (pass.auglag.penalty_init > pass.auglag.penalty_max) {
                error_msg = "auglag.penalty_init must be <= penalty_max";
                return false;
            }

            if (pass.auglag.penalty_scale <= 1.0 || !std::isfinite(pass.auglag.penalty_scale)) {
                error_msg = "auglag.penalty_scale invalid";
                return false;
            }

            if (pass.auglag.constraint_tol <= 0.0 || !std::isfinite(pass.auglag.constraint_tol)) {
                error_msg = "auglag.constraint_tol invalid";
                return false;
            }

            if (pass.auglag.total_cost_tol <= 0.0 || !std::isfinite(pass.auglag.total_cost_tol)) {
                error_msg = "auglag.total_cost_tol invalid";
                return false;
            }

            // Validate iLQR configuration
            if (pass.ilqr.max_iters < 0) {
                error_msg = "ilqr.max_iters invalid";
                return false;
            }

            if (pass.ilqr.grad_tol <= 0.0 || !std::isfinite(pass.ilqr.grad_tol)) {
                error_msg = "ilqr.grad_tol invalid";
                return false;
            }

            if (pass.ilqr.cost_tol < 0.0 || !std::isfinite(pass.ilqr.cost_tol)) {
                error_msg = "ilqr.cost_tol invalid";
                return false;
            }

            if (pass.ilqr.z_count_lim < 0) {
                error_msg = "ilqr.z_count_lim invalid";
                return false;
            }

            if (pass.ilqr.max_cost <= 0.0 || !std::isfinite(pass.ilqr.max_cost)) {
                error_msg = "ilqr.max_cost invalid";
                return false;
            }

            if (pass.ilqr.state_bound <= 0.0 || !std::isfinite(pass.ilqr.state_bound)) {
                error_msg = "ilqr.state_bound invalid";
                return false;
            }

            // Validate regularization configuration
            if (pass.reg.reg_init < 0.0 || !std::isfinite(pass.reg.reg_init)) {
                error_msg = "reg.reg_init invalid";
                return false;
            }

            if (pass.reg.reg_min < 0.0 || !std::isfinite(pass.reg.reg_min)) {
                error_msg = "reg.reg_min invalid";
                return false;
            }

            if (pass.reg.reg_max <= 0.0 || !std::isfinite(pass.reg.reg_max)) {
                error_msg = "reg.reg_max invalid";
                return false;
            }

            if (pass.reg.reg_min > pass.reg.reg_init) {
                error_msg = "reg.reg_min must be <= reg_init";
                return false;
            }

            if (pass.reg.reg_init > pass.reg.reg_max) {
                error_msg = "reg.reg_init must be <= reg_max";
                return false;
            }

            if (pass.reg.reg_scale <= 1.0 || !std::isfinite(pass.reg.reg_scale)) {
                error_msg = "reg.reg_scale invalid";
                return false;
            }

            if (pass.reg.reg_bump <= 0.0 || !std::isfinite(pass.reg.reg_bump)) {
                error_msg = "reg.reg_bump invalid";
                return false;
            }

            if (pass.reg.reg_min_cond < 0) {
                error_msg = "reg.reg_min_cond invalid";
                return false;
            }

            if (pass.reg.rand_add_ratio < 0.0 || !std::isfinite(pass.reg.rand_add_ratio)) {
                error_msg = "reg.rand_add_ratio invalid";
                return false;
            }

            // Validate line search configuration
            if (pass.linesearch.max_iters < 0) {
                error_msg = "linesearch.max_iters invalid";
                return false;
            }

            if (pass.linesearch.beta1 < 0.0 || pass.linesearch.beta1 > 1.0 || !std::isfinite(pass.linesearch.beta1)) {
                error_msg = "linesearch.beta1 invalid";
                return false;
            }

            if (pass.linesearch.beta2 <= 0.0 || !std::isfinite(pass.linesearch.beta2)) {
                error_msg = "linesearch.beta2 invalid";
                return false;
            }
        }
    }

    return true;
}

}
