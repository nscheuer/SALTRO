#include <saltro/optimizer/validation/validate_plannersettings.h>
#include <cmath>

namespace saltro::optimizer::validation {

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

    for (int i = 0; i < settings.constraints.u_max.size(); ++i) {
        if (!std::isfinite(settings.constraints.u_max(i)) || settings.constraints.u_max(i) < 0.0) {
            error_msg = "u_max contains invalid values";
            return false;
        }
    }

    if (settings.num_passes > 0) {
        for (int p = 0; p < settings.num_passes; ++p) {
            const auto& pass = settings.passes[p];

            if (pass.dt <= 0.0 || !std::isfinite(pass.dt)) {
                error_msg = "pass dt invalid";
                return false;
            }

            if (pass.cost.angle < 0.0 || !std::isfinite(pass.cost.angle)) {
                error_msg = "cost.angle invalid";
                return false;
            }

            if (pass.cost.ang_vel < 0.0 || !std::isfinite(pass.cost.ang_vel)) {
                error_msg = "cost.ang_vel invalid";
                return false;
            }

            if (pass.cost.control_mult < 0.0 || !std::isfinite(pass.cost.control_mult)) {
                error_msg = "cost.control_mult invalid";
                return false;
            }

            if (pass.auglag.max_outer_iters < 0) {
                error_msg = "auglag.max_outer_iters invalid";
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

            if (pass.auglag.penalty_scale <= 1.0 || !std::isfinite(pass.auglag.penalty_scale)) {
                error_msg = "auglag.penalty_scale invalid";
                return false;
            }

            if (pass.auglag.constraint_tol <= 0.0 || !std::isfinite(pass.auglag.constraint_tol)) {
                error_msg = "auglag.constraint_tol invalid";
                return false;
            }

            if (pass.ilqr.max_iters < 0) {
                error_msg = "ilqr.max_iters invalid";
                return false;
            }

            if (pass.ilqr.grad_tol <= 0.0 || !std::isfinite(pass.ilqr.grad_tol)) {
                error_msg = "ilqr.grad_tol invalid";
                return false;
            }

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

            if (pass.linesearch.max_iters < 0) {
                error_msg = "linesearch.max_iters invalid";
                return false;
            }

            if (pass.linesearch.beta1 < 0.0 || pass.linesearch.beta1 > 1.0 || !std::isfinite(pass.linesearch.beta1)) {
                error_msg = "linesearch.beta1 invalid";
                return false;
            }
        }
    }

    return true;
}

}
