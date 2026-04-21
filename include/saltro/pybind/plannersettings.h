#pragma once
#include <array>
#include <cmath>
#include <Eigen/Dense>
#include <saltro/limits.h>

static constexpr int MAX_OUTER_PASSES = 2;

/**
 * @brief Initial trajectory configuration.
 *
 * Settings controlling how the initial control/state trajectory is generated
 * before running the ALTRO optimization. The initialization can strongly affect
 * convergence speed and robustness of the solver.
 *
 * @param initcontroller
 * Integer flag selecting which controller or strategy is used to generate the
 * initial trajectory guess. Typical uses include zero-input rollout, PD control,
 * or a custom warm-start controller.
 */
struct InitTrajConfig {
    int initcontroller = 0;
};

/**
 * @brief Disturbance modeling configuration.
 *
 * Enables and parameterizes disturbance models that the optimizer accounts for
 * during trajectory optimization. These disturbances are incorporated into the
 * system dynamics or cost to improve robustness and realism.
 *
 * Enabled disturbances may include:
 * - Aerodynamic drag
 * - Propulsion torques
 * - Solar radiation pressure (SRP)
 * - Gravity-gradient torque
 * - Generic external torques
 * - Residual magnetic dipole effects
 *
 * Coefficient vectors represent linear scaling parameters used in disturbance
 * models. If enabled, they are applied during forward rollout and linearization.
 *
 * @param plan_for_aero Enable aerodynamic disturbance model
 * @param plan_for_prop Enable propulsion torque disturbance
 * @param plan_for_srp Enable solar radiation pressure disturbance
 * @param plan_for_gg Enable gravity-gradient disturbance
 * @param plan_for_gendist Enable generic disturbance torque
 * @param plan_for_resdipole Enable residual dipole torque
 *
 * @param srp_coeff Solar radiation pressure coefficients
 * @param drag_coeff Aerodynamic drag coefficients
 * @param coeff_N Number of disturbance coefficients used in estimation
 *
 * @param res_dipole Residual dipole torque vector
 * @param prop_torque Constant propulsion torque vector
 * @param gendist_torque Generic disturbance torque vector
 * @param J_est Estimated inertia matrix used for disturbance modeling
 */
struct DisturbanceConfig {
    bool plan_for_aero = false;
    bool plan_for_prop = false;
    bool plan_for_srp = false;
    bool plan_for_gg = false;
    bool plan_for_gendist = false;
    bool plan_for_resdipole = false;

    Eigen::Vector3d srp_coeff = Eigen::Vector3d::Zero();
    Eigen::Vector3d drag_coeff = Eigen::Vector3d::Zero();
    int coeff_N = 0;

    Eigen::Vector3d res_dipole = Eigen::Vector3d::Zero();
    Eigen::Vector3d prop_torque = Eigen::Vector3d::Zero();
    Eigen::Vector3d gendist_torque = Eigen::Vector3d::Zero();
    Eigen::Matrix3d J_est = Eigen::Matrix3d::Zero();
};

/**
 * @brief Constraint configuration.
 *
 * Defines hard and soft constraints enforced during optimization. These are
 * typically handled via the augmented Lagrangian framework inside ALTRO.
 *
 * Constraints may include actuator limits, angular velocity bounds, and
 * environmental pointing constraints (e.g., sun avoidance).
 *
 * @param control_limit_scale Scaling factor applied to actuator limits
 * @param u_max Maximum control input vector
 * @param wmax Maximum allowable angular velocity magnitude (rad/s)
 * @param sun_limit_angle Minimum allowed angle to sun direction (rad)
 */

struct ConstraintConfig {
    double control_limit_scale = 0.75;
    /// Stack-allocated bounded control limit vector (no heap allocation).
    Eigen::Matrix<double, Eigen::Dynamic, 1, 0, saltro::limits::MAX_CTRL_DIM, 1> u_max;
    double wmax = 20.0 * M_PI / 180.0;
    double sun_limit_angle = 20.0 * M_PI / 180.0;
};

/**
 * @brief Cost function configuration.
 *
 * Defines the weights and structure of the stage and terminal cost functions
 * used in the ALTRO optimization. Costs typically penalize:
 *
 * - Orientation error
 * - Angular velocity error
 * - Control effort
 * - Reaction wheel momentum
 * - Actuator stiction or saturation
 *
 * Stage costs are applied at each timestep:
 * \f[
 * \ell_k = w_\theta e_\theta^2 + w_\omega e_\omega^2 + w_u \|\mathbf{u}_k\|^2
 * \f]
 *
 * Terminal costs are applied at the final timestep:
 * \f[
 * \ell_N = w_\theta^N e_\theta^2 + w_\omega^N e_\omega^2
 * \f]
 *
 * @param angle Weight on orientation error
 * @param ang_vel Weight on angular velocity error
 * @param ang_vel_mag Weight on angular velocity magnitude
 * @param ang_vel_err_dir Weight on angular velocity direction error
 * @param control_mult Global multiplier on control cost
 *
 * @param mtq_control_weight Magnetorquer control cost weight
 * @param rw_control_weight Reaction wheel control cost weight
 * @param magic_control_weight Additional actuator control weight
 * @param rw_AM_weight Reaction wheel angular momentum penalty
 * @param rw_stic_weight Reaction wheel stiction penalty
 *
 * @param RWh_max_mult Multiplier when wheel momentum near saturation
 * @param RWh_stiction_mult Multiplier for stiction region
 * @param RWh_ok_mult Multiplier when wheel is within safe region
 *
 * @param angle_N Terminal orientation weight
 * @param ang_vel_N Terminal angular velocity weight
 * @param ang_vel_mag_N Terminal angular velocity magnitude weight
 * @param ang_vel_err_dir_N Terminal angular velocity direction weight
 *
 * @param ang_cost_func_type Type of orientation error cost function used
 * @param use_cost_hess If true, use analytic Hessians of cost
 */
struct CostConfig {
    double angle = 1e3;
    double ang_vel = 1e4;
    double ang_vel_mag = 0.0;
    double ang_vel_err_dir = 0.0;
    double control_mult = 1.0;

    double mtq_control_weight = 1e3;
    double rw_control_weight = 1e8;
    double magic_control_weight = 0.0001;
    double rw_AM_weight = 1e4;
    double rw_stic_weight = 1.0;

    double RWh_max_mult = 0.8;
    double RWh_stiction_mult = 0.01;
    double RWh_ok_mult = 0.5;

    /// Terminal weights.  **Principle**: preserve the stage ratios.  If you
    /// set `angle_N` high without matching `ang_vel_N`, the optimizer chases
    /// the target angle at max torque with no penalty for arriving at high ω
    /// — which overruns the AL control-limit penalty and produces wild
    /// actuator saturations (verified experimentally via the angle_N fine
    /// sweep, 2026-04-21: at angle_N=10× or 1000× angle with ang_vel_N=ang_vel,
    /// mtq/rw peaks at 600-1400% of u_max and the solver diverges).
    ///
    /// Use `setTerminalEmphasis(k)` to scale all terminal weights uniformly
    /// instead of editing fields individually.  Defaults already follow the
    /// principle at a modest 10× emphasis.
    double angle_N = 1e4;
    double ang_vel_N = 1e5;
    double ang_vel_mag_N = 0.0;
    double ang_vel_err_dir_N = 0.0;

    int ang_cost_func_type = 2;
    bool use_cost_hess = false;

    /// Scale all terminal weights by `k`, preserving ratios with their
    /// stage counterparts.  This is the safe way to increase terminal
    /// emphasis without introducing the weight-ratio pathology described
    /// above.  `k=1` matches stage; `k=100` is a strong terminal emphasis.
    void setTerminalEmphasis(double k) {
        angle_N = k * angle;
        ang_vel_N = k * ang_vel;
        ang_vel_mag_N = k * ang_vel_mag;
        ang_vel_err_dir_N = k * ang_vel_err_dir;
    }
};

/**
 * @brief Augmented Lagrangian configuration.
 *
 * Controls the outer-loop constraint handling in ALTRO using an augmented
 * Lagrangian method. Constraints are enforced by iteratively updating penalty
 * parameters and Lagrange multipliers:
 *
 * \f[
 * \mathcal{L}(x,u,\lambda,\mu) =
 * J(x,u) + \lambda^\top g(x,u) + \frac{\mu}{2}\|g(x,u)\|^2
 * \f]
 *
 * @param max_outer_iters Maximum number of outer iterations
 * @param lag_mult_init Initial Lagrange multiplier value
 * @param lag_mult_max Maximum allowed Lagrange multiplier
 *
 * @param penalty_init Initial penalty parameter
 * @param penalty_max Maximum penalty parameter
 * @param penalty_scale Multiplicative increase per iteration
 *
 * @param constraint_tol Constraint satisfaction tolerance
 * @param total_cost_tol Total cost convergence tolerance
 */

struct AugLagConfig {
    int max_outer_iters = 30;

    double lag_mult_init = 0.0;
    double lag_mult_max = 1e20;

    double penalty_init = 1e-1;
    double penalty_max = 1e16;
    double penalty_scale = 10.0;

    double constraint_tol = 0.002;
    double total_cost_tol = 1e-2;
};

/**
 * @brief iLQR inner-loop configuration.
 *
 * Settings for the inner iterative LQR solver used by ALTRO to optimize
 * trajectories between augmented Lagrangian updates.
 *
 * The solver iteratively linearizes dynamics and quadratizes costs:
 * \f[
 * \mathbf{x}_{k+1} \approx f(\mathbf{x}_k,\mathbf{u}_k)
 * \f]
 *
 * @param max_iters Maximum number of iLQR iterations
 * @param grad_tol Gradient norm convergence tolerance
 * @param cost_tol Cost improvement tolerance
 * @param z_count_lim Maximum number of zero-improvement steps
 *
 * @param max_cost Abort if cost exceeds this value
 * @param state_bound Maximum allowed state magnitude
 */
struct ILQRConfig {
    int max_iters = 250;
    double grad_tol = 0.0;
    double cost_tol = 1e-1;
    int z_count_lim = 10;

    /// Maximum number of backward+forward-pass retries within a single
    /// iLQR iteration (i.e., the inner regularization-raise loop).  If
    /// we hit this cap without accepting a step, bail out of the outer
    /// loop rather than letting reg cascade to reg_max via triple-bumps
    /// on every forward-pass fail.  Iterations immediately following a
    /// spike-removal substitution automatically get ~10× this budget
    /// because the discontinuous trajectory perturbation legitimately
    /// needs more re-linearization attempts to settle.
    int ls_attempts_lim = 30;

    double max_cost = 1e40;
    double state_bound = 10.0;

    /// Require strict cost decrease in line search (J_new < J_prev).
    /// Original ALTRO behavior; prevents accepting cost-increasing steps.
    bool ls_strict_decrease = false;

    /// Require BOTH grad_tol AND cost_tol for convergence (true),
    /// or allow either alone (false).  Original ALTRO uses conjunctive.
    bool conjunctive_convergence = false;

    /// Persist regularization across iLQR iterations (true = ALTRO-style),
    /// or reset to reg_init each iteration (false = legacy).
    bool persistent_reg = false;
};

/**
 * @brief Regularization configuration.
 *
 * Controls Levenberg–Marquardt-style regularization used to ensure numerical
 * stability when solving Riccati equations and backward passes.
 *
 * Regularization modifies the Hessian:
 * \f[
 * Q_{uu} \leftarrow Q_{uu} + \rho I
 * \f]
 *
 * @param reg_init Initial regularization value
 * @param reg_min Minimum regularization
 * @param reg_max Maximum regularization
 * @param reg_scale Multiplicative scaling factor
 * @param reg_bump Factor used when increasing regularization
 *
 * @param reg_min_cond Minimum condition threshold
 * @param rand_add_ratio Random diagonal perturbation ratio
 *
 * @param use_dynamics_hess Use second derivatives of dynamics
 * @param use_constraint_hess Use second derivatives of constraints
 */
struct RegularizationConfig {
    double reg_init = 1e-2;
    double reg_min = 1e-8;
    double reg_max = 1e30;
    double reg_scale = 1.6;
    double reg_bump = 10.0;

    int reg_min_cond = 2;
    double rand_add_ratio = 0.0;

    bool use_dynamics_hess = false;
    bool use_constraint_hess = false;
};

/**
 * @brief Line search configuration.
 *
 * Parameters for backtracking line search used during forward rollout.
 * The step size \f$\alpha\f$ is reduced until sufficient decrease is achieved:
 *
 * \f[
 * J(\alpha) \le J(0) + \beta_1 \alpha \nabla J^\top d
 * \f]
 *
 * @param max_iters Maximum line search iterations
 * @param beta1 Sufficient decrease parameter
 * @param beta2 Maximum allowable cost increase factor
 */
struct LineSearchConfig {
    int max_iters = 20;
    double beta1 = 1e-10;
    double beta2 = 500.0;
};

/**
 * @brief Spike removal configuration.
 *
 * Controls the homotopy-artifact spike detection and removal system
 * that runs after each accepted iLQR forward pass.
 */
struct SpikeRemovalConfig {
    bool enabled = false;
    int start_at_iter = 2;
    int max_intervention_iters = 5;
    int blend_len = 30;
    int goal_switch_buffer = 15;
    int min_consecutive = 7;
    double exit_fudge = 2.0;
    int min_prior_decrease_knots = 10;
    double min_spike_ratio = 3.0;
    /// Maximum spike window size (knots). Larger spikes are skipped.
    /// 0 = no limit.
    int max_spike_knots = 0;
    double kp_q = 0.3;
    double kd_w = 2.0;
    double rw_scale = 0.0;
    double omega_max = 0.0;
    bool verbose = false;
};

/**
 * @brief Per-pass optimization configuration.
 *
 * Each ALTRO outer pass may use different cost weights, regularization,
 * or time discretization. This allows coarse-to-fine optimization strategies.
 *
 * @param cost Cost configuration for this pass
 * @param auglag Augmented Lagrangian configuration
 * @param ilqr iLQR configuration
 * @param reg Regularization configuration
 * @param linesearch Line search configuration
 * @param spike_removal Spike detection/removal configuration
 * @param dt Timestep used for discretization
 */
struct PassConfig {
    CostConfig cost;
    AugLagConfig auglag;
    ILQRConfig ilqr;
    RegularizationConfig reg;
    LineSearchConfig linesearch;
    SpikeRemovalConfig spike_removal;
    double dt = 1.0;
};

/**
 * @brief Top-level planner settings.
 *
 * Aggregates all configuration parameters required by the ALTRO optimizer.
 * Multiple passes can be executed sequentially, each refining the trajectory.
 *
 * Typical workflow:
 * 1. Initialize trajectory
 * 2. Run ALTRO passes
 * 3. Update penalties and constraints
 * 4. Return optimized trajectory
 *
 * @param constraints Constraint configuration
 * @param disturbances Disturbance modeling configuration
 * @param init_traj Initial trajectory configuration
 *
 * @param num_passes Number of optimization passes
 * @param passes Array of pass-specific configurations
 */
struct PlannerSettings {
    ConstraintConfig constraints;
    DisturbanceConfig disturbances;
    InitTrajConfig init_traj;

    int num_passes = 0;
    std::array<PassConfig, MAX_OUTER_PASSES> passes;
};