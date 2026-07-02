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
    /// Margin on the hard RW-momentum constraint: |h| <= rw_momentum_limit_scale * h_max.
    /// <1 leaves slack below saturation (the momentum analog of control_limit_scale for
    /// torque). Default 1.0 = bind exactly at h_max (no margin).
    double rw_momentum_limit_scale = 1.0;
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
 * @param RWh_knee_frac Knee of the momentum soft cost, as a fraction of h_max
 *        (default 0.5: free* region below 50% of saturation, steep quadratic
 *        above it; *free up to the desat term below). The hard ceiling does
 *        NOT live in this cost -- enforce it with
 *        ConstraintConfig::rw_momentum_limit_scale (the AL momentum
 *        constraint), tuned per-family via AugLagConfig. Cost shapes,
 *        constraint enforces.
 * @param RWh_desat_mult Gentle desaturation quadratic applied over the whole
 *        momentum range: 0.5*rw_AM_weight*RWh_desat_mult*(h/h_max)^2. Default
 *        0.05, NOT 0: a perfectly flat free band (zero value, gradient AND
 *        curvature below the knee) lets the first inner solve exploit the
 *        wheel and over-commit before the AL penalties ramp, which was
 *        observed to grind the outer loop to MaxOuterIterations on the
 *        3MTQ+1RW vector-slew case. A small desat keeps the wheel direction
 *        informed without strangling it. Together with the stiction cost this implements
 *        bias-momentum parking: the net potential has stable minima at
 *          h* = (w_stic/h_stic) / (rw_AM_weight*desat/h_max^2 + w_stic/h_stic^2),
 *        with h_stic = RWh_stiction_mult*h_max, so the wheel idles at a bias
 *        speed instead of crossing zero. The special case
 *          rw_stic_weight = rw_AM_weight*RWh_desat_mult*(h_stic/h_max)^2
 *        parks at h* = h_stic/2. All weights stay independent -- this is a
 *        tuning recipe, not a coupling.
 * @param RWh_stiction_mult Stiction band as a fraction of h_max: below
 *        RWh_stiction_mult*h_max the kinked-quadratic stiction cost pushes
 *        |h| away from zero (wheels at rest may not restart). Deliberately
 *        kinked at h = 0: a smoothed peak would inject genuine negative
 *        curvature and create a zero-gradient point a wheel can sit on; the
 *        kink contributes no curvature to the quadratic model, and the
 *        subgradient at exactly h = 0 is 0.
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

    /// Roll-axis weight fraction for the axis-aware ω cost in vector-pointing
    /// mode (0 < ratio ≤ 1). 1.0 reduces W_ω to uniform `c.ang_vel · I`,
    /// matching the current isotropic cost. Smaller values down-weight rotation
    /// about the boresight, freeing the optimizer to spend control on the
    /// 2-DOF off-axis pointing error. Ignored in quaternion-goal mode (all
    /// 3 DOF are constrained there). Default is current behavior.
    double ang_vel_roll_ratio = 1.0;

    /// PSD-fraction knob β ∈ [0, 2) for the Lyapunov `α · err_dir · ω`
    /// crossterm.  Realized scale is `α = β · √(c.angle · λ_min(W_ω))`,
    /// keeping the (q_e_v, ω_e) block-quadratic PSD by construction.
    /// 0 (default) disables the crossterm entirely.  If `ang_vel_err_dir`
    /// is set nonzero, the back-compat path overrides α with that raw
    /// value and ignores this ratio.
    ///
    /// Derivation (Schur complement bound):
    /// The total (q_e_v, ω_e) block-quadratic with the angle cost,
    /// ω cost, and this crossterm is
    ///
    ///   ½ [q_e_v; ω_e]^T · [ w_ang · I        ½α · D^T ] · [q_e_v]
    ///                      [ ½α · D           W_ω      ]   [ω_e]
    ///
    /// where D = ∂err_dir/∂q_e_v and W_ω is the ω cost matrix
    /// (`w_av · I` in quat mode; `w_av · (roll·bs·bs^T + (I − bs·bs^T))`
    /// in vec mode with `ang_vel_roll_ratio` reduction).  The Schur
    /// complement of the bottom-right block is PSD iff
    ///
    ///   w_ang · I − ¼ α² · D^T · W_ω^{-1} · D ≽ 0
    ///   ⟺  α² · λ_max(D^T · W_ω^{-1} · D) ≤ 4 · w_ang.
    ///
    /// Since `err_dir` is a cross-product of unit vectors, ‖D‖ ≤ 1, so
    /// λ_max(D^T · W_ω^{-1} · D) ≤ 1 / λ_min(W_ω).  The conservative
    /// bound that always holds is therefore
    ///
    ///   α ≤ 2 · √(w_ang · λ_min(W_ω)).
    ///
    /// Setting α = β · √(w_ang · λ_min(W_ω)) with β ∈ [0, 2) is PSD by
    /// construction.  FD-tested in `test_satellite_cost_omega_ff.py` at β=0.5.
    double ang_vel_err_dir_ratio = 0.0;

    double mtq_control_weight = 1e3;
    double rw_control_weight = 1e8;
    double magic_control_weight = 0.0001;
    double rw_AM_weight = 1e4;
    double rw_stic_weight = 1.0;

    double RWh_stiction_mult = 0.01;
    double RWh_knee_frac = 0.5;
    double RWh_desat_mult = 0.05;

    /// Terminal weights.  **Principle**: preserve the stage ratios.  If you
    /// set `angle_N` high without matching `ang_vel_N`, the optimizer chases
    /// the target angle at max torque with no penalty for arriving at high ω.
    /// Use `setTerminalEmphasis(k)` to scale all terminal weights uniformly
    /// instead of editing fields individually.
    double angle_N = 1e4;
    double ang_vel_N = 1e5;
    double ang_vel_mag_N = 0.0;
    double ang_vel_err_dir_N = 0.0;

    int ang_cost_func_type = 2;
    bool use_cost_hess = false;

    /// Gauss-Newton mode for the angle-cost (q,q) Hessian block. When true,
    /// drop the second-order chain-rule term `f'(c)·d²c/dq²` (which can be
    /// indefinite in vec mode where `c = bs·R^T·r̂` is degree-2 in q). Keep
    /// the PwA manifold-curvature correction `−grad_dot_q · I_4` — it's the
    /// sphere-tangent projection and is PSD when `f'·c < 0`, which holds
    /// for our cost shapes in the aligned hemisphere.
    ///
    /// Effect by mode:
    ///   - **Vec mode:** drops `f'·d²c/dq²`. Empirically improves convergence
    ///     dramatically (PE_fin 6-22° → 0.2-6.6° on baseline scenarios).
    ///   - **Quat mode:** has no `f'·d²d/dq²` term (d = q_g·q is linear in q),
    ///     so this flag is a no-op.
    /// Default (false) preserves the current full-Hessian behavior.
    bool cost_hess_gauss_newton = false;

    /// Scale all terminal weights by `k`, preserving ratios with their
    /// stage counterparts.  `k=1` matches stage; `k=100` is strong terminal
    /// emphasis.
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
    double grad_tol = 1e-3;
    double cost_tol = 1e-1;
    int z_count_lim = 10;

    double max_cost = 1e40;
    double state_bound = 10.0;
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

    // DDP companion knob: when adding true second-order dynamics/constraint
    // curvature (use_dynamics_hess / use_constraint_hess), Q_uu can become
    // indefinite, which makes the LLT fail or produce an ascent direction.
    // When set, the DDP curvature contributions Q*_ddp are projected to PSD
    // (negative eigenvalues clamped to 0) before being folded into Q_uu and
    // the existing reg+LLT. Default OFF — pure Gauss-Newton is unchanged.
    bool psd_clip_quu_ddp = false;
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
 * @param dt Timestep used for discretization
 */
struct PassConfig {
    CostConfig cost;
    AugLagConfig auglag;
    ILQRConfig ilqr;
    RegularizationConfig reg;
    LineSearchConfig linesearch;
    double dt = 1.0;
};

/**
 * @brief TVLQR gain-generation configuration.
 *
 * Controls the backward-pass discretization and chunking window used when
 * generating tracking gains from an optimized trajectory.
 *
 * @param dt_tvlqr Fixed TVLQR gain discretization step. SALTRO currently
 *                 uses the planner pass dt and keeps this at 0.0.
 * @param tvlqr_len Chunk duration in seconds for gain computation.
 * @param tvlqr_overlap Overlap duration in seconds between consecutive chunks.
 */
struct TVLQRSettings {
    double dt_tvlqr;
    double tvlqr_len = 60.0;
    double tvlqr_overlap = 15.0;

    TVLQRSettings() : dt_tvlqr(0.0) {}
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
    TVLQRSettings tvlqr;

    int num_passes = 0;
    std::array<PassConfig, MAX_OUTER_PASSES> passes;
};