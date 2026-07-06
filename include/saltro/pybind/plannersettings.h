#pragma once
#include <array>
#include <cmath>
#include <vector>
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

    /**
     * @brief Goal-rate feedforward toggle (ported from OldPlanner smartbdot).
     *
     * When true and initcontroller == 3, the PD warm-start feeds forward a
     * desired body rate @c ω_des computed by finite-differencing the goal
     * trajectory between consecutive knots (OldPlanner @c wkdes), adding a
     * @c -kd·(ω - ω_des) damping term toward the goal rate instead of damping
     * ω toward zero.  Default off preserves the rate-to-zero behavior.
     */
    bool pd_goal_rate_ff_enabled = false;
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
 * @param ang_cost_func_type Type of orientation error cost function used,
 *        as a shape f(d) of the inner alignment scalar d -- in quaternion
 *        mode d = q_goal . q (post hemisphere-alignment, so d in [0, 1]);
 *        in vector-pointing mode d = c = bs . R(q)^T r_hat (cosine of the
 *        boresight-to-target angle):
 *        - 0: 1 - d            (linear)
 *        - 1: 0.5 * (1 - d)^2  (quadratic, convex)
 *        - 3: 0.5 * acos(d)^2  (squared angle, Taylor-protected at d = +1)
 *        - 5: pseudo-Huber in the angle theta = acos(d):
 *             delta^2*(sqrt(1 + (theta/delta)^2) - 1), delta =
 *             `ang_cost_huber_delta`. Quadratic near the goal (matches type
 *             3's 0.5*theta^2 to O((theta/delta)^2)), linear at large angle
 *             (bounded urgency: the demanded descent rate saturates at delta
 *             instead of growing with the error, so large slews don't demand
 *             instant acquisition), with a non-vanishing antipodal escape
 *             gradient (~delta), unlike type 0's plateau. Taylor-protected at
 *             d = +1 and antipode-clamped at d = -1 like type 3.
 *        Default is 3 (the preferred well-conditioned shape).
 *        Type 2 (acos(d), raw angle) was REMOVED: it is concave (f'' < 0,
 *        anti-PSD under Gauss-Newton) and has a genuine gradient singularity
 *        at BOTH poles, including perfect alignment (f' = -1/sqrt(1-d^2) ->
 *        -inf at d = +1, an unremovable singularity unlike type 3's). Migrate
 *        to type 3 (same acos family, convex-usable and Taylor-protected at
 *        the aligned pole) or type 0 for a linear-in-d cost.
 *        Type 4 ((1 - d)^2) was REMOVED: it is exactly type 1 with the
 *        constant 2 absorbed into the angle weight. Migrate by using type 1
 *        with doubled angle weights (2 * 0.5*(1-d)^2 == (1-d)^2).
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

    int ang_cost_func_type = 3;

    /// Pseudo-Huber crossover angle δ (rad) for `ang_cost_func_type = 5`:
    /// the pointing-angle scale at which the type-5 cost transitions from
    /// quadratic (θ ≪ δ, matches type 3's ½θ²) to linear (θ ≫ δ, slope
    /// saturates at δ — bounded urgency on large slews).  Default 0.35 rad
    /// ≈ 20°.  Must be finite and > 0 (validated); ignored by the other
    /// shapes.  Smaller δ ⇒ urgency saturates earlier (gentler large-angle
    /// demands); δ → ∞ recovers type 3 exactly.
    double ang_cost_huber_delta = 0.35;

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

    /// Opt-in curvature cap for the Gauss-Newton vec-pointing angle Hessian
    /// (only active when `cost_hess_gauss_newton == true`). Default 0.0 =
    /// disabled: exact GN behavior, zero change.
    ///
    /// **Why.** In vec mode the assembled GN angle-Hessian q-block is the
    /// rank-1 PSD term `w_ang · f''(c) · dc·dcᵀ` (c = bs·Rᵀ·r̂). Because it's
    /// rank-1, its single nonzero eigenvalue is *exactly*
    ///   λ = w_ang · f''(c) · |dc|².
    /// For the ½·acos² shape (`ang_cost_func_type == 3`) this eigenvalue
    /// diverges like 1/sin θ toward the antipode (θ → 180°): measured
    /// ≈ 71·w_ang at θ=170°, ≈ 7.2e5·w_ang at θ=179.999°. (The *full* exact
    /// Hessian is not stiff there — its divergence is the negative azimuthal
    /// cone, which regularization already handles.) The unbounded GN stiffness
    /// collapses step sizes and stalls near-antipodal slews.
    ///
    /// **What.** When > 0 and GN mode is active, the assembled rank-1
    /// eigenvalue is clamped to `gn_curvature_max · w_ang` by nonnegative
    /// scaling of the (PSD) rank-1 term — so PSD-ness and the gradient are
    /// untouched, and below the cap behavior is bit-identical. The knob is in
    /// units of the angle weight: with the standard boresight parametrization
    /// `|dc|² = 4 sin²θ`, so types 0/1 satisfy `f''·|dc|² ≤ 4` everywhere and a
    /// cap ≥ 4 leaves them exact.
    ///
    /// **Cap ⇔ angle.** A cap C bounds the effective stiffness; type-3 GN is
    /// clamped for all θ inside the angle where `f''(cos θ)·|dc|² = C`
    /// (i.e. `4·f''(cos θ)·sin²θ = C`). Larger C ⇒ cap engages only closer to
    /// the antipode. Typical useful values: C ≈ 10–50.
    double gn_curvature_max = 0.0;

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

/// Constraint families for per-family AL penalty management.
/// Each constraint in `Satellite::constraints()` belongs to one family;
/// the AL outer loop tracks μ and violation contraction per-family rather
/// than per-(constraint, timestep) so that families with different physical
/// scales can be enforced at appropriate strengths without one dominating.
enum class ConstraintFamily : int {
    AngularVelocity = 0,   ///< (|ω|² - ω_max²)/ω_max²  (vec-pointing: bound mostly for sensor/safety)
    SunAvoidance    = 1,   ///< sun_body.x() - cos(sun_limit)
    MTQSaturation   = 2,   ///< |u_mtq| - lim   (clipped physically anyway; AL just keeps optimizer honest)
    RWTorqueSat     = 3,   ///< |u_rw| - tau_lim
    RWMomentum      = 4,   ///< |h| - h_max  (the binding constraint for spinning maneuvers; needs tight enforcement)
    RWStiction      = 5,   ///< -(u·h)²  (always ≤ 0 by construction; vacuous family)
    MagicTorqueSat  = 6,
    NumFamilies     = 7,
};

struct AugLagConfig {
    int max_outer_iters = 30;

    double lag_mult_init = 0.0;
    double lag_mult_max = 1e20;

    /// Global AL penalty knobs (used as defaults when per-family overrides are
    /// empty).
    double penalty_init = 1e-1;
    double penalty_max = 1e16;
    double penalty_scale = 10.0;

    /// Per-family overrides for μ initialization, capping, and ramping. If
    /// `size() == (int)ConstraintFamily::NumFamilies`, the i-th entry applies
    /// to family i; otherwise the global scalar above is used for all families.
    /// Empty by default → fully backward-compatible (single global μ behavior).
    std::vector<double> penalty_init_per_family;
    std::vector<double> penalty_max_per_family;
    std::vector<double> penalty_scale_per_family;

    /// Per-family conditional ramping: if family violation isn't contracting
    /// (max_c_new > contraction_ratio · max_c_prev), ramp μ_family by scale_family.
    /// Otherwise leave μ_family alone (avoid over-ramping past feasibility).
    /// Set ≤ 0 (default) to disable conditional ramping (always ramp every outer iter).
    double family_contraction_ratio = 0.0;

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
 * @param psd_clamp_lxx TESTING/DIAGNOSTIC aid only -- NOT recommended for
 *        production. When true, the backward pass eigen-clamps each stage
 *        cost Hessian lxx to PSD (negative eigenvalues zeroed). Useful to
 *        prove that an indefinite cost Hessian is the culprit when a solve
 *        fails, but it runs an eigendecomposition per knot (slow) and masks
 *        model problems rather than fixing them. Default false; when false
 *        the backward pass is bitwise-identical to the unflagged code.
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
    bool psd_clamp_lxx = false;
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