Optimizer Settings Reference
============================

This page is the central reference for the SALTRO optimizer knobs exposed by
``PlannerSettings`` in ``include/saltro/pybind/plannersettings.h``.

The settings are organized by the same structs used in code:

.. code-block:: text

   PlannerSettings
   ├── constraints
   ├── disturbances
   ├── init_traj
   ├── tvlqr
   ├── num_passes
   └── passes[p]
       ├── cost
       ├── auglag
       ├── ilqr
       ├── reg
       ├── linesearch
       └── dt

At a high level, each pass solves a constrained trajectory optimization problem
with an augmented Lagrangian outer loop and an iLQR/DDP-style inner loop:

.. math::

   \min_{\{x_k, u_k\}}
   \sum_{k=0}^{N-1} \ell_k(x_k, u_k)
   + \ell_N(x_N)
   + \sum_{k=0}^{N-1} \lambda_k^\top c_k(x_k, u_k)
   + \frac{1}{2}\sum_{k=0}^{N-1} c_k(x_k, u_k)^\top \operatorname{diag}(\mu_k) c_k(x_k, u_k)

Validation ranges are enforced in
``src/validation/validate_plannersettings.cpp``. When a range matters for
practical tuning, it is called out below.

PlannerSettings
---------------

.. list-table::
   :header-rows: 1
   :widths: 28 18 54

   * - Field
     - Default
     - Meaning
   * - ``num_passes``
     - ``0``
     - Number of active optimization passes. Must satisfy ``0 <= num_passes <= MAX_OUTER_PASSES``.
   * - ``constraints``
     - struct
     - Global actuator, rate, and pointing constraints shared across passes.
   * - ``disturbances``
     - struct
     - Disturbance models included in rollout, linearization, and cost evaluation.
   * - ``init_traj``
     - struct
     - Initialization policy used to generate the first trajectory guess.
   * - ``tvlqr``
     - struct
     - Settings for post-optimization TVLQR gain generation.
   * - ``passes[p]``
     - array
     - Per-pass tuning for cost, AL, iLQR, regularization, line search, and timestep.

InitTrajConfig
--------------

.. list-table::
   :header-rows: 1
   :widths: 28 18 54

   * - Field
     - Default
     - Meaning
   * - ``initcontroller``
     - ``0``
     - Integer selector for the warm-start strategy used to build the initial trajectory before optimization. Must be nonnegative.

ConstraintConfig
----------------

The optimizer enforces hard bounds and pointing constraints through the
constraint vector ``c(x,u)`` used in the augmented Lagrangian:

.. math::

   c(x,u) \le 0

Examples include control saturation, reaction-wheel momentum bounds, angular
rate limits, and sun-angle exclusion constraints.

.. list-table::
   :header-rows: 1
   :widths: 28 18 54

   * - Field
     - Default
     - Meaning
   * - ``control_limit_scale``
     - ``0.75``
     - Margin applied to actuator limits. Effective optimizer bound is ``|u_i| <= control_limit_scale * u_max_i``. Must be finite and nonnegative.
   * - ``rw_momentum_limit_scale``
     - ``1.0``
     - Margin on reaction-wheel momentum constraints. Effective bound is ``|h_i| <= rw_momentum_limit_scale * h_max_i``.
   * - ``u_max``
     - unset
     - Per-actuator control limits. Must be nonempty and elementwise finite and nonnegative.
   * - ``wmax``
     - ``20 deg/s``
     - Maximum allowed angular-velocity magnitude used in rate constraints. Must be positive.
   * - ``sun_limit_angle``
     - ``20 deg``
     - Minimum allowed angle to the sun vector for sun-avoidance constraints. Must lie in ``[0, \pi]``.

DisturbanceConfig
-----------------

If enabled, disturbance torques are folded into the dynamics model:

.. math::

   \dot{\omega} = J^{-1}\left(\tau_{\text{control}} + \tau_{\text{disturbance}} - \omega \times J\omega\right)

The booleans decide which disturbance terms contribute to
``\tau_{\text{disturbance}}``.

.. list-table::
   :header-rows: 1
   :widths: 28 18 54

   * - Field
     - Default
     - Meaning
   * - ``plan_for_aero``
     - ``false``
     - Include aerodynamic disturbance modeling.
   * - ``plan_for_prop``
     - ``false``
     - Include constant propulsion torque disturbance.
   * - ``plan_for_srp``
     - ``false``
     - Include solar-radiation-pressure disturbance.
   * - ``plan_for_gg``
     - ``false``
     - Include gravity-gradient torque.
   * - ``plan_for_gendist``
     - ``false``
     - Include generic external disturbance torque.
   * - ``plan_for_resdipole``
     - ``false``
     - Include residual magnetic dipole disturbance.
   * - ``srp_coeff``
     - ``0``
     - SRP model coefficients.
   * - ``drag_coeff``
     - ``0``
     - Aerodynamic drag model coefficients.
   * - ``coeff_N``
     - ``0``
     - Number of disturbance coefficients used by the disturbance model. Must be nonnegative.
   * - ``res_dipole``
     - ``0``
     - Residual dipole vector.
   * - ``prop_torque``
     - ``0``
     - Constant propulsion torque vector.
   * - ``gendist_torque``
     - ``0``
     - Generic constant disturbance torque vector.
   * - ``J_est``
     - ``0``
     - Inertia estimate used when evaluating disturbance terms.

TVLQRSettings
-------------

These settings control gain generation after a nominal trajectory is already
optimized.

.. list-table::
   :header-rows: 1
   :widths: 28 18 54

   * - Field
     - Default
     - Meaning
   * - ``dt_tvlqr``
     - ``0.0``
     - TVLQR discretization step. Current SALTRO usage keeps this at ``0`` and reuses the planner pass ``dt``. Must be finite and nonnegative.
   * - ``tvlqr_len``
     - ``60.0``
     - Duration, in seconds, of each gain-computation chunk. Must be positive.
   * - ``tvlqr_overlap``
     - ``15.0``
     - Overlap, in seconds, between adjacent TVLQR chunks. Must be finite and nonnegative.

PassConfig
----------

Each active pass ``passes[p]`` defines one full ALTRO/iLQR solve at a specific
discretization:

.. math::

   t_{k+1} = t_k + \Delta t,
   \qquad \Delta t = \texttt{passes[p].dt}

.. list-table::
   :header-rows: 1
   :widths: 28 18 54

   * - Field
     - Default
     - Meaning
   * - ``dt``
     - ``1.0``
     - Timestep used for dynamics rollout, Jacobians, Hessians, and line search in this pass. Must be positive.
   * - ``cost``
     - struct
     - Stage and terminal objective weights and cost-model options.
   * - ``auglag``
     - struct
     - Augmented-Lagrangian outer-loop settings.
   * - ``ilqr``
     - struct
     - Inner iLQR stopping and safety settings.
   * - ``reg``
     - struct
     - Riccati regularization and optional second-order stabilizers.
   * - ``linesearch``
     - struct
     - Forward-pass backtracking parameters.

CostConfig
----------

The stage cost is composed of attitude error, angular-velocity penalties,
control effort, and reaction-wheel penalties:

.. math::

   \ell_k =
   w_{\theta} h_{\theta}
   + \ell_{\omega}
   + \ell_{u}
   + \ell_{h}
   + \ell_{\text{stiction}}

Terminal cost uses the corresponding ``*_N`` weights:

.. math::

   \ell_N =
   w_{\theta,N} h_{\theta,N}
   + \ell_{\omega,N}

Quaternion-goal mode uses one of these scalar attitude-error shapes with
``d = |q_{\text{goal}}^\top q|`` after hemisphere alignment:

.. math::

   h_{\theta}(d) =
   \begin{cases}
   1 - d & \texttt{ang\_cost\_func\_type = 0} \\
   \frac{1}{2}(1-d)^2 & \texttt{ang\_cost\_func\_type = 1} \\
   \arccos(d) & \texttt{ang\_cost\_func\_type = 2} \\
   \frac{1}{2}\arccos(d)^2 & \texttt{ang\_cost\_func\_type = 3}
   \end{cases}

Vector-pointing mode uses the same shape family, but with
``d = c = b_s^\top R(q)^\top \hat{r}``.

Base angular-velocity cost:

.. math::

   \ell_{\omega,\text{quad}} = \frac{1}{2} w_{\omega} \lVert \omega \rVert^2

In vector-pointing mode, ``ang_vel_roll_ratio`` reduces the penalty along the
boresight axis:

.. math::

   W_{\omega} =
   w_{\omega}\left(r_{\text{roll}}\, b_s b_s^\top + (I - b_s b_s^\top)\right)

Optional direction-error crossterm:

.. math::

   \ell_{\omega,\text{cross}} = \alpha \, e_{\text{dir}}^\top \omega,
   \qquad
   \alpha = \beta \sqrt{w_{\theta}\lambda_{\min}(W_{\omega})}

where ``beta = ang_vel_err_dir_ratio``. The implementation keeps this
quadratic block PSD by requiring ``0 <= beta < 2`` conceptually, although the
current validator does not enforce that range.

Control effort is normalized by each actuator limit:

.. math::

   \ell_u =
   \frac{1}{2}\,\texttt{control\_mult}
   \sum_i w_{u,i}\left(\frac{u_i}{u_{i,\max}}\right)^2

Reaction-wheel momentum penalty activates near saturation:

.. math::

   h_{\text{thresh}} = \texttt{RWh\_max\_mult}\, h_{\max},
   \qquad
   \ell_h =
   \frac{1}{2}\,w_h
   \left(\frac{|h| - h_{\text{thresh}}}{h_{\max} - h_{\text{thresh}}}\right)^2

for ``|h| > h_thresh``.

.. list-table::
   :header-rows: 1
   :widths: 28 18 54

   * - Field
     - Default
     - Meaning
   * - ``angle``
     - ``1e3``
     - Stage weight on the attitude-error shape ``h_theta``. Must be finite and nonnegative.
   * - ``ang_vel``
     - ``1e4``
     - Stage weight on the quadratic angular-velocity penalty. Must be finite and nonnegative.
   * - ``ang_vel_mag``
     - ``0.0``
     - Stage weight on magnetic-field-aligned angular-velocity magnitude. Present in cost and Jacobian code; Hessian support is intentionally incomplete, so the default stays zero.
   * - ``ang_vel_err_dir``
     - ``0.0``
     - Legacy cross-term coefficient. In quaternion mode it activates the older ``q``-``\omega`` coupling path. Must be finite and nonnegative.
   * - ``control_mult``
     - ``1.0``
     - Global multiplier applied to all actuator control-effort terms. Must be finite and nonnegative.
   * - ``ang_vel_roll_ratio``
     - ``1.0``
     - Vector-pointing only. Relative penalty on roll about the boresight axis. ``1`` gives isotropic ``w_omega I``; smaller values free roll motion. Intended range is ``0 < ratio <= 1``.
   * - ``ang_vel_err_dir_ratio``
     - ``0.0``
     - New PSD-scaled direction-error cross-term coefficient ``beta`` used to build ``alpha`` above. ``0`` disables the crossterm.
   * - ``mtq_control_weight``
     - ``1e3``
     - Per-unit normalized magnetorquer control penalty.
   * - ``rw_control_weight``
     - ``1e8``
     - Per-unit normalized reaction-wheel torque penalty.
   * - ``magic_control_weight``
     - ``1e-4``
     - Per-unit normalized control penalty for MAGIC actuators.
   * - ``rw_AM_weight``
     - ``1e4``
     - Weight on the reaction-wheel momentum soft penalty near saturation.
   * - ``rw_stic_weight``
     - ``1.0``
     - Weight on the reaction-wheel stiction-region penalty.
   * - ``RWh_max_mult``
     - ``0.8``
     - Fraction of ``h_max`` where the high-momentum penalty begins. Must lie in ``[0,1]``.
   * - ``RWh_stiction_mult``
     - ``0.01``
     - Fraction of ``h_max`` used to scale the stiction region. Must lie in ``[0,1]``.
   * - ``RWh_ok_mult``
     - ``0.5``
     - Additional multiplier used in the low-momentum wheel penalty shaping. Must lie in ``[0,1]``.
   * - ``angle_N``
     - ``1e4``
     - Terminal attitude weight. Must be finite and nonnegative.
   * - ``ang_vel_N``
     - ``1e5``
     - Terminal angular-velocity weight. Must be finite and nonnegative.
   * - ``ang_vel_mag_N``
     - ``0.0``
     - Terminal magnetic-field-aligned angular-velocity weight.
   * - ``ang_vel_err_dir_N``
     - ``0.0``
     - Terminal legacy direction-error cross-term weight.
   * - ``ang_cost_func_type``
     - ``2``
     - Attitude-error shape selector. The implemented and validated set is ``{0,1,2,3}``. Type ``2`` is supported but less numerically friendly near alignment than type ``3``.
   * - ``use_cost_hess``
     - ``false``
     - If true, use analytic state Hessians of the cost in the backward pass. If false, SALTRO keeps control curvature but drops potentially troublesome second-order state curvature.
   * - ``cost_hess_gauss_newton``
     - ``false``
     - Vector-pointing only. When true, drops the exact chain-rule ``f'(c) d^2c/dq^2`` term and keeps the Gauss-Newton attitude Hessian, which is often more robust.

``CostConfig::setTerminalEmphasis(k)`` is a convenience helper that preserves
stage-to-terminal ratios:

.. math::

   \texttt{angle\_N} = k\,\texttt{angle},
   \quad
   \texttt{ang\_vel\_N} = k\,\texttt{ang\_vel},
   \quad
   \texttt{ang\_vel\_mag\_N} = k\,\texttt{ang\_vel\_mag},
   \quad
   \texttt{ang\_vel\_err\_dir\_N} = k\,\texttt{ang\_vel\_err\_dir}

AugLagConfig
------------

The outer loop updates multipliers and penalties for constraints:

.. math::

   \mathcal{L}(x,u,\lambda,\mu) =
   J(x,u) + \lambda^\top c(x,u) + \frac{1}{2} c(x,u)^\top \operatorname{diag}(\mu) c(x,u)

with a typical penalty growth rule of the form:

.. math::

   \mu^{(j+1)} = \min\left(\texttt{penalty\_max},\ \texttt{penalty\_scale}\,\mu^{(j)}\right)

.. list-table::
   :header-rows: 1
   :widths: 28 18 54

   * - Field
     - Default
     - Meaning
   * - ``max_outer_iters``
     - ``30``
     - Maximum number of augmented-Lagrangian outer iterations. Must be nonnegative.
   * - ``lag_mult_init``
     - ``0.0``
     - Initial Lagrange multiplier value. Must satisfy ``0 <= lag_mult_init <= lag_mult_max``.
   * - ``lag_mult_max``
     - ``1e20``
     - Cap on Lagrange multiplier magnitude. Must be finite and nonnegative.
   * - ``penalty_init``
     - ``1e-1``
     - Initial quadratic penalty coefficient. Must satisfy ``0 < penalty_init <= penalty_max``.
   * - ``penalty_max``
     - ``1e16``
     - Maximum quadratic penalty coefficient. Must be positive.
   * - ``penalty_scale``
     - ``10.0``
     - Multiplicative penalty increase between outer iterations. Must be greater than ``1``.
   * - ``constraint_tol``
     - ``0.002``
     - Convergence tolerance on constraint violation. Must be positive.
   * - ``total_cost_tol``
     - ``1e-2``
     - Convergence tolerance on total augmented objective change. Must be positive.

ILQRConfig
----------

The inner loop iteratively linearizes the dynamics and solves local LQR
subproblems:

.. math::

   x_{k+1} \approx f(x_k, u_k),
   \qquad
   \delta u_k = d_k + K_k \delta x_k

.. list-table::
   :header-rows: 1
   :widths: 28 18 54

   * - Field
     - Default
     - Meaning
   * - ``max_iters``
     - ``250``
     - Maximum number of inner iLQR iterations. Must be nonnegative.
   * - ``grad_tol``
     - ``1e-3``
     - Gradient-based convergence threshold. Must be positive.
   * - ``cost_tol``
     - ``1e-1``
     - Cost-improvement tolerance used to detect convergence or stagnation. Must be finite and nonnegative.
   * - ``z_count_lim``
     - ``10``
     - Maximum allowed number of consecutive near-zero-improvement steps before declaring stagnation. Must be nonnegative.
   * - ``max_cost``
     - ``1e40``
     - Safety abort threshold if the trajectory cost blows up. Must be positive.
   * - ``state_bound``
     - ``10.0``
     - Safety bound on state magnitude during rollout. Must be positive.

RegularizationConfig
--------------------

Backward-pass regularization stabilizes the control Hessian:

.. math::

   Q_{uu,\text{reg}} = Q_{uu} + \rho I

where ``rho`` is adapted from the regularization schedule.

If second-order DDP curvature is enabled, SALTRO can optionally PSD-project the
extra curvature blocks before assembling ``Q_{uu}``. The newer
``psd_clamp_lxx`` diagnostic knob instead projects the stage state Hessian
itself:

.. math::

   L_{xx} = V \Lambda V^\top
   \;\rightarrow\;
   L_{xx}^{+} = V \max(\Lambda, 0) V^\top

.. warning::

   ``psd_clamp_lxx`` is intentionally a testing and diagnostic aid, not a
   production tuning recommendation. It is useful to prove that an indefinite
   cost Hessian is the failure source, but it adds an eigendecomposition per
   knot and can hide model issues instead of fixing them.

.. list-table::
   :header-rows: 1
   :widths: 28 18 54

   * - Field
     - Default
     - Meaning
   * - ``reg_init``
     - ``1e-2``
     - Initial regularization magnitude ``rho``. Must satisfy ``0 <= reg_min <= reg_init <= reg_max``.
   * - ``reg_min``
     - ``1e-8``
     - Minimum allowed regularization.
   * - ``reg_max``
     - ``1e30``
     - Maximum allowed regularization. Must be positive.
   * - ``reg_scale``
     - ``1.6``
     - Multiplicative factor used when adapting the regularization schedule. Must be greater than ``1``.
   * - ``reg_bump``
     - ``10.0``
     - Additional increase factor used on failed backward/forward steps. Must be positive.
   * - ``reg_min_cond``
     - ``2``
     - Minimum conditioning threshold used in regularization logic. Must be nonnegative.
   * - ``rand_add_ratio``
     - ``0.0``
     - Optional random diagonal perturbation ratio for robustness experiments. Must be finite and nonnegative.
   * - ``use_dynamics_hess``
     - ``false``
     - If true, include second derivatives of the dynamics in the DDP quadratic model.
   * - ``use_constraint_hess``
     - ``false``
     - If true, include second derivatives of the constraints in the DDP quadratic model.
   * - ``psd_clip_quu_ddp``
     - ``false``
     - If true, PSD-project the additional DDP curvature contributions before they are folded into the Riccati recursion.
   * - ``psd_clamp_lxx``
     - ``false``
     - If true, eigen-clamp each stage cost Hessian ``lxx`` to PSD inside the backward pass. Diagnostic only.

LineSearchConfig
----------------

The forward pass uses backtracking line search on the step size ``\alpha`` and
accepts a step when sufficient decrease is achieved:

.. math::

   J(\alpha) \le J(0) + \beta_1 \alpha \nabla J^\top d

The implementation also guards against excessive cost growth using
``beta2``.

.. list-table::
   :header-rows: 1
   :widths: 28 18 54

   * - Field
     - Default
     - Meaning
   * - ``max_iters``
     - ``20``
     - Maximum number of line-search backtracking trials. Must be nonnegative.
   * - ``beta1``
     - ``1e-10``
     - Armijo-style sufficient-decrease coefficient. Must lie in ``[0,1]``.
   * - ``beta2``
     - ``500.0``
     - Additional allowable cost-growth bound used in forward-pass acceptance logic. Must be positive.

Practical Notes
---------------

``ang_cost_func_type = 3`` is usually the safest curvature choice for pointing
problems. Type ``2`` is valid, but its raw ``acos`` curvature is concave in the
alignment scalar and is one of the reasons ``psd_clamp_lxx`` can become useful
as a diagnosis tool.

``use_cost_hess``, ``use_dynamics_hess``, and ``use_constraint_hess`` are the
main switches that move SALTRO from a mostly Gauss-Newton-style model toward a
fuller second-order DDP model. That can improve convergence, but it also makes
Hessian definiteness much more important.

For terminal retuning, prefer ``setTerminalEmphasis(k)`` over editing
``angle_N`` and ``ang_vel_N`` independently. It preserves the intended balance
between reaching the target and arriving without excessive angular velocity.
