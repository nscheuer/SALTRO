Convergence Logic Tree
======================

This page helps trace an AL-iLQR solve from the innermost line-search
decision to the final AL-iLQR status. Use it when ``trajOpt`` reports a
failure, when constraints are satisfied but the inner iLQR status is not
``Converged``, or when a trajectory appears unchanged despite a successful
return.

The three layers are:

.. raw:: html

   <div class="saltro-legend" aria-label="Convergence logic tree color legend">
     <span class="saltro-pill saltro-pill-inner">Inner: line search / forward pass</span>
     <span class="saltro-pill saltro-pill-middle">Middle: iLQR subproblem</span>
     <span class="saltro-pill saltro-pill-outer">Outer: AL-iLQR constraints</span>
   </div>


How To Read The Tree
--------------------

Start at the top of the outer layer. Each AL-iLQR outer iteration calls one
iLQR subproblem. Each iLQR iteration repeatedly tries a backward pass and a
forward-pass line search while increasing regularization. The forward pass is
the only place where the state and control trajectory are actually accepted
and overwritten.

.. raw:: html

   <div class="saltro-tree">
     <section class="saltro-layer saltro-outer">
       <header>
         <span class="saltro-layer-name">Outer Layer</span>
         <h2>AL-iLQR loop</h2>
         <p>Goal: satisfy all inequality constraints, then return a trajectory only if the inner solve made real progress or converged.</p>
       </header>

       <ol class="saltro-steps">
         <li>
           <div class="saltro-node saltro-node-action">
             <strong>Initialize augmented terms</strong>
             <p>Collect each constraint vector <code>c_k</code>. Initialize <code>lambda_aug</code> and <code>mu_aug</code> with the same per-timestep sizes.</p>
           </div>
         </li>
         <li>
           <div class="saltro-node saltro-node-call">
             <strong>Call the iLQR subproblem</strong>
             <p>Run the middle layer with the current augmented Lagrangian penalties.</p>
           </div>
           <div class="saltro-branch saltro-middle">
             <span class="saltro-branch-label">enter middle layer</span>
           </div>
         </li>
         <li>
           <div class="saltro-node saltro-node-decision">
             <strong>Did iLQR return a non-recoverable failure?</strong>
             <p>Current recoverable statuses are <code>MaxIterations</code> and <code>RegularizationExceeded</code>, because either may still leave an improved trajectory.</p>
           </div>
           <div class="saltro-outcomes">
             <div class="saltro-outcome saltro-bad">
               <strong>Yes</strong>
               <p>Return <code>ALILQRStatus::InnerFailed</code>.</p>
             </div>
             <div class="saltro-outcome saltro-good">
               <strong>No</strong>
               <p>Measure the maximum positive constraint violation.</p>
             </div>
           </div>
         </li>
         <li>
           <div class="saltro-node saltro-node-decision">
             <strong>Is <code>max_c <= constraint_tol</code>?</strong>
             <p>This is the AL feasibility test. It is checked after every iLQR subproblem, even if the iLQR status is not <code>Converged</code>.</p>
           </div>
           <div class="saltro-outcomes">
             <div class="saltro-outcome saltro-good">
               <strong>Yes, and iLQR converged or accepted at least one forward-pass step</strong>
               <p>Return <code>ALILQRStatus::Converged</code>. Constraint satisfaction is valid because the trajectory was optimized or the inner cost tolerance was reached.</p>
             </div>
             <div class="saltro-outcome saltro-bad">
               <strong>Yes, but iLQR accepted zero steps</strong>
               <p>Return <code>ALILQRStatus::InnerFailed</code>. This blocks false positives where the warm start was feasible but the optimizer never improved the trajectory.</p>
             </div>
             <div class="saltro-outcome saltro-warn">
               <strong>No</strong>
               <p>Update <code>lambda_aug</code> and <code>mu_aug</code>, then start another AL outer iteration.</p>
             </div>
           </div>
         </li>
         <li>
           <div class="saltro-node saltro-node-stop">
             <strong>Outer iteration budget exhausted</strong>
             <p>If constraints never satisfy the tolerance, return <code>ALILQRStatus::MaxOuterIterations</code>.</p>
           </div>
         </li>
       </ol>
     </section>

     <section class="saltro-layer saltro-middle">
       <header>
         <span class="saltro-layer-name">Middle Layer</span>
         <h2>iLQR subproblem</h2>
         <p>Goal: locally reduce the augmented trajectory cost by alternating backward and forward passes.</p>
       </header>

       <ol class="saltro-steps">
         <li>
           <div class="saltro-node saltro-node-action">
             <strong>Start iLQR iteration</strong>
             <p>Set <code>reg = reg_init</code>. Telemetry starts with <code>accepted_steps = 0</code>.</p>
           </div>
         </li>
         <li>
           <div class="saltro-node saltro-node-decision">
             <strong>Does the backward pass factorize successfully?</strong>
             <p>The Riccati step requires a positive-definite regularized <code>Q_uu</code>.</p>
           </div>
           <div class="saltro-outcomes">
             <div class="saltro-outcome saltro-warn">
               <strong>No</strong>
               <p>Increase regularization by <code>reg_scale</code> and retry the same iLQR iteration.</p>
             </div>
             <div class="saltro-outcome saltro-good">
               <strong>Yes</strong>
               <p>Build feedback gains <code>K</code>, feedforward terms <code>d</code>, and expected cost terms <code>deltaV</code>.</p>
             </div>
           </div>
         </li>
         <li>
           <div class="saltro-node saltro-node-call">
             <strong>Run forward-pass line search</strong>
             <p>Try to roll out a new trajectory using <code>K</code>, <code>d</code>, and decreasing line-search step sizes.</p>
           </div>
           <div class="saltro-branch saltro-inner">
             <span class="saltro-branch-label">enter inner layer</span>
           </div>
         </li>
         <li>
           <div class="saltro-node saltro-node-decision">
             <strong>Did the forward pass accept a rollout?</strong>
           </div>
           <div class="saltro-outcomes">
             <div class="saltro-outcome saltro-warn">
               <strong>No</strong>
               <p>Increase regularization by <code>reg_scale</code> and retry from the backward pass.</p>
             </div>
             <div class="saltro-outcome saltro-good">
               <strong>Yes</strong>
               <p>Overwrite <code>X</code> and <code>U</code>. Increment <code>accepted_steps</code>, store <code>last_delta_J</code>, and store <code>final_cost</code>.</p>
             </div>
           </div>
         </li>
         <li>
           <div class="saltro-node saltro-node-decision">
             <strong>Is <code>abs(J_prev - J_new) <= cost_tol</code>?</strong>
           </div>
           <div class="saltro-outcomes">
             <div class="saltro-outcome saltro-good">
               <strong>Yes</strong>
               <p>Return <code>ILQRStatus::Converged</code> and <code>true</code>.</p>
             </div>
             <div class="saltro-outcome saltro-warn">
               <strong>No</strong>
               <p>Continue to the next iLQR iteration. The trajectory may still be useful because a forward pass was accepted.</p>
             </div>
           </div>
         </li>
         <li>
           <div class="saltro-node saltro-node-stop">
             <strong>Middle-layer terminal statuses</strong>
             <p><code>RegularizationExceeded</code> means all regularization retries failed. <code>MaxIterations</code> means the cost tolerance was not reached before the iLQR iteration budget ended.</p>
           </div>
         </li>
       </ol>
     </section>

     <section class="saltro-layer saltro-inner">
       <header>
         <span class="saltro-layer-name">Inner Layer</span>
         <h2>Forward pass and line search</h2>
         <p>Goal: find one acceptable rollout that decreases cost consistently with the expected model reduction.</p>
       </header>

       <ol class="saltro-steps">
         <li>
           <div class="saltro-node saltro-node-action">
             <strong>Try line-search step <code>alpha = 1, 1/2, 1/4, ...</code></strong>
             <p>Initialize <code>X_bar</code> and <code>U_bar</code> from the current nominal trajectory.</p>
           </div>
         </li>
         <li>
           <div class="saltro-node saltro-node-decision">
             <strong>Is the rollout dynamically valid?</strong>
             <p>Reject the trial if the state is non-finite, the timestep is invalid, dynamics throw, the next state is invalid, or quaternion normalization fails.</p>
           </div>
           <div class="saltro-outcomes">
             <div class="saltro-outcome saltro-warn">
               <strong>No</strong>
               <p>Reject this <code>alpha</code> and try the next smaller step.</p>
             </div>
             <div class="saltro-outcome saltro-good">
               <strong>Yes</strong>
               <p>Compute nominal cost plus augmented Lagrangian penalty terms.</p>
             </div>
           </div>
         </li>
         <li>
           <div class="saltro-node saltro-node-decision">
             <strong>Are augmented-constraint vector sizes consistent?</strong>
             <p>The line search requires <code>lambda_aug[k]</code>, <code>mu_aug[k]</code>, and <code>c_k</code> to have matching sizes.</p>
           </div>
           <div class="saltro-outcomes">
             <div class="saltro-outcome saltro-bad">
               <strong>No</strong>
               <p>Return forward-pass failure immediately.</p>
             </div>
             <div class="saltro-outcome saltro-good">
               <strong>Yes</strong>
               <p>Evaluate the line-search acceptance ratio.</p>
             </div>
           </div>
         </li>
         <li>
           <div class="saltro-node saltro-node-decision">
             <strong>Does the line-search ratio satisfy <code>beta1 <= z <= beta2</code>?</strong>
           </div>
           <div class="saltro-outcomes">
             <div class="saltro-outcome saltro-good">
               <strong>Yes</strong>
               <p>Accept the rollout, overwrite <code>X</code> and <code>U</code>, return forward-pass success.</p>
             </div>
             <div class="saltro-outcome saltro-warn">
               <strong>No</strong>
               <p>Reject this <code>alpha</code> and keep searching.</p>
             </div>
           </div>
         </li>
         <li>
           <div class="saltro-node saltro-node-stop">
             <strong>All line-search trials rejected</strong>
             <p>Return forward-pass failure. The middle layer will usually increase regularization and retry.</p>
           </div>
         </li>
       </ol>
     </section>
   </div>


Status Reference
----------------

.. list-table:: Where each status comes from and what to inspect
   :header-rows: 1
   :widths: 20 24 30 26

   * - Layer
     - Status or return
     - Meaning
     - First places to inspect
   * - Inner
     - ``forwardPass == true``
     - A line-search trial accepted and updated the trajectory.
     - Accepted ``alpha``, ``J_prev``, ``J_new``, ``deltaV``, constraint penalties.
   * - Inner
     - ``forwardPass == false``
     - No line-search step was accepted, or augmented constraint vector sizes mismatched.
     - State validity, dynamics exceptions, ``beta1``/``beta2``, ``deltaV``, ``lambda_aug`` and ``mu_aug`` sizes.
   * - Middle
     - ``ILQRStatus::Converged``
     - The latest accepted rollout reached ``abs(J_prev - J_new) <= cost_tol``.
     - ``last_delta_J`` and ``cost_tol``.
   * - Middle
     - ``ILQRStatus::MaxIterations``
     - iLQR accepted steps but never reached the cost tolerance before ``max_iters``.
     - ``accepted_steps``, ``last_delta_J``, ``max_iters``, cost scaling.
   * - Middle
     - ``ILQRStatus::RegularizationExceeded``
     - Backward or forward pass kept failing until ``reg > reg_max``.
     - Whether ``accepted_steps`` is zero, backward-pass factorization, line-search rejection pattern, ``reg_init``/``reg_scale``/``reg_max``.
   * - Outer
     - ``ALILQRStatus::Converged``
     - Constraints are within tolerance and the iLQR subproblem either converged or accepted at least one rollout.
     - ``max_constraint_violation``, ``accepted_steps``, final trajectory quality.
   * - Outer
     - ``ALILQRStatus::InnerFailed``
     - The inner solve failed in a way AL-iLQR cannot safely accept, or constraints were satisfied with zero accepted trajectory updates.
     - Warm-start feasibility, ``accepted_steps == 0``, first backward/forward failure reason.
   * - Outer
     - ``ALILQRStatus::MaxOuterIterations``
     - AL penalty updates did not drive constraints below ``constraint_tol`` before the outer iteration budget ended.
     - Constraint component traces, penalty scaling, active constraints, actuator limits.


Common Diagnosis Paths
----------------------

.. raw:: html

   <div class="saltro-diagnosis-grid">
     <article>
       <h3>Constraints satisfied, but iLQR did not converge</h3>
       <p>This can be acceptable if <code>accepted_steps > 0</code>. It means the trajectory was optimized, but the inner cost-delta tolerance was not the final stopping reason.</p>
     </article>
     <article>
       <h3>Constraints satisfied with zero accepted steps</h3>
       <p>Treat this as unsafe. The warm start may have been feasible, but the optimizer did not produce a new trajectory.</p>
     </article>
     <article>
       <h3>Regularization exceeded after accepted steps</h3>
       <p>The latest refinement attempt stalled, but previous accepted steps may still have produced a valid AL solution if constraints are now below tolerance.</p>
     </article>
     <article>
       <h3>Max outer iterations</h3>
       <p>The middle layer may be working, but AL penalty updates are not enforcing at least one active constraint quickly enough.</p>
     </article>
   </div>


State-Slack Relaxation and the Polish Phase
-------------------------------------------

With ``auglag.use_state_slack = true`` the outer loop runs in two phases:

1. **Slack phase.** Every STATE-family constraint (AngularVelocity,
   SunAvoidance, RWMomentum) is relaxed as ``c(x) - s <= 0`` with slack cost
   ``slack_rho * s + 0.5 * slack_sigma * s^2``. The slack is eliminated in
   closed form (``include/saltro/optimizer/al_slack.h``) and never enters the
   iLQR decision space; all four AL sites (BP stage block, BP terminal seed,
   FP merit, lambda update) see the shifted constraint ``c - s*``. The
   constraint force is therefore capped at ``slack_rho + slack_sigma * s*``
   even as ``mu`` ramps, which keeps ``Q_uu`` conditioned on hard problems
   with large unavoidable transient violations. Control-family constraints
   (torque saturation) never get slacks.
2. **Polish phase.** Once the TRUE (unslacked) max violation reaches
   ``slack_off_tol``, slacks are dropped and AL iterations continue
   warm-started from the same trajectory and multipliers.
   ``ALILQRStatus::Converged`` is only ever declared in the polish phase, so
   the usual ``constraint_tol`` guarantee is unchanged and every slack run
   ends with at least one slack-free inner solve.

.. tip::

   Diagnosis additions for slack runs:

   - **Slack phase stalls above** ``slack_off_tol``: the slack price is
     below the constraint's true multiplier, so the optimizer permanently
     "buys" the violation. Observed on underactuated MTQ-only 180-degree
     slews with tight ``wmax`` (needed multiplier is huge); raising
     ``slack_rho`` alone does NOT fix it there, because with
     ``slack_sigma = 0`` the bought region also has zero curvature. The
     stall fallback (``slack_stall_iters``, default 1) detects the
     non-contracting TRUE violation and drops the slacks automatically,
     recovering exact-AL behavior (where the slack phase genuinely helps it
     contracts >5% per outer iter, so the hair-trigger default is
     empirically bit-identical there); only with the fallback disabled must
     you raise ``slack_rho``/``slack_sigma`` or loosen ``slack_off_tol`` by
     hand.
   - **Polish phase diverges after a clean slack phase**: the slack solution
     parked the trajectory somewhere the exact penalty cannot hold (e.g. the
     absorbed violation was structural, not transient). Inspect which family
     carried nonzero slack at switch time; that family's constraint is the
     real blocker.
   - **An infeasible fixed initial knot** (e.g. ``|w0| > wmax``) keeps
     ``max_c`` pinned at the k=0 violation forever - with or without slacks -
     because the initial state is not a decision variable. Slack cannot fix
     this; relax the constraint or accept ``MaxOuterIterations`` semantics.
   - ``use_state_slack = false`` (default) is bit-identical to the unslacked
     solver, and so is an uneconomically high ``slack_rho`` (regression
     anchors: the ``[alilqr][slack]`` unit tests).
