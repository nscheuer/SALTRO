# Cost function refactor — ω tracking + axis-aware W + PSD-bounded crossterm

Design doc for the `c.ang_vel` stage-cost refactor, replacing uniform `0.5·c.ang_vel·|ω|²` with a state-dependent reference tracking + axis-aware + Lyapunov-crossterm formulation.

## Motivation

Current cost `0.5·c.ang_vel·|ω|²` penalizes any angular velocity uniformly. Three problems this creates:

1. **Moving targets (nadir, ground-tracking).** Required tracking ω is nonzero — you need to rotate at a specific rate to keep pointing. Uniform `|ω|²` cost pulls toward ω=0 which is wrong.
2. **Vector pointing (2-DOF).** Roll (rotation about pointing axis) is unconstrained, but uniform `|ω|²` penalizes it, wasting control authority on something no one cares about.
3. **No feedforward guidance.** Uniform `|ω|²` gives no directional information about "this rotation reduces error." The optimizer has to discover descent directions from scratch each iteration.

The refactor addresses all three via three additions:

- **ω_ff tracking** (state-dependent reference rotation rate)
- **Axis-aware W_ω** (small roll cost, full off-axis cost) — only for vector pointing
- **Lyapunov crossterm** (first-order reward for error-reducing ω) — subsumes the current `ang_vel_err_dir` crossterm with PSD-guaranteed scaling

## Design summary

### Stage cost (replaces current ω-related terms)

```
cost_ω_k = ½ · (ω_k − ω_ff_k)ᵀ · W_ω_k · (ω_k − ω_ff_k)
         + α_k · err_dir_k^T · (ω_k − ω_ff_k)
```

Where:
- `ω_ff_k`: feedforward tracking rate, state-dependent (computed with `q_ref = current q`)
- `W_ω_k`: axis-aware weight matrix (identity for quaternion goals, diag-like for vector goals)
- `α_k · err_dir_k^T · (ω − ω_ff)`: linear Lyapunov descent reward ("reward error-reducing motion")
- `α_k = β · √(c.angle · λ_min(W_ω_k))` where β ∈ [0, 1) is the new tuning knob (replaces `ang_vel_err_dir` as a direct weight)

### Quaternion goal mode (`attitude_target[k]` is a quaternion)

```
# ω_ff from q_ref trajectory (finite-diff)
ω_ref_body_of_ref_k = finite_diff(attitude_target, k)        # in body-of-goal frame
q_e_k  = quatError(attitude_target[k], q_k)                   # shortest-path enforced
ω_ff_k = R(q_e_k)^T · ω_ref_body_of_ref_k                     # transport to current body

# No axis-aware weighting (all 3 DOF constrained)
W_ω_k  = c.ang_vel · I                                        # uniform
λ_min  = c.ang_vel

# Lyapunov-derived crossterm direction
err_dir_k = q_e_k.tail<3>()                                   # q_e,v, scales as sin(θ/2)
α_k       = β · √(c.angle · c.ang_vel)
```

### Vector-pointing mode (`attitude_target[k][0]` is NaN, last 3 elements are r̂_eci)

```
# ω_ff: tracking rate to keep bs_body pointing at r̂_eci as target moves
ṙ̂_eci_k = finite_diff(attitude_target[1:4], k)               # in ECI
ω_ff_k   = bs_body × (R(q_k)^T · ṙ̂_eci_k)                     # body frame, perp to bs

# Axis-aware weighting (roll is free DOF)
W_ω_k  = c.ang_vel · (α_roll · bs·bs^T + (I − bs·bs^T))
       = c.ang_vel · ω − c.ang_vel · (1 − α_roll) · (bs^T ω) · bs    (when applied to ω)
λ_min  = c.ang_vel · α_roll   (eigenvalue along bs)

# Lyapunov-derived crossterm direction (already perpendicular to bs — no roll component)
err_dir_k = R(q_k)^T · r̂_eci_k × bs_body                      # tangent descent, scales as sin(θ)
α_k       = β · √(c.angle · c.ang_vel · α_roll)
```

## Configuration fields

New:
- `CostConfig::ang_vel_roll_ratio` (default 0.05) — roll-axis weight fraction for vector pointing
- `CostConfig::ang_vel_err_dir_ratio` (default 0.3) — PSD-fraction knob for crossterm strength, replaces `ang_vel_err_dir` as user-facing scale

Keep:
- `CostConfig::ang_vel` — overall ω cost scale
- `CostConfig::angle` — attitude cost scale
- `CostConfig::ang_vel_err_dir` — **deprecate**. Keep for backward compat; if user sets it, override the β-derived α with raw value. Issue a warning.

## Finite-diff convention

Compute derivatives of `attitude_target` internally in saltro:

```
Interior knots (k ∈ [1, N-2]):
    diff[k] = (attitude_target[k+1] − attitude_target[k−1]) / (2·dt)
Boundary k=0:
    diff[0] = (attitude_target[1] − attitude_target[0]) / dt
Boundary k=N-1:
    diff[N-1] = (attitude_target[N-1] − attitude_target[N-2]) / dt
```

For quaternion mode: finite-diff gives `q̇_ref`, from which `ω_ref_body_of_ref = 2 · quatMult(quatConj(q_ref), q̇_ref)_vec`. This is ω in body-of-ref frame.

For vector mode: finite-diff gives `ṙ̂` directly in ECI frame.

## PSD guarantee

The stage-cost block-quadratic in `[q_e,v; ω_e]` (or `[err_dir; ω_e]`) is:

```
[ c.angle · I    α · I       ]
[ α · I          W_ω         ]
```

Schur complement gives PSD iff `W_ω ≻ α² / c.angle · I`. Since `W_ω ≻ λ_min(W_ω) · I`, sufficient condition: `α² ≤ c.angle · λ_min(W_ω)`. Our parameterization `α = β · √(c.angle · λ_min(W_ω))` with β ∈ [0, 1) guarantees it.

## Jacobian + Hessian additions

For implementation in `stageCostJacobians` / `stageCostHessians`:

### Quaternion mode

```
Let δq = q, q_ref fixed per-knot → q_e(q)
Let ω_ff(q) = R(q_e(q))^T · ω_ref_ref

∂ω_ff/∂q       = ∂R(q_e)^T/∂q · ω_ref_ref                 # via drotmatTvecdq of q_e
∂²ω_ff/∂q²     = ∂²R(q_e)^T/∂q² · ω_ref_ref               # via ddrotmatTvecdqdq of q_e
∂err_dir/∂q    = ∂q_e,v/∂q                                # G-like projection
```

### Vector mode

```
Let v_body(q) = R(q)^T · r̂_eci
∂v_body/∂q     = ∂R(q)^T/∂q · r̂_eci       (via drotmatTvecdq, already implemented)
∂²v_body/∂q²   = ∂²R(q)^T/∂q² · r̂_eci     (via ddrotmatTvecdqdq, already implemented + Bug-3 fixed)

ω_ff = bs × v_body  — linear in v_body, so:
    ∂ω_ff/∂q    = skew(bs) · ∂v_body/∂q
    ∂²ω_ff/∂q²  = skew(bs) · ∂²v_body/∂q²

err_dir = v_body × bs = −skew(bs) · v_body  — same structure
```

All infrastructure exists. Bug 3 + Bug 4 fixes ensured `drotmatTvecdq` and `ddrotmatTvecdqdq` are correct.

## Implementation plan

1. **Add config fields** to `CostConfig` with defaults; pybind bindings.
2. **Finite-diff helper** for `attitude_target` → `ω_ref` (or `ṙ̂`). Precompute once per outer iteration.
3. **Refactor `stageCost`** for the ω-related terms:
   - Branch on quaternion vs vector mode based on `isnan(attitude_target[k][0])`.
   - Compute `ω_ff_k`, `W_ω_k`, `err_dir_k`, `α_k`.
   - Return `0.5·(ω-ω_ff)^T·W·(ω-ω_ff) + α·err_dir^T·(ω-ω_ff)`.
4. **Refactor `stageCostJacobians`** — add gradients w.r.t. ω AND q (since ω_ff depends on q).
5. **Refactor `stageCostHessians`** — add Hessians (2nd derivatives). Use `ddrotmatTvecdqdq` for R-derived terms.
6. **Unit tests**:
   - FD check of gradients and Hessians for both modes.
   - PSD check on block-quadratic `[[c.angle·I, α·I], [α·I, W_ω]]`.
   - Limit check: β=0 reduces to pure velocity tracking. α_roll=1 reduces to uniform W.
   - `attitude_target` constant → ω_ff = 0 → old behavior.
7. **Backward compat**: if user sets `c.ang_vel_err_dir` directly (the old field), use it as raw α instead of β-derived. Warn.

## Risks / open questions

- **State-dependent Hessians add cost.** Each stage cost Hessian evaluation now includes `ddrotmatTvecdqdq(q, r̂)` — an O(nx²) operation per knot. Current Hessians are already O(nx²) per knot, so this is same order but more constant cost.
- **Quaternion finite-diff at boundaries.** Forward/backward diff at k=0, k=N-1 loses accuracy. May want to use extrapolation or treat those specially.
- **Vector mode at θ=π (antipodal).** `err_dir = R^T r̂ × bs = 0` when R^T r̂ = −bs. Lyapunov gradient vanishes — a saddle. Measure-zero in practice; noise breaks symmetry.
- **Roll-axis specification for quaternion mode.** We don't use it (full 3 DOF constrained), but the `ang_vel_roll_ratio` field exists. Document as "vector-mode only; ignored in quaternion mode."

## Effort estimate

- Config + pybind: 1 hour
- Finite-diff helper: 1 hour
- `stageCost` refactor: 2-3 hours
- `stageCostJacobians` refactor: 3-4 hours (state-dependent ω_ff gradients)
- `stageCostHessians` refactor: 4-6 hours (trickiest; both branches)
- Unit tests: 4-6 hours (thorough FD checks)
- Integration test on a known scenario: 2 hours

Total: **~2-3 days** for a clean, well-tested implementation.

## Order of work

1. Get reg_clamp + reproducibility diagnosis done first (understand baseline variance before changing cost function).
2. Design review of this doc.
3. Start implementation in small commits:
   a. Config fields + stubs.
   b. Vector mode first (simpler, covers the main use case).
   c. Quaternion mode second.
   d. Unit tests throughout.
4. MTQ-only A/B on the new formulation.
5. Wide A/B to integrate with other principled defaults.
