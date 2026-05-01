# PSD-clip survey — where the principle could apply

> Survey of every place we could PSD-clip an indefinite second-order term in saltro, with by-construction alternatives where they exist. Drafted 2026-04-28 to support investigating each separately.
>
> **Related work:** [memory project_ddp_status](../../../.claude/projects/c--Users-LV---Patrick-McKeen-saltro/memory/project_ddp_status.md) — the DDP indefinite-Quu fix is item #4 in this survey. The diagnostic infrastructure landed for DDP (SALTRO_DDP_DIAGNOSE, `backward_pass(return_quu=True)`) extends naturally to cost-side eigenvalue tracing.

## Background: two flavors of "make it PSD"

1. **PSD-clip via eigendecomp**: compute eigenvalues, project negative ones to zero (or floor). Generic, works on any symmetric matrix. Cost: O(n³) per clip; for our blocks (4×4 cost-Hess, 6×6 reduced-state), this is microseconds.
2. **PSD by construction**: rewrite the math so the term is structurally PSD without runtime checking. Cleaner, faster, but only works for terms that *can* be reformulated. The classic example: cost as `½·r²` for a vector residual `r` — Gauss-Newton Hessian `J_r^T·J_r` is automatically PSD, no eigendecomp needed.

By-construction is preferred where possible because it's cheaper, semantically cleaner, and doesn't hide model errors behind a clamp.

---

## Where indefinite second-order terms appear

### 1. Cost angle (q,q) Hessian — CHAIN TERM

**Term**: `f'(c) · d²c/dq²` in `lxx[q,q]` for vec mode.
- `f'(c)`: scalar, sign depends on cost shape and `c`.
- `d²c/dq²`: 4×4 tensor, no sign guarantee (degree-2 in q makes this unstructured).
- Combined: indefinite in general.

**Currently**: dropped via `cost_hess_gauss_newton` flag. Today's main fix.

**By-construction option**: YES.
For cost shapes of the form `½·w·r(q)²` (cases 1, 3, 4 are all of this form with different `r`):
```
case 1: r = (1-c),         J_r = -dc/dq         → GN Hess = (dc/dq)(dc/dq)^T
case 3: r = phi = arccos(c), J_r = -dc/dq/√(1-c²) → GN Hess = (dc/dq)(dc/dq)^T / (1-c²)
case 4: r = (1-c),         J_r = -dc/dq         → GN Hess = (dc/dq)(dc/dq)^T (×2 from f''=2)
```
All rank-1 outer products, structurally PSD. No eigendecomp needed.

The current `cost_hess_gauss_newton=True` flag computes `f''(c)·(dc/dq)(dc/dq)^T` which agrees with proper Gauss-Newton for cases 1 and 4 but differs for case 3 by `phi·c/((1-c²)^1.5)` (since `f''(c) = 1/(1-c²) − phi·c/((1-c²)^1.5)` for case 3, and proper GN is `1/(1-c²)`). Case 3's `f''` includes an indefinite scalar even in the outer-product term. So technically:

**Action for PR 3 (later)**: Replace the `cost_hess_gauss_newton` flag's behavior with proper Gauss-Newton-by-construction: `J_r^T · w · J_r` where `r = arccos(c)` for case 3, `r = (1-c)` for cases 1/4. PSD by construction, no clipping. Slightly different math from current GN flag for case 3 only.

**PwA correction**: `−grad_dot_q · I_4` is sign-uncertain too (depends on `f'·c`). Currently kept because it's PSD in aligned hemisphere. By-construction alternative: project to tangent plane explicitly (working in 3D reduced state) — eliminates the I_4 ambiguity. PhD does this naturally; we've been adding it as a correction.

**Convergence-impact estimate**: HIGH (already validated in today's sweep).

### 2. Cost (q,ω) cross-Hessian

**Term**: `α · ∂err_dir/∂q` (when crossterm active).
- `∂err_dir/∂q = -S · J_rhat^T`: 3×4 matrix, sign-uncertain.
- This is the OFF-DIAGONAL cross block. Symmetric-Hessian property requires it to appear in `lxx[q,ω]` and its transpose in `lxx[ω,q]`.

The full block-symmetric quadratic on `[ω; q]` has eigenvalues that include cross-block contributions. If `(ω,ω)` block is `w_av·I_3` (PSD) and `(q,q)` block is the angle-cost Hess, then for the joint to be PSD we need a Schur-complement condition involving the cross block.

**Currently**: cross block is computed and added to `lxx`. Not clipped.

**By-construction option**: PARTIAL.
The cross-cost `α · err_dir(q) · ω` is bilinear in (q, ω) when err_dir is independent of ω (which it is). The Hessian is structurally:
```
[ 0                    α · ∂err_dir/∂q ]
[ α · (∂err_dir/∂q)^T            0     ]
```
This is **always indefinite** (has eigenvalues ±‖α·∂err_dir/∂q‖_F when the diagonals are zero). No way to make a non-zero cross block PSD without changing the cost.

**PSD-clip option**: project the joint `[ω; q]` block to nearest PSD via eigendecomp. Could matter when the cross dominates the on-diagonal contributions. Worth profiling if BP-step direction issues emerge.

**Convergence-impact estimate**: LOW-MEDIUM. Cross terms are usually small relative to diagonals when `α` is well-tuned. Only matters in pathological regimes.

### 3. Cost ω quadratic `(ω,ω)`

**Term**: `0.5 · w_av · ‖ω‖²` and (vec-mode) `−0.5·w_av·(1−roll)·(bs·ω)²`.
- First: `w_av · I_3` — PSD with `w_av ≥ 0`.
- Second: `−w_av·(1−roll)·bs·bs^T` — rank-1 negative.
- Sum: `w_av · (roll·bs·bs^T + (I − bs·bs^T))` — PSD when `roll ∈ [0, 1]`.

**Currently**: by-construction PSD (we worked this out when adding the axis-aware reduction).

**Action**: nothing needed; by-construction PSD already.

### 4. DDP dynamics terms (`Qxx_ddp`, `Quu_ddp`, `Qux_ddp`)

**Term**: `Σ_l V_x[l] · F_xx[l, ⋅, ⋅]` etc., where `F_xx` is RK4-integrated dynamics Hessian.
- `V_x[l]`: scalar component of value gradient, sign-uncertain.
- `F_xx[l, ⋅, ⋅]`: 7×7 (or reduced 6×6) tensor slice, no sign guarantee.
- Combined: indefinite in general.

**Currently**: PSD-clipped via `psd_clip_quu_ddp` and `psd_clip_qxx_ddp` (per memory `project_ddp_status.md`). Eigendecomp on the 6×6 (or 3×3 for Q_uu) block.

**By-construction option**: **probably NO** for this generality. The dynamics nonlinearity is genuinely arbitrary — gyroscopic terms, MTQ/B-field cross products, reaction-wheel coupling — there's no clean "residual squared" decomposition. The closest analog is iLQR-without-DDP, which is exactly Gauss-Newton on the dynamics (drop the contraction with `V_x` entirely).

**Selective inclusion** (the other analyst's idea): instead of all-or-nothing or eigen-clip, include only the terms with known good structure:
- `V_x[ω] · ∂²f_ω/∂x²`: rigid-body gyroscopic Hessian. Closed form `[p̃]× J − J [p̃]×` (skew-symmetric). FD-verifiable in isolation. Could include without clipping.
- `V_x[q] · ∂²f_q/∂x²`: quaternion-output dynamics Hessian. More complex; likely the source of indefiniteness. Either clip or skip.
- `V_x[h] · ∂²f_h/∂x²`: RW dynamics linear → identically zero. Skip.

**Action for separate investigation**: see `project_ddp_status.md`. We have the PSD-clip; could compare against selective inclusion.

**Convergence-impact estimate**: HIGH (already validated; iLQR-default is the saltro choice, DDP off by default).

### 5. AL constraint Hessian

**Term**: AL contributes to `lxx`:
```
∂²(½μc²)/∂x² + ∂²(λ·c)/∂x²
  = μ · cx·cx^T + (λ + μ·c) · cxx
```
- First: rank-1 outer product, `μ ≥ 0` → PSD.
- Second: `cxx` (constraint Hessian) is sign-uncertain in general; coefficient `(λ + μ·c)` is sign-uncertain (`λ ≥ 0` for inequality, `c` either sign, `μ ≥ 0`).
- Combined: second term can be indefinite.

**Currently**: not PSD-clipped. The `use_constraint_hess` flag exists in `RegularizationConfig` (per pybind binding) but I don't know if it's wired in or what the default is.

**By-construction option**: YES for many constraint types.
- Linear constraints (`c = a^T x − b`): `cxx = 0`, no second-order term to worry about.
- Quadratic constraints: `cxx = const`, sign known a priori. PSD-clip at compile time if needed.
- General nonlinear: can write `c = r(x)` and use Gauss-Newton form `J_r^T · J_r` if cost-as-LSQ structure exists. For our typical constraints (control bounds, ω limits, sun angle), the Gauss-Newton form usually applies.

**Action for separate investigation**: audit our constraint set. If all are linear or simple quadratic, no work needed. If any are nonlinear, GN-by-construction or PSD-clip.

**Convergence-impact estimate**: LOW for current constraints (mostly linear); HIGH if we add nonlinear constraints later (sun-pointing, RW soft limits with smooth penalties, etc.).

### 6. PwA manifold correction

**Term**: `−grad_dot_q · I_4` added to `lxx[q,q]` (vec mode and quat mode).
- `grad_dot_q = f'(c) · 2c` (from Euler degree-2 homogeneity).
- Sign of `grad_dot_q`: depends on `f'·c`. PSD when `f'·c < 0` (aligned hemisphere); indefinite when `f'·c > 0` (off-target).

**Currently**: always added.

**By-construction option**: YES.
The PwA correction exists because we're working in 4D ambient with a unit-norm constraint. Working directly in 3D reduced state (PhD's approach via `findGMat`) eliminates the need for this correction — the manifold curvature is built into the parameterization.

**Action**: see PORT_AUDIT recommendation #1 ("smartbdot warm-start"... wait, different item). The reparameterization is mentioned as a deeper fix in the regression note; would require restructuring stageCost*/Jacobians*/Hessians* to operate in 3D throughout. Significant refactor.

**Convergence-impact estimate**: MEDIUM. Today's GN flag works around the indefiniteness for cases where it matters; full reparameterization would be a cleaner long-term solution.

### 7. State-mag cost Hessian (`w_avmag`)

**Term**: would be `Hess of (w_avmag · |ω · b_body|)` w.r.t. (ω, q).
- Currently NOT computed (gradient yes, Hess gap documented in today's commit).
- The `|·|` makes it non-smooth → Hess is technically a distribution (delta function on the sign-flip).

**Currently**: gap. `(void)w_avmag;` in stageCostHessians.

**By-construction option**: depends on smoothing. If we replace `|x|` with smooth approximation `√(x² + ε)`, Hess is well-defined and structurally PSD-ish.

**Action**: separate investigation (low priority; default `ang_vel_mag = 0` so this is dormant).

---

## Summary table

| # | Location | Status | By-construction PSD? | Investigation cost |
|---|---|---|---|---|
| 1 | Cost (q,q) chain | DROPPED via GN flag | YES (Gauss-Newton on residual) | LOW (refine current flag) |
| 2 | Cost (q,ω) cross | added, not clipped | NO (cross is intrinsically indefinite) | LOW (profile if needed) |
| 3 | Cost (ω,ω) | by-construction PSD | YES (already done) | NONE |
| 4 | DDP dynamics | PSD-clipped (eigendecomp) | NO; selective inclusion partial | HIGH (real refactor) |
| 5 | AL constraint Hess | not currently clipped | YES for linear/quadratic constraints | MEDIUM (audit constraint set) |
| 6 | PwA correction | always added | YES via 3D reduced reparameterization | HIGH (full refactor) |
| 7 | `w_avmag` Hess | not implemented | YES with smoothing | LOW (dormant) |

## Recommended order of investigation

1. **By-construction Gauss-Newton for cost (#1)** — refine the GN flag to use proper `J_r^T·J_r` form. Mostly affects case 3. Small incremental change. Validates the by-construction approach end-to-end.
2. **Cross-block Schur-complement profiling (#2)** — quick check whether the (ω,q) cross is dominating in any failing scenarios. Likely no-op.
3. **AL constraint Hess audit (#5)** — survey our constraint set, confirm by-construction PSD applies, document.
4. **DDP selective inclusion (#4)** — only if we want to revisit DDP. Lower priority since iLQR is default.
5. **3D reparameterization (#6)** — deeper fix, biggest payoff for vec mode but biggest investment.
6. **`w_avmag` (#7)** — defer until ever activated.

The most surgical first step is **(1)**: replace the current "drop chain term" Gauss-Newton with "proper Jacobian-only Gauss-Newton on the residual." Eliminates the only place in the cost Hessian where the user-facing flag does an ad-hoc subtraction; replaces it with the standard NLS form.
