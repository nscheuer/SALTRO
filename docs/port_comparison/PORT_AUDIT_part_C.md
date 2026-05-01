# Port Audit Part C: recent PhD C++ ↔ saltro current branch

> **HIGHEST PRIORITY** — active port drift between recent PhD (`OldPlanner.cpp`+`Satellite.cpp`) and saltro `PKMN_antispike` working tree. Agent-generated 2026-04-27.

## 1. Vec-Pointing Cost

| Source | File:Line | Formula |
|---|---|---|
| PhD | `Satellite.cpp:893-1038` | 4 cases (`whichAngCostFunc` 0-3): linear `1−c`, `½(1−c)²`, `acos(c)`, `½·acos(c)²`. W_ang linear cross `w_avang·ω·(R^T·r̂×bs)` (line 1012-1016). |
| saltro | `satellite.cpp:1072-1109` | 5 cases (0-4): cases 0-3 identical to PhD; **case 4 `(1−c)²` is NEW**. `is_eci_format` flag gates vec/quat. Synthetic `q_goal` constructed via `processAttitudeTarget()` (line 143-202). |

**Drift note:** Saltro adds case 4 (smooth bowl over `c ∈ [-1,1]`, avoids quat-mode `(1-d²)` antipodal-zero issue). saltro builds a synthetic q_goal as a *computational intermediate* — PhD computes everything directly from `angerrvec = cross(R^T·ê, ŝ)`. Saltro's synth-q enables uniform downstream code but the computation only kicks in for cross-cost (legacy path) or fallback cases.

**Impact: LOW** (case 4 is opt-in, default behavior preserved).

## 2. Quaternion-Pointing Cost

| Source | Formula |
|---|---|
| PhD | 6 cases (`Satellite.cpp:1087-1217`): includes special **case 2** `½·norm(angerrvec)²/qerr(0)²` (line 1185-1196). Double-cover via sign-check at 1156-1161. |
| saltro | 5 cases (`satellite.cpp:1111-1135`): case 2 IS implemented but as `acos(qdot_aligned)`. **PhD's case-2 special form not present.** Hemisphere flip via `qdot_aligned` (line 1065-1069). |

**Drift note:** Saltro **omits PhD's case 2 special form** (`norm²/qerr(0)²`). Saltro's case 2 is just `acos(qdot)` — not equivalent. Likely intentional removal of esoteric mode but should verify no mission depends on it. Hemisphere correction is mathematically equivalent.

**Impact: LOW** if PhD case 2 unused; **MEDIUM** if any mission relied on it.

## 3. ω-Side Cost

| Source | Formula |
|---|---|
| PhD | `Satellite.cpp:1006-1018`: `½·w_av·\|ω\|²` + linear `−w_avang·sign(qdot)·(q_g^T·W·ω)` + mag `w_avmag·\|ω·b̂\|`. |
| saltro | `satellite.cpp:1139-1222`: V2 path (NEW) `½·w_av·\|ω−ω_ff\|²` + Lyapunov `α·err_dir·(ω−ω_ff)` with `α = β·√(angle·λ_min)`. Legacy path activated when `w_avang≠0` (lines 1156-1158). **Axis-aware reduction** (vec mode) `−½·w_av·(1−roll_ratio)·(b̂·ω)²` (lines 1217-1222). |

**Drift note:** Saltro adds **3 saltro-only features**: ω_ff tracking (PhD: zero feedforward), Lyapunov α-from-β cross-coupling (PhD: raw w_avang), axis-aware ω reduction in vec-mode via `roll_ratio` (PhD: isotropic). All opt-in; legacy path with `w_avang≠0` recovers PhD form.

**Just-fixed (today):** Legacy `w_avang` path in vec mode now routes through V2 with `α=w_avang`, `ω_ff=0` — matches PhD's linear cross exactly.

**Impact: MEDIUM** (V2 features powerful but opt-in; legacy matches PhD); **HIGH if `roll_ratio<1.0`** without tuning.

## 4. Control Cost

| Source | Formula |
|---|---|
| PhD | `Satellite.cpp:1024-1028`: `useRawControlCost` flag selects `½·u_cost·(u−u_prev)²` (delta) or `½·u_cost·u²` (raw). Per-actuator weights via `diagmat(MTQ_cost)`, `diagmat(RW_cost)`, `diagmat(magic_cost)`. k=N-1 skipped. |
| saltro | `satellite.cpp:1232-1245`: **Always raw**, normalized per-actuator: `½·w_u_mult·mtq_control_weight·(u_i/u_max)²` etc. **No delta-control option.** |

**Drift note:** Saltro **removes delta-control path** (PhD `useRawControlCost=0`). Saltro **normalizes by per-actuator limits** — design improvement (scale-independent) but **breaks PhD-tuned mission params**. Missions tuned to PhD raw-control require re-tuning.

**Impact: MEDIUM** (affects actuator authority balance).

## 5. RW Angular Momentum & Stiction

| Source | Formula |
|---|---|
| PhD | Softplus-based: `½·RW_AM_cost·shifted_softplus(h, thresh)²` + smoothstep stiction. C² everywhere. |
| saltro | Piecewise-quadratic: if `h>h_thresh`, `½·rw_AM_weight·((h-h_thresh)/(h_max-h_thresh))²`; else `½·rw_AM_weight·RWh_ok_mult·(h/h_max)²`. Stiction similar piecewise. **C¹ but not C²** (kink at threshold). |

**Drift note:** Saltro replaces softplus with piecewise-quadratic. Computationally simpler, avoids softplus singularities at large h. Functional behavior similar. Sign handling at `lkx[h]` correct (`sign_h = safeSign(h)`).

**Impact: LOW** (well-behaved penalty shape change).

## 6. AL Outer Loop

| Source | Behavior |
|---|---|
| PhD | `OldPlanner.cpp:2050+`: exit on `max_constraint ≤ tol`. λ ← λ + μ·c. μ ← μ·penScale. |
| saltro | `alilqr.cpp:130-197`: **Dual-gate convergence** — exit only if (`max_c ≤ constraint_tol` AND (`inner_ok=Converged` AND `outer ≥ min_outer_iters`)) OR (`max_c ≤ constraint_tol_strict`). λ clamped to `[0, lag_mult_max]`. **PenMax exit** (line 187): break if μ saturates everywhere AND `max_c > constraint_tol`. |

**Drift note:** Saltro adds **maturity gate** (`min_outer_iters=3`, default) and `strict_path` (disabled by default). Correctness fix: PhD's single-gate could declare victory after inner `MaxIterations` bailout. The line 143-154 comment documents this PhD-alignment (2026-04-23).

**Impact: MEDIUM** (tighter constraint satisfaction; potentially longer outer loops).

## 7. Inner iLQR

| Source | Behavior |
|---|---|
| PhD | `OldPlanner.cpp:~1900-2050`: gradient-norm + line-search z-ratio + cost stagnation. Single tolerance. Uniform Tikhonov reg `Q_uu + ρI`. |
| saltro | `iLQR.cpp:220-290`: **Two-tier convergence** (`cost_tol` outer + `ilqr_cost_tol` strict inner). Stagnation count `z_count_lim=10`. **Spike removal hook** (line 211). Choice of Tikhonov OR eigen-modification. |

**Drift note:** Saltro adds two-tier tolerance, spike-removal hook (saltro-only), eigen-mod regularization path. Default uniform Tikhonov should match PhD.

**Important:** the Python alilqr/ilqr wrappers (`tests/debug/optimizer/alilqr_python/ilqr.py`) only check `cost_tol` — they don't use `ilqr_cost_tol`/`z_count_lim`/`grad_tol`. The C++ inner-iLQR has all four. **Asymmetry between debug Python and production C++.**

**Impact: MEDIUM**.

## 8. Backward Pass — Q assembly & DDP

| Source | Formula |
|---|---|
| PhD | `OldPlanner.cpp:2147-2167`: `Q_xx = lxx + A^T·P·A + (constraints)`. Optional `useDynamicsHess` flag (line 2149-2151). Manifold projection via `G_k`. |
| saltro | `backwardpass.cpp:426-431`: `Q_xx = lxx + A^T·P·A + Qxx_ddp`, `Q_uu = luu + B^T·P·B + Quu_ddp`. **DDP terms** computed via RK4 Hessian integration (300-423) with **PSD-clipping** (401-423). **Unregularized Q_uu** in Riccati (lines 50-53, 104-114) — prevents reg inflation. |

**Drift note:** Saltro adds DDP (2nd-order dynamics) Hessian integration with PSD-clipping. PhD has optional `useDynamicsHess` flag. saltro's PSD-clipping (negative eigenvalues clipped to floor) hides indefiniteness — could mask real curvature; could also stabilize. Unregularized Q_uu in Riccati is mathematically principled.

**Manifold projection:** `backwardpass.cpp:213` does `G_k * lxx_full * G_k^T` — equivalent to PhD's reduced-state Hessian. Verified.

**Impact: MEDIUM**.

## 9. Forward Pass / Line Search

| Source | Behavior |
|---|---|
| PhD | Armijo-style backtracking on cost improvement. |
| saltro | `forwardpass.cpp`: predicted-improvement ratio `z = (J_new − J_old) / (dV[0] + dV[1])`. Accept if `z > beta1` (1e-4); aggressive backtrack/reset if `z < beta2` (1e-8). Triple-bump reg on LS failure. |

**Drift note:** Saltro's z-ratio LS is theoretically aligned with iLQR; PhD's Armijo is simpler. Both converge.

**Impact: LOW**.

## 10. Constraints

Both: inequality (control sat, ω bounds, sun angle). Quadratic AL. Same dual update. Saltro's collection is modular (separate `satellite.constraints()` call per timestep), PhD embeds in cost assembly.

**Impact: LOW** (architectural difference, same math).

## 11. Warm-Start ⚠️ CRITICAL GAP

| Source | Behavior |
|---|---|
| PhD | `OldPlanner.cpp:680-790`: **smartbdot** (line 689+) MTQ control + RW desat. **QuatGain** (line 84) PD pointing. |
| saltro | `plannersettings.h:21-23`: `InitTrajConfig::initcontroller` enum stub. **Implementation incomplete.** PDController exists at `pdcontroller.cpp` but not integrated into main warm-start pipeline. **smartbdot logic NOT yet ported.** |

**Drift note:** **smartbdot warm-start NOT ported.** Saltro has a placeholder structure; actual generation logic missing. Spike removal includes a PD controller but doesn't use it for initial trajectory.

**Impact: HIGH** — incomplete feature blocks flight-readiness. Action: port smartbdot before integration.

## 12. Spike Removal / Homotopy

PhD: none. Saltro: 824 lines new in `spike_removal.cpp`. Saltro-specific. Includes mode-specific heuristics that may bake assumptions about vec-vs-quat cost structure. Recommendation: document mode assumptions explicitly.

**Impact: MEDIUM** (helpful but heuristic; risks masking optimizer issues).

## 13. Numerics & Defaults

| Param | PhD (inferred) | saltro |
|---|---|---|
| reg_init | finite (e.g., 1e-6) | **0.0** (`plannersettings.h:357`) |
| reg_max | reasonable bound | **1e30** (effectively infinite) |
| reg_scale | ~1.6-2.0 | 1.6 |
| cost_tol | strict single | **two-tier**: cost_tol=1e-1, ilqr_cost_tol=1e0 |
| min_outer_iters | (none/implicit 1) | 3 |
| z_count_lim | n/a (no stagnation count) | 10 |

**Drift note:** Saltro's `reg_init=0` is aggressive (PhD likely uses small finite damping). Two-tier tolerance allows looser inner iLQR — missions tuned to PhD strict tol may not converge as tightly in saltro. **Re-tune `ilqr_cost_tol` from flight data.**

**Impact: LOW-MEDIUM**.

## Actionable Discrepancies (Ranked)

### 1. 🔴 HIGHEST: Warm-start (smartbdot) not ported
- saltro's `InitTrajConfig::initcontroller` is a stub.
- Action: port PhD's smartbdot from `OldPlanner.cpp:680-790`. Test against PhD reference traj. Effort: medium (~50-100 lines + tests).

### 2. 🟡 HIGH: Spike-removal mode assumptions undocumented
- Mode-specific logic at `spike_removal.cpp:713`, `:755` may bake vec/quat assumptions.
- Action: document assumptions explicitly. Add config flags to disable per-mode.
- Effort: low.

### 3. 🟡 MEDIUM: Control-cost normalization differs from PhD
- saltro normalizes by `u_max`; PhD uses raw weights. Missions tuned to PhD will have different actuator authority balance.
- Action: extract PhD weights, map via `weight_new = weight_old * u_max_old / u_max_new`.
- Effort: low.

### 4. 🟡 MEDIUM: iLQR two-tier tolerance looser than PhD
- saltro `ilqr_cost_tol=1e0` allows sub-converged inner exit.
- Action: A/B against PhD; tighten to 1e-1 if outer loop count diverges.
- Effort: medium.

### 5. 🟡 MEDIUM: DDP Hessian integration & PSD-clipping untested
- saltro's RK4-Hessian integration + PSD-clipping not in PhD reference.
- Action: numerical FD verification of dynamics Hessians; test PSD-clip behavior.
- Effort: medium.

### 6. 🟢 LOW: Quaternion-cost case 2 differs
- saltro's case 2 is `acos(qdot)`; PhD's is special form `½·norm²/qerr(0)²`.
- Action: query PhD missions for case 2 usage; document if dead code.
- Effort: low.

### 7. 🟢 LOW: Vec-cost case 4 is saltro extension
- New `(1-c)²` shape; not in PhD. **Today's testing shows this is the BEST shape** for vec-mode convergence (PE_fin=4.67° vs type-3's 9.5°).
- Action: validate further; consider making default for vec mode.

### 8. 🟢 LOW: Axis-aware ω reduction (`roll_ratio<1.0`) untested
- Action: keep `roll_ratio=1.0` default. Document parameter.
- Effort: low.

## Summary

saltro's port is **~90% complete** with mathematically sound extensions (DDP, two-tier tol, axis-aware ω, PSD-clipping, vec-mode case 4).

**Critical gaps:**
1. **Warm-start (smartbdot) not ported** — blocks flight.
2. **Control-cost normalization differs** — needs mission re-tuning.
3. **Spike removal assumptions undocumented** — risks hidden failures.

**Convergence risk:** **MEDIUM**. Missions will converge but may require outer-loop tuning and warm-start completion.
