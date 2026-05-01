# Saltro Port Audit — Cross-Source Comparison

> **Scope:** 5-source comparison of the trajectory-optimizer port — MATLAB → early C++/Py → recent PhD C++ → saltro current branch (`PKMN_antispike`) → saltro main. Synthesized 2026-04-27 from four parallel agent passes (Parts A-D).
>
> **Detail in:** [Part A](PORT_AUDIT_part_A.md) (MATLAB ↔ early C++) · [Part B](PORT_AUDIT_part_B.md) (early C++ ↔ recent PhD) · [Part C](PORT_AUDIT_part_C.md) (recent PhD ↔ saltro current) · [Part D](PORT_AUDIT_part_D.md) (saltro current ↔ main).

## TL;DR

The port has been an evolving line of code through ≥4 generations. Each generation added correctness or features, but also introduced drift. **No single source is the "ground truth"** — MATLAB has cost-shape options the C++ ports never had; early C++ uses Gauss-Newton Hessian where recent PhD uses full Hessian; saltro has DDP machinery, axis-aware ω cost, and case-4 angle shape that don't exist anywhere upstream.

**Most actionable findings (ranked by impact on convergence/flight-readiness):**

1. **🔴 smartbdot warm-start not ported** (saltro vs PhD) — flight-readiness blocker.
2. **🟡 Cost-shape options dropped between MATLAB and early C++** (`useAngSquared`, `useExpAngle`).
3. **🟡 Cost Hessian inconsistency** — early C++ Gauss-Newton, recent PhD full Hessian, saltro full Hessian + PwA.
4. **🟡 Control-cost normalization changed** between PhD and saltro (raw → per-actuator-normalized).
5. **🟡 Spike removal assumptions undocumented** (saltro extension; mode-specific heuristics).
6. **🟢 Vec-mode case 4 in saltro** — newly empirically validated (PE_fin = 4.67° on 00_baseline, best-converging vec shape we have).

**Convergence risk for current branch shipping today:** MEDIUM. Missions will likely converge, but require warm-start completion and parameter re-tuning.

## Source-by-source genealogy

```
   MATLAB (PhD original, ALiLQR_BC_PM.m)
       │   useAngSquared, useExpAngle cost-shape options
       │   penInit=100, penMax=1e18, penScale=20
       │   su=500 (control weight)
       ▼
   early C++/Py  (altro_general.cpp, ~2018-2020 era)
       │   Lost: useAngSquared/useExpAngle cost shapes
       │   Reg: shrink-on-success added (line 1959)
       │   Numerics: su=1, penMax=1e8, swslew=0
       │   Cost Hess: Gauss-Newton (dphi^T Q dphi only)
       ▼
   recent PhD C++  (OldPlanner.cpp + Satellite.cpp ~2023-2024)
       │   Restructured: cost (Satellite.cpp) ⟂ planner (OldPlanner.cpp)
       │   ADDED: full second-order cost Hessian (case 3: dphi*dphi^T + ddphi*phi)
       │   ADDED: explicit state normalization, 3-Jac projection chain
       │   ADDED: case 3 (½·phi²) for vec mode
       │   ADDED: more constraints (RW angmom, sun-pointing, ω limits)
       ▼
   saltro main  (current production branch)
       │   Mostly aligned with recent PhD
       ▼
   saltro current branch (PKMN_antispike)  ← in-flight
           ADDED: vec-mode case 4 ((1-c)²)
           ADDED: V2 ω cost (ω_ff tracking, α=β·sqrt cross)
           ADDED: axis-aware ω reduction (roll_ratio)
           ADDED: DDP Hessians + PSD-clip
           ADDED: spike removal (824 lines, saltro-only)
           ADDED: PD controller (saltro-only, for spike sub)
           FIXED: RK4 chain rule, quaternion norm Hessian
           FIXED (today): legacy w_avang vec-mode routing
           NOT YET: smartbdot warm-start port
           NOT YET: PhD's control-cost-as-delta-mode option
```

## Subsystem-level synthesis

### Vec-pointing cost (4-DOF flow)

| Stage | Status |
|---|---|
| MATLAB | 4 cost shapes (linear, quadratic, exp). |
| early C++ | **Lost** quadratic and exp shapes. Cases 0-2 only. |
| recent PhD | Cases 0-3 (added case 3 = `½·phi²`). Full Hessian. State normalization. |
| saltro main | Aligned with recent PhD. |
| saltro current | Cases 0-4. Case 4 (`(1-c)²`) is saltro extension; **today's testing showed it's the best vec-mode shape**, achieving PE_fin = 4.67° on 00_baseline (vs 9.5° with type-3). |
| **saltro current (today's fix)** | Legacy `w_avang` vec-mode bug fixed: was using synthetic q_goal (3-DOF mismatch); now routes through PhD-style 2-DOF linear cross. |

**Open issue:** Vec-mode `ang_cost_func_type=2` (`f(c)=acos(c)`) diverges at c→1 because `f'(c)=-1/√(1-c²)` is singular. PhD uses different `phi=arccos(2·mag²-1)` parameterization. Type-2 should be flagged unsupported in vec mode pending reparameterization.

### Cost Hessian representation

| Stage | Hessian form |
|---|---|
| MATLAB | Standard analytic (full) |
| early C++ | **Gauss-Newton: `dphi^T·Q·dphi` only** (line 1215) |
| recent PhD | **Full: `dphi*dphi^T + ddphi*phi`** (case 3, line 993). Adds `ddphi` (second-order angle curvature). |
| saltro current | Full Hessian + PwA manifold correction `−grad_dot_q · I_4`. Projects to reduced 3D via `G_k * lxx_full * G_k^T` (`backwardpass.cpp:213`). Mathematically equivalent to PhD reduced-state form. |

**Drift impact:** Gauss-Newton vs full Hessian causes 20-50% iteration count difference. Saltro's full Hessian + PwA is correctly equivalent to PhD's direct-3D approach (verified earlier today via `backwardpass.cpp:213`).

### ω-side cost & cross-coupling

| Stage | Form |
|---|---|
| MATLAB | `0.5·w_av·\|ω\|²` + linear `w_avang·ω·n̂`. No ω·B field. |
| early C++ | Same as MATLAB. |
| recent PhD | Same: linear cross `−w_avang·sign(qdot)·(q_g^T·W·ω)`. |
| saltro main | Aligned with recent PhD. |
| saltro current | **V2: `½·w_av·\|ω-ω_ff\|² + α·err_dir·(ω-ω_ff)`** (Lyapunov, PSD-bounded). **Axis-aware reduction** in vec mode via `roll_ratio`. **Legacy w_avang path** activatable for back-compat — **today's fix routes legacy w_avang through V2 with α=w_avang, ω_ff=0 (PhD form exact)**. |

**Saltro extensions (intentional):** ω_ff tracking, axis-aware reduction. Off by default; opt-in via `ang_vel_err_dir_ratio>0` and `ang_vel_roll_ratio<1`.

### Reg & tolerances

| Stage | Reg policy |
|---|---|
| MATLAB | Grow only (drho ↑ on FP fail). penMax=1e18, penScale=20. costTol=1e-4. |
| early C++ | **Sign-flipped**: shrink drho on success (line 1959). penMax=1e8, penScale=10 (GITHUB variant). Adds **conditioning restart** at `cond(Q_uu)>50`. |
| recent PhD | Standard backward pass with reg growth. |
| saltro current | reg_init=1e-12 (very small), reg_max=1e30, reg_scale=1.6. Two-tier cost tol (`cost_tol`, `ilqr_cost_tol`). Stagnation count `z_count_lim=10`. Min_outer_iters=3. **Maturity gate** + strict_path. |

**Drift impact:** MEDIUM. Saltro's two-tier tol is looser than PhD strict; missions tuned to PhD may not converge as tightly. Re-tune from flight data. Note: Python `ilqr.py` debug wrapper only uses `cost_tol` — asymmetry from production C++.

### Backward pass DDP

| Stage | DDP terms |
|---|---|
| MATLAB / early C++ / recent PhD | None (standard iLQR). PhD has optional `useDynamicsHess` flag. |
| saltro current | **Full DDP** (Qxx_ddp, Quu_ddp, Qux_ddp from RK4 Hessian integration). **PSD-clipping** of indefinite blocks (`backwardpass.cpp:401-423`). Off by default. |

**Saltro extension (intentional).** PSD-clip is a stabilization mechanism for indefinite DDP terms. Hides indefiniteness — could mask real curvature; could also stabilize. Untested numerically against FD on dynamics Hessians.

### Warm-start ⚠️

| Stage | Approach |
|---|---|
| MATLAB / PhD | **smartbdot** (MTQ magnetic alignment) + **QuatGain** (PD). Critical pre-flight init. |
| early C++ | Same as MATLAB. |
| saltro current | **Stub:** `InitTrajConfig::initcontroller` enum placeholder. **smartbdot NOT ported.** PD controller exists in saltro (`pdcontroller.cpp`) but isn't wired to warm-start pipeline. |

**Status: BLOCKER for flight-readiness.** Missions will fall back to zero-input or PD rollout, suboptimal compared to smartbdot. **Action: port `OldPlanner.cpp:680-790`.**

### Spike removal & PD substitution (saltro-only)

824 lines of new C++ (`spike_removal.cpp`) + 143 lines of PD controller. Saltro-specific addition with no upstream reference. Multiple failed iterations (winding-number detector reverted). Mean-PE guard added as band-aid for false positives. Heuristic thresholds may not generalize.

**Impact: MEDIUM**. Useful in real-world control to smooth oscillations, but the mode-specific logic (MTQ vs RW vs vec-vs-quat) bakes in assumptions that need explicit documentation.

### Numerical defaults (most-drifted parameters)

| Param | MATLAB | early C++ | PhD recent | saltro current |
|---|---|---|---|---|
| `su` (control weight) | 500 | 1 | 1 (parameterized) | per-actuator normalized (`/u_max`) |
| `swslew` | 1e-6 | 0 | parameterized | parameterized |
| `penMax` | 1e18 | 1e8 | parameterized | `penalty_max` (parameterized) |
| `penScale` | 20 | 10 | parameterized | `penalty_scale` (parameterized) |
| `cost_tol` | 1e-4 | 1e-4 | single | two-tier: `cost_tol=1e-1`, `ilqr_cost_tol=1e0` |
| `reg_init` | 0 | 0 | parameterized | **1e-12** (very small) |
| Maturity | n/a | n/a | n/a | `min_outer_iters=3` |

**Action:** mission tuning needs to be ported, not assumed-equivalent.

## Cross-source actionable list (ranked)

### 🔴 HIGHEST priority

**1. Port smartbdot warm-start** (PhD `OldPlanner.cpp:680-790` → saltro). Flight-readiness blocker. Effort: medium (~50-100 lines + tests).

### 🟡 HIGH priority

**2. Cost-shape options absent from C++ ports** — MATLAB had `useAngSquared` and `useExpAngle`; lost in early C++ and never re-added. May not matter for current cases 0-4 if shapes overlap functionally, but worth confirming. Effort: low (verify) / medium (re-add).

**3. Document spike-removal mode assumptions** — `spike_removal.cpp:713`/`:755` baked-in vec/quat heuristics. Add config flags. Effort: low.

**4. Re-tune control-cost weights** for saltro's per-actuator normalization. PhD-tuned weights need mapping `weight_new = weight_old · u_max_old/u_max_new`. Effort: low.

**5. Legacy w_avang vec-mode bug — FIXED today.** `stageCost`, `stageCostJacobians`, `stageCostHessians` now route legacy `w_avang≠0` in vec mode through new path with `α=w_avang`, `ω_ff=0` (matches PhD veccostJacobians linear form). Type-4 with this fix gives PE_fin=4.67°.

### 🟡 MEDIUM priority

**6. iLQR two-tier tolerance looser than PhD** — saltro's `ilqr_cost_tol=1e0` allows sub-converged inner exit. A/B against PhD; tighten to 1e-1 if outer loop counts diverge.

**7. DDP Hessian integration & PSD-clipping untested numerically** — verify against FD on dynamics Hessians. If `use_dynamics_hess=false`, this is moot.

**8. Quaternion-cost case 2 differs from PhD** — saltro's `acos(qdot)` vs PhD's special `½·norm²/qerr(0)²`. Verify no mission depends on PhD form.

**9. Fix or document type-2 vec-mode divergence** — `f'(c)=-1/√(1-c²)` blows up at c→1 in our parameterization. PhD uses `phi=arccos(2·mag²-1)` reparameterization. Either implement PhD's smoother param, or document type-2 as unsupported in vec mode.

### 🟢 LOW priority

**10. Vec-mode case 4 validated as best shape** — today's testing showed `(1-c)²` gives best vec convergence. Consider making default. **Action: change default `ang_cost_func_type` for vec mode to 4.**

**11. Axis-aware ω reduction (`roll_ratio<1.0`) untested** — keep default 1.0; document.

**12. Reg shrink-on-success** (early C++ line 1959) — was that intentional? Probably an experimental change that didn't propagate. Document why saltro doesn't have it.

## Disagreements between agents

Two agents disagreed on whether recent PhD uses Gauss-Newton or full Hessian. Resolved by reading `Satellite.cpp:993` directly: case 3 is `dphi*dphi.t() + ddphi*phi` — **full Hessian** (Part B is correct, the earlier Part A-style claim was wrong). Recent PhD has full second-order; only early C++ uses Gauss-Newton.

## Verification status of in-flight saltro changes

| Change | FD-validated | Tested at scale | Status |
|---|---|---|---|
| RK4 Jacobian fix | ✅ | ✅ (4 baseline scenarios) | Bug fix, validated |
| Quaternion norm Hessian fix | ✅ | ✅ | Bug fix, validated |
| Cost Hessians (`use_cost_hess`) | ✅ | ✅ | Optional, well-tested |
| Vec-mode case 4 | ✅ | ✅ (today, PE_fin=4.67°) | Best-converging shape |
| V2 ω cost (ω_ff, α-from-β) | ✅ (46 FD tests) | ✅ | Opt-in, well-tested |
| Axis-aware ω reduction | ✅ | partial | Opt-in via `roll_ratio<1` |
| DDP Hessians | partial | partial | Optional, off by default |
| PSD-clip | n/a | partial | Stabilization, not validated |
| Legacy w_avang vec fix (today) | ✅ | ✅ | New today, FD-validated |
| Spike removal | n/a | ✅ (4 baselines) | Heuristic, mode-specific |
| smartbdot warm-start | ❌ | ❌ | **NOT YET PORTED** |

## Recommendations

**Pre-merge for vec-pointing PR (today's session):**

1. ✅ Legacy `w_avang` vec-mode fix is FD-validated and improves convergence (PE_fin 11.5°→9.5° type-3, 56°→4.67° type-4). Consider including in PR.
2. ⚠️ Vec-mode case 4 is the best-converging shape; if making it default, add a deprecation note for case-3-as-default.
3. ⚠️ Type-2 in vec mode should be guarded or warned.

**For longer-horizon merge to main:**

1. Port smartbdot warm-start (highest priority).
2. Document spike-removal assumptions; add per-mode config flags.
3. Validate DDP Hessians numerically (FD vs analytic).
4. Re-tune control-cost weights for missions.
5. Decide on PhD's quat case-2 (port or document as removed).
6. Fix vec-mode type-2 reparameterization or remove as unsupported.

## Caveats

- Some line numbers in agent outputs (especially Part A's "line 1959") are agent-claimed and have not been independently verified against the actual file content. Treat as approximate; verify with Read before acting.
- Agents could not Write directly (read-only `Explore` agents); their findings were re-emitted as text and saved to disk by the synthesis pass.
- Part A and Part B partially disagreed on Hessian form; Part B's reading from `Satellite.cpp:993` (full Hessian) is correct.

---

*This audit complements the in-progress vec-pointing diagnostic work. The today's session findings (cost Hessian innocent, ω-rate-space curvature insufficient, structural plateau at 10° pre-fix, type-4 + legacy-fix giving 4.67°) are not duplicated here — see [project_vector_pointing_regression.md](../../../.claude/projects/c--Users-LV---Patrick-McKeen-saltro/memory/project_vector_pointing_regression.md) for that thread.*
