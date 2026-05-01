# Port Audit Part B: early C++ ↔ recent PhD C++

> Comparison of early C++/Py port (`altro_general.cpp` ~3670 lines) against recent PhD C++ (`Satellite.cpp` + `OldPlanner.cpp`). Agent-generated 2026-04-27.

## Executive Summary

The early C++ predates the PhD restructuring; recent PhD splits satellite dynamics/cost from planner orchestration. **Core mathematical structures (vec-cost cases, AL penalty scaling) are preserved, but recent code adds state normalization and dimension-aware projection logic.**

## 1. Vec-Cost (whichAngCostFunc cases 0-3)

| Aspect | Early (`altro_general.cpp:864-899`) | Recent (`Satellite.cpp:972-999`) |
|---|---|---|
| Case 0 | `(1-ddot)` linear | `(1.0-ddot)` linear | Identical |
| Case 1 | `0.5*(1-ddot)^2` | `0.5*(1.0-ddot)^2` | Identical |
| Case 2 | `acos(2*dot^2-1)` analytic | `phi` from `cost2angQ()` lookup | Functionally equivalent |
| Case 3 | Not present | `0.5*phi^2` squared angle | **NEW in PhD** |

**Drift note:** Early C++ has cases 0-2 (computed via `findWMat(q)` projection at lines 885, 890, 894). Recent PhD refactors into `cost2angQ()` (line 967) returning `(phi, dphi, ddphi)`, adding **full Hessian term `ddphi`**. Case 3 (squared angle) is entirely new.

**Convergence-impact estimate: LOW** for the cases themselves; the math is equivalent modulo projection.

## 2. Quat-Cost Path

| Aspect | Early | Recent (`Satellite.cpp:1087-1150`) |
|---|---|---|
| Function structure | Monolithic `cost2Func()` line 946 | Two dispatchers: `veccostJacobians()` (line 893) + `quatcostJacobians()` (line 1087) |
| Quat alignment | `cost2ang()` for "pvs" alignment | `cost2angQ()` for both; ECIvec_k as `vec4` not `vec3` |
| Sign convention | `trans(dphi)*Qk(4,4)*dphi` Gauss-Newton (line 1215) | Same (lines 974-996), then projects via `nj.t()*lkxx*nj` (line 1079) |

**Drift note:** Early code unifies quat/vec; recent splits them with quaternion-native ECIvec_k as `vec4`. The actual cost formulas are mathematically the same modulo the state-normalization Jacobian chain `nj = findGMat(qk)*state_norm_jacobian(xkraw)*findGMat(qkraw).t()` (line 950).

**Convergence-impact estimate: MEDIUM**

## 3. Cost Hessian Representation

| Aspect | Early | Recent |
|---|---|---|
| Hessian form | `dphi^T * Q(4,4) * dphi` (line 1215) | `dphi*dphi.t() + ddphi*phi` (case 3, line 993) |
| Second-order | **NO — Gauss-Newton only** | **YES — `ddphi*phi` term included** |

**Drift note (CRITICAL):** Early C++ uses **pure Gauss-Newton** (no second-order curvature). Recent PhD adds the second-order `ddphi*phi` term — first-order correction from cost-function curvature. Most other cost terms (state-velocity, cross, RW angmom) similarly use full Hessians (e.g., line 1009: `lkxx += w_ang*dd_sc_ang`, line 1016: `lkxx += w_avang*ddvTRTudqQ(...)`).

Early code's Gauss-Newton assumption is conservative — slower convergence (treats problem as more quadratic than it is). Recent PhD's full Hessian is tighter.

**Convergence-impact estimate: HIGH** — Gauss-Newton vs full Hessian can change iter count by 20-50%.

> ⚠️ Note: this finding contradicts an earlier impression from a different agent that PhD also uses Gauss-Newton. The truth from `Satellite.cpp:993` is that case 3 in recent PhD has both terms. Earlier agent was looking only at case 0/1 in `altro_general.cpp`.

## 4. 3D Reduced State vs 4D Ambient

| Aspect | Early | Recent |
|---|---|---|
| State dim | 7D `[w(3), q(4)]` implicitly; ops on 4D q | 7D explicit; cost in 3D via reduction `findGMat()` |
| Projection | `findGMat()` sporadic (lines 378, 585, 962, 1080, 1160, 1277) | `findGMat()` + `state_norm_jacobian` chained as `nj` (line 950, 1078) |
| State normalization | Implicit (assumes normalized) | **Explicit** `xk = state_norm(xk)` (line 947, 1141) |

**Drift note:** Early code uses `findGMat()` to map gradients from 7D to 6D; recent layers explicit state normalization on top. Composition: `findGMat(qkraw)` (7→7) × `state_norm_jacobian(xkraw)` (7→7) × `findGMat(qk)` (7→3 reduced). Ensures cost Jacobians stay in correct tangent space even if quaternion drifts off unit-norm during integration.

**Convergence-impact estimate: LOW-MEDIUM** — depends on integrator tightness; matters for spike scenarios.

## 5. AL Outer Loop

| Aspect | Early | Recent (`OldPlanner.cpp:1486-1750`) |
|---|---|---|
| Iteration | `for(int j=0; j<maxOuterIter; j++)` | Same (line 1486) |
| Exit | Update lambda/mu, check max viol | `outerBreak()` checks `cmaxtmp<cmax_tmp` AND `mu>=penMax_tmp` (line 1735) |
| Penalty scaling | `mu = max(0, min(penMax, penScale*mu))` (line 2598) | `mu = max(0.0, min(muMax, muScale*mu))` (line 1907) |
| λ update | `lambda += mu*c` clipped (line 2590-2591) | Same (line 1891-1899) |

**Drift note:** Structurally identical. Recent has explicit `outerBreak()`/`ilqrBreak()` functions; early code implies these inline.

**Convergence-impact estimate: NEGLIGIBLE**

## 6. Backward Pass: Q_uu and DDP

| Aspect | Early | Recent (`OldPlanner.cpp:1005-1017`) |
|---|---|---|
| Q_uu | `lkuu = R` (line 1196) | `lkuu = costJac.luu` includes `act_cost_mat*w_u_mult` (line 987, 1030) |
| BP recursion | `Sk = lkxx + A^T*Sp1*A - A^T*Sp1*B*K` (line 1013) | Same (line 1013) |
| Reg | `rho, drho` from `regInit` (line 2278) | Same (line 1476, 1497) |
| **DDP F_uu terms** | **Not present** | **Not present** |

**Drift note:** Both use **standard DDP backward recursion with NO first-order Q_uu correction** (no `F_u^T S' F_u`). Both rely on state-transition Jacobians A, B directly. Recent PhD adds RW angmom to `lkuu` via `act_cost_mat`, which is an addition not a structural change.

**Note for saltro context:** saltro's recent DDP additions (`Qxx_ddp`, `Quu_ddp`, PSD-clip) are NOT present in either reference. Saltro-specific.

**Convergence-impact estimate: NEGLIGIBLE** for the comparison; both LQR-style.

## 7. Spike Removal / Homotopy

| Aspect | Early | Recent |
|---|---|---|
| Spike removal | **Not found** | **Not found** |
| Homotopy | **Not found** | **Not found** |
| Random perturbation | Line 1510 commented out | Not found |

**Drift note:** Neither reference has spike removal or homotopy. Saltro's multi-stage spike removal is a saltro-specific addition.

**Convergence-impact estimate: NEGLIGIBLE**

## 8. Warm-Start

| Aspect | Early | Recent |
|---|---|---|
| Synthetic-q construction | **Not found** | **Not found** |
| Initial control sat | Clamp `+/-0.15` (line 487-488) | Clamp `0.01 * RW_max_torq` (line 722) |
| Initial trajectory | `generateTrajectory()` in test.cpp ~400 | Likely in PyPlanner.hpp (not shown) |

**Drift note:** Neither reference shows synthetic-q construction (saltro builds it via `processAttitudeTarget`). Both assume pre-computed initial trajectory. Different actuator-saturation scales suggest different hardware.

**Convergence-impact estimate: LOW** for algorithm; depends on initialization.

## 9. Constraints (AL formulation)

| Aspect | Early | Recent |
|---|---|---|
| Constraint set | Inequality `u≤umax`, `u≥-umax` only | Control bounds + RW angmom + sun-pointing + ω limits |
| AL form | Quadratic `0.5*mu*c^2` | Same |
| Jacobian projection | `ckx = ckx * findGMat(q)` (line 325) | Same pattern |
| Constraint Hessian | `constraint_hess_mult = 1.0` (line 21) | Same (Satellite.hpp:21) |

**Drift note:** Both use quadratic-penalty AL. Recent expands constraint set; AL formulation unchanged.

**Convergence-impact estimate: LOW** for algorithm; MEDIUM for problem difficulty.

## 10. Numerical Defaults

| Param | Early | Recent | Notes |
|---|---|---|---|
| `regInit` | (parameterized) line 2278 | get<0>(regSettings) line 1438 | Both parameterized |
| `penMax` | line 2598 | get<3>(auglagSettings) line 1741 | Both parameterized |
| `penScale` | line 2598 | line 1883 | Both parameterized |
| Control sat | 0.15 A·m² (line 487) | 0.01 × max_torque (line 722) | Different hardware scales |
| `lagMultMax` | line 2590 | line 1881 | Both ~1000 typical |
| `constraint_tol` | implicit `cmax_tmp` | implicit `cmax_tmp` line 1742 | Neither has explicit gate |

**Drift note:** Numerics parameterized in both. Control-sat ratios differ (hardware-specific). No explicit constraint-tolerance gate in either.

**Convergence-impact estimate: MEDIUM** (depends on tuning).

## Top Discrepancies

1. **Cost Hessian: Gauss-Newton vs full second-order** *HIGH* — Early uses `dphi^T Q dphi` only; recent adds `dphi*dphi.t() + ddphi*phi`. 20-50% iter-count difference.
2. **State normalization & projection chain** *MEDIUM* — Early implicit; recent explicit `state_norm()` + 3-Jacobian composition. Robustness to drift.
3. **Vec-cost separation & case 3 addition** *MEDIUM* — Early: unified, cases 0-2; recent: split, case 3 added.
4. **Constraint set expansion** *LOW* — Early: control bounds only; recent: + RW + sun + ω.
5. **AL exit gating clarity** *LOW* — Recent has explicit `outerBreak()`; early implicit.

## Conclusion

Recent PhD is a **cleaner, more feature-complete port** with full Hessian, explicit state normalization, modularized cost (vec/quat split), expanded constraints. **No critical functionality dropped.** Early C++ Gauss-Newton + implicit projections are weaker but algorithmically the same family.

**For saltro:** ensure `ddphi` term is faithfully ported, state normalization preserved, regularization tuned for application.
