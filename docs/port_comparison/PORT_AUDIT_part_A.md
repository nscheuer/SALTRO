# Port Audit Part A: MATLAB ↔ early C++/Py

> Comparison of PhD MATLAB original (`ALiLQR_BC_PM.m`, `ALILQR_GITHUB.m`, etc.) against early C++/Python port (`altro_general.cpp` family). Agent-generated 2026-04-27.

## 1. Vector-Pointing Cost (`cost2ang` function)

| Aspect | MATLAB (`ALiLQR_BC_PM.m`) | C++ (`altro_general.cpp`) | Differences |
|---|---|---|---|
| Angle definition | `acos(2*dot(qk,q0k)^2-1)` (lines 1010, 1018, 1023) | `acos(2*pow(dot(q,q0k),2)-1)` (line 893) | Identical |
| Cost shape options | `useAngSquared` (`ang^2`), `useExpAngle` (`exp(abs(ang))`) (lines 1035-1043) | **No branching for squared/exp** — only base case (lines 882-895) | C++ lacks `useAngSquared` and `useExpAngle` cases; cost is always `ang` in degrees |
| Units | Radians implicit | Explicit conversion `ang = ang*180/π` (line 897) | C++ outputs degrees; MATLAB radians |
| Derivative (dang) | `(-2*(la*a.' + lb*b.'))*Wmat(qk)/mag/(sqrt(1-mag^2)+eps)` (line 1013) | `((-2*(la*trans(a)+lb*trans(b))/(mag*pow(1-mag*mag,0.5))))*findWMat(q)` (line 894) | C++ may miss final `/mag` division in some branches |

**Drift note:** The C++ port loses two cost-shaping options (squared, exponential angle) that allow tuning small-angle gradient. The unit conversion is explicit in C++ but implicit in MATLAB — possible factor-of-180 scale mismatch in early integration. Agent flagged a possible `/mag` discrepancy in dang.

**Convergence-impact estimate: HIGH** (cost shape options matter for small-angle convergence)

## 2. Quaternion-Pointing Cost

| Aspect | MATLAB | C++ |
|---|---|---|
| Q-cost form | `dphi.'*Qk(4,4)*phi` (line 924) | `trans(dphi)*Qk(4,4)*phi` (line 1214) | Same |
| `useAngSquared`/`useExpAngle` | Available | **Not implemented** | Same drift as vec-mode |
| Block-diagonal Q | `[Qkww 0; 0 Qkqq]` (line 518 MATLAB) | Identical via `findQ` | Match |

**Drift note:** Same structural omission as vec-pointing — no quadratic/exp angle shape options. Asymmetric cost shaping between vec and quat objectives.

**Convergence-impact estimate: MEDIUM**

## 3. Angular-Velocity (ω-side) Cost

| Aspect | MATLAB | C++ |
|---|---|---|
| ω penalty | `Qk(1:3,1:3) = swpoint*eye(3,3)` or `swslew*eye(3,3)` (lines 517, 525) | `Qkww = Qkww*swpoint` / `*swslew` (lines 219, 252) | Identical |
| ω·B-field penalty | Not in cost (only dynamics) | Not in cost | Both omit |
| Slew/point phase | `sratioslew`, `swslew` overrides (lines 49-50, 520, 525) | Same (lines 249, 252) | Match |

**Drift note:** Both omit ω·B penalty in cost. Phase scheduling identical.

**Convergence-impact estimate: LOW**

## 4. Control Cost

| Aspect | MATLAB | C++ |
|---|---|---|
| Penalty matrix | `R = su*eye(3,3)` | `R = su*eye(3,3)` | Match |
| Raw vs delta-control | `0.5*uk'*R*uk` raw | `0.5*trans(uk)*R*uk` raw | Match |
| Boundary cases | k=N: `lkuu = R*0`, `lku=uk*0` (lines 887-888) | Same (lines 1169-1170) | Match |
| Per-actuator weights | Global `su` scalar | Global `su` scalar | No diversity |

**Drift note:** Identical raw-control formulation, identical terminal handling.

**Convergence-impact estimate: LOW**

## 5. RW Angular Momentum / Stiction

Neither implements RW momentum penalty in cost — both are pure trajectory-optimization without RW modeling. Slew/point ω penalty handled via `swslew`/`swpoint`.

**Convergence-impact estimate: LOW** (consistent design)

## 6. AL/iLQR Loop Control

| Aspect | MATLAB | C++ |
|---|---|---|
| Two-tier cost tol | `costTol = 1e-4`, `ilqrCostTol = 50*costTol` (lines 115, 122) | Both present, multiplier may differ | Same structure, possibly different ratio |
| AL outer stop | `dLA < costTol AND grad < gradTol`, or `cmaxtmp < cmax` (lines 2348-2364) | Conditional on `costTol` and constraint sat (line 2362) | Logically equivalent |
| Inner iLQR stop | `dLA < ilqrCostTol AND grad < gradTol`, or stale-iter | Same structure | Equivalent |
| Reg growth | `regScale = 1.6`; `drho = drho*regScale` (line 2246); init `regInit = 0` (line 83) | `regScale = 1.6`; growth on FP fail; **also `drho = min(drho/regScale, 1/regScale)` on success (line 1959)** | **DRIFT: C++ shrinks drho on success; MATLAB does not** |
| Penalty schedule | `penInit = 100`, `penScale = 20`, `penMax = 1e18` (lines 102-103) | `penInit = 100`, `penScale = 20`, `penMax = 1e8` | **DRIFT: penMax 1e18 vs 1e8** |

**Drift note:** Most critical: C++ proactively reduces regularization on successful BP (line 1959 in early C++); MATLAB only grows. This is a **sign-flipped regularization policy**. Penalty cap difference (1e18 vs 1e8) could cause C++ to hit ceiling earlier and stall.

**Convergence-impact estimate: HIGH**

## 7. Backward Pass

| Aspect | MATLAB | C++ |
|---|---|---|
| Dynamics Jacobians | `dynamicsJacobians()` linearize+integrate | `rk4Jacobians()` (lines 1834-1842) | Match |
| Q-function | `Qkxx = lkxx + Aqk.'*Pkp1*Aqk` (no dyn-Hess) | `Qkxx = lkxx + trans(Aqk)*Pkp1*Aqk` (line 1899) | iLQR-style (no DDP) |
| Reg mechanism | Single-pass reg growth | **Conditioning check + restart loop** (line 1882: `if(cond(Qkuureg)>50.0) k=N-2`) | C++ has explicit conditioning restart |
| Q_uu reg | `Qkuureg = Qkuu + rho*eye(m,m)` (implicit) | `mat Qkuureg = Qkuu + rho*mat(3,3).eye()` (line 1879) | Match |

**Drift note:** C++ implements a "bad cond → restart BP → reduce reg on success" loop; MATLAB doesn't show this mechanism in the available excerpt. The line-1959 shrinking is a significant algorithmic difference.

**Convergence-impact estimate: HIGH**

## 8. Constraints

| Aspect | MATLAB | C++ |
|---|---|---|
| Control saturation | `u_max - u >= 0`, `-u_max - u >= 0` (line 649) | `join_cols(u-umax, -u-umax)` (line 114) | Identical |
| Slacks | Optional `useSlacks` parameter | Supported but less prominent | Slight drift |
| Pointing | Soft cost-based, not hard constraint | Same | Match |
| AL penalty | `mu`, `penInit = 100`, `penScale = 20` | Same | Match |

**Convergence-impact estimate: LOW**

## 9. Numerical Defaults

| Parameter | MATLAB | C++ | Drift |
|---|---|---|---|
| `dt` | 1 s | 1 s | Match |
| `N` | 3600 | template variable | Same order |
| `swpoint` (w_av) | `0.0001 × (π/180)²` ≈ 3e-8 | `0.0001` | **Possible 3e-4× units mismatch** |
| `sv1` (w_avang) | 500 | input parameter | Variable |
| `swslew` | 1e-6 | 0 | **DRIFT: C++ disables slew weight** |
| `su` (control) | 500 | 1 | **MAJOR DRIFT: 500× difference** |
| `costTol` | 1e-4 | 1e-4 | Match |
| `maxIter` (outer) | 700 | 500 | 200-iter difference |
| `maxIlqrIter` | 25 | 100 | 4× difference |
| `regInit` | 0 | 0 | Match |
| `regScale` | 1.6 | 1.6 | Match |
| `penInit` | 100 | 100 (BC) / 1 (GITHUB) | Conflicting C++ defaults |
| `penMax` | 1e18 | 1e8 | **10× difference** |
| `penScale` | 20 | 10 (GITHUB) | 2× difference |

**Drift note:** Many numerical defaults diverged. The control-weight 500× is the most consequential — it changes the Pareto trade-off between state accuracy and control effort. `swpoint` units may be off (squared rad vs raw). `swslew = 0` in C++ removes off-trajectory feedback during slew.

**Convergence-impact estimate: HIGH**

## Top Discrepancies (Impact on Convergence)

1. **Cost-shape options dropped (`useAngSquared`, `useExpAngle`)** — C++ always uses raw angle. Breaks small-angle convergence shaping. *HIGH*
2. **Reg shrinking in BP (C++ line 1959)** — C++ divides drho by regScale on success; MATLAB doesn't. Sign-flipped policy. *HIGH*
3. **Control weight `su` 500× mismatch** — MATLAB 500, C++ 1. Drastic Pareto shift. *HIGH*
4. **Conditioning-restart in BP (C++ line 1882-1893)** — C++ restarts BP when `cond(Q_uu_reg) > 50`; MATLAB doesn't. May cause stalls. *MEDIUM*
5. **Penalty cap `penMax` 1e18 vs 1e8** — C++ ceiling 10× lower. Earlier stalling. *MEDIUM*

**Overall:** Significant algorithmic drift between MATLAB and early C++. Cost-function simplification, regularization policy reversal, numerical weight mismatches, and BP loop structure changes are compounding. Early C++ port appears incomplete/pre-validation.

> ⚠️ **Caveat:** Some line-number citations above (especially around line 1959 / 1882) are agent-claimed and would benefit from manual verification against the actual `altro_general.cpp` content.
