# Port Audit Part D: saltro current branch (`PKMN_antispike`) ↔ saltro main

> Branch `PKMN_antispike` is 35 commits ahead of main. Agent-generated 2026-04-27.

## Overview

In-flight changes on `PKMN_antispike`: spike-removal subsystem (~824 new lines C++), cost Hessian + terminal weighting refactor, RK4 Jacobian fixes, quaternion helpers, extensive test harness.

## 1. Spike Removal Subsystem

**Files:**
- `include/saltro/optimizer/spike_removal.h` (79 lines, NEW)
- `src/optimizer/spike_removal.cpp` (824 lines, NEW)
- `tests/unit/optimizer/test_spike_removal.py` (564 lines, NEW)

**Status:** EXPERIMENTAL — core stabilized after multi-round R&D.

**Key changes:**
- C++ spike detection + removal pipeline; multiple detection passes (local-outlier, hemisphere-flip)
- `isSaturated()` scales by `control_limit_scale` (commit ee2d5a5) — detects actuator saturation at AL ceiling
- `torqueOpposesError()` checks if actuator is fighting pointing error
- Mean-PE guard prevents spike removal from destroying converged trajectories (commit 768ca86)
- Python sync at `tests/debug/optimizer/alilqr_python/spike_removal.py` (1080 lines)

**Commits:** ec17b6e → ee2d5a5 (10 commits)

**Risk:** **MEDIUM**. Multiple failed iterations (winding-number detector reverted in 06e5229). Mean-PE guard is band-aid suggesting false-positive rate. Heuristic thresholds may not generalize beyond 4 test scenarios.

## 2. Cost Function & Settings

**Files:**
- `include/saltro/pybind/plannersettings.h` (+74 lines)
- `python/pybind/plannersettings_py.cpp` (+32 lines)
- `src/pybind/satellite.cpp` (+190 lines)

**Status:** WELL-TESTED.

**Key changes:**
- New cost fields: `ang_vel_err_dir` (crossterm weight), `ang_vel_roll_ratio` (axis-aware W_ω reduction), `ang_vel_err_dir_ratio` (gating)
- Terminal versions of all new fields
- `setTerminalEmphasis(k)` (commit 9e52fbc) — uniformly scales terminal weights by k
- ECI format support: detect NaN in first quaternion element → ECI vector goal
- Analytic cost Hessians (commits 10ae031, 08fd6ce) — `use_cost_hess` flag
- Vec-mode case 4 cost; `is_eci_format` branches in 3 functions
- **Just-added:** legacy `w_avang` vec-mode routing fix (today's session)

**Risk:** LOW. New weights default to zero (back-compat). Terminal emphasis is convenience scaling. Analytic Hessians optional.

## 3. RK4 & Quaternion Math Fixes

**Files:**
- `include/saltro/math/integrators/rk4.h` (+127 lines, major refactor)
- `src/math/quaternion.cpp` (+27 lines, fixes + helpers)

**Status:** BUG FIX.

**Key changes:**
- **RK4 Jacobians:** complete rewrite of `rk4_jacobians()` with explicit chain rule through all 4 stages. Discrete-time A, B match forward integration exactly (critical for iLQR stability).
- **Quaternion norm Hessian fix** (d12d305, refined 93b6114): `ddrotmatTvecdqdq()` was missing diagonal term and symmetric partner. Now correct: `H_k[1+a, 1+b] = -2 δ_{ab} v_k + 2 v_a δ_{bk} + 2 v_b δ_{ak}`
- **New quaternion helpers** (24a1b03): `quatConj()`, `quatMult()`, `quatAngle()`

**Risk:** LOW. Math library fixes with explicit test coverage.

## 4. Optimizer Core (iLQR, BP, FP)

**Files:**
- `src/optimizer/iLQR.cpp` (+102 lines)
- `src/optimizer/backwardpass.cpp` (+242 lines)
- `src/optimizer/forwardpass.cpp` (+82 lines)

**Status:** WELL-TESTED.

**Key changes:**
- `reg_init = 1e-12` (very small); retried with `reg *= reg_scale` (1.6) on failure
- **Gradient-based convergence** (08fd6ce): two-tier `cost_tol` + `grad_tol`
- **BP** (08fd6ce): P_k propagates with UNREGULARIZED Q_uu (only K/d use regularized) — prevents Riccati inflation
- **FP**: linesearch with `α = 2^{-iter}`, MRP-based reduced-state attitude error
- **AL outer loop** (069543e): `min_outer_iters`, `strict_path` gating

**Risk:** MEDIUM. Reg floor 1e-12 is aggressive. Gradient tolerance heuristic. DDP machinery partially visible but incomplete.

## 5. PD Controller (Spike Substitution)

**Files:**
- `include/saltro/pybind/controller/pdcontroller.h` (89 lines, NEW)
- `src/pybind/controller/pdcontroller.cpp` (143 lines, NEW)

**Status:** EXPERIMENTAL.

**Key changes:**
- Actuator-agnostic PD via numerical Jacobian (20767e4)
- Round-2: authority weighting + scale-to-max (fff95f3)
- Substitutes impulsive controls with smooth PD feedback during spike removal

**Risk:** MEDIUM. Numerical Jacobian (FD) can be unstable. Authority weighting heuristic — may need tuning per vehicle.

## 6. Test Harness

**Files:**
- `tests/unit/optimizer/test_spike_removal.py` (564 lines)
- `tests/debug/optimizer/alilqr_python/spike_removal.py` (1080 lines)
- 40+ new debug/visualization scripts (`animate_convergence.py`, `convergence_diagnose.py`, `wide_test_runner_vec.py`, etc.)
- 8 new GIF animations, 20+ log files from 4 baseline scenarios

**Status:** EXTENSIVE.

**Coverage:**
- 4 baseline scenarios (30°/90° slew, detumble, 90° large-ω)
- 47% cost reduction in Release build (f5c2e01)
- BdotCtrl warm-start eliminates spikes at source (6317d05)
- Sub-degree convergence with high terminal weights (06ef908)

**Risk:** LOW. Visual validation. Failed prototypes explicitly reverted.

## 7. Documentation

- `docs/cost_function_refactor.md` (NEW, ~9 KB) — cost evolution + vec-mode machinery
- `docs/port_comparison/` (NEW directory for audit outputs)

**Risk:** LOW.

## Risk Summary

### 🔴 HIGH

1. **Spike detection heuristics** — multiple failed iterations, mean-PE guard band-aid, thresholds may not generalize.
2. **PD authority weighting** — numerical Jacobian, heuristic gains.

### 🟡 MEDIUM

1. **Reg floor 1e-12** — very small initial reg.
2. **Gradient-tol convergence** — heuristic two-tier.
3. **Cost Hessian integration** — wired into BP; defaults off but if wrong, breaks Riccati.

### 🟢 LOW

1. **RK4 Jacobian fix** — bug fix, validated.
2. **Quaternion helpers** — library functions.
3. **Terminal emphasis** — convenience.
4. **Warm-start partial results** — API extension.

## By-subsystem table

| Subsystem | Commits | Lines | Status | Risk |
|---|---|---|---|---|
| Spike Removal | 10 | 1467 | EXPERIMENTAL | 🔴 HIGH |
| Cost & Settings | 4 | 296 | WELL-TESTED | 🟢 LOW |
| Optimizer Core | 5 | 426 | WELL-TESTED | 🟡 MEDIUM |
| RK4 & Quat Math | 3 | 154 | BUG FIX | 🟢 LOW |
| PD Controller | 2 | 232 | EXPERIMENTAL | 🟡 MEDIUM |
| Tests & Debug | ~40 | ~3000 | EXTENSIVE | 🟢 LOW |

**Total:** 35 commits, ~3200 core lines, 40+ debug scripts, 8 test animations.

## Shipping readiness

**Recommended phase-in:**
- Phase 1: RK4/quat fixes + cost Hessian (low risk, required)
- Phase 2: Spike detection with monitoring + mean-PE guard
- Phase 3: PD substitution as optional feature flag
- Phase 4: Terminal-emphasis scaling

**Substantial in-flight effort with good test coverage but experimental core subsystems.** Spike removal and PD control need broader validation before production.
