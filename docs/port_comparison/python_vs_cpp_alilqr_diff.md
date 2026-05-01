# C++ ↔ Python AL/iLQR Diff

## Summary

The Python AL/iLQR wrappers call C++ backward_pass and forward_pass for the actual BP/FP math, so the trajectory-update math is unified. However, the **loop structure, convergence criteria, regularization scheduling, and stopping conditions** have drifted significantly. This document details each gap and prescribes the Python fix needed for behavioral parity.

---

## Section 1: Inner iLQR (`iLQR.cpp` ↔ `ilqr.py`)

### 1.1 Two-Tier Cost Tolerance (CRITICAL)

**C++** (iLQR.cpp:220-230): Has two distinct tolerances:
- `ilqr_cost_tol` (loose, ~1e0): used for inner-loop exit via disjunctive convergence
- `cost_tol` (strict, ~1e-1): used for stagnation counting and outer-loop maturity

Inner-loop converges when `delta_J <= max(ilqr_cost_tol, cost_tol)` in disjunctive mode.

**Python** (ilqr.py:293): Only checks `cost_tol`, never uses loose `ilqr_cost_tol`. Causes Python inner loop to run ~10x longer than C++ on typical problems.

**Fix**: Apply two-tier logic in disjunctive branch:
```python
inner_tol = max(passsettings.ilqr.ilqr_cost_tol, passsettings.ilqr.cost_tol)
if delta_J <= inner_tol and not passsettings.ilqr.conjunctive_convergence:
    return X, U, "converged", snapshots, transitions
```

### 1.2 Gradient-Norm Convergence

**C++** (iLQR.cpp:232-240): Computes `max_k ||d_k||` and checks vs `grad_tol`.

**Python** (ilqr.py): No gradient-norm convergence check at all.

**Fix**: Add after forward-pass success:
```python
grad_converged = False
if passsettings.ilqr.grad_tol > 0.0:
    max_d_norm = max(np.linalg.norm(d_k) for d_k in d_list)
    grad_converged = (max_d_norm <= passsettings.ilqr.grad_tol)
```

### 1.3 Conjunctive vs. Disjunctive Convergence

**C++** (iLQR.cpp:242-261): Two modes:
- **Conjunctive** (true): require BOTH `outer_cost_converged` AND gradient-ok
- **Disjunctive** (false, default): exit on ANY of {loose inner cost, gradient}

**Python** (ilqr.py:293): Only checks strict `cost_tol`, no conjunctive flag or gradient.

**Fix**: Implement flag-gated branching per C++ logic.

### 1.4 Stagnation Counter (z_count_lim)

**C++** (iLQR.cpp:267-275): Increments counter each iteration where `delta_J <= cost_tol`. On reaching `z_count_lim` consecutive steps with no progress, exit as Converged (prevents burning max_iters on cost plateau).

**Python** (ilqr.py): No stagnation counter at all.

**Fix**: Initialize before main loop, increment on `delta_J <= cost_tol`, reset on progress.

### 1.5 Regularization Decrease Clamping

**C++** (iLQR.cpp:137-149): After successful BP, `reg` is clamped to `reg_min`, preventing discontinuity.

**Python** (ilqr.py:156): Snaps `reg` to 0 instead of clamping. Creates chatter between reg=0 and reg=reg_min.

**Fix**: Change `reg = 0.0` to `reg = max(dreg * reg, reg_min)`.

---

## Section 2: AL Outer (`alilqr.cpp` ↔ `alilqr.py`)

### 2.1 min_outer_iters Maturity Gate

**C++** (alilqr.cpp:155-164): Exit only if `max_c <= constraint_tol` AND inner reported `Converged` AND at least `min_outer_iters` iterations completed.

**Python** (alilqr.py:218-220): Exits immediately on `max_c <= constraint_tol`, without checking maturity.

**Fix**: Require `inner_ok AND outer_matured OR strict_path` (via `constraint_tol_strict`).

### 2.2 Penalty Maximum Exit (penMax reached)

**C++** (alilqr.cpp:187-191): If all penalties saturate at `penalty_max` yet `max_c > constraint_tol`, break and return failure.

**Python** (alilqr.py): Missing entirely.

**Fix**: Add check after lambda/mu update to detect saturation and break loop.

---

## Section 3: Forward-Pass / Line-Search

Python delegates all FP logic to C++ via `saltro_py.forward_pass()`. No drift — the math is unified.

**No fix needed** — the FP math is unified.

---

## Section 4: Backward-Pass Regularization Mechanics

C++ applies triple-bump on FP failure: scale-up, add bump, scale-up.

Python implements triple-bump identically.

**No fix needed** — regularization bump logic is correct.

---

## Section 5: Spike Removal Timing & Configuration

C++ calls spike removal after successful FP step accepted.

Python also calls after FP success, with additional constraint-violation gate (Python enhancement).

**No fix needed** — placement and gate are compatible.

---

## Section 6: PhD MATLAB Zcount Semantics

**MATLAB** increments when `dLA == 0` (exactly zero).

**C++** increments when `delta_J <= cost_tol` (any improvement below tolerance).

C++ is more permissive and aligned with iLQR intent. Python needs Section 1.4 fix.

---

## Section 7: Missing Flags & Config Reads

**C++ parameters not read by Python**:
- `conjunctive_convergence`, `grad_tol`, `z_count_lim` (iLQR config)
- `min_outer_iters`, `constraint_tol_strict` (AugLag config)

These must be read and applied per Sections 1 and 2.

---

## Summary of Required Changes to Python

| Item | File | Fix |
|------|------|-----|
| Two-tier cost tolerance | ilqr.py:293 | Use max(ilqr_cost_tol, cost_tol) in disjunctive path |
| Gradient-norm check | ilqr.py | Compute max_k norm(d_k), check vs grad_tol |
| Conjunctive/disjunctive | ilqr.py | Add flag-gated branching logic |
| Stagnation counter | ilqr.py | Add z_count increment/reset on delta_J <= cost_tol |
| Regularization clamp | ilqr.py:156 | Change reg=0.0 to reg=max(dreg*reg, reg_min) |
| min_outer_iters gate | alilqr.py:218 | Require inner_ok AND outer_matured OR strict_path |
| Penalty saturation exit | alilqr.py | Check any_mu < penalty_max and break if saturated |
| Config reads | ilqr.py, alilqr.py | Ensure all flags from configs are read |

---

**Document version**: 2026-04-29  
**Scope**: Python AL/iLQR wrapper behavioral parity with C++ production code.  
**Status**: Read-only diagnosis — source files not modified.
