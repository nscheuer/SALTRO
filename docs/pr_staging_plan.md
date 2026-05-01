# PR Staging Plan — vec-pointing math + GN flag

> Branch `PKMN_antispike` has 35+ commits with mixed work (vec-pointing, GN flag, DDP, spike removal, RK4 fixes, etc.). This plan separates the vec-pointing math fix and the GN flag into two reviewable PRs against `main`. Other in-flight work stays on the branch for later.

## PR 1 — Vector-pointing cost (math fix + cleanup)

**Goal:** Fix vec-pointing math (2-DOF angle cost), correct legacy `w_avang` cross-cost geometry in vec mode, fix quat case-4 (concave→convex), and clean up dormant `omega_ref`/`omega_ff` machinery.

**Files (cost subsystem only):**

| File | What changes |
|---|---|
| `include/saltro/pybind/satellite.h` | Remove `omega_ref` from 6 cost-function signatures; remove unused `q_current` from `processAttitudeTarget`. |
| `src/pybind/satellite.cpp` | Add `is_eci_format` branches in stageCost / Jacobians / Hessians for vec-mode 2-DOF angle cost (cases 0-4). Route legacy `w_avang` in vec mode through PhD-style linear cross. Fix quat case 4 from `1-d²` (concave) to `(1-d)²` (convex). Remove `omega_ref`/`omega_ff` machinery. Remove unused `dV_dq`/`d2V_dq2` in `disturbanceTorque`. Document `w_avmag` Hessian gap. |
| `include/saltro/pybind/plannersettings.h` | Add `ang_vel_err_dir_ratio` (β for new-path crossterm), `ang_vel_roll_ratio` (axis-aware W_ω reduction), and terminal versions. |
| `python/pybind/plannersettings_py.cpp` | Bind the new fields. |
| `python/pybind/satellite_py.cpp` | Drop `omega_ref` py::arg from 6 method bindings. |
| `tests/unit/pybind/test_satellite_cost_omega_ff.py` | Rewrite without `omega_ref` parametrization; keep coverage of vec-mode angle cost (5 cases × FD), crossterm, axis-aware W_ω. 33 tests. |

**Out of scope for this PR (stays on branch):**
- `cost_hess_gauss_newton` flag (PR 2)
- Optimizer changes (`src/optimizer/*`, `include/saltro/optimizer/*`)
- `src/math/integrators/rk4.h` (RK4 chain rule rewrite)
- `src/math/quaternion.cpp` (Hessian fix + helpers)
- Spike removal (`src/optimizer/spike_removal.{cpp,h}`)
- PD controller (`src/pybind/controller/pdcontroller.{cpp,h}`)
- All debug harnesses under `tests/debug/`
- Documentation (`docs/cost_function_refactor.md`, port audit, sweep results, this file)

**How to create PR 1:**

```bash
git checkout main
git pull
git checkout -b vec-pointing-cost
git checkout PKMN_antispike -- \
  include/saltro/pybind/satellite.h \
  src/pybind/satellite.cpp \
  include/saltro/pybind/plannersettings.h \
  python/pybind/plannersettings_py.cpp \
  python/pybind/satellite_py.cpp \
  tests/unit/pybind/test_satellite_cost_omega_ff.py
# Inspect, then revert any GN-flag-specific lines in satellite.cpp /
# plannersettings.h that don't belong (see "GN-flag lines to exclude" below).
git add -A
git commit -m "fix(cost): vec-pointing 2-DOF math + cleanup omega_ref"
gh pr create --base main --title "Vec-pointing cost fix + cleanup" --body "..."
```

**GN-flag lines to exclude from PR 1** (these belong to PR 2):
- `include/saltro/pybind/plannersettings.h`: the `cost_hess_gauss_newton` field + comment block.
- `python/pybind/plannersettings_py.cpp`: the `def_readwrite("cost_hess_gauss_newton", ...)` line.
- `src/pybind/satellite.cpp`: the two `if (!cost_cfg.cost_hess_gauss_newton)` guards inside `stageCostHessians` (one in vec branch around L1815, one in quat branch around L1875).

## PR 2 — Gauss-Newton cost-Hessian flag

**Goal:** Add opt-in `cost_hess_gauss_newton` flag that drops the `f'·d²c/dq²` chain term in the angle-cost Hessian. PSD by construction (when `f''≥0`); makes vec-mode converge to sub-degree (otherwise plateaus at ~10°). No-op in quat mode (no chain term to drop).

**Files (3 small additions, stacked on PR 1):**

| File | What changes |
|---|---|
| `include/saltro/pybind/plannersettings.h` | Add `bool cost_hess_gauss_newton = false;` field with doc comment. |
| `python/pybind/plannersettings_py.cpp` | Add `def_readwrite` binding. |
| `src/pybind/satellite.cpp` | Add 2 `if (!cost_cfg.cost_hess_gauss_newton)` guards in `stageCostHessians`: one wrapping `Hqq_ang += dh_dc * d2c_dq2;` (vec branch), and one wrapping the quat-mode PwA correction skip (so flag is no-op in quat). |

**How to create PR 2 (after PR 1 lands):**

```bash
git checkout main
git pull
git checkout -b gauss-newton-cost-hess
# Apply just the 3 GN-specific hunks (see "GN-flag lines to include" below).
git add -A
git commit -m "feat(cost): cost_hess_gauss_newton flag for vec-mode convergence"
gh pr create --base main --title "Gauss-Newton cost Hessian flag" --body "..."
```

**Default value question:** Should `cost_hess_gauss_newton` default to `true` or `false`?

- **`false` (current):** Conservative. Quat users see no change. Vec users must opt-in to get sub-degree convergence (otherwise stuck at ~10°).
- **`true`:** Vec works out of the box. Quat is mathematically unchanged (chain term doesn't exist there) so zero risk.

Recommendation: ship as **default `false`** in PR 2, then a follow-up PR can flip to `true` after broader confidence. Or auto-enable inside the vec-mode branch only (more opinionated). User to decide.

## What stays on `PKMN_antispike` (not in either PR)

All of these are valid in-flight work but not part of this round:

- DDP machinery (`Qxx_ddp`, `Quu_ddp`, PSD-clip in `backwardpass.cpp`)
- Spike removal (`spike_removal.{cpp,h}`, ~824 new lines)
- PD controller (`pdcontroller.{cpp,h}`)
- iLQR two-tier convergence, stagnation count, grad-tol
- AL min_outer_iters maturity gate
- RK4 Jacobian + quaternion norm Hessian fixes (`rk4.h`, `quaternion.cpp`)
- Reg defaults (1e-12 floor)
- All debug/visualization scripts
- Port audit, sweep, and dormant-code-audit docs

Each of these can become its own PR or stay branched until needed.

## Verification before opening either PR

For PR 1:
- [ ] `pytest tests/unit/pybind/test_satellite_cost_omega_ff.py` — 33 tests pass.
- [ ] Smoke test: vec-mode 00_baseline runs without crashing (PE_fin will plateau at ~10° without GN — that's expected; PR 2 fixes it).

For PR 2:
- [ ] Same FD tests still pass (GN flag default false → same behavior).
- [ ] Vec-mode 00_baseline with `c.cost_hess_gauss_newton = True` reaches sub-degree PE_fin.
- [ ] Quat-mode 00_baseline with GN on/off gives identical numbers (no-op).

## Caveats

- **Tests file naming**: `test_satellite_cost_omega_ff.py` was renamed in spirit but kept at the original path for git history. Could be renamed to `test_satellite_cost_v2.py` or merged into existing `test_satellite_cost.py` in a follow-up if desired.
- **`omega_ff` removal is irreversible without a feature add-back later**: If trajectory tracking with non-zero ω_ref is needed in the future (e.g., following a moving target), the BP plumbing would need to be added back. Spec is described in `docs/dormant_code_audit.md`.
- **`magic_control_weight`**: kept (the magic-actuator type is still a TODO per `project_magic_actuator_todo.md`).
- **Type 2 in vec mode**: still has `f'(c) = -1/√(1-c²)` divergence at c=1; not fixed in either PR. Document as not recommended for vec mode.
