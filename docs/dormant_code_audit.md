# Dormant Code Audit — PKMN_antispike

> Generated 2026-04-28 by Explore agent. Findings to be addressed before splitting into PRs.

## Summary

| ID | Item | File | Action | Lines | Note |
|---|---|---|---|---|---|
| 1 | `omega_ref` parameter & `omega_ff` machinery | `satellite.cpp/.h` (3 functions) | **DELETE** | ~170 | User-flagged; BP never passes nonzero |
| 2 | Legacy `w_avang` back-compat paths | `satellite.cpp` (3 functions) | **KEEP** | ~80 | User explicitly wants kept |
| 3 | `w_avmag` Hessian gap | `satellite.cpp:1717` | DOCUMENT | ~50 | Incomplete; document as known gap |
| 4 | Unused disturbance derivatives (`dV_dq`, `d2V_dq2`) | `satellite.cpp:254-257` | **DELETE** | ~4 | Computed but never used |
| 5 | `q_current_unused` parameter | `satellite.cpp:143-145` | **DELETE** | ~5 | Parameter never referenced |
| 6 | `magic_control_weight` config | `plannersettings.h:171` | KEEP | ~10 | TODO: magic actuator port (see memory) |

## Detail

### 1. `omega_ref` / `omega_ff` (HIGH)

`omega_ref` is wired through stageCost / stageCostJacobians / stageCostHessians signatures with default `Vec3::Zero()`. BP at `backwardpass.cpp:166-167, 192-193` never passes nonzero — always uses default. The new-path branches `if (omega_ref.squaredNorm() > 0.0)` therefore never fire.

Comments at `satellite.cpp:1028, 1294, 1668` say "wired into signature; new-path activation lands in commit 3" — that commit never happened. Code volume: ~170 lines across the three functions.

After deletion, both legacy and new-path crossterms collapse to the same shape `coefficient · ω · err_dir`, differing only in coefficient computation (legacy: raw `w_avang`; new path: `α = β·√(w_ang·λ_min)`).

### 2. Legacy `w_avang` paths (KEEP per user direction)

Defaults to 0 in CostConfig. Active when `c.ang_vel_err_dir != 0` (test runners set it explicitly). The user wants to keep this for back-compat and for tuning flexibility.

After omega_ref cleanup, the legacy path simplifies considerably and remains as the "manual coefficient" alternative to the new path's "auto coefficient via β".

### 3. `w_avmag` Hessian gap (DOCUMENT)

stageCostHessians computes nothing for the `w_avmag·|ω·b_body|` term — line 1717 has `(void)w_avmag;`. The gradient IS computed (in stageCostJacobians) but the Hessian path skips this term entirely.

Action: add a comment at line 1717 noting Hessian is not implemented for this term. Users enabling `use_cost_hess` while having nonzero `ang_vel_mag` will get an inconsistent quadratic model. Not blocking for this PR.

### 4. Unused disturbance derivatives (DELETE)

`disturbanceTorque` at lines 254-257 computes `dV_dq` (Jacobian) and `d2V_dq2` (Hessian) of `R^T · V_eci` but never uses them. Cast to `(void)` to silence warnings.

```cpp
Mat34 dV_dq = saltro::math::drotmatTvecdq(q, V_eci).transpose();
auto d2V_dq2 = saltro::math::ddrotmatTvecdqdq(q, V_eci);
(void)B_eci;
(void)rho;
(void)dV_dq;
(void)d2V_dq2;
```

Wasted computation. Delete.

### 5. `q_current_unused` parameter (DELETE)

`processAttitudeTarget(attitude_target, boresight_body, q_current_unused)` — the third parameter is named `_unused` and the comment confirms it's not referenced. Three call sites pass `q` but it's ignored.

Delete from signature; update 3 call sites.

### 6. `magic_control_weight` (KEEP)

There's an existing memory note `project_magic_actuator_todo.md` indicating the magic-actuator type was stripped during port but should be restored. The config field is dormant *for now* but is a real TODO. Keep it.

## Cleanup order

This PR (vec-pointing math + cleanup):
1. Delete `omega_ref` / `omega_ff` machinery
2. Delete `q_current_unused`
3. Delete unused disturbance derivatives
4. Add documentation comment for `w_avmag` Hessian gap
5. Verify all 46 FD tests pass
6. Smoke test on 00_baseline

Other PRs (separate):
- GN flag (own PR)
- `magic_control_weight` activation (when magic actuator is restored)
