# Inner-solve failure-mode diagnostic (2026-05-05)

Findings from running [inner_solve_diagnostic.py](../tests/debug/optimizer/alilqr_python/inner_solve_diagnostic.py) on four scenarios with `SALTRO_FP_VERBOSE=1`.

All runs **without spike removal** (`use_spike=False`) — exposes raw inner-solve failure patterns that spike removal otherwise masks in production.

## Summary table

| scenario | iters | stop | bp_fails | fp_fails | ls_rejects | rollout_fails | neg-z | %neg-z |
|---|---|---|---|---|---|---|---|---|
| 10_short_dt1_ict1 | 580 | converged @ 40°PE | 0 | 38 | 8009 | 3 | 548/563 | **97%** |
| 11_long_dt30_ict1 | 374 | penalty_max | 290 | 2 | 116 | **2421** | 79/371 | 21% |
| 12_5x_ict1 | 2041 | penalty_max @ 63°PE | 0 | 262 | 13470 | 7573 | 1934/2026 | **95%** |
| 12_5x_ict1e-3 | 2101 | penalty_max @ 127°PE | 0 | 409 | 16729 | 6753 | 2005/2087 | **96%** |

## Two distinct failure modes

### Mode A — Cost-side Q_uu indefiniteness (`10_short`, `12_omega_5x`)

**Fingerprint:**
- 95–97% of iters have **negative `last_z`** (BP-proposed direction ascends in AL cost)
- **0 BP failures** — the BP itself succeeds; reg stays at floor
- High LS-reject count (8k–17k); LS rescues with tiny α (down to 1e-7)
- AL constraint violation stalls at ~9 (9000× tolerance), μ ratchets to 1e15
- Eventually exits via `ls_attempts_exceeded` after cost grows to 1e15–1e16

**Mechanism:**
The cost Hessian `lxx` (specifically the (q,q) block + off-diagonal cross-terms) is producing an *indefinite* Q_uu after Riccati propagation. BP returns a direction along a negative-eigenvalue mode — the predicted ΔV is negative (descent claimed) but actual cost increases at α=1. LS rescues by shrinking α until a coincidental nearby trajectory is slightly better. Progress is glacial.

**Why ict=1.0 vs 1e-3 doesn't change the root cause:** the curvature is wrong regardless of when we exit the inner loop.

**Why this doesn't kill convergence in production:** **spike removal**. When the optimizer gets stuck in negative-z LS-fishing, spike removal substitutes a PD-driven trajectory and breaks out. Without spike removal (this diagnostic), the optimizer fights itself indefinitely. The synced sweep run with spike removal converged 12_5x_ict1 in 396 iters at 3°PE — vs 2041 iters / 63°PE here.

### Mode B — Dynamics-side Q_uu indefiniteness (`11_long_dt30`)

**Fingerprint:**
- 21% neg-z (much lower than Mode A)
- **290 BP failures** — Q_uu non-PD even after regularization
- **2421 rollout_fails** — RK4 dynamics produce NaN / invalid quaternions
- Ends in a `reg_exceeded` loop (BP keeps failing)
- Constraint violation stalls at 78 (78,000× tolerance)

**Mechanism:**
With dt=30s and τ_max=1e-3 N·m, J=0.07 kg·m², the per-knot torque kick is ~0.43 rad/s — way past the natural rotational timescale. RK4 step error compounds, dynamics linearization (used by BP) is grossly inaccurate, Q_uu inherits the integration error. Regularization can't rescue because the underlying Hessian is wildly wrong.

**Fix path:** auto-substep in `rk4_step` (deferred plan in [memory](../../.claude/projects/c--Users-LV---Patrick-McKeen-saltro/memory/project_auto_substep_idea.md)). Take `n_substeps = ceil(dt / dt_natural)` where `dt_natural = 0.1·J/(τ_max·ω_scale)`.

## Implications

1. **The ict=1.0 → 1e-3 regression on 12_5x is not a Q_uu issue per se** — same indefiniteness pattern at both tolerances. The visible regression must be a **spike-removal × ict interaction**: tighter ict changes when the inner exits, which shifts when spike removal triggers, which produces a different intervention trajectory. Confirming this needs a 4-way A/B (with/without spike × ict=1/1e-3).

2. **Mode A (cost-side) is the prevalent failure** across most "high-omega"/"large slew" scenarios. It is currently masked by spike removal but causes the slow convergence and the fragile A/B behavior we see. Fix path: PSD-clip on cost-side Q_uu (analogous to existing `psd_clip_quu_ddp` for DDP), or eigen-modification on the assembled iLQR Q_uu.

3. **Mode B (dynamics-side) is dt-dependent.** Will keep biting at long horizons until auto-substep ships.

## Confirming next step

Wire Q_uu eigenvalue logging from BP (per [deep-twirling-minsky plan](../../.claude/plans/deep-twirling-minsky.md)). Per-iter, per-knot λ_min(Q_uu) trace:
- Mode A scenarios: should show λ_min(Q_uu) < 0 at iters where last_z < 0. Confirms the cost Hessian assembly is producing indefinite Q_uu.
- Mode B scenarios: should show λ_min(Q_uu) < 0 paired with BP failures (reg can't restore PD-ness).

The plan was written for DDP diagnosis but applies identically to iLQR. Execute it next.
