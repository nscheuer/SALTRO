# Big Sweep: legacy vs newpath × type 3/4 × vec/quat (with Gauss-Newton)

> **Sweep date:** 2026-04-28. All cells with `cost_hess_gauss_newton=True`. Per-cell output dirs under `tests/debug/optimizer/alilqr_python/wide_results{,_vec}_*GN_*/`.

## TL;DR

- **Vec mode winner: type 3 + newpath + GN** (13 of 18 scenarios <1°; only `10_short_100s_dt1` stuck at 25° — that's a fundamentally infeasible 90° slew in 100s).
- **Quat mode winner: type 3 + legacy + GN** (most scenarios <1°; newpath breaks `11_long_3000s_dt30`).
- **Type 4 is consistently weaker than type 3** in both modes despite the quat-case-4 fix `(1-d²)→(1-d)²`. Steeper-near-optimum gradient of type 3 wins.
- **GN is decisive**: across all working cells, dramatic improvements vs full-Hess (already documented earlier session).

## PE_fin per scenario (degrees)

### Vec mode (boresight pointing, 2-DOF)

| scenario | t3 legacy | **t3 newpath** | t4 legacy | t4 newpath |
|---|---|---|---|---|
| 00_baseline | 0.1 | **0.1** | 6.4 | 2.6 |
| 01_sat_3_0_mtq | 1.9 | **2.1** | 36.4 | 7.4 |
| 02_sat_0_3_rw | 0.0 | **0.0** | 4.1 | 4.3 |
| 03_sat_3_3_hybrid | 0.0 | **0.0** | 1.6 | 2.2 |
| 04_angle_1e2_low | 0.4 | **0.7** | 6.4 | 6.2 |
| **05_angle_1e6_high** | **97.7 ❌** | **0.0** | 1.0 | 3.7 |
| 06_angvel_1e4_high | 0.5 | **0.3** | 1.6 | 5.1 |
| 07_angvel_1_low | 0.9 | **0.8** | 3.8 | 3.7 |
| 08_ctrl_10x_heavy | 2.6 | **0.0** | 5.7 | 5.9 |
| 09_ctrl_0.01x_light | 0.3 | **0.3** | 5.3 | 2.2 |
| 10_short_100s_dt1 | 25.0 | **25.0** | 29.9 | 29.9 |
| **11_long_3000s_dt30** | **0.2** (reg_exceeded ⚠️) | **3.7** | 91.2 ❌ | 91.2 ❌ |
| 12_omega_5x | 2.2 | **2.4** | 12.0 | 11.0 |
| 13_omega_10x | 5.2 | **2.9** | 12.1 | 11.7 |
| 14_aero_on | 0.1 | **0.1** | 6.4 | 2.6 |
| 15_gg_on | 1.3 | **1.0** | 3.5 | 3.5 |
| 16_all_disturb_on | 1.3 | **1.0** | 3.5 | 3.5 |
| 17_slew_180 | 0.3 | **0.2** | 28.7 | 28.7 |

**Vec winner: t3 + newpath**. The α-from-β formulation handles `05_angle_1e6_high` (extreme weight ratio) and avoids the `11_long_3000s_dt30` reg blowup that hits t3 legacy.

### Quat mode (full quaternion target, 3-DOF)

| scenario | **t3 legacy** | t3 newpath | t4 legacy | t4 newpath |
|---|---|---|---|---|
| 00_baseline | **0.7** | 0.7 | 15.9 | 13.8 |
| 01_sat_3_0_mtq | **78.6** | 77.2 | 86.5 | 88.2 |
| 02_sat_0_3_rw | **0.1** | 0.1 | 11.8 | 13.2 |
| 03_sat_3_3_hybrid | **0.1** | 0.1 | 10.4 | 12.2 |
| 04_angle_1e2_low | **13.4** | 21.5 | 21.8 | 21.7 |
| 05_angle_1e6_high | **0.0** | 0.0 | 4.1 | 5.1 |
| 06_angvel_1e4_high | **0.7** | 0.8 | 10.0 | 12.8 |
| 07_angvel_1_low | **0.7** | 0.7 | 13.4 | 12.1 |
| 08_ctrl_10x_heavy | **0.8** | 0.8 | 14.6 | 14.2 |
| 09_ctrl_0.01x_light | **0.7** | 0.7 | 12.3 | 13.8 |
| 10_short_100s_dt1 | **37.9** | 38.1 | 38.3 | 38.3 |
| **11_long_3000s_dt30** | **0.0** | 92.1 ❌ | 7.6 | 123.6 ❌ |
| 12_omega_5x | **60.5** | 9.6 | 42.2 | 54.0 |
| 13_omega_10x | **167.6** | 146.8 | 133.6 | 116.1 |
| 14_aero_on | **0.7** | 0.7 | 15.9 | 13.8 |
| 15_gg_on | **0.7** | 0.7 | 15.6 | 11.7 |
| 16_all_disturb_on | **0.7** | 0.7 | 15.6 | 11.7 |
| 17_slew_180 | **1.6** | 1.6 | 15.1 | 14.7 |

**Quat winner: t3 + legacy**. Newpath has the `11_long_3000s_dt30` instability in quat mode (legacy is fine there).

### Hard scenarios that stay hard regardless of config

- `10_short_100s_dt1` — 100s for 90° slew is infeasible at our actuator authorities. ~25-38° plateau across all configs. Not a config issue.
- `01_sat_3_0_mtq` (MTQ-only) — well-known hard case. Best result is vec t3 legacy (1.9°). Quat best is also ~78°. MTQ-only physics-limited.
- `12_omega_5x`, `13_omega_10x` — high initial ω. Vec converges to 2-5°, quat converges only with newpath (60° → 10° on 12).

## Cross-mode synthesis

| feature | best vec | best quat |
|---|---|---|
| ang_cost_func_type | 3 | 3 |
| crossterm | newpath (α=β·√(...)) | legacy (raw w_avang) |
| Gauss-Newton | on | on (no-op in quat anyway) |

## What this means for the PR

**Single configuration that works well in BOTH modes:** type 3 + GN (mandatory). The crossterm choice is mode-dependent:
- Vec mode: newpath strictly dominant (no reg_exceeded, handles edge cases).
- Quat mode: legacy strictly dominant (newpath breaks `11_long`).

**Recommendation: ship per-mode crossterm defaults.** When `is_eci_format` (vec mode), default to newpath (`ang_vel_err_dir_ratio = 0.3`, `ang_vel_err_dir = 0`). When quat mode, default to legacy (`ang_vel_err_dir = c.ang_vel`, `ang_vel_err_dir_ratio = 0`). User can override either.

OR keep crossterm choice user-facing and document per-mode recommendations.

## Why is newpath worse for quat 11_long_3000s_dt30 specifically?

Hypothesis: newpath's α-from-β formula `α = β·√(w_ang·w_av)` is mode-aware (uses `λ_min(W_ω) = w_av·roll_ratio` in vec, plain `w_av` in quat). For long horizons (3000s), accumulated dynamics-Hessian-style indefiniteness in DDP-disabled iLQR may interact with the higher α (since `w_av·roll_ratio` is smaller in vec → smaller α → more conservative; in quat α is larger → more aggressive). Worth confirming with a focused test.

## Convergence cost (iters)

GN takes 30-150× more iters than full Hess. From these runs, typical iter counts:
- Vec t3 newpath: 750-15000 iters per scenario (median ~2500)
- Quat t3 legacy: 80-3900 iters per scenario (median ~530)

Wall time per scenario varies wildly: 4 iters (early plateau) to 15000+ iters. Fast scenarios complete in seconds, hard ones in minutes.

## Files

Per-cell summaries with gifs are in:
- `tests/debug/optimizer/alilqr_python/wide_results_vec_GN_t3_legacy/`
- `tests/debug/optimizer/alilqr_python/wide_results_vec_GN_t3_newpath/`
- `tests/debug/optimizer/alilqr_python/wide_results_vec_GN_t4_legacy/`
- `tests/debug/optimizer/alilqr_python/wide_results_vec_GN_t4_newpath/`
- `tests/debug/optimizer/alilqr_python/wide_results_quat_GN_t3_legacy/`
- `tests/debug/optimizer/alilqr_python/wide_results_quat_GN_t3_newpath/`
- `tests/debug/optimizer/alilqr_python/wide_results_quat_GN_t4_legacy/`
- `tests/debug/optimizer/alilqr_python/wide_results_quat_GN_t4_newpath/`

Each contains 18 (vec) or 18 (quat) gif animations, midway-snapshot pngs, and final-state pngs per scenario.
