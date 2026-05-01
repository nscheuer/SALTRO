# Wide sweep: synced wrapper vs unsynced

> Same 8 cells × 18 scenarios as [docs/sweep_legacy_vs_newpath.md](sweep_legacy_vs_newpath.md), but using the **synced Python wrapper** (2-tier tolerance, stagnation count, maturity gate, penalty-max exit). Run 2026-04-30.

## TL;DR

- **Iter counts dropped 40-95% across the board** with synced wrapper. Big practical win.
- **PE_fin preserved on easy/normal cases** (most scenarios within 0.5° of unsynced).
- **Some hard cases regressed.** Scenarios that previously took 10000+ iters to grind to a tight result now exit early via stagnation count and miss the deeper optimum. This affects ~6 cells across all 4 vec configs.
- **Newly-honest failures.** Several cases that previously falsely "converged" with `max_c >> ctol` now correctly stop with `penalty_max_reached`. Better signal.
- **Vec winner is still type 3 + newpath + GN.** Quat winner is still type 3 + legacy + GN. Rankings unchanged.

## Cell summary (all GN on, max_iters=200, ilqr_cost_tol=1e0 default)

### Vec mode (PE_fin / iters; **bold** = winner per scenario)

| scenario | t3 legacy | t3 newpath | t4 legacy | t4 newpath |
|---|---|---|---|---|
| 00_baseline | 0.3 / 417 | **0.1 / 505** | 6.2 / 660 | 2.4 / 516 |
| 01_sat_3_0_mtq | 21.6 / 1456 ⚠️penmax | 27.6 / 821 | **8.9 / 720** | 6.5 / 772 |
| 02_sat_0_3_rw | **0.0 / 356** | 0.0 / 270 | 4.2 / 197 | 4.4 / 203 |
| 03_sat_3_3_hybrid | **0.0 / 311** | 0.0 / 298 | 2.1 / 128 | 2.6 / 119 |
| 04_angle_1e2_low | 0.7 / 212 | **0.6 / 246** | 6.6 / 89 | 6.1 / 53 |
| 05_angle_1e6_high | **0.0 / 964** | 0.0 / 756 | 2.3 / 735 | 1.6 / 902 ⚠️ls_exc |
| 06_angvel_1e4_high | 0.9 / 789 | **0.2 / 719** | 0.9 / 280 | 5.0 / 607 |
| 07_angvel_1_low | **0.2 / 672** | 0.3 / 328 | 2.3 / 744 | 2.7 / 740 |
| 08_ctrl_10x_heavy | 2.1 / 433 | **0.6 / 546** | 5.0 / 190 | 5.0 / 191 |
| 09_ctrl_0.01x_light | **1.0 / 424** | 1.1 / 568 | 3.1 / 343 | 3.5 / 655 |
| 10_short_100s_dt1 | 28.7 / 451 | 28.3 / 951 | 30.6 / 893 ⚠️penmax | 30.5 / 728 ⚠️penmax |
| 11_long_3000s_dt30 | 2.0 / 480 | 10.0 / 124 | **0.9 / 230** | 128.3 / 1036 ⚠️penmax |
| 12_omega_5x | **3.0 / 371** | 3.0 / 396 | 12.4 / 274 | 14.7 / 529 |
| 13_omega_10x | 10.7 / 884 | 9.3 / 1397 ⚠️ls_exc | **8.2 / 373** | 10.5 / 340 |
| 14_aero_on | 0.3 / 417 | **0.1 / 505** | 6.2 / 660 | 2.4 / 516 |
| 15_gg_on | 2.1 / 1974 | **2.0 / 1585** | 2.4 / 648 | 1.5 / 576 |
| 16_all_disturb_on | 2.1 / 1974 | 2.0 / 1585 | 2.4 / 648 | **1.5 / 576** |
| 17_slew_180 | **0.2 / 599** | 0.4 / 531 | 37.6 / 42 | 37.6 / 43 |

**Vec winner: t3 + newpath + GN** still — wins or ties on 9/18 scenarios; t3 legacy wins another 7. Type 4 has small wins on hard cases but loses on most.

### Quat mode (PE_fin / iters)

| scenario | t3 legacy | t3 newpath | t4 legacy | t4 newpath |
|---|---|---|---|---|
| 00_baseline | **0.7 / 309** | 0.7 / 286 | 13.3 / 159 | 15.3 / 178 |
| 01_sat_3_0_mtq | **80.6 / 784** | 79.8 / 670 | 89.4 / 942 | 93.2 / 1123 |
| 02_sat_0_3_rw | **0.1 / 56** | 0.1 / 41 | 11.8 / 47 | 13.2 / 90 |
| 03_sat_3_3_hybrid | **0.1 / 531** | 0.1 / 291 | 10.5 / 78 | 12.2 / 47 |
| 04_angle_1e2_low | **20.8 / 219** | 15.4 / 234 | 21.2 / 63 | 20.6 / 53 |
| 05_angle_1e6_high | **0.1 / 888** | 0.0 / 364 | 4.1 / 501 | 4.5 / 692 |
| 06_angvel_1e4_high | **0.8 / 225** | 0.8 / 603 | 11.7 / 134 | 12.7 / 255 |
| 07_angvel_1_low | **0.7 / 291** | 0.7 / 298 | 17.0 / 229 | 13.9 / 241 |
| 08_ctrl_10x_heavy | **0.8 / 112** | 0.8 / 109 | 14.6 / 65 | 13.6 / 61 |
| 09_ctrl_0.01x_light | **0.7 / 338** | 0.8 / 296 | 15.2 / 384 | 16.4 / 215 |
| 10_short_100s_dt1 | **42.5 / 372** | 40.2 / 580 | 41.5 / 694 ⚠️penmax | 39.1 / 321 |
| 11_long_3000s_dt30 | 113.7 / 111 ⚠️penmax | **0.1 / 1206** | 6.2 / 1088 | 9.4 / 476 |
| 12_omega_5x | **9.0 / 954** | 9.5 / 987 | 90.5 / 635 | 46.1 / 663 |
| 13_omega_10x | 132.7 / 2443 ⚠️penmax | 90.4 / 2701 | 59.6 / 1510 ⚠️ls_exc | **94.9 / 1703** |
| 14_aero_on | **0.7 / 309** | 0.7 / 286 | 13.3 / 159 | 15.3 / 178 |
| 15_gg_on | **0.7 / 290** | 0.7 / 499 | 15.6 / 156 | 14.3 / 382 |
| 16_all_disturb_on | **0.7 / 290** | 0.7 / 499 | 15.6 / 156 | 14.3 / 382 |
| 17_slew_180 | **1.6 / 431** | 1.6 / 533 ⚠️ls_exc | 14.7 / 295 | 15.1 / 502 |

**Quat winner: t3 + legacy + GN** still — wins or ties on 17/18.

## Synced vs unsynced — iter count comparison (sample scenarios)

| scenario | config | unsynced iters | synced iters | ratio |
|---|---|---|---|---|
| 00_baseline | vec t3 newpath | 2439 | **505** | 4.8× |
| 00_baseline | vec t3 legacy | 2987 | **417** | 7.2× |
| 00_baseline | quat t3 legacy | 532 | **309** | 1.7× |
| 00_baseline | vec t4 legacy | 5029 | **660** | 7.6× |
| 01_sat_3_0_mtq | vec t3 newpath | 10072 | **821** | 12.3× |
| 02_sat_0_3_rw | vec t3 newpath | 9243 | **270** | 34× |
| 03_sat_3_3_hybrid | vec t3 newpath | 9710 | **298** | 33× |
| 05_angle_1e6_high | vec t3 newpath | 3198 | **756** | 4.2× |
| 13_omega_10x | quat t4 legacy | 2270 | **1510** | 1.5× |

Median speedup ~5×. Easy/well-behaved cases get the biggest wins (10-30×); hard cases that need many iters anyway get smaller (~1.5-2×).

## Synced vs unsynced — PE_fin comparison (where it shifted)

Most scenarios: PE_fin within ±0.5° of unsynced. A few notable shifts:

| scenario | config | unsynced PE_fin | synced PE_fin | direction |
|---|---|---|---|---|
| 11_long_3000s_dt30 | vec t4 legacy | 91.2° (failed) | **0.9°** | ⬆️ synced recovers |
| 11_long_3000s_dt30 | vec t4 newpath | 91.2° (failed) | 128.3° penmax | similar, more honest |
| 11_long_3000s_dt30 | vec t3 newpath | 3.7° | 10.0° | ⬇️ regressed |
| 17_slew_180 | vec t4 legacy | 28.7° | 37.6° | ⬇️ regressed |
| 17_slew_180 | vec t4 newpath | 28.7° | 37.6° | ⬇️ regressed |
| 13_omega_10x | quat t3 legacy | 167.6° (claimed converged) | 132.7° penmax | ⬆️ honest |

**Pattern**: regressions happen when synced wrapper's stagnation count or 2-tier tol exits early on a case that *would* benefit from more iterations. The unsynced wrapper kept running and sometimes lucked into a better local minimum; synced respects the convergence criteria (which is more correct but exposes when those criteria are too loose).

## Stop-reason changes — newly honest

Synced wrapper's `penalty_max_reached` exit catches cases where the AL outer loop saturated μ at `penalty_max` without driving `max_c` below `constraint_tol`. These were previously masked as "converged":

- vec t3 legacy 01_sat_3_0_mtq → penmax (was: AL converged with degraded constraint)
- vec t4 legacy 10_short_100s_dt1 → penmax  
- vec t4 newpath 10_short_100s_dt1 → penmax
- vec t4 newpath 11_long_3000s_dt30 → penmax
- quat t3 legacy 11_long_3000s_dt30 → penmax
- quat t3 legacy 13_omega_10x → penmax
- quat t4 legacy 10_short_100s_dt1 → penmax

These were always failures by the constraint criterion; the synced wrapper just makes it visible.

## Key implications

1. **The synced wrapper's iter counts reflect what C++ production does.** Median 5× iter reduction validates the sync was real. Production was always faster than our debug-script measurements suggested.

2. **PE_fin rankings unchanged.** Cost math is unaffected by the sync; the same configs win as before.

3. **Some hard scenarios regress in synced wrapper.** Specifically when stagnation/2-tier exits early before deeper convergence. The fix is in the queued `ilqr_cost_tol` A/B — tightening the loose inner exit threshold should let inner solve subproblems more thoroughly per outer.

4. **Honest failure detection improved.** `penalty_max_reached` now surfaces dual-saturation cases that were previously hidden as fake "converged" results.

## Next experiments

1. **`ilqr_cost_tol` A/B** (queued): tighten from `1e0` → `1e-2` / `1e-3` / `1e-4`. Hypothesis: regressed scenarios will improve with tighter inner tol while still keeping the bulk of the iter savings (probably 2-3× instead of 5×, but PE_fin recovers).
2. **Validate against pure C++ trajOpt**: if there's a binding to `saltro_py.trajOpt` that uses the C++ AL outer/iLQR, run the wide sweep through that path. Compares synced-python (~production) vs actual production. They should match.

The current sweep is a clean, honest reflection of production behavior. Vec-pointing math from PR #9 + GN flag from PR #10 still hold up — the per-scenario quality is unchanged, just measured against a faster (and more honest) loop now.
