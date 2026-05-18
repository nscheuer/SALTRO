# saltro wide_results gallery

Mobile-friendly view of the wide-test-runner outputs for the `PKMN_antispike` branch baseline. Each scenario has three artifacts:

- `_final.png` — converged trajectory snapshot (PE / quaternion / controls / ω)
- `_midway.png` — ~10 iteration snapshots showing optimizer evolution
- `.gif` — full convergence animation

Generated 2026-05-18 from baseline commit on `PKMN_antispike`.

---

## 00_baseline

![](wide_results/00_baseline_final.png)

[full midway grid](wide_results/00_baseline_midway.png) · [gif](wide_results/00_baseline.gif)

## 01_sat_3_0_mtq — MTQ-only stress (auto-skip)

![](wide_results/01_sat_3_0_mtq_final.png)

[midway](wide_results/01_sat_3_0_mtq_midway.png) · [gif](wide_results/01_sat_3_0_mtq.gif)

## 02_sat_0_3_rw — RW-only

![](wide_results/02_sat_0_3_rw_final.png)

[midway](wide_results/02_sat_0_3_rw_midway.png) · [gif](wide_results/02_sat_0_3_rw.gif)

## 03_sat_3_3_hybrid — 3 MTQ + 3 RW

![](wide_results/03_sat_3_3_hybrid_final.png)

[midway](wide_results/03_sat_3_3_hybrid_midway.png) · [gif](wide_results/03_sat_3_3_hybrid.gif)

## 04_angle_1e2_low — light angle weight

![](wide_results/04_angle_1e2_low_final.png)

[midway](wide_results/04_angle_1e2_low_midway.png) · [gif](wide_results/04_angle_1e2_low.gif)

## 05_angle_1e6_high — heavy angle weight (eigen-mod stress)

![](wide_results/05_angle_1e6_high_final.png)

[midway](wide_results/05_angle_1e6_high_midway.png) · [gif](wide_results/05_angle_1e6_high.gif)

## 06_angvel_1e4_high

![](wide_results/06_angvel_1e4_high_final.png)

[midway](wide_results/06_angvel_1e4_high_midway.png) · [gif](wide_results/06_angvel_1e4_high.gif)

## 07_angvel_1_low

![](wide_results/07_angvel_1_low_final.png)

[midway](wide_results/07_angvel_1_low_midway.png) · [gif](wide_results/07_angvel_1_low.gif)

## 08_ctrl_10x_heavy

![](wide_results/08_ctrl_10x_heavy_final.png)

[midway](wide_results/08_ctrl_10x_heavy_midway.png) · [gif](wide_results/08_ctrl_10x_heavy.gif)

## 09_ctrl_0.01x_light

![](wide_results/09_ctrl_0.01x_light_final.png)

[midway](wide_results/09_ctrl_0.01x_light_midway.png) · [gif](wide_results/09_ctrl_0.01x_light.gif)

## 10_short_100s_dt1 — physics-bound (short t)

![](wide_results/10_short_100s_dt1_final.png)

[midway](wide_results/10_short_100s_dt1_midway.png) · [gif](wide_results/10_short_100s_dt1.gif)

## 11_long_3000s_dt30 — long dt

![](wide_results/11_long_3000s_dt30_final.png)

[midway](wide_results/11_long_3000s_dt30_midway.png) · [gif](wide_results/11_long_3000s_dt30.gif)

## 12_omega_5x

![](wide_results/12_omega_5x_final.png)

[midway](wide_results/12_omega_5x_midway.png) · [gif](wide_results/12_omega_5x.gif)

## 13_omega_10x — high-tumble stress

![](wide_results/13_omega_10x_final.png)

[midway](wide_results/13_omega_10x_midway.png) · [gif](wide_results/13_omega_10x.gif)

## 14_aero_on

![](wide_results/14_aero_on_final.png)

[midway](wide_results/14_aero_on_midway.png) · [gif](wide_results/14_aero_on.gif)

## 15_gg_on

![](wide_results/15_gg_on_final.png)

[midway](wide_results/15_gg_on_midway.png) · [gif](wide_results/15_gg_on.gif)

## 16_all_disturb_on

![](wide_results/16_all_disturb_on_final.png)

[midway](wide_results/16_all_disturb_on_midway.png) · [gif](wide_results/16_all_disturb_on.gif)

## 17_slew_180

![](wide_results/17_slew_180_final.png)

[midway](wide_results/17_slew_180_midway.png) · [gif](wide_results/17_slew_180.gif)

## 18_omega_10x_rw_aligned — RW axis aligned with spin (stress)

![](wide_results/18_omega_10x_rw_aligned_final.png)

[midway](wide_results/18_omega_10x_rw_aligned_midway.png) · [gif](wide_results/18_omega_10x_rw_aligned.gif)

## 19_omega_10x_rw_perp — RW perpendicular to spin (stress)

![](wide_results/19_omega_10x_rw_perp_final.png)

[midway](wide_results/19_omega_10x_rw_perp_midway.png) · [gif](wide_results/19_omega_10x_rw_perp.gif)

## 20_omega_10x_mtq_only

![](wide_results/20_omega_10x_mtq_only_final.png)

[midway](wide_results/20_omega_10x_mtq_only_midway.png) · [gif](wide_results/20_omega_10x_mtq_only.gif)

## 21_vector_point — fixed inertial target vector

![](wide_results/21_vector_point_final.png)

[midway](wide_results/21_vector_point_midway.png) · [gif](wide_results/21_vector_point.gif)

## 22_sequential_45_90 — slew to A then to B

![](wide_results/22_sequential_45_90_final.png)

[midway](wide_results/22_sequential_45_90_midway.png) · [gif](wide_results/22_sequential_45_90.gif)

## 23_nadir_track — time-varying nadir pointing

![](wide_results/23_nadir_track_final.png)

[midway](wide_results/23_nadir_track_midway.png) · [gif](wide_results/23_nadir_track.gif)
