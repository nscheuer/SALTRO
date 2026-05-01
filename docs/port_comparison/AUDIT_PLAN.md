# Port-comparison audit — full plan (run overnight)

Goal: Build `docs/port_comparison/PORT_AUDIT.md` — a single subsystem-organized
document showing where each subsystem drifted across the 5 sources, ranked
by suspected impact on convergence/correctness.

## The 5 sources

| # | Source | Path |
|---|---|---|
| 1 | **MATLAB** (PhD original) | `docs/port_comparison/phd_reference/unzipped/**/*.m`. Key: `ALiLQR_BC_PM.m`, `ALILQR_GITHUB.m`, `ALiLQR_complextraj.m` (in `unzipped/beavercube-adcs-master_thesis/`) |
| 2 | **Early C++/Py** | `docs/port_comparison/phd_reference/unzipped/re/altro_general.cpp`, `altro_beavercube.cpp`, `altro_helper.cpp`. Note: `bc/bc/TrajectoryPlanning/MatrixTest/` has same files (verify identical) |
| 3 | **Recent PhD C++** (now local, fetched via gh API) | `docs/port_comparison/phd_reference/OldPlanner.{cpp,hpp}`, `Satellite.{cpp,hpp}`, `PlannerUtil.cpp`, `PyPlanner.hpp`. Source-of-truth lives at `github.com/patrickmckeen/PhD_Dissertation_Code/GeneralizedADS/ADCS/trajectory_planner/src/planner/` |
| 4 | **saltro current branch** (`PKMN_antispike`, working tree) | `c:/Users/LV - Patrick McKeen/saltro/{include,src,python,tests}/...` |
| 5 | **saltro main branch** | `git show main:<path>` for any file |

## Subsystem outline (sections of the audit doc)

For each subsystem, fill out a small per-source matrix (file:line, formula or
key code, and a "drift note"):

1. **Cost — state-side angle** (vec & quat): cases of `whichAngCostFunc` / `ang_cost_func_type`, formula for `sc_ang`, `d_sc_ang`, `dd_sc_ang`. Highlight: case 0/1/2/3 mappings, which is default, factor-of-2 differences.
2. **Cost — state-side ω**: isotropic `0.5·w_av·‖ω‖²`, ω-cross-pointing terms (`w_avang` linear vs our `α·err_dir·(ω−ω_ff)` quadratic), ω-along-B (`w_avmag` — flag as deprecated).
3. **Cost — RW**: angular-momentum softplus penalty, stiction term. Verify the softplus thresholds and shapes match.
4. **Cost — control-side**: raw-control vs delta-control branching (`useRawControlCost`), per-actuator weights (mtq/rw/magic), `act_cost_mat` shape, k=0 / k=N-1 special cases.
5. **Cost Jacobians/Hessians representation**: 3D reduced vs 4D ambient. Specifically the conversion via `findGMat`/`findWMat`/`state_norm_jacobian` (PhD) vs PwA correction `−grad_dot_q · I_4` (saltro). Verify our BP consumes a representation that's **mathematically equivalent** to what PhD passes — sign and projection mechanics included.
6. **Dynamics**: integrator (RK4 details, substeps), torque sources (MTQ/RW/magic — note: magic stripped in saltro per `project_magic_actuator_todo.md`), bias/noise treatment.
7. **Dynamics Jacobians/Hessians**: how A_q, B_q are computed, presence of full F_xx/F_xu/F_uu tensors, whether DDP is used.
8. **AL/iLQR loop**: backward-pass Q_uu assembly (line 2167 in PhD), regularization growth/shrink schedule, line search criteria, BP failure handling (eigen-modification? abort? regrow?).
9. **Constraints**: terminal pointing as cost-only (PhD line 935: `if(ECIvec_k.is_zero()) w_ang = 0`) vs AL terminal constraint, control saturation, RW-AM softplus.
10. **Warm-start**: `smartbdot`, PD, Excitation. Per source: which warm-start exists, what `initcontroller` codes mean.
11. **Spike removal / homotopy**: present in saltro only — no PhD analog. Document as a saltro-specific layer.
12. **Numerics**: tolerances, dt, step counts, default weights.

## Output format

Each subsection: 
- One-paragraph overview of what the subsystem does.
- Matrix: rows = sources, columns = (file:line, formula, behavior diff vs source N+1).
- "Drift note": one paragraph explicitly describing where the math/behavior diverged.
- "Convergence-impact estimate": low / medium / high.

Final section: **Actionable port discrepancies** ranked high→low impact, each
with a one-line proposed fix or follow-up experiment.

## Execution

Dispatch in parallel via Explore agents (one per source-pair to keep
context tractable):

- **Agent A**: MATLAB ↔ early-C++ (`altro_general.cpp` family). Output: subsections 1–6 across these two sources.
- **Agent B**: early-C++ ↔ recent-PhD (`OldPlanner.cpp`+`Satellite.cpp`). Output: same subsections plus 7-9.
- **Agent C**: recent-PhD ↔ saltro current branch. **Highest priority** — this is where active drift sits. All 12 subsections.
- **Agent D**: saltro current ↔ main. Smaller diff, all subsections, focused on what's in-flight on `PKMN_antispike`.

Each agent writes its section into `docs/port_comparison/PORT_AUDIT_part_<X>.md`. Final synthesis pass merges into `PORT_AUDIT.md` with a "TL;DR" / executive summary at the top.

## Estimated scale

~30–60 min wall, ~4 parallel agents, ~3–5k word final doc. Each agent
produces ~800–1500 words of structured comparison. Final synthesis pass
removes duplication and ranks the actionable items.

## Trigger phrase to start

When ready, say something like "run the port audit" and I'll dispatch the 4
Explore agents in parallel and synthesize when they return.
