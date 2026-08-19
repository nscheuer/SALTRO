# MC dataset generation for imitation learning

Generates MTQ-only trajectory-optimization demonstrations: BeaverCube-2 (3U)
with 3x 0.4 A·m² body-axis MTQs at 100% effective dipole, circular 550 km SSO,
random epoch / initial quaternion / goal quaternion (rest goal) / initial rate
(uniform direction, half-normal magnitude, 1 deg/s at 1 sigma), 1000 s
trajectories at dt=10, IGRF-13 + J2 (hardcoded in `saltro_py.trajOpt`),
no disturbances.

## Setup (Linux / Threadripper)

```bash
git clone -b batch/imitation-datagen https://github.com/nscheuer/SALTRO.git
cd SALTRO
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt matplotlib
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSALTRO_BUILD_PYTHON=ON \
      -DSALTRO_WARNINGS_AS_ERRORS=OFF -DPYBIND11_FINDPYTHON=ON \
      -DPython_EXECUTABLE=$PWD/venv/bin/python
cmake --build build -j
(cd build && ctest)            # expect 48/48
(cd tests && pytest unit -q)   # expect ~961 passed
```

First configure needs network (FetchContent pulls Eigen 3.4.0 + pybind11).
`-DSALTRO_WARNINGS_AS_ERRORS=OFF` is only needed if the compiler trips the
pre-existing sign-conversion warnings (clang does; gcc may not).

## Run

The solver is single-threaded and holds the GIL, so parallelism is
process-level. On a 64-core machine leave a few cores for the OS/writer:

```bash
nohup python mc_datagen/gen_dataset.py \
    --n-trials 100000 --workers 60 --out /data/saltro_mc_v1 \
    > /data/saltro_mc_v1.log 2>&1 &
```

- **Resume**: re-running with the same `--out` skips trials already in
  `index.jsonl` (seeds are a pure function of `--seed` + trial index, so a
  resumed run produces identical trials).
- **Throughput**: ~0.1 s/solve → a 64-core day is tens of millions of
  core-seconds; 100k trials take well under an hour. Size `--n-trials` to
  what you want, not to the wall clock.
- **Storage**: ~14 MB per 1000 trials (`.npz` per trial: X, U, B, R, V +
  sampled inputs; `--save-gains` adds TVLQR K, ~2x size).
- **Failures**: `trajOpt` raises on non-convergence; failures are caught and
  logged per-trial in `index.jsonl` (`status: failed`), never kill the batch.

## Cost recipe (defaults; pilot-validated on this exact case, 2026-08-19)

Running `angle=1, ang_vel=1e2, mtq_control_weight=1e-1`; terminal
`angle_N=100, ang_vel_N=1000`; `ang_cost_func_type=5` (pseudo-Huber,
delta=0.35); IntegratedBdot warm start; `outer_iters=30`; sun keepout
disabled (`sun_limit_angle=0`; random goals must be feasible);
`control_limit_scale=1.0`.

Validation at n=1000: 100% convergence, final error p50/p90/p99 =
0.9/3.3/8.3 deg, final rate p90 = 0.5 deg/s, 97% of trials fully spike-free
(no isolated control jumps, no rail chatter), solve p90 = 0.14 s.

Notes from the pilot grid:
- Terminal weights are load-bearing: without them the optimizer "converges"
  with 80% of trials ending >10 deg from goal at these short horizons.
- dt=10 beat dt=5/dt=2 on final error at fixed iteration caps (more knots
  make the caps bind) and is 3-6x cheaper. Controls are smooth at dt=10.
- `ang_vel=1e1, mtq_weight=1e-2` (the older 1U recipe) converges but
  chatters at the rails on BC-2 — don't use it here.
- Alternative `--cost-type 3` (squared angle) gives slightly tighter
  mid-course tracking, slightly worse finals/smoothness.
- ~1/3 of trials show large post-acquisition excursions ("momentum tours"):
  with a random initial rate the vehicle often cannot brake at first goal
  passage — real underactuated physics, not solver artifacts (4% at w0=0).
  Cull via `analyze.py`'s excursion metric if unwanted.

## Analyze

```bash
python mc_datagen/analyze.py /data/saltro_mc_v1 --plot sample.png
```
