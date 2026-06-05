# Spike Removal — usage guide

The spike removal pass detects "spike" candidates in the inner-loop trajectory
and replaces them with a PD-controlled segment, blended back into the iLQR
trajectory tail with feedback-gain correction. It runs once per accepted
forward pass when enabled. **Default: disabled** — no behavior change unless
explicitly opted in.

## When to enable it

Spike removal is targeted at the failure mode where the inner iLQR exhibits
transient pointing-error excursions ("spikes") around SO(3) homotopy artifacts
during early outer-loop iterations — for instance, attitude slews through the
180° antipode where the cost landscape has a removable singularity that some
homotopies traverse via a brief blow-up of pointing error before settling.

Signatures of a spike candidate the detector looks for:

- a run of consecutive *increasing* pointing-error knots,
- preceded by a period of *decreasing* PE (i.e. the trajectory had been
  converging),
- with a peak-to-entry ratio above `min_spike_ratio` (default 3.0),
- that eventually returns to *near* the entry-error level (within `exit_fudge`),
- and is **not** physics-limited (actuator saturated while opposing the spike
  correctly — that's good behavior, not a homotopy artifact).

When detected, the spike window is replaced with a PD-controlled segment using
`saltro::controller::PDController` (allocator handles MTQ + RW distribution;
see `rw_scale`), then blended back into the iLQR trajectory tail over
`blend_len` knots with feedback-gain correction (the saved nominal controls
`U_bar` + feedback gains `K` rebuild the tail). The rest of the trajectory is
left untouched.

## Enabling it from Python

```python
import saltro_py as S

settings = S.PlannerSettings()
# ... configure your problem as usual ...

# Turn on spike removal for pass 0
settings.passes[0].spike_removal.enabled = True

# Optional tuning (defaults shown)
spike = settings.passes[0].spike_removal
spike.start_at_iter            = 2     # don't intervene before iter 2
spike.max_intervention_iters   = 5     # cap total interventions per pass
spike.blend_len                = 30    # knots of blend back into iLQR tail
spike.goal_switch_buffer       = 15    # skip spikes near goal transitions
spike.min_consecutive          = 7     # min run of consecutive PE increase
spike.exit_fudge               = 2.0   # how close to entry PE to count "exited"
spike.min_prior_decrease_knots = 10    # require convergence before the spike
spike.min_spike_ratio          = 3.0   # min peak/entry PE ratio
spike.max_spike_knots          = 0     # 0 = no limit; skip spikes wider than this
spike.kp_q                     = 0.3   # PD proportional gain (rad)
spike.kd_w                     = 2.0   # PD damping gain
spike.rw_scale                 = -1.0  # auto-split MTQ/RW (-1=auto, 0=MTQ only, 1=RW eq.)
spike.omega_max                = 0.0   # 0 = no rate cap on PD substitution
spike.verbose                  = False # set True to log detector decisions
```

## Recommended starting point

For most slew problems, defaults work. For high-cost-weight cases that exhibit
visible mid-trajectory spikes, enable it with verbose logging on the first
run so you can see what the detector catches:

```python
spike.enabled = True
spike.verbose = True
```

Inspect the resulting trajectory; if the detector intervenes incorrectly,
tighten the gates (raise `min_spike_ratio`, raise `min_consecutive`, lower
`max_spike_knots`).

## Interaction with the rest of the optimizer

- Runs **once per accepted iLQR forward pass**, between forward and the next
  backward pass — so it does not directly interfere with backward-pass logic.
- Uses **PDController** to generate the substitute segment. The PD gains are
  tuned to clamp the spike, not to be optimal — iLQR continues to refine the
  surrounding trajectory in subsequent iterations.
- Does **not** affect convergence detection. The modified trajectory is simply
  the next iterate; convergence checks fire on cost changes as usual.
- Honors `start_at_iter` and `max_intervention_iters` so it can't run too
  early (before iLQR has settled the basin) or too often (oscillating with
  the optimizer).

## Default-off

`SpikeRemovalConfig::enabled = false` by default. Without that flag, spike
removal is a no-op at every callsite — no detection runs, no PD allocation,
no overhead, no behavior change.

## See also

- `include/saltro/optimizer/spike_removal.h` — public API (`detectSpikes`, `applySpikeRemoval`).
- `tests/unit/optimizer/test_spike_removal.py` — unit tests, including injected-spike scenarios.
- `include/saltro/pybind/controller/pdcontroller.h` — `PDController` used for the substitute segment.
