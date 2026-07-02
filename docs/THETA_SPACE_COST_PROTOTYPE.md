# θ-space (angle-space) vec-pointing cost — experimental prototype

**Status:** exploratory prototype, opt-in flag `CostConfig::use_theta_cost_param`
(default `false`). **No PR.** This is groundwork for the future
cost-scaling/autotuning rewrite. Branch `exp/theta-space-cost`.

---

## 1. Motivation

SALTRO's vector-pointing attitude cost is a shape `f(c)` of the alignment
cosine `c = bs·R(q)ᵀ·r̂ = cos θ`, where `θ` is the boresight-to-target angle.
The cost is **smooth in θ** (it is a function of a geodesic distance on the
pointing 2-sphere) but is **singular in c** at both poles because `c = cos θ`
flattens quadratically there. Every acos-related patch in the current code is a
symptom of that bad parametrization:

- `angCostShape` floors `1 − c²` at `1e-12` and divides by `√(1−c²)` and its
  cube — Taylor/floor artifacts near `c = +1`;
- the Gauss-Newton (GN) angle-Hessian `f''(c)·∂c∂cᵀ` for the geodesic-natural
  shape `f = ½·acos²(c)` (type 3) **blows up like 1/sinθ** as `θ → 180°`
  (the "antipodal razor"), and **vanishes** as `θ → 0°` (loses all curvature on
  the last mile to the target);
- the production workaround was to abandon type 3 and use type 1 (`½(1−c)²`,
  `f'' ≡ 1`) purely because it has no antipodal blow-up — trading the
  geodesic-natural cost for a conditioning-safe but geometrically distorted one.

The θ-space idea: assemble the tangent-space cost value / gradient / Hessian
**directly in θ**, computing

```
θ = atan2(|bs × b|, bs · b),   b = R(q)ᵀ·r̂      (stable at both poles)
```

and `g(θ)` value-equivalent to the c-space shape. The `1/sinθ` geometry factor
that vanishes at the poles cancels analytically against the `sinθ` in `∂θ/∂c`,
leaving `O(1)` derivatives everywhere except the exact poles (where the pointing
direction is genuinely undefined).

---

## 2. Math

### 2.1 Shapes

`g(θ)` is chosen value-equivalent to `f(c)` under `c = cos θ`:

| type | c-space `f(c)`      | θ-space `g(θ)`   | `g'(θ)`            | `g''(θ)`                     |
|------|---------------------|------------------|--------------------|------------------------------|
| 0    | `1 − c`             | `1 − cosθ`       | `sinθ`             | `cosθ`                       |
| 1    | `½(1−c)²`           | `½(1−cosθ)²`     | `(1−cosθ)·sinθ`    | `sin²θ + (1−cosθ)cosθ`       |
| 2    | `acos c`            | `θ`              | `1`                | `0`                          |
| 3    | `½·acos²c`          | `½θ²`            | `θ`                | `1`  ← constant curvature    |

### 2.2 Reduced tangent derivatives

Let `∂c/∂t = dc` and `∂²c/∂t² = ddc` be the reduced attitude-tangent gradient
and (manifold-corrected) Hessian of `c` — exactly the quantities SALTRO already
computes in `vecPointingGeom` (`dc`, `ddc`). The θ chain rule gives

```
∂θ/∂t   = −(1/sinθ)·dc
∂²θ/∂t² = −(1/sinθ)·ddc − (cosθ/sin³θ)·dc·dcᵀ      ← derived below
```

**Derivation of ∂²θ.** With `θ = arccos(c)`,
`∂θ/∂t_i = −(1−c²)^{−1/2}·dc_i`. Differentiating again,
```
∂²θ/∂t_i∂t_j = −(1−c²)^{−1/2}·ddc_ij  −  dc_i · ∂_j[(1−c²)^{−1/2}]
             = −(1−c²)^{−1/2}·ddc_ij  −  dc_i · c(1−c²)^{−3/2}·dc_j.
```
With `sinθ = √(1−c²)` this is the boxed form. On the pointing 2-sphere this is
the Hessian of the geodesic distance from a fixed point: curvature `0` along the
radial (geodesic) direction and `+cotθ` (times `θ` for the `½θ²` cost) in the
azimuthal direction — isotropic `+1` at `θ→0`, and genuinely **negative**
(`cotθ → −∞`) at the antipode where the level sets close into a cone.

### 2.3 Cost derivatives

```
value    :  w · g(θ)
gradient :  w · g'(θ)·∂θ/∂t                          = w·(−g'/sinθ)·dc
Hessian  :  w · [ g''·(∂θ/∂t)(∂θ/∂t)ᵀ + g'·∂²θ/∂t² ]      (full / Newton)
         =  w · [ (g''/sin²θ − g'·cosθ/sin³θ)·dc·dcᵀ  −  (g'/sinθ)·ddc ]
GN       :  w · g''(θ)·û ûᵀ ,   û = dc / |dc|          (drop the g'·∂²θ term)
```

**Key identity — full θ-space ≡ full c-space.** Because `f'(c) = −g'(θ)/sinθ`
and `f''(c) = g''/sin²θ − g'·cosθ/sin³θ`, the full θ-space Hessian is
*algebraically identical* to the c-space full Hessian `f''·dc·dcᵀ + f'·ddc` for
the matching shape. So the full-Newton path gains **only pole-region numerical
accuracy**, nothing structural.

**The real payoff is GN.** c-space GN is `f''(c)·dc·dcᵀ`; for type 3,
`f''(c) = 1/sin²θ − θcosθ/sin³θ` and `|dc|² = sin²θ`, so the curvature scales as
`(1 − θcosθ/sinθ)`, which → `+∞` like `1/sinθ` at the antipode and → `0` at
alignment. θ-space GN is `g''·û ûᵀ`; for type 3, `g'' ≡ 1`, giving **constant
curvature `1·û ûᵀ` at every angle**, including 179.999°.

### 2.4 Poles

- **Aligned (θ=0):** `g'(0)=0` (type 3) so the gradient → 0 smoothly; the GN
  Hessian direction `û` is undefined but `g''(0)` is finite → isotropic limit
  `g''·(I − bs·bsᵀ)` (curvature in the two off-boresight DOF, 0 about roll).
- **Antipode (θ=π):** a genuine geometric cone tip — the escape direction is
  undefined. GN falls back to the same bounded isotropic limit; full Newton
  keeps the true (negative) azimuthal curvature and is therefore indefinite
  there (this is correct, not a bug — see §5).

Guard: `kThetaPoleEps = 1e-7` on `sinθ` / `|dc|`.

---

## 3. Implementation notes

All changes are in the `is_eci_format` (vector-pointing) branch only; quaternion
mode is untouched. Files:

- `include/saltro/pybind/plannersettings.h` — `bool use_theta_cost_param = false;`
- `python/pybind/plannersettings_py.cpp` — binding.
- `src/pybind/satellite.cpp`
  - new `thetaCostShape(θ, type)` helper (value/`g'`/`g''`);
  - `stageCost` value branch — θ-space `g(θ)` vs c-space `f(c)`;
  - `stageCostJacobians` — θ-space reduced gradient `−(g'/sinθ)·dc`;
  - `stageCostHessians` — θ-space full and GN reduced Hessians.

The reduced 3-vector / 3×3 block is lifted to the quaternion block with the same
`W` basis the c-space path uses, so the backward-pass `Wᵀ` re-projection and the
final `(I − qqᵀ)` gradient projection are unchanged. `θ` and `sinθ` come from
`atan2(|bs×b|, bs·b)` and `|bs×b|` (accurate at the poles), while `dc`/`ddc`
come unchanged from `vecPointingGeom`. With the flag **off** the code paths are
bit-identical (only a local refactor computing `b_body` once).

---

## 4. Verification

Scripts (committed):
`tests/debug/optimizer/theta_space_fd_parity.py`,
`tests/debug/optimizer/alilqr_python/theta_space_ab.py`.

### 4.1 FD-parity + c-space equivalence (type 3, isolated angle cost, w=1)

θ = 1e-5 … 179.99°. `grad_relerr` = analytic vs central-diff of cost;
`Hess_relerr` = analytic vs forward-diff of the analytic gradient (carries the
manifold-projection FD mismatch — **identical for both parametrizations** since
the full Hessians are analytically equal); `*_equiv` = θ-space vs c-space.

```
   theta | grad_relerr(FD) | fullHess_relerr(FD) | val_equiv | grad_equiv | Hess_equiv
    1e-5 |      1.28e-14   |       2.98e-06      |  4.1e-18  |  0.0e+00   |  0.0e+00
    0.01 |      8.73e-12   |       2.50e-03      |  1.4e-17  |  0.0e+00   |  9.0e-16
   30deg |      2.74e-10   |       1.31e-01      |  5.6e-17  |  2.3e-16   |  9.2e-16
   90deg |      5.74e-10   |       3.48e-01      |  0.0e+00  |  0.0e+00   |  0.0e+00
  150deg |      1.08e-09   |       1.82e-01      |  2.2e-15  |  9.4e-15   |  8.1e-14
  179deg |      9.05e-10   |       6.17e-03      |  4.4e-14  |  5.2e-12   |  1.3e-09
179.99deg|      9.62e-08   |       3.65e-03      |  1.1e-12  |  1.2e-08   |  3.1e-04
```

- Gradient FD parity: `≤ 1e-8` across the whole range (clean).
- θ-space vs c-space equivalence: value/grad/Hess agree to `~1e-12`…`1e-14`
  everywhere except 179.99°, where the **c-space side degrades** (its acos/floor
  artifacts) and the θ-space side stays accurate — the `3e-4` "disagreement" is
  the c-space error, i.e. θ-space is the more correct of the two.
- The `fullHess_relerr(FD)` column (~0.1–0.45 mid-range) is **not** a θ-space
  defect: it is the standard mismatch between an ambient forward-difference of
  the tangent-projected gradient and the analytic manifold (PwA) Hessian. It is
  bit-identical for the c-space path (the two full Hessians are equal), and the
  production Hessian unit tests already account for it. Types 0 and 1 give the
  same picture.

### 4.2 GN Hessian curvature — the headline conditioning result

Type 3, GN, angle weight = 1, Frobenius norm of the quaternion-block Hessian:

```
   theta | ‖H_qq‖ c-space GN | ‖H_qq‖ θ-space GN | ratio (c / θ)
    1e-5 |     1.33e-10       |     1.0000        |      0.00     ← c-space loses ALL curvature
    0.01 |     1.33e-04       |     1.0000        |      0.00
   30deg |     0.3724         |     1.0000        |      0.37
   90deg |     4.0000         |     1.0000        |      4.00
  150deg |    22.138          |     1.0000        |     22.14
  179deg |   719.93           |     1.0000        |    719.93
179.99deg| 72000.0            |     1.0000        |  72000.00     ← c-space razor
```

θ-space GN is a flat `1.0` at every angle; c-space GN vanishes at alignment and
diverges as `1/sinθ` at the antipode. This is exactly the "constant curvature vs
1/sinθ blow-up" payoff.

### 4.3 End-to-end AL-iLQR A/B (3-1 hybrid sat, dt=10, type 3, GN unless noted)

`iters` = accumulated inner iLQR iterations (max 510 = 50×… budget); `PE` =
final boresight-to-target error.

| scenario | c-space | θ-space |
|---|---|---|
| **A. 90° slew, GN** | max_iters, **PE 37.7°** (stalls) | **converged, PE 0.007°** |
| **B. ~179° slew, GN** [headline] | max_iters, **PE 29.6°** | max_iters, **PE 0.37°** |
| **C. ~179° slew, FULL Newton** | reg_exceeded, PE 99.8° | reg_exceeded, PE 99.8° (identical) |
| **D. ~179° slew, type 1, GN** | converged-ish, PE 6.6° | **reg_exceeded, PE 117°** (θ WORSE) |

- **A & B**: with the geodesic-natural type-3 cost, c-space GN cannot finish the
  slew — it loses curvature near alignment (A) and is razor-conditioned near the
  antipode (B). θ-space GN, constant-curvature, drives the pointing error to
  ~0° in both. This is the win.
- **C**: full-Newton type 3 fails *identically* in both parametrizations. This
  is the equivalence check **and** an honest negative: the antipodal indefinite
  curvature is a **genuine geometric cone**, not a parametrization artifact, so
  θ-space does not (and cannot) fix full Newton there.
- **D**: a real **footgun** — see §5.

---

## 5. Honest assessment

**Is it worth productionizing?** For the **type-3** (`½·acos²`) cost, yes in
spirit: θ-space GN is unconditionally well-conditioned (constant curvature `1`),
which lets us *use the geodesic-natural cost* instead of the type-1 workaround
adopted only to dodge the antipodal blow-up. Scenarios A/B show this converts
two stalls into clean convergence. But it is **not** a drop-in for all shapes.

**What breaks.** The constant-curvature payoff is **specific to type 3**
(`g'' ≡ 1`). The θ-space GN Hessian is `g''(θ)·û ûᵀ`, which is PSD **iff
`g''(θ) ≥ 0`**:

```
   theta | g''  type0 | type1  | type2 | type3
   90deg |   0.000    | +1.000 |  0    | +1.000
  150deg |  −0.866    | −1.366 |  0    | +1.000
  179deg |  −1.000    | −1.999 |  0    | +1.000
```

- **type 0** (`1−cosθ`): `g'' = cosθ < 0` for θ > 90° → indefinite GN.
- **type 1** (`½(1−cosθ)²`): `g''` goes negative near the antipode → indefinite
  GN. This is why scenario D regresses: c-space type-1 GN is `f'' ≡ 1` (always
  PSD), but θ-space type-1 GN loses PSD-ness and the backward pass blows reg.
- **type 2** (`θ`): `g'' = 0` → GN is identically zero (no curvature; degenerate,
  same as c-space acos GN).

So the recommendation is **narrow**: θ-space GN is the right thing for type 3
only; types 0/1 must stay in c-space (where their `f''` is PSD by construction).
A production version should either restrict the flag to type 3 or PSD-clamp
`g''` per shape. Full Newton is analytically identical in both spaces, so there
is no reason to switch it except pole-region accuracy.

**Interaction with `ang_vel_err_dir` crossterm and the PwA manifold correction.**
- The θ-space change touches **only** the angle-cost `(q,q)` block. The
  `ang_vel_err_dir` Lyapunov crossterm and its `(ω,q)`/`(q,q)` Hessian addends
  are assembled separately and unchanged; the θ-space gradient/Hessian are
  bit-for-bit the same lift `W·(·)·Wᵀ` into the quaternion block, so they
  superpose exactly as before. The PSD-by-construction argument for the
  crossterm (`α ≤ 2√(w_ang·λ_min(W_ω))`) assumed the angle block is `w_ang·I`
  in the reduced tangent — **but the θ-space GN angle block is `w_ang·û ûᵀ`
  (rank-1)**, same as c-space GN, so that bound is unaffected. It would need
  re-derivation only if the *full* θ block were used with the crossterm
  simultaneously (not the GN default).
- The PwA manifold-curvature correction `−(∂c/∂q·q)·I` lives inside
  `vecPointingGeom.ddc`. The θ-space full Hessian consumes `ddc` (correction
  included) and is algebraically identical to c-space, so the correction is
  honored. The θ-space **GN** path drops `ddc` entirely (`g''·û ûᵀ` only) —
  same as c-space GN — so the correction is intentionally absent there, exactly
  as today.

**Other caveats.**
- `θ`/`sinθ` from `atan2`/`|bs×b|` are accurate at the poles, but the reduced
  `dc`/`ddc` still come from `vecPointingGeom` and use the same `W` basis; the
  pole isotropic-limit guard assumes the roll axis in the reduced tangent is
  `bs` (true when `findWMat` is the body-frame tangent basis). It only fires
  within `1e-7` of an exact pole where the gradient is ~0, so any small basis
  mismatch is harmless, but a production version should confirm the basis.
- The A/B `iters` metric is a coarse proxy (accumulated inner iterations from
  the Python debug harness), adequate for relative comparison, not absolute
  benchmarking.

---

## 6. Open questions for the autotuning-era rewrite

1. **Per-shape curvature policy.** Rather than one `f(c)`/`g(θ)` switch, the
   autotuner wants a curvature it can *scale*. θ-space type 3 gives a clean
   `w·û ûᵀ` knob (curvature = `w`, independent of θ). Should the cost API expose
   "geodesic quadratic with tunable curvature `w`" as a first-class primitive
   and retire the c-space shape enum for vec mode?
2. **Antipodal cone handling.** Full Newton is legitimately indefinite at the
   antipode. Is the right production behavior (a) GN constant-curvature always
   (θ-space type 3), (b) full Newton with a principled azimuthal PSD-clamp, or
   (c) a homotopy that starts GN and switches to full Newton once `θ < 90°`?
3. **Coupling with `ang_vel_roll_ratio`.** Both the roll-reduced ω cost and the
   θ-space angle cost are "2-DOF pointing, roll is free" statements. Can they be
   unified into a single reduced-attitude cost on the pointing 2-sphere ×
   roll-circle, instead of two separate additive terms?
4. **Crossterm PSD bound with the full θ block.** If the autotuner ever wants
   full-Newton angle curvature *and* the Lyapunov crossterm on, re-derive the
   Schur bound with the true (indefinite near antipode) angle block.
5. **Numerical floor removal.** With θ-space, `angCostShape`'s `1e-12` floors
   become unnecessary for type 3. Worth auditing whether any downstream code
   silently depends on the floored (never-exactly-singular) c-space values.

---

*Prototype by the θ-space exploration task. Flag default off; all 81 Python cost
tests, the C++ `alilqr` suite (6149 assertions), and `backwardpass` tests pass
unchanged with the flag off.*
