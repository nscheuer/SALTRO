# Test Audit: Silent UB Bugs Hidden by Release-Mode Builds

**Status:** TODO — not yet started. Drafted 2026-05-19 after fixing one instance.

## Problem

`.github/workflows/build-and-test.yml` originally configured CI with
`-DCMAKE_BUILD_TYPE=Release` only.  Release builds define `NDEBUG`,
which makes `eigen_assert` (and `assert`) compile to no-ops. Eigen still
produces the correct *value* for ops with matching dimensions, but bad
ops (e.g. `Block<Ref<Matrix>>::resize()` with a size mismatch) become
silent no-ops instead of aborting. The subsequent assignment loop then
copies whatever bytes overlap — often *happens to look fine* in trivial
test setups but is undefined behavior.

This PR adds a Debug job to CI so this class of bug surfaces
automatically going forward. This document describes the
audit-the-existing-tree task — a follow-up to systematically check
existing tests under Debug.

## Known instance (fixed 2026-05-19)

`tests/unit/optimizer/test_forwardpass_al.cpp` —
`ForwardPassALFixture::ForwardPassALFixture()` initialized `x0` using
`satellite.stateDim()` *in the member-init list*, before
`configureSatellite()` added 3 RWs in the constructor body. That meant
`x0` was size 7 (no RWs) but `X` (built later via the same `stateDim()`)
was size 10. `warm_start` then did `X.col(0) = x0` which tries to resize
the column block of `Ref<MatrixXd>` X.

- **Release:** `eigen_assert` no-op, the copy loop copies 7 entries into
  the first 7 slots of the 10-slot column, leaving RW momentum at 0.
  Test runs, RW=0 happens to be a valid IC, ok_fp returns true,
  REQUIRE passes. Bug invisible.
- **Debug:** `eigen_assert` fires → SIGABRT before any assignment runs.
  Test fails with crash.

Fix was to re-init `x0` after `configureSatellite()` ran so it has the
right size. Applied across PRs #12-#16's stacked test commit.

## Symptom patterns to look for

Search the tree for any of these and verify the order is correct:

1. **Member-init list using `.stateDim()` or `.controlDim()` before
   actuators are added in the constructor body.** Same class of bug.
   Grep test fixtures for member-init list usage of these calls.

2. **`Ref<MatrixXd>` columns assigned from VectorXd of different size.**
   Look for `X.col(k) = ...` where X is a `Ref<MatrixXd>` parameter and
   the RHS comes from `Satellite::VecX` or `VectorXd::Zero(some_size)`.
   Verify the sizes match.

3. **`MatrixXd::Zero(rows, cols)` where `rows` is computed from a
   satellite object that hasn't been configured yet** (e.g. the
   satellite was passed in but the test fixture configured RWs after
   reading stateDim).

4. **Mixed `U` shapes** — passing U with `N` cols when the function
   expects `N-1` (or vice versa). The
   `if (U_bar.cols() == U.cols()) { U = U_bar; }`-style guards in
   `forwardpass.cpp:264-268` suggest historical confusion. Tests that
   pass mismatched-N-cols U can trip Block-of-Ref::resize.

5. **`Eigen::Ref<>` parameters being assigned to with `operator=`** —
   `X = X_bar` at `src/optimizer/forwardpass.cpp:263` is fine when sizes
   match but risky if the caller passed a sub-block. Audit Ref callers
   to confirm dimensions.

## How to run the audit

```bash
# Build Debug locally
mkdir -p build-debug && cd build-debug
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build . -j4

# Run the full test suite. Any tests that pass in Release but FAIL or
# SIGABRT here are the audit targets.
ctest --output-on-failure
```

With the Debug CI job added (this PR), the audit can also use the CI
results directly — any Debug-only failures on this branch's CI are
audit targets.

## Test files to audit (highest-risk first)

| Priority | File | Why |
|---|---|---|
| 1 | `tests/unit/optimizer/test_forwardpass.cpp` | Same FP code path as the fixed test |
| 1 | `tests/unit/optimizer/test_backwardpass.cpp` | BP code path; uses fixtures with addRW |
| 1 | `tests/unit/optimizer/test_backwardpass_al.cpp` | AL-specific BP test |
| 1 | `tests/unit/optimizer/test_alilqr.cpp` | Top-level inner solve |
| 2 | `tests/unit/optimizer/test_warm_start.cpp` | warm_start is where the original SIGABRT was triggered |
| 2 | `tests/unit/pybind/test_satellite_cost.cpp` | Cost path with Ref<MatrixXd> params |
| 2 | `tests/unit/pybind/test_satellite_constraints.cpp` | Constraints take Ref params too |
| 3 | `tests/unit/pybind/test_satellite_dynamics.cpp` | Dynamics step path |
| 3 | All other `tests/unit/**/test_*.cpp` | Lower priority but worth a sweep |

## Per-file checklist

For each file above:

- [ ] Build Debug, run the file's test binary
- [ ] Note any SIGABRT or `eigen_assert` failures
- [ ] For each failure: identify the resize source (gdb backtrace
  works well — `gdb -batch -ex 'run' -ex 'bt'`)
- [ ] Determine: is the test logically buggy (e.g. init order),
  or is the production code buggy?
- [ ] Apply the smallest correct fix
- [ ] Verify in both Release AND Debug

## Out of scope

- Performance tests under Debug (Debug builds are slow; don't expect
  benchmarks to pass thresholds)
- Tests with intentional `assert`s for sanity checks unrelated to
  Eigen size matching

## Related

- This PR: adds Debug CI job
- Fix for the FP_al instance: PR #12's `8a57ec9` (x0 init order)
- Fix for the FP_al stability test: PR #12's `3239484` (accept
  ok_fp=false under random Q_uu-indefinite AL multipliers)
