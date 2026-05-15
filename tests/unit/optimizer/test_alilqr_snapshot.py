"""Deterministic snapshot regression for the alilqr optimizer.

Why this test exists
--------------------
The existing optimizer tests in `test_alilqr.py` and friends are
*quality* tests — they assert thresholds like "final pointing error
< 5°" and "trajOpt returns ok=True". They're sensitive to convergence
edge cases and currently have several known failures tied to the
active inner-solve tuning work. See
`memory/project_p4_optimizer_test_failures_deferred.md`.

This file complements those by pinning a *known-good easy-converging*
trajectory's output numerically. The configuration here converges to
~0.001° pointing error and is far from any threshold edge, so it should
stay stable across legitimate optimizer changes. The values are
deterministic (verified by running twice — bit-identical output).

Drift on this test means: a code change altered the optimizer's
behavior on a case that was previously trivial — worth a closer look
*before* the change ships, even if the quality tests still pass.

When values intentionally change
--------------------------------
If a planned change shifts the snapshot (e.g., a numerical
algorithm tweak or a default tuning update), update the constants
below in one commit alongside the code change, and reference the
PR in the commit message.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(ROOT / "tests" / "debug" / "optimizer" / "configs"))
sys.path.insert(0, str(ROOT / "tests" / "unit" / "optimizer"))

import saltro_py
from sat_0_3_rw import create_satellite as create_satellite_rw
from test_alilqr import (
    SEC_PER_CENTURY,
    create_planner_settings,
    quat_pointing_error_deg,
)


# ----------------------------------------------------------------------------
# Configuration — keep these synced with the values captured below
# ----------------------------------------------------------------------------

DT_SECONDS = 10.0
TF_SECONDS = 200.0

# Initial state
W0 = np.array([-0.01, 0.02, 0.03], dtype=float)
Q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
H0 = np.array([0.0, 0.0, 0.0], dtype=float)

# Goal: 90° rotation about Y (qgoal = [cos(45°), 0, 0, sin(45°)])
SQRT2_2 = float(np.sqrt(2.0) / 2.0)
QGOAL_FINAL = np.array([SQRT2_2, 0.0, 0.0, SQRT2_2], dtype=float)

# Initial orbit state
R0 = np.array([7000e3, 0.0, 0.0], dtype=float)
V0 = np.array([0.0, 7.5e3, 0.0], dtype=float)


# ----------------------------------------------------------------------------
# Captured snapshot values (2026-05-15, branch PKMN_antispike,
# commit 0fbdab4)
# ----------------------------------------------------------------------------

SNAPSHOT_POINTING_ERROR_DEG = 0.001045
SNAPSHOT_OMEGA_NORM = 1.886728e-6
SNAPSHOT_U_FROBENIUS = 3.541955e-4
SNAPSHOT_FINAL_RW_MOMENTUM_NORM = 2.594e-3  # ‖[h_x, h_y, h_z]‖ at terminal step

# Generous tolerance bands. We're looking for drift, not bit-identity:
# - 50% relative window on each scalar.
# - Plus a small absolute floor for quantities that may legitimately
#   round to ~0 in some configurations.
REL_TOL = 0.5
ABS_FLOOR_OMEGA = 5e-6
ABS_FLOOR_POINTING = 0.005  # deg


# ----------------------------------------------------------------------------
# Test fixture
# ----------------------------------------------------------------------------


def _run_known_good_trajopt():
    """Run the pinned configuration end-to-end and return its outputs."""
    plannersettings = create_planner_settings(DT_SECONDS)
    satellite = create_satellite_rw(plannersettings)

    jtime = np.array(
        [0.22, 0.22 + TF_SECONDS / SEC_PER_CENTURY], dtype=float
    )
    qgoal = np.array(
        [
            [SQRT2_2, SQRT2_2],
            [0.0, 0.0],
            [0.0, 0.0],
            [SQRT2_2, SQRT2_2],
        ],
        dtype=float,
    )
    boresight = np.array(
        [[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]], dtype=float
    )

    x0 = np.hstack((W0, Q0, H0))

    ok, X, U, _K = saltro_py.trajOpt(
        plannersettings, satellite, x0, R0, V0, jtime, qgoal, boresight
    )
    return ok, X, U, qgoal


# ----------------------------------------------------------------------------
# Snapshot assertions
# ----------------------------------------------------------------------------


def _assert_close(actual: float, expected: float, rel_tol: float,
                  abs_floor: float, name: str) -> None:
    """Snapshot assertion: actual close to expected within rel_tol, with an
    absolute floor for quantities that legitimately round to ~0."""
    tol = max(rel_tol * abs(expected), abs_floor)
    assert abs(actual - expected) <= tol, (
        f"snapshot drift in {name}: actual={actual:.6e}, "
        f"expected={expected:.6e}, tol=±{tol:.2e}"
    )


def test_alilqr_snapshot_converges_on_known_good_config():
    """Basic sanity: the pinned config must continue to solve."""
    ok, X, U, _ = _run_known_good_trajopt()
    assert ok, "snapshot config failed to converge — this should never happen"

    assert X.shape[1] == 21  # 200s / 10s + 1 = 21 nodes
    assert U.shape[1] == 21
    assert np.all(np.isfinite(X))
    assert np.all(np.isfinite(U))


def test_alilqr_snapshot_final_pointing_error_within_tolerance():
    """Pointing error at terminal step must stay near 0.001°. A 50% shift
    on this near-zero value would indicate the convergence behavior on
    the easy-case has changed materially."""
    ok, X, _U, qgoal = _run_known_good_trajopt()
    assert ok

    pe = quat_pointing_error_deg(X[3:7, -1], qgoal[:, -1])
    _assert_close(
        pe, SNAPSHOT_POINTING_ERROR_DEG,
        rel_tol=REL_TOL, abs_floor=ABS_FLOOR_POINTING,
        name="final_pointing_error_deg",
    )


def test_alilqr_snapshot_final_omega_norm_within_tolerance():
    """Final angular velocity must remain near zero (~1.9e-6). Drift here
    suggests the velocity-damping balance shifted."""
    ok, X, _U, _ = _run_known_good_trajopt()
    assert ok

    omega_norm = float(np.linalg.norm(X[0:3, -1]))
    _assert_close(
        omega_norm, SNAPSHOT_OMEGA_NORM,
        rel_tol=REL_TOL, abs_floor=ABS_FLOOR_OMEGA,
        name="final_omega_norm",
    )


def test_alilqr_snapshot_total_control_effort_within_tolerance():
    """‖U‖_F captures total integrated control activity. Drift here is
    the most likely signal of cost-weight or BP gain changes."""
    ok, _X, U, _ = _run_known_good_trajopt()
    assert ok

    u_frob = float(np.linalg.norm(U))
    _assert_close(
        u_frob, SNAPSHOT_U_FROBENIUS,
        rel_tol=REL_TOL, abs_floor=1e-5,
        name="‖U‖_Frobenius",
    )


def test_alilqr_snapshot_final_rw_momentum_within_tolerance():
    """The 3-RW configuration accumulates a small terminal momentum. Drift
    in this value indicates the slew profile changed shape (e.g., a
    different time-allocation of RW torque)."""
    ok, X, _U, _ = _run_known_good_trajopt()
    assert ok

    h_final = X[saltro_py.Satellite.RW_MOMENTUM_INDEX:
                saltro_py.Satellite.RW_MOMENTUM_INDEX + 3, -1]
    h_norm = float(np.linalg.norm(h_final))
    _assert_close(
        h_norm, SNAPSHOT_FINAL_RW_MOMENTUM_NORM,
        rel_tol=REL_TOL, abs_floor=1e-4,
        name="final_rw_momentum_norm",
    )


def test_alilqr_snapshot_is_deterministic():
    """Running the same config twice in one process must produce
    bit-identical X and U. If this fails, there's hidden non-determinism
    (uninitialized memory, RNG without seed, hash-order dependence)."""
    ok1, X1, U1, _ = _run_known_good_trajopt()
    ok2, X2, U2, _ = _run_known_good_trajopt()

    assert ok1 and ok2
    assert np.array_equal(X1, X2), "X differs between runs — non-determinism"
    assert np.array_equal(U1, U2), "U differs between runs — non-determinism"
