import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "build"))

import saltro_py


# API contract: jtime is Julian centuries T (not Julian Date).
def make_jcenturies(N, span_days, T0=0.22):
    return T0 + np.linspace(0.0, span_days, N) / 36525.0


def valid_initial_conditions():
    Re = 6378136.3
    r0 = np.array([Re + 400e3, 0.0, 0.0])
    v0 = np.array([0.0, 7660.0, 0.0])  # valid LEO velocity
    return r0, v0


def test_generate_orbit_basic_dimensions():
    N = 8
    r0, v0 = valid_initial_conditions()
    jtime = make_jcenturies(N, span_days=0.01)

    ok, R, V, B, S, rho = saltro_py.generate_orbit(
        r0, v0, jtime,
        orbit_model=0,
        magnetic_model=0,
        sun_model=0,
        eclipse_model=0,
        density_model=0
    )

    assert ok
    assert R.shape == (3, N)
    assert V.shape == (3, N)
    assert B.shape == (3, N)
    assert S.shape == (3, N)
    assert rho.shape == (N,)

    assert np.all(np.isfinite(R))
    assert np.all(np.isfinite(V))
    assert np.all(np.isfinite(B))
    assert np.all(np.isfinite(S))
    assert np.all(np.isfinite(rho))


def test_generate_orbit_single_point():
    N = 1
    r0, v0 = valid_initial_conditions()
    jtime = make_jcenturies(N, span_days=0.0)

    ok, R, V, B, S, rho = saltro_py.generate_orbit(
        r0, v0, jtime,
        0, 0, 0, 0, 0
    )

    assert ok
    assert R.shape == (3, 1)
    assert V.shape == (3, 1)
    assert B.shape == (3, 1)
    assert S.shape == (3, 1)
    assert rho.shape == (1,)


def test_generate_orbit_invalid_empty_jtime():
    r0, v0 = valid_initial_conditions()
    jtime = np.array([])

    with pytest.raises(RuntimeError):
        saltro_py.generate_orbit(
            r0, v0, jtime,
            0, 0, 0, 0, 0
        )


def test_generate_orbit_non_increasing_time():
    r0, v0 = valid_initial_conditions()

    jtime = np.array([0.22, 0.22, 0.221])  # not strictly increasing

    ok, *_ = saltro_py.generate_orbit(
        r0, v0, jtime,
        0, 0, 0, 0, 0
    )

    assert not ok


def test_generate_orbit_time_not_finite():
    r0, v0 = valid_initial_conditions()
    jtime = np.array([0.22, np.nan, 0.23])

    ok, *_ = saltro_py.generate_orbit(
        r0, v0, jtime,
        0, 0, 0, 0, 0
    )

    assert not ok


def test_generate_orbit_time_is_julian_date_rejected():
    r0, v0 = valid_initial_conditions()

    # Typical Julian Date ~ 2.4e6 (should be rejected by validator)
    jtime = np.array([2459000.0, 2459000.0001])

    ok, *_ = saltro_py.generate_orbit(
        r0, v0, jtime,
        0, 0, 0, 0, 0
    )

    assert not ok


def test_generate_orbit_invalid_r0_magnitude():
    r0, v0 = valid_initial_conditions()
    r0_bad = np.array([1.0e5, 0.0, 0.0])  # too small
    jtime = make_jcenturies(5, span_days=0.01)

    ok, *_ = saltro_py.generate_orbit(
        r0_bad, v0, jtime,
        0, 0, 0, 0, 0
    )

    assert not ok


def test_generate_orbit_invalid_v0_magnitude():
    r0, v0 = valid_initial_conditions()
    v0_bad = np.array([0.0, 100.0, 0.0])  # too small
    jtime = make_jcenturies(5, span_days=0.01)

    ok, *_ = saltro_py.generate_orbit(
        r0, v0_bad, jtime,
        0, 0, 0, 0, 0
    )

    assert not ok


def test_generate_orbit_nonfinite_state():
    r0, v0 = valid_initial_conditions()
    r0_bad = np.array([np.nan, 0.0, 0.0])
    jtime = make_jcenturies(5, span_days=0.01)

    ok, *_ = saltro_py.generate_orbit(
        r0_bad, v0, jtime,
        0, 0, 0, 0, 0
    )

    assert not ok


def test_generate_orbit_exceeds_max_length():
    r0, v0 = valid_initial_conditions()

    # create intentionally too-large array
    N = saltro_py.limits.MAX_LENGTH_TRAJ + 1
    jtime = make_jcenturies(N, span_days=0.01)

    with pytest.raises(RuntimeError):
        saltro_py.generate_orbit(
            r0, v0, jtime,
            0, 0, 0, 0, 0
        )