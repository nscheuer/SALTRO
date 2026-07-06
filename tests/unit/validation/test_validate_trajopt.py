import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "build"))

import saltro_py

# Python twin of tests/unit/validation/test_validate_trajopt.cpp.
#
# validatetrajOpt() is the top-level input gate-keeper for a planning problem:
# it chains the per-field validators (settings/satellite/x0/orbit/jtime/q_goal/
# boresight) and then the cross-context dimension checks. The per-field
# validators have their own python twins; this pins the orchestration and, in
# particular, the cross-context dimension checks (N, state_dim, input_dim,
# jtime length) through the bound python surface.


def valid_settings():
    s = saltro_py.PlannerSettings()
    s.num_passes = 1
    s.constraints.control_limit_scale = 0.75
    s.constraints.u_max = np.array([1.0, 1.0, 1.0])
    s.constraints.wmax = 0.3
    s.constraints.sun_limit_angle = 0.35
    s.disturbances.coeff_N = 3
    s.init_traj.initcontroller = 0
    s.passes[0].dt = 1.0
    return s


def valid_inertia():
    return np.diag([0.05, 0.06, 0.07])


class Fixture:
    """3 MTQs -> controlDim 3 (matches u_max size 3), no RW -> stateDim 7."""

    def __init__(self):
        self.settings = valid_settings()
        self.sat = saltro_py.Satellite(valid_inertia(), self.settings)
        self.sat.addMTQ(np.array([1.0, 0.0, 0.0]), 0.2)
        self.sat.addMTQ(np.array([0.0, 1.0, 0.0]), 0.2)
        self.sat.addMTQ(np.array([0.0, 0.0, 1.0]), 0.2)
        self.x0 = np.array([0.1, 0.05, -0.02, 1.0, 0.0, 0.0, 0.0])
        self.r0 = np.array([7000e3, 0.0, 0.0])
        self.v0 = np.array([0.0, 7546.0, 0.0])
        # Julian centuries since J2000, in [0.20, 0.40]
        self.jtime = np.array([0.22, 0.22 + 1e-6])
        self.q_goal = np.column_stack(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        )
        self.boresight = np.column_stack(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        self.N = 2

    def run(self, state_dim, input_dim, n):
        return saltro_py.validatetrajOpt(
            self.settings,
            self.sat,
            self.x0,
            self.r0,
            self.v0,
            self.jtime,
            self.q_goal,
            self.boresight,
            state_dim,
            input_dim,
            n,
        )


def test_consistent_problem_passes():
    f = Fixture()
    ok, err = f.run(f.sat.stateDim, f.sat.controlDim, f.N)
    assert ok, (
        f"stateDim={f.sat.stateDim} controlDim={f.sat.controlDim} err={err}"
    )
    assert err == ""


def test_n_must_be_greater_than_1():
    f = Fixture()
    ok, err = f.run(f.sat.stateDim, f.sat.controlDim, 1)
    assert not ok
    assert "N must be > 1" in err


def test_state_dim_must_match_the_satellite():
    f = Fixture()
    ok, err = f.run(f.sat.stateDim + 1, f.sat.controlDim, f.N)
    assert not ok
    assert "state_dim" in err


def test_input_dim_must_match_the_satellite():
    f = Fixture()
    ok, err = f.run(f.sat.stateDim, f.sat.controlDim + 1, f.N)
    assert not ok
    assert "input_dim" in err


def test_jtime_length_must_match_n():
    f = Fixture()
    # jtime has 2 entries; claim N=3 -> cross-context length mismatch.
    ok, _err = f.run(f.sat.stateDim, f.sat.controlDim, 3)
    assert not ok
