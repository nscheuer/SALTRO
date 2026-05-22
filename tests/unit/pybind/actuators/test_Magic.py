"""
Python tests for the Magic (direct body-torque) actuator binding.

Mirror of test_Magic.cpp -- exercises ``saltro_py.Magic`` through the
pybind11 surface. Magic actuators apply ``τ = u * axis`` with no
environmental dependence and no internal momentum state, so all
derivatives w.r.t. base state are zero and ``∂τ/∂u`` is exactly the
constant axis row.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))

import saltro_py


def valid_axis():
    return np.array([1.0, 0.0, 0.0])


def valid_base_state():
    x = np.zeros(7)
    x[3] = 1.0
    return x


def test_magic_constructor_with_valid_inputs():
    axis = valid_axis()
    max_torque = 0.01
    magic = saltro_py.Magic(axis, max_torque)
    assert np.allclose(magic.axis, axis)
    assert magic.u_max == max_torque


def test_magic_normalizes_non_unit_axis():
    axis = np.array([2.0, 0.0, 0.0])
    magic = saltro_py.Magic(axis, 0.05)
    assert np.isclose(np.linalg.norm(magic.axis), 1.0)
    assert np.isclose(magic.axis[0], 1.0)


def test_magic_torque_linear_in_u_along_axis():
    axis = np.array([0.0, 1.0, 0.0])
    magic = saltro_py.Magic(axis, 0.1)
    x = valid_base_state()
    u = 0.04
    tau = magic.torque(u, x)
    assert np.allclose(tau, axis * u)


def test_magic_torque_independent_of_base_state():
    """Same u must produce the same τ regardless of ω or q."""
    axis = np.array([1.0, 0.0, 0.0])
    magic = saltro_py.Magic(axis, 0.1)
    x1 = valid_base_state()
    x2 = np.array([0.3, -0.1, 0.7,
                   np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    u = 0.02
    assert np.allclose(magic.torque(u, x1), magic.torque(u, x2))
    assert np.allclose(magic.torque(u, x1), axis * u)


def test_magic_torque_zero_at_zero_command():
    magic = saltro_py.Magic(valid_axis(), 0.1)
    tau = magic.torque(0.0, valid_base_state())
    assert np.allclose(tau, 0.0)


def test_magic_dtorq_du_equals_axis_row():
    axis = np.array([0.0, 0.0, 1.0])
    magic = saltro_py.Magic(axis, 0.05)
    # pybind11 collapses Mat13 (1x3) to a length-3 ndarray.
    jac = np.asarray(magic.dtorq_du(0.03, valid_base_state())).ravel()
    assert jac.shape == (3,)
    assert np.allclose(jac, axis)


def test_magic_dtorq_du_constant_in_u_and_x():
    axis = np.array([0.5, 0.5, np.sqrt(0.5)])  # unit-norm
    magic = saltro_py.Magic(axis, 0.1)
    jac0 = np.asarray(magic.dtorq_du(0.0, valid_base_state())).ravel()
    x_other = np.array([0.1, 0.2, 0.3,
                        np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0])
    jac1 = np.asarray(magic.dtorq_du(0.07, x_other)).ravel()
    assert np.allclose(jac0, jac1)


def test_magic_dtorq_dbasestate_is_zero():
    magic = saltro_py.Magic(valid_axis(), 0.1)
    J = np.asarray(magic.dtorq_dbasestate(0.05, valid_base_state()))
    assert J.shape == (7, 3)
    assert np.allclose(J, 0.0)


# The ddtorq_* methods return Tensor3<...> objects, which aren't yet
# registered in the pybind module (same gap as MTQ's mixed Hessian
# binding). The C++ side has them covered in tests/unit/pybind/actuators/
# test_Magic.cpp -- once Tensor3 is bound to Python these tests will
# light up. The C++ implementations are pinned to zero so this is
# safe behaviour to defer.
@pytest.mark.skip(reason="Tensor3<...> not yet bound to Python")
def test_magic_ddtorq_dudu_is_zero():
    magic = saltro_py.Magic(valid_axis(), 0.1)
    H = np.asarray(magic.ddtorq_dudu(0.02, valid_base_state()))
    assert np.allclose(H, 0.0)


@pytest.mark.skip(reason="Tensor3<1,7,3> not yet bound to Python")
def test_magic_ddtorq_dudbasestate_is_zero():
    magic = saltro_py.Magic(valid_axis(), 0.1)
    H = np.asarray(magic.ddtorq_dudbasestate(0.02, valid_base_state()))
    assert np.allclose(H, 0.0)


@pytest.mark.skip(reason="Tensor3<7,7,3> not yet bound to Python")
def test_magic_ddtorq_dbasestatedbasestate_is_zero():
    magic = saltro_py.Magic(valid_axis(), 0.1)
    H = np.asarray(
        magic.ddtorq_dbasestatedbasestate(0.02, valid_base_state())
    )
    assert np.allclose(H, 0.0)


def test_magic_torque_is_odd_in_u():
    magic = saltro_py.Magic(valid_axis(), 0.1)
    x = valid_base_state()
    assert np.allclose(magic.torque(+0.03, x), -magic.torque(-0.03, x))


# -----------------------------------------------------------------------------
# Satellite integration: addMagic / numMagic / getMagic / controlDim
# -----------------------------------------------------------------------------

def _identity_inertia():
    return 0.05 * np.eye(3)


def test_satellite_addMagic_grows_controlDim():
    sat = saltro_py.Satellite(_identity_inertia(),
                              saltro_py.PlannerSettings())
    sat.addMTQ(np.array([1.0, 0.0, 0.0]), 0.2)
    sat.addRW(np.array([0.0, 1.0, 0.0]), 0.001, 1e-5, 0.0, 0.01)

    base_cdim = sat.controlDim
    assert sat.numMagic == 0
    sat.addMagic(np.array([0.0, 0.0, 1.0]), 0.05)
    assert sat.numMagic == 1
    assert sat.controlDim == base_cdim + 1


def test_satellite_getMagic_returns_added_actuator():
    sat = saltro_py.Satellite(_identity_inertia(),
                              saltro_py.PlannerSettings())
    sat.addMagic(np.array([0.0, 1.0, 0.0]), 0.07)

    m = sat.getMagic(0)
    assert np.allclose(m.axis, np.array([0.0, 1.0, 0.0]))
    assert m.u_max == 0.07


def test_satellite_getMagic_out_of_range_raises():
    sat = saltro_py.Satellite(_identity_inertia(),
                              saltro_py.PlannerSettings())
    with pytest.raises(Exception):
        sat.getMagic(0)
    sat.addMagic(np.array([1.0, 0.0, 0.0]), 0.05)
    with pytest.raises(Exception):
        sat.getMagic(1)


def test_satellite_addMagic_overflow_raises():
    """At the configured limit, addMagic should raise."""
    sat = saltro_py.Satellite(_identity_inertia(),
                              saltro_py.PlannerSettings())
    # Add up to the compile-time limit.
    for i in range(8):
        try:
            sat.addMagic(np.array([1.0, 0.0, 0.0]), 0.05)
        except Exception:
            # We've hit the limit; the previous count is the max.
            break
    else:
        pytest.skip("Did not reach MAX_NUM_MAGIC within 8 attempts -- limit "
                    "raised? Skipping overflow assertion.")
    with pytest.raises(Exception):
        sat.addMagic(np.array([1.0, 0.0, 0.0]), 0.05)


def test_satellite_actuatorTorque_includes_magic_contribution():
    """Magic-only satellite: u along magic axis should produce τ = u*axis."""
    sat = saltro_py.Satellite(np.eye(3), saltro_py.PlannerSettings())
    sat.addMagic(np.array([1.0, 0.0, 0.0]), 0.5)

    # State: ω=0, q=identity, no RW momenta
    x = np.zeros(7)
    x[3] = 1.0  # q0
    u = np.array([0.3])
    # Body-frame magnetic field is irrelevant for magic; pass arbitrary ECI B
    B_eci = np.array([0.1, 0.0, 0.0])

    tau = np.asarray(sat.actuatorTorque(x, u, B_eci))
    assert np.allclose(tau, np.array([0.3, 0.0, 0.0]))
