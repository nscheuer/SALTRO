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


def valid_magnetic_field():
    return np.array([0.0, 1.0e-5, 0.0])


def test_mtq_constructor():
    axis = valid_axis()
    max_dipole = 0.2

    mtq = saltro_py.MTQ(axis, max_dipole)
    assert np.allclose(mtq.axis, axis)
    assert mtq.u_max == max_dipole


def test_mtq_torque_computation():
    axis = np.array([1.0, 0.0, 0.0])
    mtq = saltro_py.MTQ(axis, 0.2)

    x = valid_base_state()
    B_body = np.array([0.0, 1.0e-5, 0.0])
    u = 0.1

    tau = mtq.torque(u, x, B_body)

    expected = np.array([0.0, 0.0, 1.0e-6])
    
    assert np.allclose(tau, expected, atol=1e-12)


def test_mtq_torque_with_different_configurations():
    axis = np.array([0.0, 1.0, 0.0])
    mtq = saltro_py.MTQ(axis, 1.0)

    x = valid_base_state()
    B = np.array([2.0e-5, 0.0, 0.0])
    
    tau = mtq.torque(0.5, x, B)
    
    expected = np.array([0.0, 0.0, -1.0e-5])
    
    assert np.allclose(tau, expected, atol=1e-12)


def test_mtq_torque_is_zero_when_b_parallel_to_axis():
    axis = np.array([1.0, 0.0, 0.0])
    mtq = saltro_py.MTQ(axis, 1.0)

    x = valid_base_state()
    B = np.array([1.0e-5, 0.0, 0.0])

    tau = mtq.torque(1.0, x, B)
    
    assert np.linalg.norm(tau) < 1e-15


def test_mtq_torque_is_zero_when_u_is_zero():
    axis = valid_axis()
    mtq = saltro_py.MTQ(axis, 1.0)

    x = valid_base_state()
    B = valid_magnetic_field()

    tau = mtq.torque(0.0, x, B)
    
    assert np.allclose(tau, 0.0)


def test_mtq_dtorq_du():
    axis = np.array([1.0, 0.0, 0.0])
    mtq = saltro_py.MTQ(axis, 1.0)

    x = valid_base_state()
    B = np.array([0.0, 1.0e-5, 0.0])

    J = mtq.dtorq_du(0.5, x, B)

    B_cross_axis = np.cross(B, axis)
    expected = -B_cross_axis.reshape(1, 3)
    
    assert np.allclose(J, expected, atol=1e-12)


def test_mtq_dtorq_dbasestate_with_zero_dB_dq():
    axis = valid_axis()
    mtq = saltro_py.MTQ(axis, 1.0)

    x = valid_base_state()
    B = valid_magnetic_field()
    dB_dq = np.zeros((4, 3))

    J = mtq.dtorq_dbasestate(0.5, x, B, dB_dq)

    assert np.allclose(J, 0.0)


@pytest.mark.skip(reason="Tensor3<1,7,3> not yet bound to Python")
def test_mtq_ddtorq_dudbasestate_with_non_zero_dB_dq():
    axis = valid_axis()
    mtq = saltro_py.MTQ(axis, 1.0)

    x = valid_base_state()
    B = valid_magnetic_field()
    
    dB_dq = np.ones((4, 3))

    H = mtq.ddtorq_dudbasestate(0.5, x, B, dB_dq)

    assert H.shape[0] == 1
    assert H.shape[1] == 7
