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


def test_rw_constructor_with_valid_inputs():
    axis = valid_axis()
    max_torque = 0.01
    J = 2.0e-5
    h0 = 0.0
    h_max = 0.1

    rw = saltro_py.RW(axis, max_torque, J, h0, h_max)
    assert np.allclose(rw.axis, axis)
    assert rw.u_max == max_torque
    assert rw.wheelInertia == J
    assert rw.momentum == h0
    assert rw.momentumMax == h_max


def test_rw_rejects_non_positive_inertia():
    axis = valid_axis()
    
    with pytest.raises(Exception):
        saltro_py.RW(axis, 0.01, 0.0, 0.0, 0.1)
    
    with pytest.raises(Exception):
        saltro_py.RW(axis, 0.01, -1.0e-5, 0.0, 0.1)


def test_rw_rejects_negative_h_max():
    axis = valid_axis()
    
    with pytest.raises(Exception):
        saltro_py.RW(axis, 0.01, 2.0e-5, 0.0, -0.1)


def test_rw_accepts_zero_h_max():
    axis = valid_axis()
    
    rw = saltro_py.RW(axis, 0.01, 2.0e-5, 0.0, 0.0)
    assert rw.momentumMax == 0.0


def test_rw_set_momentum_and_get_momentum():
    axis = valid_axis()
    rw = saltro_py.RW(axis, 0.01, 2.0e-5, 0.0, 0.1)

    assert rw.momentum == 0.0
    
    rw.momentum = 0.05
    assert rw.momentum == 0.05
    
    rw.momentum = -0.03
    assert rw.momentum == -0.03


def test_rw_torque_computation():
    axis = np.array([1.0, 0.0, 0.0])
    rw = saltro_py.RW(axis, 0.01, 2.0e-5, 0.0, 0.1)

    x = valid_base_state()
    u = 0.005

    tau = rw.torque(u, x)

    expected = axis * u
    
    assert np.allclose(tau, expected)


def test_rw_torque_with_different_axis():
    axis = np.array([0.0, 0.0, 1.0])
    rw = saltro_py.RW(axis, 0.01, 2.0e-5, 0.0, 0.1)

    x = valid_base_state()
    u = -0.003

    tau = rw.torque(u, x)

    expected = np.array([0.0, 0.0, -0.003])
    
    assert np.allclose(tau, expected)


def test_rw_storage_torque_computation():
    axis = valid_axis()
    rw = saltro_py.RW(axis, 0.01, 2.0e-5, 0.0, 0.1)

    x = valid_base_state()
    u = 0.005

    h_dot = rw.storageTorque(u, x)

    assert h_dot == -u


def test_rw_dtorq_du():
    axis = np.array([0.0, 1.0, 0.0])
    rw = saltro_py.RW(axis, 0.01, 2.0e-5, 0.0, 0.1)

    x = valid_base_state()

    J = rw.dtorq_du(0.5, x)

    expected = axis.reshape(1, 3)
    
    assert np.allclose(J, expected)


def test_rw_dtorq_dbasestate_is_zero():
    axis = valid_axis()
    rw = saltro_py.RW(axis, 0.01, 2.0e-5, 0.0, 0.1)

    x = valid_base_state()

    J = rw.dtorq_dbasestate(0.5, x)

    assert np.allclose(J, 0.0)


def test_rw_dstor_torq_du():
    axis = valid_axis()
    rw = saltro_py.RW(axis, 0.01, 2.0e-5, 0.0, 0.1)

    x = valid_base_state()

    J = rw.dstor_torq_du(0.5, x)

    assert J == -1.0


def test_rw_dstor_torq_dbasestate_is_zero():
    axis = valid_axis()
    rw = saltro_py.RW(axis, 0.01, 2.0e-5, 0.0, 0.1)

    x = valid_base_state()

    J = rw.dstor_torq_dbasestate(0.5, x)

    assert np.allclose(J, 0.0)
