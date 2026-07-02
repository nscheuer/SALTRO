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


def test_actuator_constructor_with_valid_inputs():
    axis = valid_axis()
    u_max = 1.0

    act = saltro_py.Actuator(axis, u_max)
    assert np.allclose(act.axis, axis)
    assert act.u_max == u_max


def test_actuator_normalizes_axis():
    axis = np.array([3.0, 4.0, 0.0])
    u_max = 1.0

    act = saltro_py.Actuator(axis, u_max)
    
    assert np.isclose(np.linalg.norm(act.axis), 1.0, rtol=1e-10)
    assert np.allclose(act.axis, axis / np.linalg.norm(axis))


def test_actuator_takes_absolute_value_of_u_max():
    axis = valid_axis()
    
    act1 = saltro_py.Actuator(axis, -5.0)
    assert act1.u_max == 5.0
    
    act2 = saltro_py.Actuator(axis, 5.0)
    assert act2.u_max == 5.0


def test_actuator_rejects_zero_axis():
    axis = np.zeros(3)
    u_max = 1.0

    with pytest.raises(Exception):
        saltro_py.Actuator(axis, u_max)


def test_actuator_rejects_non_finite_axis():
    axis = np.array([np.nan, 0.0, 0.0])
    u_max = 1.0

    with pytest.raises(Exception):
        saltro_py.Actuator(axis, u_max)


def test_actuator_rejects_non_finite_u_max():
    axis = valid_axis()
    
    with pytest.raises(Exception):
        saltro_py.Actuator(axis, np.nan)
    
    with pytest.raises(Exception):
        saltro_py.Actuator(axis, np.inf)


def test_actuator_clamp_function():
    axis = valid_axis()
    u_max = 2.5
    act = saltro_py.Actuator(axis, u_max)

    assert act.clamp(1.0) == 1.0
    assert act.clamp(-1.0) == -1.0
    assert act.clamp(2.5) == 2.5
    assert act.clamp(-2.5) == -2.5
    assert act.clamp(5.0) == 2.5
    assert act.clamp(-5.0) == -2.5


def test_actuator_base_class_torque_returns_zero():
    axis = valid_axis()
    act = saltro_py.Actuator(axis, 1.0)
    x = valid_base_state()

    tau = act.torque(0.5, x)
    assert np.allclose(tau, 0.0)


def test_actuator_base_class_derivatives_return_zero():
    axis = valid_axis()
    act = saltro_py.Actuator(axis, 1.0)
    x = valid_base_state()

    J_u = act.dtorq_du(0.5, x)
    J_x = act.dtorq_dbasestate(0.5, x)

    assert np.allclose(J_u, 0.0)
    assert np.allclose(J_x, 0.0)


def test_actuator_base_class_hessians_are_callable_finite_and_zero():
    """Smoke test: the Tensor3-returning base-class Hessians must actually be
    callable from Python (requires the Tensor3<->numpy caster to be registered
    in the actuator binding TU) and return finite, correctly-shaped zeros."""
    axis = valid_axis()
    act = saltro_py.Actuator(axis, 1.0)
    x = valid_base_state()

    # Tensor3<R,C,S> -> numpy (slices, rows, cols)
    H_uu = act.ddtorq_dudu(0.5, x)
    assert H_uu.shape == (3, 1, 1)
    assert np.all(np.isfinite(H_uu))
    assert np.allclose(H_uu, 0.0)

    H_ux = act.ddtorq_dudbasestate(0.5, x)
    assert H_ux.shape == (3, 1, 7)
    assert np.all(np.isfinite(H_ux))
    assert np.allclose(H_ux, 0.0)

    H_xx = act.ddtorq_dbasestatedbasestate(0.5, x)
    assert H_xx.shape == (3, 7, 7)
    assert np.all(np.isfinite(H_xx))
    assert np.allclose(H_xx, 0.0)


def test_actuator_input_len_constant():
    assert saltro_py.Actuator.input_len == 1
