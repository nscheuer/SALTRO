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


def test_rw_torque_is_independent_of_attitude_quaternion():
    """RW torque is body-fixed: τ = u·axis_hat. It must NOT depend on the
    attitude quaternion. This guards against accidental world-frame
    rotation of the body torque."""
    axis = np.array([0.5, 0.2, np.sqrt(1 - 0.25 - 0.04)])
    rw = saltro_py.RW(axis, 0.05, 1e-5, 0.0, 0.1)

    rng = np.random.default_rng(20260514)
    u = 0.012

    quats = [
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.707, 0.707, 0.0, 0.0]),
        np.array([0.5, 0.5, 0.5, 0.5]),
        rng.normal(size=4),
    ]

    tau_ref = None
    for q_raw in quats:
        q = q_raw / np.linalg.norm(q_raw)
        x = np.zeros(7)
        x[3:7] = q

        tau = rw.torque(u, x)
        if tau_ref is None:
            tau_ref = tau
            # Sanity: matches the documented formula
            assert np.allclose(tau, u * rw.axis, atol=1e-15)
        else:
            assert np.allclose(tau, tau_ref, atol=1e-15), (
                f"RW torque changed with q: q={q}, τ={tau}, ref={tau_ref}"
            )


def test_rw_torque_scales_linearly_with_u():
    """τ_RW = u · axis must be linear in u (no clamping inside .torque)."""
    axis = np.array([0.0, 1.0, 0.0])
    rw = saltro_py.RW(axis, 0.01, 2e-5, 0.0, 0.1)
    x = valid_base_state()

    tau_unit = rw.torque(1.0, x)
    for u in [-3.7, -0.5, 0.0, 0.1, 2.4]:
        assert np.allclose(rw.torque(u, x), u * tau_unit, atol=1e-15)


def test_rw_storage_torque_is_negative_input():
    """Storage torque (rate of change of h) equals -u for the wheel."""
    axis = valid_axis()
    rw = saltro_py.RW(axis, 0.01, 2.0e-5, 0.0, 0.1)
    x = valid_base_state()

    for u in [-0.05, -0.001, 0.0, 0.002, 0.04]:
        assert rw.storageTorque(u, x) == -u


def test_rw_hessians_are_callable_finite_and_zero():
    """Smoke test: RW inherits the Tensor3-returning ddtorq_* Hessians from the
    Actuator base class; they must be bound (with the Tensor3 caster registered)
    and return finite, correctly-shaped zeros -- RW torque is affine in u and
    independent of the base state."""
    axis = valid_axis()
    rw = saltro_py.RW(axis, 0.01, 2.0e-5, 0.0, 0.1)
    x = valid_base_state()

    # Tensor3<R,C,S> -> numpy (slices, rows, cols)
    H_uu = rw.ddtorq_dudu(0.5, x)
    assert H_uu.shape == (3, 1, 1)
    assert np.all(np.isfinite(H_uu))
    assert np.allclose(H_uu, 0.0)

    H_ux = rw.ddtorq_dudbasestate(0.5, x)
    assert H_ux.shape == (3, 1, 7)
    assert np.all(np.isfinite(H_ux))
    assert np.allclose(H_ux, 0.0)

    H_xx = rw.ddtorq_dbasestatedbasestate(0.5, x)
    assert H_xx.shape == (3, 7, 7)
    assert np.all(np.isfinite(H_xx))
    assert np.allclose(H_xx, 0.0)


def test_rw_hessians_match_finite_difference_of_jacobians():
    """FD cross-check: differencing dtorq_du / dtorq_dbasestate must reproduce
    the (zero) Hessians, confirming the bound accessors agree with the bound
    first derivatives."""
    axis = np.array([0.0, 1.0, 0.0])
    rw = saltro_py.RW(axis, 0.01, 2.0e-5, 0.0, 0.1)
    x = valid_base_state()
    u = 0.3
    h = 1e-6

    # d(dtorq_du)/du vs ddtorq_dudu: slice s holds d^2 tau_s / du^2 (1x1).
    fd_uu = (rw.dtorq_du(u + h, x) - rw.dtorq_du(u - h, x)) / (2 * h)  # (3, 1)
    H_uu = rw.ddtorq_dudu(u, x)  # (3, 1, 1)
    assert np.allclose(H_uu[:, 0, 0], fd_uu.ravel(), atol=1e-9)

    # d(dtorq_dbasestate)/du vs ddtorq_dudbasestate: slice s is (1, 7).
    # dtorq_dbasestate returns (7, 3): rows are base states, cols torque comps.
    fd_ux = (rw.dtorq_dbasestate(u + h, x) - rw.dtorq_dbasestate(u - h, x)) / (2 * h)  # (7, 3)
    H_ux = rw.ddtorq_dudbasestate(u, x)  # (3, 1, 7)
    assert np.allclose(H_ux[:, 0, :], fd_ux.T, atol=1e-9)

    # d(dtorq_dbasestate)/dx_j vs ddtorq_dbasestatedbasestate: slice s is (7, 7).
    H_xx = rw.ddtorq_dbasestatedbasestate(u, x)  # (3, 7, 7)
    for j in range(7):
        dx = np.zeros(7)
        dx[j] = h
        fd_xx_j = (rw.dtorq_dbasestate(u, x + dx) - rw.dtorq_dbasestate(u, x - dx)) / (2 * h)  # (7, 3)
        assert np.allclose(H_xx[:, :, j], fd_xx_j.T, atol=1e-9)
