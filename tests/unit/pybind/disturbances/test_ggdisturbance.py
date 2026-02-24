import numpy as np
import saltro_py


MU_EARTH = 3.986004418e14  # m^3/s^2


def make_dist_cfg(plan_for_gg):
    cfg = saltro_py.DisturbanceConfig()
    cfg.plan_for_gg = plan_for_gg
    return cfg


def test_ggdisturbance_returns_zero_when_disabled():
    J = np.eye(3)
    gg = saltro_py.GGDisturbance(J)
    dist = make_dist_cfg(False)

    x = np.zeros(7)
    r_body = np.array([1.0e6, 0.0, 0.0])

    torque = gg.torque(x, dist, r_body, J)
    assert np.allclose(torque, np.zeros(3))


def test_ggdisturbance_returns_zero_when_inactive():
    J = np.eye(3)
    gg = saltro_py.GGDisturbance(J)
    gg.setActive(False)
    dist = make_dist_cfg(True)

    x = np.zeros(7)
    r_body = np.array([1.0e6, 0.0, 0.0])

    torque = gg.torque(x, dist, r_body, J)
    assert np.allclose(torque, np.zeros(3))


def test_ggdisturbance_returns_zero_for_zero_position():
    J = np.eye(3)
    gg = saltro_py.GGDisturbance(J)
    dist = make_dist_cfg(True)

    x = np.zeros(7)
    r_body = np.zeros(3)

    torque = gg.torque(x, dist, r_body, J)
    assert np.allclose(torque, np.zeros(3))


def test_ggdisturbance_returns_zero_for_spherical_inertia():
    """Spherical satellite (J ∝ I) produces zero GG torque"""
    J = 10.0 * np.eye(3)
    gg = saltro_py.GGDisturbance(J)
    dist = make_dist_cfg(True)

    x = np.zeros(7)
    r_body = np.array([6.5e6, 1.0e6, 2.0e6])

    torque = gg.torque(x, dist, r_body, J)
    
    # For spherical inertia, J*n = c*n, so n × (J*n) = n × (c*n) = 0
    assert np.linalg.norm(torque) < 1e-6


def test_ggdisturbance_torque_for_elongated_satellite():
    """Elongated satellite along x-axis"""
    J = np.array([[1.0, 0.0, 0.0],
                  [0.0, 10.0, 0.0],
                  [0.0, 0.0, 10.0]])

    gg = saltro_py.GGDisturbance(J)
    dist = make_dist_cfg(True)

    x = np.zeros(7)
    
    # Position along z-axis (nadir = -z direction)
    r = 7.0e6  # meters
    r_body = np.array([0.0, 0.0, r])

    torque = gg.torque(x, dist, r_body, J)

    # nadir = (0, 0, -1)
    # J * nadir = (0, 0, -10)
    # nadir × (J * nadir) = (0, 0, -1) × (0, 0, -10) = (0, 0, 0)
    # Torque should be zero when aligned with principal axis
    assert np.linalg.norm(torque) < 1e-6


def test_ggdisturbance_torque_for_misaligned_satellite():
    """Elongated satellite along x-axis"""
    J = np.array([[1.0, 0.0, 0.0],
                  [0.0, 10.0, 0.0],
                  [0.0, 0.0, 10.0]])

    gg = saltro_py.GGDisturbance(J)
    dist = make_dist_cfg(True)

    x = np.zeros(7)
    
    # Position at 45 degrees in x-z plane
    r = 7.0e6
    r_body = np.array([r / np.sqrt(2.0), 0.0, r / np.sqrt(2.0)])

    torque = gg.torque(x, dist, r_body, J)

    # Compute expected torque manually
    r_hat = r_body / np.linalg.norm(r_body)
    nadir = -r_hat
    const_term = 3.0 * MU_EARTH / (r * r * r)
    expected = const_term * np.cross(nadir, J @ nadir)

    assert np.allclose(torque, expected, rtol=1e-10, atol=0.0)


def test_ggdisturbance_torque_magnitude_scales_with_inverse_r3():
    """Torque should scale as 1/r^3"""
    J = np.array([[1.0, 0.0, 0.0],
                  [0.0, 5.0, 0.0],
                  [0.0, 0.0, 8.0]])

    gg = saltro_py.GGDisturbance(J)
    dist = make_dist_cfg(True)

    x = np.zeros(7)
    
    r1 = 7.0e6
    r2 = 14.0e6  # 2x distance
    
    r_body1 = np.array([r1 / np.sqrt(3.0), r1 / np.sqrt(3.0), r1 / np.sqrt(3.0)])
    r_body2 = np.array([r2 / np.sqrt(3.0), r2 / np.sqrt(3.0), r2 / np.sqrt(3.0)])

    torque1 = gg.torque(x, dist, r_body1, J)
    torque2 = gg.torque(x, dist, r_body2, J)

    # Torque should scale as 1/r^3, so doubling distance should reduce by factor of 8
    ratio = np.linalg.norm(torque1) / np.linalg.norm(torque2)
    assert np.allclose(ratio, 8.0, rtol=1e-10, atol=0.0)


def test_ggdisturbance_jacobian_is_zero_when_dr_dq_is_zero():
    J = np.array([[2.0, 0.0, 0.0],
                  [0.0, 3.0, 0.0],
                  [0.0, 0.0, 4.0]])

    gg = saltro_py.GGDisturbance(J)
    dist = make_dist_cfg(True)

    x = np.zeros(7)
    r_body = np.array([7.0e6, 0.0, 0.0])
    dr_dq = np.zeros((3, 4))

    jac = gg.dtorque_dq(x, dist, r_body, J, dr_dq)
    
    assert np.allclose(jac, np.zeros((3, 4)))


def test_ggdisturbance_jacobian_returns_zero_when_disabled():
    J = np.eye(3)
    gg = saltro_py.GGDisturbance(J)
    dist = make_dist_cfg(False)

    x = np.zeros(7)
    r_body = np.array([7.0e6, 1.0e6, 0.0])
    dr_dq = np.random.randn(3, 4)

    jac = gg.dtorque_dq(x, dist, r_body, J, dr_dq)
    
    assert np.allclose(jac, np.zeros((3, 4)))


def test_ggdisturbance_jacobian_has_correct_dimensions():
    J = np.array([[2.0, 0.1, 0.0],
                  [0.1, 3.0, 0.2],
                  [0.0, 0.2, 4.0]])

    gg = saltro_py.GGDisturbance(J)
    dist = make_dist_cfg(True)

    x = np.zeros(7)
    r_body = np.array([6.8e6, 1.0e6, 0.5e6])
    
    # Create a non-trivial dr_dq
    dr_dq = np.array([[1.0, 0.5, 0.2, 0.1],
                      [0.3, 1.0, 0.4, 0.2],
                      [0.1, 0.2, 1.0, 0.5]]) * 1.0e5

    jac = gg.dtorque_dq(x, dist, r_body, J, dr_dq)
    
    # Jacobian should be 3x4
    assert jac.shape == (3, 4)
    
    # Should not be all zeros for non-trivial inputs
    assert np.linalg.norm(jac) > 0.0


def test_ggdisturbance_jacobian_numerical_check():
    """Verify Jacobian against numerical finite differences"""
    J = np.array([[5.0, 0.0, 0.0],
                  [0.0, 8.0, 0.0],
                  [0.0, 0.0, 10.0]])

    gg = saltro_py.GGDisturbance(J)
    dist = make_dist_cfg(True)

    x = np.zeros(7)
    r_body = np.array([7.0e6, 1.0e6, 0.5e6])
    
    dr_dq = np.array([[2.0e5, 1.0e5, 0.5e5, 0.2e5],
                      [1.0e5, 2.0e5, 1.0e5, 0.5e5],
                      [0.5e5, 1.0e5, 2.0e5, 1.0e5]])

    jac = gg.dtorque_dq(x, dist, r_body, J, dr_dq)
    
    # Numerical derivative check
    eps = 1.0e-6
    for j in range(4):
        r_plus = r_body + eps * dr_dq[:, j]
        r_minus = r_body - eps * dr_dq[:, j]
        
        torque_plus = gg.torque(x, dist, r_plus, J)
        torque_minus = gg.torque(x, dist, r_minus, J)
        
        numerical_deriv = (torque_plus - torque_minus) / (2.0 * eps)
        
        assert np.allclose(jac[:, j], numerical_deriv, rtol=1e-5, atol=0.0)


def test_ggdisturbance_hessian_is_zero_when_derivatives_are_zero():
    J = np.array([[2.0, 0.0, 0.0],
                  [0.0, 3.0, 0.0],
                  [0.0, 0.0, 4.0]])

    gg = saltro_py.GGDisturbance(J)
    dist = make_dist_cfg(True)

    x = np.zeros(7)
    r_body = np.array([7.0e6, 1.0e6, 0.0])
    dr_dq = np.zeros((3, 4))
    d2r_dq2 = [np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 4))]

    hess = gg.ddtorque_dqdq(x, dist, r_body, J, dr_dq, d2r_dq2)

    for k in range(3):
        assert np.allclose(hess[k], np.zeros((4, 4)))


def test_ggdisturbance_hessian_returns_zero_when_disabled():
    J = np.eye(3)
    gg = saltro_py.GGDisturbance(J)
    dist = make_dist_cfg(False)

    x = np.zeros(7)
    r_body = np.array([7.0e6, 1.0e6, 0.0])
    dr_dq = np.random.randn(3, 4)
    d2r_dq2 = [np.random.randn(4, 4), np.random.randn(4, 4), np.random.randn(4, 4)]

    hess = gg.ddtorque_dqdq(x, dist, r_body, J, dr_dq, d2r_dq2)

    for k in range(3):
        assert np.allclose(hess[k], np.zeros((4, 4)))


def test_ggdisturbance_hessian_has_correct_dimensions():
    J = np.array([[3.0, 0.1, 0.0],
                  [0.1, 5.0, 0.2],
                  [0.0, 0.2, 7.0]])

    gg = saltro_py.GGDisturbance(J)
    dist = make_dist_cfg(True)

    x = np.zeros(7)
    r_body = np.array([6.8e6, 1.0e6, 0.5e6])
    
    dr_dq = np.array([[1.0e5, 0.5e5, 0.2e5, 0.1e5],
                      [0.3e5, 1.0e5, 0.4e5, 0.2e5],
                      [0.1e5, 0.2e5, 1.0e5, 0.5e5]])

    d2r_dq2 = [1.0e4 * np.random.randn(4, 4) for _ in range(3)]

    hess = gg.ddtorque_dqdq(x, dist, r_body, J, dr_dq, d2r_dq2)
    
    # Each slice should be 4x4
    for k in range(3):
        assert hess[k].shape == (4, 4)


def test_ggdisturbance_hessian_is_symmetric():
    """Hessian should be symmetric: H_ij = H_ji"""
    J = np.array([[4.0, 0.0, 0.0],
                  [0.0, 6.0, 0.0],
                  [0.0, 0.0, 9.0]])

    gg = saltro_py.GGDisturbance(J)
    dist = make_dist_cfg(True)

    x = np.zeros(7)
    r_body = np.array([6.8e6, 1.2e6, 0.8e6])
    
    dr_dq = np.array([[2.0e5, 1.0e5, 0.5e5, 0.3e5],
                      [1.5e5, 2.0e5, 1.0e5, 0.6e5],
                      [0.8e5, 1.2e5, 2.0e5, 1.5e5]])

    # Create symmetric second derivatives
    d2r_dq2 = []
    for k in range(3):
        temp = np.random.randn(4, 4)
        d2r_dq2.append(1.0e4 * (temp + temp.T))

    hess = gg.ddtorque_dqdq(x, dist, r_body, J, dr_dq, d2r_dq2)
    
    # Each slice should be symmetric
    for k in range(3):
        assert np.allclose(hess[k], hess[k].T, rtol=1e-10, atol=1e-10)


def test_ggdisturbance_hessian_numerical_check():
    """Verify Hessian against numerical finite differences"""
    J = np.array([[3.0, 0.0, 0.0],
                  [0.0, 5.0, 0.0],
                  [0.0, 0.0, 7.0]])

    gg = saltro_py.GGDisturbance(J)
    dist = make_dist_cfg(True)

    x = np.zeros(7)
    r_body = np.array([7.0e6, 1.0e6, 0.5e6])
    
    dr_dq = np.array([[1.5e5, 0.8e5, 0.4e5, 0.2e5],
                      [0.9e5, 1.5e5, 0.7e5, 0.4e5],
                      [0.4e5, 0.6e5, 1.5e5, 0.8e5]])

    d2r_dq2 = [np.zeros((4, 4)) for _ in range(3)]

    hess = gg.ddtorque_dqdq(x, dist, r_body, J, dr_dq, d2r_dq2)
    
    # Numerical derivative check (2nd derivative w.r.t. first quaternion element)
    eps = 1.0e-6
    r_plus = r_body + eps * dr_dq[:, 0]
    r_minus = r_body - eps * dr_dq[:, 0]
    
    jac_plus = gg.dtorque_dq(x, dist, r_plus, J, dr_dq)
    jac_minus = gg.dtorque_dq(x, dist, r_minus, J, dr_dq)
    
    numerical_hess_slice = (jac_plus - jac_minus) / (2.0 * eps)
    
    # Check first row of Hessian (corresponds to first component of torque)
    for j in range(4):
        analytical = hess[0][0, j]
        numerical = numerical_hess_slice[0, j]
        
        # Use absolute tolerance for values near zero, relative for larger values
        if abs(analytical) < 1e-6 and abs(numerical) < 1e-6:
            assert abs(analytical - numerical) < 1e-6
        else:
            assert np.allclose(analytical, numerical, rtol=1e-3, atol=0.0)
