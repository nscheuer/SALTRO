import numpy as np
import saltro_py


def make_face(area, centroid, normal, cd):
    return saltro_py.GeometryFace(area, centroid, normal, 0.0, 0.0, 0.0, cd)


def make_dist_cfg(plan_for_aero):
    cfg = saltro_py.DisturbanceConfig()
    cfg.plan_for_aero = plan_for_aero
    return cfg


def test_dragdisturbance_returns_zero_when_disabled():
    config = saltro_py.GeometryConfig()
    config.addFace(make_face(1.0, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), 2.0))

    drag = saltro_py.DragDisturbance(config)
    dist = make_dist_cfg(False)

    x = np.zeros(7)
    v_body = np.array([0.0, 3.0, 0.0])

    torque = drag.torque(x, dist, v_body)
    assert np.allclose(torque, np.zeros(3))


def test_dragdisturbance_returns_zero_when_inactive():
    config = saltro_py.GeometryConfig()
    config.addFace(make_face(1.0, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), 2.0))

    drag = saltro_py.DragDisturbance(config)
    drag.setActive(False)
    dist = make_dist_cfg(True)

    x = np.zeros(7)
    v_body = np.array([0.0, 3.0, 0.0])

    torque = drag.torque(x, dist, v_body)
    assert np.allclose(torque, np.zeros(3))


def test_dragdisturbance_single_face_torque_matches_expected():
    config = saltro_py.GeometryConfig()
    config.addFace(make_face(2.0, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), 2.0))

    drag = saltro_py.DragDisturbance(config)
    dist = make_dist_cfg(True)

    x = np.zeros(7)
    v_body = np.array([0.0, 3.0, 0.0])

    torque = drag.torque(x, dist, v_body)
    expected = np.array([0.0, 0.0, -18.0])

    assert np.allclose(torque, expected, rtol=1e-12, atol=0.0)


def test_dragdisturbance_ignores_faces_with_negative_incidence():
    config = saltro_py.GeometryConfig()
    config.addFace(make_face(1.0, np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]), 2.0))

    drag = saltro_py.DragDisturbance(config)
    dist = make_dist_cfg(True)

    x = np.zeros(7)
    v_body = np.array([1.0, 0.0, 0.0])

    torque = drag.torque(x, dist, v_body)
    assert np.allclose(torque, np.zeros(3))


def test_dragdisturbance_sums_torque_across_multiple_faces():
    config = saltro_py.GeometryConfig()
    config.addFace(make_face(1.0, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), 1.0))
    config.addFace(make_face(2.0, np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0]), 2.0))

    drag = saltro_py.DragDisturbance(config)
    dist = make_dist_cfg(True)

    x = np.zeros(7)
    v_body = np.array([0.0, 2.0, 0.0])

    torque = drag.torque(x, dist, v_body)

    inc = 2.0
    summed = (1.0 * 1.0 * inc) * np.array([1.0, 0.0, 0.0]) + (2.0 * 2.0 * inc) * np.array([0.0, 1.0, 0.0])
    expected = -0.5 * np.cross(summed, v_body)

    assert np.allclose(torque, expected, rtol=1e-12, atol=0.0)


def test_dragdisturbance_jacobian_is_zero_when_dv_dq_is_zero():
    config = saltro_py.GeometryConfig()
    config.addFace(make_face(1.0, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), 1.0))

    drag = saltro_py.DragDisturbance(config)
    dist = make_dist_cfg(True)

    x = np.zeros(7)
    v_body = np.array([0.0, 2.0, 0.0])
    dV_dq = np.zeros((3, 4))

    jac = drag.dtorque_dq(x, dist, v_body, dV_dq)
    assert np.allclose(jac, np.zeros((3, 4)))


def test_dragdisturbance_jacobian_calculation_with_nonzero_dv_dq():
    config = saltro_py.GeometryConfig()
    config.addFace(make_face(2.0, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), 1.0))

    drag = saltro_py.DragDisturbance(config)
    dist = make_dist_cfg(True)

    x = np.zeros(7)
    v_body = np.array([0.0, 3.0, 0.0])

    # Set up dV_dq: how velocity changes with state
    dV_dq = np.zeros((3, 4))
    dV_dq[0, 0] = 1.0   # dv_x / dq_0 = 1.0
    dV_dq[1, 1] = 2.0   # dv_y / dq_1 = 2.0
    dV_dq[2, 2] = 1.5   # dv_z / dq_2 = 1.5

    jac = drag.dtorque_dq(x, dist, v_body, dV_dq)

    # Manually compute expected jacobian
    # C = CD * A * (n . v) * centroid = 1.0 * 2.0 * 3.0 * (1.0, 0.0, 0.0) = (6.0, 0.0, 0.0)
    # tau = -0.5 * C.cross(v) = -0.5 * (6,0,0).cross(0,3,0) = -0.5 * (0,0,18) = (0,0,-9)
    #
    # dtau/dq_j = -0.5 * (dC/dq_j . cross(v) + C . cross(dv/dq_j))
    # 
    # dC/dq_0 = CD * A * (n . dv/dq_0) * centroid = 1.0 * 2.0 * (0,1,0).(1,0,0) * (1,0,0) = (0,0,0)
    # dC/dq_1 = CD * A * (n . dv/dq_1) * centroid = 1.0 * 2.0 * (0,1,0).(0,2,0) * (1,0,0) = (4,0,0)
    # dC/dq_2 = CD * A * (n . dv/dq_2) * centroid = 1.0 * 2.0 * (0,1,0).(0,0,1.5) * (1,0,0) = (0,0,0)
    #
    # dtau/dq_0 = -0.5 * ((0,0,0).cross(0,3,0) + (6,0,0).cross(1,0,0)) = -0.5 * (0 + (0,0,0)) = (0,0,0)
    # dtau/dq_1 = -0.5 * ((4,0,0).cross(0,3,0) + (6,0,0).cross(0,2,0)) = -0.5 * ((0,0,12) + (0,0,12)) = (0,0,-12)
    # dtau/dq_2 = -0.5 * ((0,0,0).cross(0,3,0) + (6,0,0).cross(0,0,1.5)) = -0.5 * (0 + (0,-9,0)) = (0,4.5,0)
    # dtau/dq_3 = -0.5 * ((0,0,0).cross(0,3,0) + (6,0,0).cross(0,0,0)) = (0,0,0)

    expected = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 4.5, 0.0],
        [0.0, -12.0, 0.0, 0.0]
    ])

    assert np.allclose(jac, expected, rtol=1e-12, atol=1e-12)


def test_dragdisturbance_hessian_is_zero_when_derivatives_are_zero():
    config = saltro_py.GeometryConfig()
    config.addFace(make_face(1.0, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), 1.0))

    drag = saltro_py.DragDisturbance(config)
    dist = make_dist_cfg(True)

    x = np.zeros(7)
    v_body = np.array([0.0, 2.0, 0.0])
    dV_dq = np.zeros((3, 4))
    d2V_dq2 = [np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 4))]

    hess = drag.ddtorque_dqdq(x, dist, v_body, dV_dq, d2V_dq2)
    for k in range(3):
        assert np.allclose(hess[k], np.zeros((4, 4)))


def test_dragdisturbance_hessian_calculation_with_nonzero_derivatives():
    config = saltro_py.GeometryConfig()
    config.addFace(make_face(1.0, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), 2.0))

    drag = saltro_py.DragDisturbance(config)
    dist = make_dist_cfg(True)

    x = np.zeros(7)
    v_body = np.array([0.0, 2.0, 0.0])

    # First derivative of velocity with respect to state
    dV_dq = np.zeros((3, 4))
    dV_dq[0, 0] = 1.0   # dv_x / dq_0 = 1.0
    dV_dq[1, 1] = 1.0   # dv_y / dq_1 = 1.0
    dV_dq[2, 2] = 1.0   # dv_z / dq_2 = 1.0

    # Second derivatives of velocity with respect to state pairs
    d2V_dq2 = [np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 4))]
    # d²v_y / dq_0 dq_0 = 0.5 (constant second derivative)
    d2V_dq2[1][0, 0] = 0.5
    d2V_dq2[1][1, 0] = 0.5
    d2V_dq2[1][0, 1] = 0.5
    d2V_dq2[1][1, 1] = 0.5

    hess = drag.ddtorque_dqdq(x, dist, v_body, dV_dq, d2V_dq2)

    # Basic verifications:
    # 1. When velocity derivatives are symmetric and simple, second derivatives should follow pattern
    # 2. The hessian should be non-zero due to the non-zero d2V_dq2
    has_nonzero = False
    for k in range(3):
        for i in range(4):
            for j in range(4):
                if np.abs(hess[k][i, j]) > 1e-12:
                    has_nonzero = True
                    break
            if has_nonzero:
                break
        if has_nonzero:
            break
    assert has_nonzero

    # Verify specific elements based on the geometry and derivatives
    # C = CD * A * (n . v) * centroid = 2.0 * 1.0 * (0,1,0).(0,2,0) * (1,0,0) = (4,0,0)
    # tau = -0.5 * (4,0,0).cross(0,2,0) = -0.5 * (0,0,8) = (0,0,-4)
    #
    # For mixed partials involving y-velocity second derivatives
    # The y-component (index 1) of the hessian should show contributions from d2V_dq2[1]
    
    # Check that mixing q_0 and q_1 with nonzero d2V_dq2[1] gives nonzero contributions
    has_nonzero_mixed = (
        np.abs(hess[1][0, 1]) > 1e-12 or np.abs(hess[2][0, 1]) > 1e-12 or
        np.abs(hess[0][0, 1]) > 1e-12 or 
        np.abs(hess[1][1, 0]) > 1e-12 or np.abs(hess[2][1, 0]) > 1e-12 or
        np.abs(hess[0][1, 0]) > 1e-12
    )
    assert has_nonzero_mixed
