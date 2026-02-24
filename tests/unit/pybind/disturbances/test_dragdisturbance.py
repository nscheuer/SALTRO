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
