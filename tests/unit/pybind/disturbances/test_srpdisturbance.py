import numpy as np
import saltro_py


SOLAR_CONSTANT = 1361.0
C_LIGHT = 299792458.0
P_SRP = SOLAR_CONSTANT / C_LIGHT


def make_face(area, centroid, normal, eta_s, eta_d, eta_a):
    return saltro_py.GeometryFace(
        area,
        np.array(centroid, dtype=float),
        np.array(normal, dtype=float),
        float(eta_s),
        float(eta_d),
        float(eta_a),
        0.0,
    )


def make_dist_cfg(plan_for_srp):
    cfg = saltro_py.DisturbanceConfig()
    cfg.plan_for_srp = plan_for_srp
    return cfg


def torque_from_dq(srp, dist, v0, dV_dq, dq):
    x = np.zeros(7)
    v = v0 + dV_dq @ dq
    return srp.torque(x, dist, v)


def test_srpdisturbance_returns_zero_when_disabled():
    config = saltro_py.GeometryConfig()
    config.addFace(make_face(1.0, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.2, 0.3, 0.1))

    srp = saltro_py.SRPDisturbance(config)
    dist = make_dist_cfg(False)

    x = np.zeros(7)
    v_body = np.array([1.0, 0.0, 0.0])

    torque = srp.torque(x, dist, v_body)
    assert np.allclose(torque, np.zeros(3))


def test_srpdisturbance_returns_zero_when_inactive():
    config = saltro_py.GeometryConfig()
    config.addFace(make_face(1.0, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.2, 0.3, 0.1))

    srp = saltro_py.SRPDisturbance(config)
    srp.setActive(False)
    dist = make_dist_cfg(True)

    x = np.zeros(7)
    v_body = np.array([1.0, 0.0, 0.0])

    torque = srp.torque(x, dist, v_body)
    assert np.allclose(torque, np.zeros(3))


def test_srpdisturbance_returns_zero_for_near_zero_sun_vector():
    config = saltro_py.GeometryConfig()
    config.addFace(make_face(1.0, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.2, 0.3, 0.1))

    srp = saltro_py.SRPDisturbance(config)
    dist = make_dist_cfg(True)

    x = np.zeros(7)
    v_body = np.array([0.0, 0.0, 0.0])

    torque = srp.torque(x, dist, v_body)
    assert np.allclose(torque, np.zeros(3))


def test_srpdisturbance_ignores_faces_with_zero_area():
    config = saltro_py.GeometryConfig()
    config.addFace(make_face(0.0, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.2, 0.3, 0.1))

    srp = saltro_py.SRPDisturbance(config)
    dist = make_dist_cfg(True)

    x = np.zeros(7)
    v_body = np.array([1.0, 0.0, 0.0])

    torque = srp.torque(x, dist, v_body)
    assert np.allclose(torque, np.zeros(3))


def test_srpdisturbance_ignores_faces_with_invalid_normals():
    config = saltro_py.GeometryConfig()
    config.addFace(make_face(1.0, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.2, 0.3, 0.1))

    srp = saltro_py.SRPDisturbance(config)
    dist = make_dist_cfg(True)

    x = np.zeros(7)
    v_body = np.array([1.0, 0.0, 0.0])

    torque = srp.torque(x, dist, v_body)
    assert np.allclose(torque, np.zeros(3))


def test_srpdisturbance_ignores_negative_incidence():
    config = saltro_py.GeometryConfig()
    config.addFace(make_face(1.0, [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], 0.2, 0.3, 0.1))

    srp = saltro_py.SRPDisturbance(config)
    dist = make_dist_cfg(True)

    x = np.zeros(7)
    v_body = np.array([1.0, 0.0, 0.0])

    torque = srp.torque(x, dist, v_body)
    assert np.allclose(torque, np.zeros(3))


def test_srpdisturbance_single_face_torque_matches_expected():
    config = saltro_py.GeometryConfig()
    config.addFace(make_face(2.0, [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], 0.1, 0.3, 0.2))

    srp = saltro_py.SRPDisturbance(config)
    dist = make_dist_cfg(True)

    x = np.zeros(7)
    v_body = np.array([1.0, 0.0, 0.0])

    m_s = 2.0 * (0.2 + 0.3)
    m_n = 2.0 * (2.0 * 0.1 * 1.0 + (2.0 / 3.0) * 0.3)
    expected_z = P_SRP * (m_s + m_n)

    torque = srp.torque(x, dist, v_body)

    assert abs(torque[0]) < 1e-12
    assert abs(torque[1]) < 1e-12
    assert np.isclose(torque[2], expected_z, rtol=1e-12, atol=0.0)


def test_srpdisturbance_normalizes_face_normals():
    config = saltro_py.GeometryConfig()
    config.addFace(make_face(1.0, [0.0, 1.0, 0.0], [2.0, 0.0, 0.0], 0.0, 0.0, 1.0))

    srp = saltro_py.SRPDisturbance(config)
    dist = make_dist_cfg(True)

    x = np.zeros(7)
    v_body = np.array([1.0, 0.0, 0.0])

    expected_z = P_SRP * 1.0
    torque = srp.torque(x, dist, v_body)

    assert np.isclose(torque[2], expected_z, rtol=1e-12, atol=0.0)


def test_srpdisturbance_sums_torque_across_multiple_faces():
    config = saltro_py.GeometryConfig()
    config.addFace(make_face(1.0, [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], 0.0, 0.0, 1.0))
    config.addFace(make_face(2.0, [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], 0.0, 0.0, 1.0))

    srp = saltro_py.SRPDisturbance(config)
    dist = make_dist_cfg(True)

    x = np.zeros(7)
    v_body = np.array([1.0, 0.0, 0.0])

    t1 = np.cross(np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    t2 = np.cross(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
    expected = -P_SRP * (1.0 * t1 + 2.0 * t2)

    torque = srp.torque(x, dist, v_body)

    assert np.allclose(torque, expected, rtol=1e-12, atol=0.0)


def test_srpdisturbance_jacobian_is_zero_when_dv_dq_is_zero():
    config = saltro_py.GeometryConfig()
    config.addFace(make_face(1.0, [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], 0.1, 0.2, 0.3))

    srp = saltro_py.SRPDisturbance(config)
    dist = make_dist_cfg(True)

    x = np.zeros(7)
    v_body = np.array([1.0, 0.0, 0.0])
    dV_dq = np.zeros((3, 4))

    jac = srp.dtorque_dq(x, dist, v_body, dV_dq)
    assert np.allclose(jac, np.zeros((3, 4)))


def test_srpdisturbance_jacobian_matches_finite_difference():
    config = saltro_py.GeometryConfig()
    config.addFace(make_face(2.0, [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], 0.2, 0.1, 0.1))

    srp = saltro_py.SRPDisturbance(config)
    dist = make_dist_cfg(True)

    x = np.zeros(7)
    v_body = np.array([1.0, 0.2, -0.1])

    dV_dq = np.zeros((3, 4))
    dV_dq[0, 0] = 0.1
    dV_dq[1, 1] = -0.2
    dV_dq[2, 2] = 0.15
    dV_dq[0, 3] = -0.05

    eps = 1.0e-6
    jac = srp.dtorque_dq(x, dist, v_body, dV_dq)

    for j in range(4):
        dq = np.zeros(4)
        dq[j] = eps
        f_plus = torque_from_dq(srp, dist, v_body, dV_dq, dq)
        dq[j] = -eps
        f_minus = torque_from_dq(srp, dist, v_body, dV_dq, dq)
        fd = (f_plus - f_minus) / (2.0 * eps)

        assert np.allclose(jac[:, j], fd, rtol=1e-5, atol=0.0)


def test_srpdisturbance_hessian_is_zero_when_disabled():
    config = saltro_py.GeometryConfig()
    config.addFace(make_face(1.0, [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], 0.1, 0.2, 0.3))

    srp = saltro_py.SRPDisturbance(config)
    dist = make_dist_cfg(False)

    x = np.zeros(7)
    v_body = np.array([1.0, 0.0, 0.0])
    dV_dq = np.zeros((3, 4))
    d2V_dq2 = [np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 4))]

    hess = srp.ddtorque_dqdq(x, dist, v_body, dV_dq, d2V_dq2)
    for k in range(3):
        assert np.allclose(hess[k], np.zeros((4, 4)))


def test_srpdisturbance_hessian_matches_finite_difference():
    config = saltro_py.GeometryConfig()
    config.addFace(make_face(1.5, [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], 0.2, 0.2, 0.1))

    srp = saltro_py.SRPDisturbance(config)
    dist = make_dist_cfg(True)

    x = np.zeros(7)
    v_body = np.array([1.2, -0.3, 0.4])

    dV_dq = np.zeros((3, 4))
    dV_dq[0, 0] = 0.2
    dV_dq[1, 1] = -0.1
    dV_dq[2, 2] = 0.15
    dV_dq[0, 3] = -0.05

    d2V_dq2 = [np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 4))]

    eps = 2.0e-6
    hess = srp.ddtorque_dqdq(x, dist, v_body, dV_dq, d2V_dq2)

    def torque_at(dq):
        return torque_from_dq(srp, dist, v_body, dV_dq, dq)

    for i in range(2):
        for j in range(2):
            dq_pp = np.zeros(4)
            dq_pm = np.zeros(4)
            dq_mp = np.zeros(4)
            dq_mm = np.zeros(4)

            dq_pp[i] = eps
            dq_pp[j] = eps
            dq_pm[i] = eps
            dq_pm[j] = -eps
            dq_mp[i] = -eps
            dq_mp[j] = eps
            dq_mm[i] = -eps
            dq_mm[j] = -eps

            f_pp = torque_at(dq_pp)
            f_pm = torque_at(dq_pm)
            f_mp = torque_at(dq_mp)
            f_mm = torque_at(dq_mm)

            fd = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps * eps)

            tol_abs = 2e-7
            tol_rel = 1e-4

            for k in range(3):
                if abs(fd[k]) < tol_abs:
                    assert np.isclose(hess[k][i, j], fd[k], atol=tol_abs, rtol=0.0)
                else:
                    assert np.isclose(hess[k][i, j], fd[k], rtol=tol_rel, atol=0.0)
