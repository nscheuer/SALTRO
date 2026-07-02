import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))

import saltro_py


PI = 3.14159265358979323846


def make_simple_satellite():
    sat = saltro_py.Satellite()
    J = np.array([
        [0.05, 0.0, 0.0],
        [0.0, 0.06, 0.0],
        [0.0, 0.0, 0.04],
    ], dtype=float)
    sat.setInertia(J)
    sat.addMTQ(np.array([1.0, 0.0, 0.0]), 0.2)
    sat.addMTQ(np.array([0.0, 1.0, 0.0]), 0.2)
    sat.addMTQ(np.array([0.0, 0.0, 1.0]), 0.2)
    sat.addRW(np.array([0.0, 0.0, 1.0]), 1.0e-3, 1.0e-5, 0.0, 1.6e-2)
    return sat


def make_satellite_with_magic():
    sat = make_simple_satellite()
    sat.addMagic(np.array([1.0, 0.0, 0.0]), 2.0e-2)
    sat.addMagic(np.array([0.0, 1.0, 0.0]), 2.0e-2)
    return sat


def make_state(omega, q):
    x = np.zeros(8, dtype=float)
    x[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3] = omega
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = q
    return x


def rotation_matrix(q):
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    q0, q1, q2, q3 = q
    return np.array([
        [1.0 - 2.0 * (q2 * q2 + q3 * q3), 2.0 * (q1 * q2 - q0 * q3), 2.0 * (q1 * q3 + q0 * q2)],
        [2.0 * (q1 * q2 + q0 * q3), 1.0 - 2.0 * (q1 * q1 + q3 * q3), 2.0 * (q2 * q3 - q0 * q1)],
        [2.0 * (q1 * q3 - q0 * q2), 2.0 * (q2 * q3 + q0 * q1), 1.0 - 2.0 * (q1 * q1 + q2 * q2)],
    ], dtype=float)


def random_unit(rng):
    v = np.array(rng.normal(0.0, 1.0, size=3), dtype=float)
    n = np.linalg.norm(v)
    if n > 1e-9:
        return v / n
    return np.array([0.0, 0.0, 1.0], dtype=float)


def test_pdcontroller_returns_small_command_when_aligned_quaternion_goal():
    sat = make_simple_satellite()
    pd = saltro_py.PDController(sat)

    x = make_state(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    B_eci = np.array([2.2e-5, -1.6e-5, 3.1e-5], dtype=float)
    q_goal = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    boresight = np.array([0.0, 0.0, 1.0], dtype=float)

    u = np.asarray(pd.find_u(x, B_eci, q_goal, boresight))
    assert u.size == 4
    assert np.linalg.norm(u) < 1e-6


def test_pdcontroller_accepts_nan_flagged_vector_goal_and_produces_finite_output():
    sat = make_simple_satellite()
    pd = saltro_py.PDController(sat)

    x = make_state(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    B_eci = np.array([2.2e-3, -1.6e-3, 3.1e-3], dtype=float)
    q_goal_vec = np.array([np.nan, 1.0, 0.0, 0.0], dtype=float)
    boresight = np.array([0.0, 0.0, 1.0], dtype=float)

    u = np.asarray(pd.find_u(x, B_eci, q_goal_vec, boresight))
    assert u.size == 4
    assert np.all(np.isfinite(u))
    assert np.linalg.norm(u) > 1e-9
    assert abs(u[0]) > abs(u[1])
    assert abs(u[0]) > abs(u[3])


def test_pdcontroller_vec_goal_returns_small_command_when_boresight_aligned():
    sat = make_simple_satellite()
    pd = saltro_py.PDController(sat)

    x = make_state(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    B_eci = np.array([2.2e-5, -1.6e-5, 3.1e-5], dtype=float)
    q_goal_vec = np.array([np.nan, 0.0, 0.0, 1.0], dtype=float)
    boresight = np.array([0.0, 0.0, 1.0], dtype=float)

    u = np.asarray(pd.find_u(x, B_eci, q_goal_vec, boresight))
    assert u.size == 4
    assert np.all(np.isfinite(u))
    assert np.linalg.norm(u) < 1e-6


def test_pdcontroller_vec_goal_handles_antipodal_singularity_finitely():
    sat = make_simple_satellite()
    pd = saltro_py.PDController(sat)

    x = make_state(np.array([0.0, 0.0, 0.05]), np.array([1.0, 0.0, 0.0, 0.0]))
    B_eci = np.array([2.2e-5, -1.6e-5, 3.1e-5], dtype=float)
    q_goal_vec = np.array([np.nan, 0.0, 0.0, -1.0], dtype=float)
    boresight = np.array([0.0, 0.0, 1.0], dtype=float)

    u = np.asarray(pd.find_u(x, B_eci, q_goal_vec, boresight))
    assert u.size == 4
    assert np.all(np.isfinite(u))


def test_pdcontroller_vec_goal_produces_direction_correct_torque_on_random_samples():
    seed = 20260526
    num_samples = 20

    rng = np.random.default_rng(seed)
    sat = make_simple_satellite()
    pd = saltro_py.PDController(sat)

    direction_correct = 0
    for _ in range(num_samples):
        axis = random_unit(rng)
        angle = float(rng.normal(0.0, 1.0) * PI)
        q = np.zeros(4, dtype=float)
        q[0] = math.cos(0.5 * angle)
        q[1:] = axis * math.sin(0.5 * angle)
        q /= np.linalg.norm(q)

        omega = rng.uniform(-0.05, 0.05, size=3)
        B_eci = random_unit(rng) * rng.uniform(1.0e-4, 2.0e-3)
        r_eci = random_unit(rng)
        bs = random_unit(rng)

        x = make_state(omega, q)
        q_goal_vec = np.array([np.nan, r_eci[0], r_eci[1], r_eci[2]], dtype=float)

        R = rotation_matrix(q)
        r_body = R.transpose() @ r_eci
        tau_des = pd.kp_q * np.cross(bs, r_body) - pd.kd_w * omega
        if np.linalg.norm(tau_des) < 1e-12:
            continue
        tau_des_hat = tau_des / np.linalg.norm(tau_des)

        u = np.asarray(pd.find_u(x, B_eci, q_goal_vec, bs))
        assert np.all(np.isfinite(u))
        tau_actual = np.asarray(sat.actuatorTorque(x, u, B_eci))
        assert np.all(np.isfinite(tau_actual))

        tau_actual_norm = np.linalg.norm(tau_actual)
        if tau_actual_norm > 1e-15:
            cos_sim = float(np.dot(tau_actual, tau_des_hat) / tau_actual_norm)
            if cos_sim > 0.0:
                direction_correct += 1

    assert direction_correct >= num_samples * 3 // 4


def test_warm_start_dispatches_initcontroller_3_to_pdcontroller():
    sat = make_simple_satellite()

    settings = saltro_py.PlannerSettings()
    settings.init_traj.initcontroller = 3
    settings.num_passes = 1
    settings.passes[0].dt = 1.0

    N = 5
    dt_sec = 1.0
    sec_per_century = 36525.0 * 86400.0

    jtime = np.zeros(N, dtype=float)
    q_goal = np.zeros((4, N), dtype=float)
    boresight = np.zeros((3, N), dtype=float)
    R = np.zeros((3, N), dtype=float)
    V = np.zeros((3, N), dtype=float)
    B = np.zeros((3, N), dtype=float)
    S = np.zeros((3, N), dtype=float)
    rho = np.zeros(N, dtype=float)

    for k in range(N):
        jtime[k] = 0.25 + (k * dt_sec) / sec_per_century
        q_goal[0, k] = 1.0
        boresight[:, k] = np.array([0.0, 0.0, 1.0], dtype=float)
        B[:, k] = np.array([2.2e-5, -1.6e-5, 3.1e-5], dtype=float)

    x0 = make_state(np.array([0.05, -0.03, 0.04]), np.array([1.0, 0.0, 0.0, 0.0]))

    ok, X, U = saltro_py.warm_start(
        settings, sat, x0, jtime, q_goal, boresight, R, V, B, S, rho
    )
    assert ok
    assert np.all(np.isfinite(X))
    assert np.all(np.isfinite(U))


def test_pdcontroller_returns_full_controldim_output_when_magic_actuators_exist():
    sat = make_satellite_with_magic()
    pd = saltro_py.PDController(sat)

    x = make_state(np.array([0.02, -0.01, 0.03]), np.array([1.0, 0.0, 0.0, 0.0]))
    B_eci = np.array([2.2e-3, -1.6e-3, 3.1e-3], dtype=float)
    q_goal_vec = np.array([np.nan, 1.0, 0.0, 0.0], dtype=float)
    boresight = np.array([0.0, 0.0, 1.0], dtype=float)

    u = np.asarray(pd.find_u(x, B_eci, q_goal_vec, boresight))
    assert u.size == sat.controlDim
    assert np.all(np.isfinite(u))

    tau_actual = np.asarray(sat.actuatorTorque(x, u, B_eci))
    assert np.all(np.isfinite(tau_actual))
    assert np.linalg.norm(tau_actual) > 1e-9


def test_warm_start_initcontroller_3_works_with_magic_actuators_present():
    sat = make_satellite_with_magic()

    settings = saltro_py.PlannerSettings()
    settings.init_traj.initcontroller = 3
    settings.num_passes = 1
    settings.passes[0].dt = 1.0

    N = 5
    dt_sec = 1.0
    sec_per_century = 36525.0 * 86400.0

    jtime = np.zeros(N, dtype=float)
    q_goal = np.zeros((4, N), dtype=float)
    boresight = np.zeros((3, N), dtype=float)
    R = np.zeros((3, N), dtype=float)
    V = np.zeros((3, N), dtype=float)
    B = np.zeros((3, N), dtype=float)
    S = np.zeros((3, N), dtype=float)
    rho = np.zeros(N, dtype=float)

    for k in range(N):
        jtime[k] = 0.25 + (k * dt_sec) / sec_per_century
        q_goal[0, k] = 1.0
        boresight[:, k] = np.array([0.0, 0.0, 1.0], dtype=float)
        B[:, k] = np.array([2.2e-5, -1.6e-5, 3.1e-5], dtype=float)

    x0 = make_state(np.array([0.05, -0.03, 0.04]), np.array([1.0, 0.0, 0.0, 0.0]))

    ok, X, U = saltro_py.warm_start(
        settings, sat, x0, jtime, q_goal, boresight, R, V, B, S, rho
    )
    assert ok
    assert np.all(np.isfinite(X))
    assert np.all(np.isfinite(U))
