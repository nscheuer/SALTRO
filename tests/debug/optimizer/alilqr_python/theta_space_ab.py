"""
θ-space vec-pointing cost prototype: end-to-end A/B.

Compares c-space (use_theta_cost_param=False) vs θ-space (True) on vector-pointing
slews solved with AL-iLQR, holding everything else fixed.

Scenarios:
  A) 90deg slew, type 3 (acos^2 / theta^2), GN Hessian   -> regression: match
  B) near-antipode (~179deg) slew, type 3, GN Hessian     -> HEADLINE
  C) near-antipode (~179deg) slew, type 3, FULL Newton    -> both (analytically equal)
"""
import sys
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "configs"))

import saltro_py
from sat_3_1_hybrid import create_satellite
from trajOpt import trajOpt

NAN = float("nan")


def base_settings(gn, theta_param, ftype=3, angle_w=2e2):
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = 2
    ps.num_passes = 1
    ps.passes[0].dt = 10.0
    ps.passes[0].ilqr.cost_tol = 1e-3
    ps.passes[0].ilqr.max_iters = 50
    ps.passes[0].auglag.max_outer_iters = 10
    ps.passes[0].auglag.constraint_tol = 1e-3
    c = ps.passes[0].cost
    c.angle = angle_w
    c.ang_vel = 1e1
    c.ang_vel_mag = 0.0
    c.ang_vel_err_dir = 0.0
    c.control_mult = 1.0
    c.mtq_control_weight = 1e-1
    c.rw_control_weight = 1e3
    c.magic_control_weight = 0.0
    c.rw_AM_weight = 0.0
    c.rw_stic_weight = 0.0
    c.RWh_max_mult = 0.0
    c.RWh_stiction_mult = 0.0
    c.RWh_ok_mult = 0.0
    c.angle_N = angle_w
    c.ang_vel_N = 1e1
    c.ang_cost_func_type = ftype
    c.use_cost_hess = True
    c.cost_hess_gauss_newton = gn
    c.use_theta_cost_param = theta_param
    for f in ("aero", "gg", "srp", "prop", "gendist", "resdipole"):
        setattr(ps.disturbances, f"plan_for_{f}", False)
    ps.passes[0].reg.reg_init = 1e-6
    ps.passes[0].reg.reg_max = 1e10
    ps.passes[0].reg.reg_scale = 10.0
    ps.passes[0].linesearch.max_iters = 24
    ps.passes[0].linesearch.beta1 = 1e-10
    ps.passes[0].linesearch.beta2 = 5000.0
    return ps


def pointing_error_deg(satellite, X, bs_body, r_eci):
    """Boresight-to-target angle at the final knot."""
    q = X[3:7, -1]; q = q / np.linalg.norm(q)
    Rt = np.asarray(saltro_py.rotationMatrix(q)).T
    b = Rt @ (r_eci / np.linalg.norm(r_eci))
    bs = bs_body / np.linalg.norm(bs_body)
    c = np.clip(bs @ b, -1, 1)
    return np.degrees(np.arccos(c))


def run(ps, target_vec, init_err_note):
    satellite = create_satellite(ps)
    jtime = np.array([0.22, 0.22 + 1000/(36525*86400)])
    qgoal = np.array([[np.nan, np.nan],
                      [target_vec[0], target_vec[0]],
                      [target_vec[1], target_vec[1]],
                      [target_vec[2], target_vec[2]]])
    boresight = np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    w0 = np.array([0.01, 0.01, 0.01])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    h0 = np.array([0.0])
    x0 = np.hstack((w0, q0, h0))
    r0 = np.array([7000e3, 0.0, 0.0]); v0 = np.array([0.0, 7.5e3, 0.0])
    try:
        X, U, stop, snaps, trans, dt, ctol, elapsed = trajOpt(
            ps, satellite, x0, r0, v0, jtime, qgoal, boresight, debug=True)
    except Exception as e:
        return {"stop": f"EXC:{e}", "iters": None, "J": None, "PE": None, "t": None}
    PE = pointing_error_deg(satellite, X, np.array([1.0, 0, 0]), np.array(target_vec))
    return {"stop": stop, "iters": len(snaps), "J": snaps[-1]["J"] if snaps else None,
            "PE": PE, "t": elapsed}


def fmt(r):
    j = f"{r['J']:.4e}" if r['J'] is not None else "None"
    pe = f"{r['PE']:.4f}" if r['PE'] is not None else "None"
    t = f"{r['t']:.2f}s" if r['t'] is not None else "-"
    return f"stop={r['stop']:<14} iters={str(r['iters']):>4}  J={j}  PE={pe}deg  {t}"


def main():
    # target for ~90deg: inertial -Y (body +X starts at +X)
    tgt90 = [0.0, -1.0, 0.0]
    # target for ~179deg: nearly inertial -X, tilted 1deg into +Y
    a = np.radians(179.0)
    tgt179 = [np.cos(a), np.sin(a), 0.0]   # angle from +X == 179deg
    print(f"tgt179 = {tgt179}, initial boresight-to-target angle ~ 179 deg\n")

    print("===== Scenario A: 90deg slew, type 3, GN (regression) =====")
    ra_c = run(base_settings(gn=True, theta_param=False, ftype=3), tgt90, "90")
    ra_t = run(base_settings(gn=True, theta_param=True,  ftype=3), tgt90, "90")
    print(f"  c-space : {fmt(ra_c)}")
    print(f"  θ-space : {fmt(ra_t)}")

    print("\n===== Scenario B: ~179deg slew, type 3, GN  [HEADLINE] =====")
    rb_c = run(base_settings(gn=True, theta_param=False, ftype=3), tgt179, "179")
    rb_t = run(base_settings(gn=True, theta_param=True,  ftype=3), tgt179, "179")
    print(f"  c-space GN : {fmt(rb_c)}")
    print(f"  θ-space GN : {fmt(rb_t)}")

    print("\n===== Scenario C: ~179deg slew, type 3, FULL Newton =====")
    rc_c = run(base_settings(gn=False, theta_param=False, ftype=3), tgt179, "179")
    rc_t = run(base_settings(gn=False, theta_param=True,  ftype=3), tgt179, "179")
    print(f"  c-space FULL : {fmt(rc_c)}")
    print(f"  θ-space FULL : {fmt(rc_t)}")

    print("\n===== Scenario D: ~179deg slew, type 1 (bounded), GN (control) =====")
    rd_c = run(base_settings(gn=True, theta_param=False, ftype=1), tgt179, "179")
    rd_t = run(base_settings(gn=True, theta_param=True,  ftype=1), tgt179, "179")
    print(f"  c-space GN : {fmt(rd_c)}")
    print(f"  θ-space GN : {fmt(rd_t)}")


if __name__ == "__main__":
    main()
