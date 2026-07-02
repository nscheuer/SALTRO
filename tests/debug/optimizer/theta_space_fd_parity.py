"""
θ-space vec-pointing cost prototype: FD-parity + c-space equivalence probe.

Flag: CostConfig.use_theta_cost_param (opt-in). attitude_target ECI format is
[nan, x, y, z]; boresight is a body 3-vector.

Checks:
 (1) FD parity: analytic gradient vs central-diff of cost; full Hessian vs
     forward-diff of the analytic gradient, across
     θ ∈ {1e-5, 0.01, 30°, 90°, 150°, 179°, 179.99°}.
 (2) Equivalence: θ-space full-Newton value/grad/Hess == c-space (type 3).
 (3) GN curvature: θ-space (constant) vs c-space (∝1/sinθ) near the antipode.
"""
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "build"))
import saltro_py as saltro

np.set_printoptions(precision=6, suppress=True, linewidth=200)
NAN = float("nan")


def make_sat():
    J = np.diag([0.067, 0.067, 0.067])
    sat = saltro.Satellite(J, saltro.PlannerSettings())
    for ax in np.eye(3):
        sat.addRW(ax, 0.01, 0.001, 0.0, 0.01)
    for ax in np.eye(3):
        sat.addMTQ(ax, 0.5)
    return sat


def quat_mul(a, b):
    w0, x0, y0, z0 = a; w1, x1, y1, z1 = b
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1])


def state_for_theta(sat, theta, roll=0.3):
    qz = np.array([np.cos(theta/2), 0, 0, np.sin(theta/2)])
    qr = np.array([np.cos(roll/2), np.sin(roll/2), 0, 0])
    q = quat_mul(qr, qz); q /= np.linalg.norm(q)
    x = np.zeros(sat.stateDim)
    x[sat.AV_INDEX:sat.AV_INDEX+3] = [0.01, -0.02, 0.015]
    x[sat.QUAT_INDEX:sat.QUAT_INDEX+4] = q
    for i in range(3):
        x[sat.RW_MOMENTUM_INDEX + i] = 0.001*(i+1)
    return x


def cfg(theta_param, gn, ftype=3):
    c = saltro.CostConfig()
    c.ang_cost_func_type = ftype
    c.use_cost_hess = True
    c.cost_hess_gauss_newton = gn
    c.use_theta_cost_param = theta_param
    c.angle = 1.0; c.angle_N = 1.0
    c.ang_vel = 0.0; c.ang_vel_N = 0.0; c.ang_vel_mag = 0.0
    c.control_mult = 0.0
    c.rw_AM_weight = 0.0; c.rw_stic_weight = 0.0; c.RWh_ok_mult = 0.0
    return c


def fd_grad(sat, x, u, bs, tgt, B, c):
    eps = 1e-7; g = np.zeros(sat.stateDim)
    for i in range(sat.stateDim):
        xp = x.copy(); xp[i] += eps; xm = x.copy(); xm[i] -= eps
        g[i] = (sat.stageCost(0, 10, xp, u, bs, tgt, B, c)
                - sat.stageCost(0, 10, xm, u, bs, tgt, B, c))/(2*eps)
    return g


def fd_hess(sat, x, u, bs, tgt, B, c):
    eps = 1e-6; nx = sat.stateDim; H = np.zeros((nx, nx))
    lx0, _, _ = sat.stageCostJacobians(0, 10, x, u, bs, tgt, B, c)
    for j in range(nx):
        xp = x.copy(); xp[j] += eps
        lxp, _, _ = sat.stageCostJacobians(0, 10, xp, u, bs, tgt, B, c)
        H[:, j] = (lxp - lx0)/eps
    return 0.5*(H + H.T)


def main():
    sat = make_sat()
    bs = np.array([1.0, 0.0, 0.0])
    tgt = np.array([NAN, 1.0, 0.0, 0.0])   # ECI target r=[1,0,0]
    B = np.array([0.0, 0.0, 0.0])
    u = np.zeros(sat.controlDim)

    thetas = [1e-5, 0.01, np.deg2rad(30), np.deg2rad(90),
              np.deg2rad(150), np.deg2rad(179), np.deg2rad(179.99)]
    labels = ["1e-5", "0.01", "30deg", "90deg", "150deg", "179deg", "179.99deg"]

    print("\n=== verify pointing angle == theta (type-2 cost returns theta) ===")
    for th, lab in zip(thetas, labels):
        x = state_for_theta(sat, th)
        c2 = cfg(True, False, ftype=2)   # g(theta)=theta
        val = sat.stageCost(0, 10, x, u, bs, tgt, B, c2)
        print(f"  {lab:>10}: theta_set={th:.6e}  cost(type2,theta-space)={val:.6e}  err={abs(val-th):.2e}")

    for ftype in (3, 1, 0):
        print(f"\n########## ang_cost_func_type = {ftype} ##########")
        print(f"{'theta':>10} | grad_relerr(FD) | fullHess_relerr(FD) | val_equiv | grad_equiv | Hess_equiv")
        for th, lab in zip(thetas, labels):
            x = state_for_theta(sat, th)
            ct = cfg(True, False, ftype)   # theta-space, full Newton
            cc = cfg(False, False, ftype)  # c-space, full Newton

            # (1) FD parity for theta-space full Newton
            g_an, _, _ = sat.stageCostJacobians(0, 10, x, u, bs, tgt, B, ct)
            g_fd = fd_grad(sat, x, u, bs, tgt, B, ct)
            H_an, _, _ = sat.stageCostHessians(0, 10, x, u, bs, tgt, B, ct)
            H_fd = fd_hess(sat, x, u, bs, tgt, B, ct)
            # focus on quaternion block for angle cost
            qsl = slice(sat.QUAT_INDEX, sat.QUAT_INDEX+4)
            gden = max(np.linalg.norm(g_fd[qsl]), 1e-12)
            grelerr = np.linalg.norm(g_an[qsl]-g_fd[qsl])/gden
            Hb_an = H_an[qsl, qsl]; Hb_fd = H_fd[qsl, qsl]
            Hden = max(np.linalg.norm(Hb_fd), 1e-12)
            Hrelerr = np.linalg.norm(Hb_an-Hb_fd)/Hden

            # (2) equivalence theta-space vs c-space full Newton
            vt = sat.stageCost(0, 10, x, u, bs, tgt, B, ct)
            vc = sat.stageCost(0, 10, x, u, bs, tgt, B, cc)
            gc, _, _ = sat.stageCostJacobians(0, 10, x, u, bs, tgt, B, cc)
            Hc, _, _ = sat.stageCostHessians(0, 10, x, u, bs, tgt, B, cc)
            v_eq = abs(vt-vc)
            g_eq = np.linalg.norm(g_an[qsl]-gc[qsl])
            H_eq = np.linalg.norm(Hb_an - Hc[qsl, qsl])
            print(f"{lab:>10} | {grelerr:14.3e} | {Hrelerr:18.3e} | {v_eq:.2e} | {g_eq:.2e} | {H_eq:.2e}")

    print("\n########## GN Hessian curvature: theta-space (const) vs c-space (1/sinθ) ##########")
    print("type 3, cost_hess_gauss_newton=True, angle weight=1")
    print(f"{'theta':>10} | ||H_qq|| c-space GN | ||H_qq|| theta GN | ratio(c/theta)")
    for th, lab in zip(thetas, labels):
        x = state_for_theta(sat, th)
        qsl = slice(sat.QUAT_INDEX, sat.QUAT_INDEX+4)
        ct = cfg(True, True, 3); cc = cfg(False, True, 3)
        Ht, _, _ = sat.stageCostHessians(0, 10, x, u, bs, tgt, B, ct)
        Hc, _, _ = sat.stageCostHessians(0, 10, x, u, bs, tgt, B, cc)
        nt = np.linalg.norm(Ht[qsl, qsl]); nc = np.linalg.norm(Hc[qsl, qsl])
        print(f"{lab:>10} | {nc:18.4e} | {nt:16.4e} | {nc/max(nt,1e-30):12.2f}")


if __name__ == "__main__":
    main()
