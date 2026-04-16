"""Check cost gradient behavior near 180 deg rotation."""
import sys, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from sat_3_1_hybrid import create_satellite

ps = saltro_py.PlannerSettings()
ps.num_passes = 1
c = ps.passes[0].cost
c.angle = 1e6; c.ang_vel = 1e4; c.ang_cost_func_type = 3; c.use_cost_hess = True
c.mtq_control_weight = 1e-1; c.rw_control_weight = 1.0
c.angle_N = 1e6; c.ang_vel_N = 1e4
sat = create_satellite(ps)

qg = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
bs = np.array([1.0, 0.0, 0.0])
B = np.array([3e-5, 1e-5, -2e-5])
u = np.zeros(int(sat.controlDim))
N = 10

def findWMat(q):
    W = np.zeros((4,3))
    W[0,:] = [-q[1], -q[2], -q[3]]
    W[1,:] = [ q[0], -q[3],  q[2]]
    W[2,:] = [ q[3],  q[0], -q[1]]
    W[3,:] = [-q[2],  q[1],  q[0]]
    return W

def G_full(q, nRW=1):
    G = np.zeros((6+nRW, 7+nRW))
    G[0:3, 0:3] = np.eye(3)
    G[3:6, 3:7] = findWMat(q).T
    G[6, 7] = 1.0
    return G

print(f"{'pe_deg':>8s}  {'q0':>7s}  {'|grad_q|':>12s}  {'grad_reduced[3:6]':>35s}  {'|grad_r|':>10s}")
print("-" * 90)

for angle_deg in [0, 30, 60, 90, 120, 150, 170, 175, 179, 179.9]:
    a = np.radians(angle_deg) / 2
    q = np.array([np.cos(a), 0, np.sin(a), 0])
    if q[0] < 0:
        q = -q

    x = np.hstack([np.zeros(3), q, np.zeros(1)])

    lx, lu, lux = sat.stageCostJacobians(5, N, x, u, bs, qg, B, c)
    lx = np.asarray(lx).flatten()

    grad_q = lx[3:7]
    G = G_full(q)
    grad_reduced = G @ lx
    grad_att = grad_reduced[3:6]

    pe = 2 * np.degrees(np.arccos(min(abs(float(np.dot(q, qg))), 1.0)))
    print(f"{pe:8.1f}  {q[0]:+.5f}  {np.linalg.norm(grad_q):12.2f}"
          f"  [{grad_att[0]:+11.2f} {grad_att[1]:+11.2f} {grad_att[2]:+11.2f}]"
          f"  {np.linalg.norm(grad_att):10.2f}")

# Also check: does the Hessian degenerate near 180?
print("\nHessian eigenvalues (attitude block) vs angle:")
for angle_deg in [0, 90, 150, 170, 179, 179.9]:
    a = np.radians(angle_deg) / 2
    q = np.array([np.cos(a), 0, np.sin(a), 0])
    if q[0] < 0:
        q = -q
    x = np.hstack([np.zeros(3), q, np.zeros(1)])

    lxx, luu, lux = sat.stageCostHessians(5, N, x, u, bs, qg, B, c)
    lxx = np.asarray(lxx)

    G = G_full(q)
    lxx_r = G @ lxx @ G.T
    # Attitude block eigenvalues
    eigs = np.linalg.eigvalsh(lxx_r[3:6, 3:6])
    pe = 2 * np.degrees(np.arccos(min(abs(float(np.dot(q, qg))), 1.0)))
    print(f"  pe={pe:6.1f}  eigs=[{eigs[0]:+.4e}, {eigs[1]:+.4e}, {eigs[2]:+.4e}]")
