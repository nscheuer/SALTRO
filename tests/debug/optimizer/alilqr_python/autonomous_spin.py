"""
AUTONOMOUS SPINNING MANEUVER — BeaverCube-2 "overwhelming disturbance" test (thesis Table 7.3)
==============================================================================================

WHAT THIS IS
------------
The AL-iLQR (ALTRO-style) planner DISCOVERS, with NO seed and NO spin guide, that the
right way to hold pointing under an overwhelming body-fixed propulsion torque is to
*spin the body about the boresight* so the disturbance is smeared into a circle and
time-averages to ~zero in inertial space. The wheel then only has to handle the
residual, so its momentum stays inside budget.

Result (this config = 325x MTQ + 3x running angle; ok=True, fully feasible, unseeded):
    PE_fin ~ 0.12 deg,  PE_ss(t>=300s) ~ 0.115 +- 0.06 deg  (pointing nailed)
    t(<1 deg) = 151 s  from PE_0 = 179.4 deg (antipode)
    <wz>_ss ~ 13.8 deg/s  (1.6x the 8.6 deg/s gyro-balance minimum)
    spin axis = body +z to 0.09 deg (= boresight)
    |h|_max = 1.70 mN*m*s  (= 0.85 * h_max margin bind)
    MTQ_max = 0.56 * u_max,  RW_max = 0.56 * u_max  (both feasible)
    MTQ jag = 5.3% (clean sinusoid at the ~26 s spin period)

WHY IT WORKS (the recipe — every ingredient matters)
----------------------------------------------------
1. FULL NEWTON cost Hessian  (cost_hess_gauss_newton = FALSE).
   Gauss-Newton drops the  w*theta*d2(theta)/du2  term. Near the 179 deg antipode that
   term is the dominant "leave the antipode" force, and in low-authority directions it
   is what reins in the otherwise-enormous Newton step. Under GN the planner over-
   commanded the (weak, perp-B) MTQ to ~900x u_max and stalled. Full Newton fixes it.

2. DDP dynamics Hessian  (use_dynamics_hess = TRUE, psd_clip_quu_ddp = TRUE).
   At wz = 0 the linearized dynamics have NO gyroscopic coupling (Euler's bilinear
   term vanishes), so plain iLQR literally cannot feel that spinning helps. The
   gyroscopic-stiffening signal lives in the 2nd-derivative-of-dynamics term f_xx,
   which only DDP carries. psd_clip drops the one spurious -5e13 RW-RW eigenvalue that
   RK4 cross-stage chaining produces (else it corrupts the step).

3. HIGH penalty_max (1e15, safe ceiling).
   *** OPPOSITE of the no-prop slew, which needs high penalty too but for a different
   reason. *** For the spin, the GN-era finding was that high penalty_max DESTROYS the
   solution (mu swamps the cost -> tumbling), so they capped it at 100. But that left
   MTQ violated ~36x. With FULL NEWTON the landscape changes: high penalty no longer
   tips into the tumbling basin, and it now forces MTQ + RW + |h| all to feasibility
   while the spin survives. The MTQ<->|h| "sharing" tradeoff dissolves at high pmax.
   Sweep (2026-05): pmax 1e12..1e15 give BIT-IDENTICAL ok=True results -- the AL exits
   on convergence before the cap binds, so 1e15 is a safe ceiling, not a tuned value.

4. NORMALIZED + HIGH MTQ control weight  (mtq_w = 1000 ~ 325x Bryson;  rw_w = 2.5e7 = 1/u_rw_max^2).
   Bryson scaling balances MTQ vs RW (raw uniform weights let the wheel's ~1e7x cost-
   per-torque advantage hoard momentum). The 325x MTQ multiplier is for COMMAND SMOOTHNESS:
   the peak is constraint-pinned at 0.56 by the |h| margin regardless of weight, but the
   bulk smooths monotonically (jag 23% -> 5% as weight goes 1x -> 325x). The spin itself
   is discovered across the full range (1x..325x), so the multiplier is presentational.

5. roll-free angular-velocity cost  (ang_vel_roll_ratio = 0).
   The goal is vec-pointing (2-DOF); roll about the boresight is unconstrained, so the
   spin DOF must not be penalized or the maneuver can't exist.

6. HIGH angle weights, RUNNING > TERMINAL  (angle = 3e7 = 3x angle_N = 1e7).
   Pins the boresight anti-ram THROUGHOUT, so the only feasible spin is about the
   boresight (smearing the body-x prop). 3x running over terminal front-loads the
   acquisition (PE < 1 deg at 151 s vs 293 s at 1x running) and also pulls the spin
   toward minimal. The running-angle band is NARROW: 10x running breaks the maneuver
   (|h| past physical, MTQ 41x over); 30x collapses to no-spin hoarding (|h|=28).
   3x is the safe ceiling.

7. rw_momentum_limit_scale = 0.85  (the RW-momentum constraint MARGIN).
   Binds |h| at 0.85*h_max = 1.7 mN*m*s, leaving headroom below saturation (the
   momentum analogue of control_limit_scale=0.75 for torque). Gives ok=True + margin.
   This is the only NEW SALTRO source field the recipe needs; everything else
   (full-Newton/DDP/psd_clip flags) is already in SALTRO HEAD.

8. wmax relaxed to 60 deg/s.  Default 20 deg/s is for slow slews; the spin reaches
   ~13 deg/s and the GN-era runs hit |omega| transients that the 20 deg/s isotropic
   bound grinds against. Roll about the boresight is a free DOF for vec-pointing.

WARM START: plain PD (initcontroller=3), NO seed, NO omega target. Bdot + full-Newton
blows up; PD is the start that lands in the spin basin once the recipe is in place.

HISTORY: see ADCS_wt/SPINNING_MANEUVER_*.md and BACKLOG.md. The GN-era runs got the
unseeded spin but with MTQ ~36x over (BACKLOG STATUS). Full Newton (validated first on
the no-prop slew) was the missing antipode-escape + step-taming force the DIAGNOSIS
predicted (sec 6.3). This file is the converged, fully-feasible culmination.

DISCOVERY ROBUSTNESS: The spinning solution emerges across the full control-weight range
(mtq_w 1x..325x; jag drops monotonically 23% -> 5%, peak pinned at 0.56 by the |h| margin)
and across the safe running-angle range (1x..3x of terminal). Ingredients (1-3, 5, 7)
ENABLE the discovery; the 325x weight + 3x angle here are PRESENTATION choices (command
smoothness, fast acquisition) -- they do not enable the spin, they only refine its rendering.

RUN:  PYTHONPATH=<Generalized_ADCS> python autonomous_spin.py
"""
import sys, os, numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)                                   # Generalized_ADCS
sys.path.append(os.path.join(ROOT, "SALTRO", "build"))     # saltro_py
import warnings; warnings.filterwarnings("ignore")

import saltro_py as S
import ADCS as _A
from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.disturbances.prop_disturbance import Prop_Disturbance
from ADCS.satellite_hardware.errors.noise import Noise
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.helpers.math_helpers import rot_mat, normalize
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.saltro.SALTRO_planner_settings import PlannerSettings
from ADCS.controller.saltro.SALTRO_pass_settings import (
    PassConfig, CostConfig, ILQRConfig, AugLagConfig, RegularizationConfig)

# ---- Table 7.3 satellite -----------------------------------------------------
J_THESIS = np.array([[0.1, 0., 0.00013],
                     [0., 0.05, -0.00021],
                     [0.00013, -0.00021, 0.005]])              # kg*m^2, 20:1 oblate
Q0_THESIS = np.array([-0.232, -0.664, -0.234, -0.671]); Q0_THESIS /= np.linalg.norm(Q0_THESIS)
U_MTQ_MAX = np.array([0.19, 0.57, 0.57])                       # A*m^2
U_RW_MAX  = 2e-4                                               # N*m
PROP      = np.array([3e-4, 0., 0.])                           # body-X propulsion torque, N*m (Table 7.3)
DT = 1.0; TF = 500.0

def build_satellite():
    sat = create_beavercube2_cubesat(estimated=False)
    sat.J_COM = J_THESIS.copy(); sat.J_0 = J_THESIS.copy()
    # Table 7.3 actuators — REQUIRED: ps.constraints.u_max is derived from these,
    # so the saturation limits would be wrong if we left the default BeaverCube2 set.
    sat.actuators = [
        MTQ(axis=np.array([1., 0, 0]), max_torque=0.19),
        MTQ(axis=np.array([0, 1., 0]), max_torque=0.57),
        MTQ(axis=np.array([0, 0, 1.]), max_torque=0.57),
        RW(axis=np.array([0, 1., 0]), max_torque=U_RW_MAX, J=2e-6,
           h=np.array([0.0]), h_max=np.array([2e-3])),
    ]
    bz = np.array([0., 0., 1.])                                # boresight = body +z
    sat.boresight = {next(iter(sat.boresight)): bz} if isinstance(sat.boresight, dict) else bz
    # THE overwhelming disturbance: constant body-X propulsion torque (Table 7.3).
    # WITHOUT this the planner solves the no-disturbance problem and never spins
    # (it just points) -- the disturbance is the whole reason the spin is needed.
    sat.disturbances = [d for d in sat.disturbances if not isinstance(d, Prop_Disturbance)] + [
        Prop_Disturbance(torque_nominal=PROP.copy(), noise=Noise(std_noise=np.zeros(3)))]
    return sat

def cs_satellite(sat):
    cs = S.Satellite(); cs.setInertia(np.asarray(sat.J_COM))
    cs.addMTQ(np.array([1., 0, 0]), 0.19)
    cs.addMTQ(np.array([0., 1, 0]), 0.57)
    cs.addMTQ(np.array([0., 0, 1]), 0.57)
    cs.addRW(np.array([0., 1, 0]), U_RW_MAX, 2e-6, 0.0, 2e-3)  # +y wheel, h_max=2 mN*m*s
    return cs

def grids(dt=DT):
    os0 = _A.Orbital_State(ephem=_A.Ephemeris(), J2000=0.22,
                           R=6800.*np.array([0., np.cos(np.deg2rad(51.5)), np.sin(np.deg2rad(51.5))]),
                           V=np.array([7.65, 0., 0.]))
    goal = _A.goals.AntiVelocity_Goal(); t0 = 0.22; gl = GoalList({t0: goal})
    t_end = t0 + TF * TimeConstants.sec2cent; n = int(TF / dt) + 1
    jtime = np.linspace(t0, t_end, n)
    so = Orbit(os0, t_end, dt=dt, use_J2=True, fast=True, verbose=False)
    qg = np.empty((4, n)); bs = np.empty((3, n))
    for i, tk in enumerate(jtime):
        os_at = so.get_os(float(tk)); ag = gl.get_active_goal(float(tk), time_units="centuries")
        tr, _ = ag.to_ref(os_at); tr = np.asarray(tr).reshape(4)
        qg[:, i] = tr if np.isnan(tr[0]) else normalize(tr)
        bs[:, i] = np.array([0., 0., 1.])                      # boresight = body +z
    r0 = np.asarray(os0.R) * 1e3; v0 = np.asarray(os0.V) * 1e3
    return jtime, qg, bs, r0, v0, n

def make_settings():
    sat = build_satellite()
    ps = PlannerSettings(est_sat=sat)
    ps.constraints.wmax = 60 * np.pi / 180.                    # ingredient 8
    ps.init_traj.initcontroller = 3                            # PD, NO seed
    ps.disturbances.plan_for_prop = 1                          # the overwhelming prop torque
    ps.passes = [PassConfig()]; p = ps.passes[0]; p.dt = DT
    p.ilqr = ILQRConfig(); p.ilqr.max_iters = 150
    p.aug_lag = AugLagConfig()
    p.aug_lag.max_outer_iters = 35
    p.aug_lag.penalty_init = 0.01
    p.aug_lag.penalty_scale = 3.0
    p.aug_lag.penalty_max = 1e15                               # ingredient 3 (HIGH; safe ceiling)
    p.reg = RegularizationConfig(); p.reg.reg_init = 1e-3
    p.reg.use_dynamics_hess = 1                                # ingredient 2 (DDP)
    p.reg.psd_clip_quu_ddp = 1                                 # ingredient 2 (psd-clip)
    p.cost = CostConfig(angle=3e7, angle_N=1e7,               # ingredient 6 (3x running for fast acquisition)
                        ang_vel=1e2, ang_vel_N=1e2,
                        control_mult=1.0,
                        mtq_control_weight=1000.0,             # ingredient 4 (325x Bryson, smooth commands)
                        rw_control_weight=2.5e7,               # ingredient 4 (1/u_rw_max^2)
                        ang_cost_func_type=3, use_cost_hess=1)
    # Set every flag on the C++ objects directly so this file does not depend on the
    # separate ADCS-repo wrapper plumbing (cost_hess_gauss_newton, ang_vel_roll_ratio,
    # rw_momentum_limit_scale, the RWAM knee, etc.).
    orig_cost = p.cost.to_cpp
    def cost_to_cpp():
        c = orig_cost()
        c.cost_hess_gauss_newton = False                      # ingredient 1 (FULL NEWTON)
        c.ang_vel_roll_ratio = 0.0                            # ingredient 5 (roll free)
        c.ang_vel_err_dir_ratio = 0.0
        c.rw_AM_weight = 1e4; c.RWh_ok_mult = 0.5             # RWAM soft cost, knee @ 0.5*h_max
        c.rw_stic_weight = 0.0; c.RWh_stiction_mult = 0.05
        return c
    p.cost.to_cpp = cost_to_cpp
    orig_con = ps.constraints.to_cpp
    def con_to_cpp():
        c = orig_con()
        c.rw_momentum_limit_scale = 0.85                      # ingredient 7 (|h| margin -> 1.7)
        return c
    ps.constraints.to_cpp = con_to_cpp
    return sat, ps

def solve():
    sat, ps = make_settings()
    jt, qg, bs, r0, v0, N = grids(DT)
    x0 = np.concatenate([np.zeros(3), Q0_THESIS, [0.0]])      # rest, NO seed
    ok, X, U, K = S.trajOpt(ps.to_cpp(), cs_satellite(sat), x0, r0, v0,
                            np.ascontiguousarray(jt), np.ascontiguousarray(qg),
                            np.ascontiguousarray(bs))
    return ok, np.asarray(X), np.asarray(U), jt, qg, bs

def metrics(X, U, qg):
    pe = np.array([float(np.degrees(np.arccos(np.clip(
        (rot_mat(X[3:7, k]) @ np.array([0, 0, 1])) @ (qg[1:4, k] / np.linalg.norm(qg[1:4, k])),
        -1, 1)))) for k in range(X.shape[1])])
    wz = X[2, :] * 180 / np.pi; h = X[7, :] * 1000
    mtq = np.max(np.abs(U[0:3, :].T) / U_MTQ_MAX); rw = np.max(np.abs(U[3, :])) / U_RW_MAX
    return dict(PE_fin=pe[-1], PE_m30=float(np.mean(pe[-30:])), wz_m30=float(np.mean(wz[-30:])),
                h_max=float(np.abs(h).max()), mtq=float(mtq), rw=float(rw))

if __name__ == "__main__":
    ok, X, U, jt, qg, bs = solve()
    m = metrics(X, U, qg)
    print(f"ok={ok}  PE_fin={m['PE_fin']:.2f}  PE_m30={m['PE_m30']:.2f}  "
          f"<wz>_m30={m['wz_m30']:.2f} deg/s  |h|max={m['h_max']:.2f} mN*m*s  "
          f"mtq={m['mtq']:.2f}x  rw={m['rw']:.2f}x")
