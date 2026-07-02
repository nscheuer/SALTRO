"""Paper-2 (P2.2) autonomous SPINNING maneuver — 179 deg acquisition, self-contained.

THE paper spin case (fig_spin_pe/omega/h in the manuscript; regenerable via
Generalized_ADCS papers/Planner/generate_p2.2_spin.py). From rest at the
antipode (179 deg from an anti-velocity vector goal) under an OVERWHELMING
body-fixed prop torque (3e-4 N m — 18x the best-axis MTQ authority), the
planner must both acquire pointing AND discover that spinning the body about
the boresight smears the disturbance into a circle that time-averages to zero,
so the wheel only handles the residual and stays inside its momentum budget.

Reference result (thesis Table 7.3 satellite; canonical run 2026-05-29,
reproduced 2026-07-02 on main + PR #26 + PR #28):
    ok=True, PE_fin ~ 0.07-0.12 deg, t(<1 deg) ~ 151 s from PE_0 = 179.4 deg
    <w_z>_ss ~ 13.7 deg/s (spin axis = boresight = body +z)
    |h|_max = 1.70 mN m s (binds at the 0.85*h_max margin), MTQ/RW peak 0.56x

THE RECIPE (every ingredient matters — full derivation in ADCS_wt/
SPINNING_MANEUVER_*.md and the PKMN_antispike autonomous_spin.py header):
  1. FULL NEWTON cost Hessian (cost_hess_gauss_newton = False). Near the
     antipode the dropped GN term is the dominant "leave the antipode" force;
     under GN the planner over-commands the weak MTQ ~900x and stalls.
  2. DDP (use_dynamics_hess + psd_clip_quu_ddp). At w=0 the linearized
     dynamics have no gyroscopic coupling — the "spinning helps" signal lives
     in f_xx, which only DDP carries.
  3. HIGH penalty_max (1e15): with full Newton, high penalty forces
     MTQ+RW+|h| feasibility without tipping into the tumbling basin.
  4. Bryson-normalized control weights (mtq 1000, rw 2.5e7 = 1/u_max^2).
  5. Roll-free ang-vel cost (ang_vel_roll_ratio = 0) — the spin DOF must be free.
  6. Running angle 3x terminal (3e7 / 1e7) — front-loads acquisition.
  7. rw_momentum_limit_scale = 0.85 — |h| margin bind at 1.7 of 2.0 mN m s.
  8. wmax = 60 deg/s.
  Warm start: plain PD (initcontroller = 3), NO seed, NO omega target.

DEPENDENCIES: needs PR #26 (vec-pointing full-Newton Hessian, roll ratio,
rw_AM knee) and PR #28 (PDController warm start, initcontroller=3) on top of
main >= 2026-06-23 (DDP #62, momentum margin #39). Until #28 is in this
branch's history, run against an integration build of both.

Fully self-contained: Table 7.3 satellite inline, anti-velocity goal from a
closed-form circular two-body orbit (matches the GenADCS J2 orbit to <0.1 deg
over the 500 s horizon), saltro_py resolved from this worktree's own build/.

Usage:
    python debug_3_1_spin179_dt1.py
    python debug_3_1_spin179_dt1.py --plot spin179.png
    python debug_3_1_spin179_dt1.py --gauss-newton      # ablation: stalls/overcommands
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))

import saltro_py  # noqa: E402
from final_viewer import plot_final_trajectory  # noqa: E402

SEC2CENT = 1.0 / (36525.0 * 86400.0)
MU_EARTH = 3.986004418e14  # m^3/s^2

# ---- Table 7.3 satellite (thesis; 20:1 oblate) --------------------------------
J_THESIS = np.array([[0.1, 0.0, 0.00013],
                     [0.0, 0.05, -0.00021],
                     [0.00013, -0.00021, 0.005]])
Q0_THESIS = np.array([-0.232, -0.664, -0.234, -0.671])
Q0_THESIS = Q0_THESIS / np.linalg.norm(Q0_THESIS)
U_MTQ_MAX = np.array([0.19, 0.57, 0.57])   # A m^2
U_RW_MAX = 2e-4                            # N m
PROP = np.array([3e-4, 0.0, 0.0])          # body-X prop torque, N m
DT = 1.0
TF = 500.0


def create_satellite(ps):
    sat = saltro_py.Satellite(J_THESIS, ps)
    sat.addMTQ(np.array([1.0, 0.0, 0.0]), 0.19)
    sat.addMTQ(np.array([0.0, 1.0, 0.0]), 0.57)
    sat.addMTQ(np.array([0.0, 0.0, 1.0]), 0.57)
    # addRW(axis, max_torque, J, h0, h_max); +y wheel, h_max = 2 mN m s
    sat.addRW(np.array([0.0, 1.0, 0.0]), U_RW_MAX, 2e-6, 0.0, 2e-3)
    return sat


def create_planner_settings(gauss_newton):
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = 3            # plain PD to goal (PR #28), NO seed
    ps.constraints.wmax = np.deg2rad(60.0)     # ingredient 8
    ps.constraints.rw_momentum_limit_scale = 0.85  # ingredient 7

    ps.num_passes = 1
    p = ps.passes[0]
    p.dt = DT
    p.ilqr.max_iters = 150
    p.auglag.max_outer_iters = 35
    p.auglag.penalty_init = 0.01
    p.auglag.penalty_scale = 3.0
    p.auglag.penalty_max = 1e15                # ingredient 3

    p.reg.reg_init = 1e-3
    p.reg.use_dynamics_hess = True             # ingredient 2 (DDP)
    p.reg.psd_clip_quu_ddp = True

    c = p.cost
    c.angle = 3e7                              # ingredient 6 (3x running)
    c.angle_N = 1e7
    c.ang_vel = 1e2
    c.ang_vel_N = 1e2
    c.control_mult = 1.0
    c.mtq_control_weight = 1000.0              # ingredient 4
    c.rw_control_weight = 2.5e7
    c.ang_cost_func_type = 3
    c.use_cost_hess = True
    c.cost_hess_gauss_newton = bool(gauss_newton)  # ingredient 1: FULL NEWTON default
    c.ang_vel_roll_ratio = 0.0                 # ingredient 5 (roll free)
    c.ang_vel_err_dir_ratio = 0.0
    c.rw_AM_weight = 1e4
    c.RWh_ok_mult = 0.5
    c.rw_stic_weight = 0.0
    c.RWh_stiction_mult = 0.05

    d = ps.disturbances
    d.plan_for_aero = False
    d.plan_for_gg = False
    d.plan_for_srp = False
    d.plan_for_gendist = False
    d.plan_for_resdipole = False
    d.plan_for_prop = True                     # THE overwhelming disturbance
    d.prop_torque = PROP.copy()
    return ps


def grids(dt=DT):
    """Anti-velocity vector goal along a closed-form circular two-body orbit
    (6800 km, 51.5 deg inclination — the paper case's orbit)."""
    r_hat0 = np.array([0.0, np.cos(np.deg2rad(51.5)), np.sin(np.deg2rad(51.5))])
    v_hat0 = np.array([1.0, 0.0, 0.0])
    r0 = 6800.0e3 * r_hat0
    v0 = 7.65e3 * v_hat0
    n_rate = np.linalg.norm(v0) / np.linalg.norm(r0)  # rad/s, circular

    t0 = 0.22
    n = int(TF / dt) + 1
    jtime = np.linspace(t0, t0 + TF * SEC2CENT, n)
    t_sec = (jtime - t0) / SEC2CENT

    qg = np.empty((4, n))
    bs = np.empty((3, n))
    for i, ts in enumerate(t_sec):
        v_hat = -np.sin(n_rate * ts) * r_hat0 + np.cos(n_rate * ts) * v_hat0
        qg[0, i] = np.nan                      # vec mode
        qg[1:, i] = -v_hat                     # anti-velocity target
        bs[:, i] = np.array([0.0, 0.0, 1.0])   # boresight = body +z
    return jtime, qg, bs, r0, v0


def rot_bz_eci(q):
    w, x, y, z = q
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gauss-newton", action="store_true",
                    help="ablation: GN Hessian instead of full Newton "
                         "(over-commands the MTQ near the antipode).")
    ap.add_argument("--plot", metavar="PNG", default=None)
    args = ap.parse_args()

    ps = create_planner_settings(args.gauss_newton)
    sat = create_satellite(ps)
    jtime, qg, bs, r0, v0 = grids()
    x0 = np.concatenate([np.zeros(3), Q0_THESIS, [0.0]])  # rest at the antipode

    pe0 = np.degrees(np.arccos(np.clip(
        rot_bz_eci(Q0_THESIS) @ (qg[1:, 0] / np.linalg.norm(qg[1:, 0])), -1, 1)))
    print("P2.2 autonomous spin — 179 deg acquisition under overwhelming prop")
    print(f"  saltro_py = {saltro_py.__file__}")
    print(f"  PE_0 = {pe0:.1f} deg, prop = {PROP[0]*1e6:.0f} uN m body-x, "
          f"tf={TF:.0f}s dt={DT:.1f}s, full_newton={not args.gauss_newton}")

    t = time.perf_counter()
    ok, X, U, _K = saltro_py.trajOpt(ps, sat, x0, r0, v0,
                                     np.ascontiguousarray(jtime),
                                     np.ascontiguousarray(qg),
                                     np.ascontiguousarray(bs))
    wall = time.perf_counter() - t
    X = np.asarray(X); U = np.asarray(U)
    print(f"  -> trajOpt ok = {ok}  ({wall:.1f}s, X{X.shape} U{U.shape})")

    N = X.shape[1]
    t_sec = (jtime - jtime[0]) / SEC2CENT
    pe = np.array([float(np.degrees(np.arccos(np.clip(
        rot_bz_eci(X[3:7, k]) @ (qg[1:, k] / np.linalg.norm(qg[1:, k])), -1, 1))))
        for k in range(N)])
    w_deg = X[0:3, :] * 180.0 / np.pi
    h_mnms = X[7, :] * 1000.0
    mtq = float(np.max(np.abs(U[0:3, :].T) / U_MTQ_MAX))
    rw = float(np.max(np.abs(U[3, :])) / U_RW_MAX)
    below1 = np.nonzero(pe < 1.0)[0]
    t_acq = t_sec[below1[0]] if below1.size else float("nan")

    print(f"  PE_fin={pe[-1]:.2f} deg  PE_m30={float(np.mean(pe[-30:])):.2f} deg  "
          f"t(<1deg)={t_acq:.0f} s")
    print(f"  <wz>_m30={float(np.mean(w_deg[2, -30:])):.2f} deg/s  "
          f"|h|max={float(np.abs(h_mnms).max()):.2f} mN m s (limit 2.0, margin 1.7)")
    print(f"  MTQ peak={mtq:.2f}x u_max  RW peak={rw:.2f}x u_max")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
        axes[0].plot(t_sec, pe)
        axes[0].set_ylabel("pointing err (deg)")
        axes[0].set_title(f"P2.2 spin acquisition (ok={ok}, "
                          f"full_newton={not args.gauss_newton})")
        for i, lab in enumerate(("wx", "wy", "wz")):
            axes[1].plot(t_sec, w_deg[i], label=lab)
        axes[1].plot(t_sec, np.linalg.norm(w_deg, axis=0), "r-", label="|w|")
        axes[1].legend(); axes[1].set_ylabel("deg/s")
        axes[2].plot(t_sec, h_mnms)
        axes[2].axhline(2.0, ls=":", c="red"); axes[2].axhline(-2.0, ls=":", c="red")
        axes[2].set_ylabel("RW h (mN m s)"); axes[2].set_xlabel("t (s)")
        for ax in axes:
            ax.grid(alpha=0.4)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"  wrote {args.plot}")

    plot_final_trajectory(
        X,
        U,
        ps.passes[0].dt,
        satellite=sat,
        q_goal=qg,
        boresight_body=bs,
        jtime=jtime,
        title=f"P2.2 spin 179 deg (ok={ok}, full_newton={not args.gauss_newton})",
    )


if __name__ == "__main__":
    main()
