"""Monte Carlo dataset generator for imitation learning.

3-MTQ-only BeaverCube-2 (3U) satellite, circular SSO orbit, random epoch /
initial quaternion / goal quaternion (rest goal) / initial angular velocity.
IGRF-13 magnetic field (hardcoded inside saltro_py.trajOpt along with J2+RK4
propagation, NOAA sun, cylinder eclipse, Harris-Priester density).
No disturbances.

Each trial writes one .npz (states X, controls U, field B, sampled inputs)
plus one row in index.jsonl with convergence status and spike metrics.
Safe to re-run with the same --out: finished trials are skipped (resume).

Run (from repo root, python 3.12 with the built module in build/):
    python mc_datagen/gen_dataset.py --n-trials 100000 --workers 60 \
        --out /data/saltro_mc_v1
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build"))

SEC_PER_CENTURY = 36525.0 * 86400.0
MU = 3.986004418e14
R_EARTH = 6378137.0
J2 = 1.08262668e-3
SSO_NODE_RATE = 2.0 * np.pi / (365.2421897 * 86400.0)  # rad/s
SHARD_SIZE = 1000  # trials per output subdirectory

# BeaverCube-2 (3U) inertia, kg m^2 — from tests/debug/optimizer/alilqr_python/debug_bc2.py
BC2_INERTIA = [
    [0.03136490806, 5.88304e-05, -0.00671361357],
    [5.88304e-05, 0.03409127827, -0.00012334756],
    [-0.00671361357, -0.00012334756, 0.01004091997],
]
BC2_BORESIGHT = [0.0, 1.0, 0.0]


def jd_from_ymd(y, m, d):
    a = (14 - m) // 12
    yy = y + 4800 - a
    mm = m + 12 * a - 3
    jdn = d + (153 * mm + 2) // 5 + 365 * yy + yy // 4 - yy // 100 + yy // 400 - 32045
    return jdn - 0.5  # 00:00 UTC


def centuries_since_j2000(jd):
    return (jd - 2451545.0) / 36525.0


def sso_inclination(a):
    n = np.sqrt(MU / a**3)
    cos_i = -SSO_NODE_RATE / (1.5 * J2 * (R_EARTH / a) ** 2 * n)
    if not -1.0 <= cos_i <= 0.0:
        raise ValueError(f"no SSO inclination for a={a}")
    return np.arccos(cos_i)


def sso_state(rng, alt_m):
    """Random circular SSO: RAAN and argument-of-latitude uniform."""
    a = R_EARTH + alt_m
    inc = sso_inclination(a)
    raan = rng.uniform(0.0, 2.0 * np.pi)
    u = rng.uniform(0.0, 2.0 * np.pi)
    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)
    P = np.array([cO, sO, 0.0])                 # node direction
    Q = np.array([-sO * ci, cO * ci, si])       # in-plane perpendicular
    r0 = a * (np.cos(u) * P + np.sin(u) * Q)
    vmag = np.sqrt(MU / a)
    v0 = vmag * (-np.sin(u) * P + np.cos(u) * Q)
    return r0, v0


def random_quat(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    if q[0] < 0.0:
        q = -q
    return q


def random_omega(rng, sigma_deg):
    """Uniform random direction, half-normal magnitude (1-sigma = sigma_deg)."""
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    mag = abs(rng.normal(0.0, np.deg2rad(sigma_deg)))
    return mag * direction


def quat_angle_deg(q1, q2):
    d = min(1.0, abs(float(np.dot(q1, q2))))
    return np.rad2deg(2.0 * np.arccos(d))


def make_settings(cfg, saltro_py):
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = 2  # IntegratedBdot warm start
    ps.num_passes = 1
    p0 = ps.passes[0]
    p0.dt = cfg["dt"]
    p0.ilqr.cost_tol = cfg["cost_tol"]
    p0.ilqr.max_iters = cfg["ilqr_iters"]
    p0.auglag.max_outer_iters = cfg["outer_iters"]
    p0.auglag.constraint_tol = 1e-3

    c = p0.cost
    c.angle = cfg["angle"]
    c.ang_vel = cfg["ang_vel"]
    c.ang_vel_mag = 0.0
    c.ang_vel_err_dir = 0.0
    c.control_mult = 1.0
    c.mtq_control_weight = cfg["mtq_weight"]
    c.rw_control_weight = 1.0
    c.magic_control_weight = 0.0
    c.rw_AM_weight = 0.0
    c.rw_stic_weight = 0.0
    c.RWh_stiction_mult = 0.0
    c.RWh_knee_frac = 0.0
    c.angle_N = cfg["angle_N"]
    c.ang_vel_N = cfg["ang_vel_N"]
    c.ang_vel_mag_N = 0.0
    c.ang_vel_err_dir_N = 0.0
    c.ang_cost_func_type = cfg["cost_type"]
    c.ang_cost_huber_delta = cfg["huber_delta"]
    c.use_cost_hess = True

    for f in ("plan_for_aero", "plan_for_gg", "plan_for_srp",
              "plan_for_prop", "plan_for_gendist", "plan_for_resdipole"):
        setattr(ps.disturbances, f, False)

    ps.constraints.sun_limit_angle = 0.0  # keepout off: random goals must be feasible
    ps.constraints.control_limit_scale = cfg["control_limit_scale"]

    p0.reg.reg_init = 1e-6
    p0.reg.reg_max = 1e10
    p0.reg.reg_scale = 10.0
    p0.reg.use_dynamics_hess = False
    p0.reg.use_constraint_hess = False

    p0.linesearch.max_iters = 24
    p0.linesearch.beta1 = 1e-10
    p0.linesearch.beta2 = 5000.0
    return ps


def spike_metrics(U, u_lim, tail_frac=0.1):
    """Control-quality metrics. An 'isolated spike' is a step where the
    per-axis control jumps by more than half the actuator range and jumps
    back within two steps — the K*dz artifact shape — as opposed to a
    sustained bang-bang reversal, which stays on the new rail."""
    u_norm = np.linalg.norm(U, axis=0)
    n = U.shape[1]
    n_tail = max(1, int(n * tail_frac))
    med = float(np.median(u_norm))
    du = np.diff(U, axis=1)
    step = np.abs(du).max(axis=0) if n > 1 else np.zeros(1)
    spikes = 0
    for k in np.nonzero(step > u_lim)[0]:  # jump > half range (rails are ±u_lim)
        k_back = min(k + 2, n - 1)
        if np.abs(U[:, k_back] - U[:, k]).max() > u_lim:
            spikes += 1
    return {
        "u_max": float(u_norm.max()),
        "u_median": med,
        "sat_frac": float((u_norm > 0.99 * u_lim * np.sqrt(3)).mean()),
        "tail_max_u": float(u_norm[-n_tail:].max()),
        "tail_spike_ratio": float(u_norm[-n_tail:].max() / (med + 1e-12)),
        "max_du_step": float(step.max()),
        "isolated_spikes": spikes,
    }


def run_trial(trial_idx, seed, cfg):
    import saltro_py  # per-process import

    rng = np.random.default_rng(seed)
    t_epoch = rng.uniform(cfg["epoch_lo"], cfg["epoch_hi"])  # julian centuries
    r0, v0 = sso_state(rng, cfg["alt_km"] * 1e3)
    q0 = random_quat(rng)
    qg = random_quat(rng)
    w0 = random_omega(rng, cfg["av_sigma_deg"])
    duration = float(rng.choice(cfg["durations"]))

    ps = make_settings(cfg, saltro_py)
    sat = saltro_py.Satellite(np.array(cfg["inertia"]), ps)
    for ax in np.eye(3):
        sat.addMTQ(ax, cfg["mtq_am2"])

    jtime = np.array([t_epoch, t_epoch + duration / SEC_PER_CENTURY])
    qgoal = np.column_stack([qg, qg])
    boresight = np.column_stack([cfg["boresight"], cfg["boresight"]])
    x0 = np.hstack((w0, q0))

    rec = {
        "trial": trial_idx,
        "seed": int(seed),
        "duration_s": duration,
        "epoch_c": t_epoch,
        "slew_deg": quat_angle_deg(q0, qg),
        "w0_degs": float(np.rad2deg(np.linalg.norm(w0))),
    }
    t_start = time.time()
    try:
        ok, X, U, K = saltro_py.trajOpt(ps, sat, x0, r0, v0, jtime, qgoal, boresight)
    except RuntimeError as e:
        rec.update(status="failed", error=str(e), solve_s=time.time() - t_start)
        return rec, None

    rec["solve_s"] = time.time() - t_start
    rec["status"] = "converged"

    N = X.shape[1]
    dt = cfg["dt"]
    q_final = X[3:7, -1]
    rec["final_angle_deg"] = quat_angle_deg(q_final / np.linalg.norm(q_final), qg)
    rec["final_w_degs"] = float(np.rad2deg(np.linalg.norm(X[0:3, -1])))
    rec["max_w_degs"] = float(np.rad2deg(np.linalg.norm(X[0:3, :], axis=0).max()))
    u_lim = cfg["mtq_am2"] * cfg["control_limit_scale"]
    rec.update(spike_metrics(U, u_lim))

    # regenerate the environment on the solver's grid for IL features
    jt_fine = t_epoch + np.arange(N) * dt / SEC_PER_CENTURY
    _, R, V, B, S, rho = saltro_py.generate_orbit(r0, v0, jt_fine, 1, 2, 0, 0, 0)

    arrays = dict(X=X, U=U, B=B[:, :N], R=R[:, :N], V=V[:, :N],
                  q0=q0, w0=w0, qgoal=qg, r0=r0, v0=v0,
                  jtime0=np.array([t_epoch]), dt=np.array([dt]))
    if cfg["save_gains"]:
        arrays["K"] = K
    return rec, arrays


def traj_path(out, trial_idx):
    return out / "trajs" / f"shard_{trial_idx // SHARD_SIZE:04d}" / f"trial_{trial_idx:06d}.npz"


def load_done(out):
    """Trials already recorded in the index (converged or failed)."""
    done = set()
    idx = out / "index.jsonl"
    if idx.exists():
        for line in idx.open():
            try:
                done.add(json.loads(line)["trial"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=100)
    ap.add_argument("--durations", type=float, nargs="+", default=[1000.0])
    ap.add_argument("--dt", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=str, default="mc_out")
    ap.add_argument("--alt-km", type=float, default=550.0)
    ap.add_argument("--epoch-start", type=str, default="2026-01-01")
    ap.add_argument("--epoch-end", type=str, default="2027-01-01")
    ap.add_argument("--av-sigma-deg", type=float, default=1.0)
    ap.add_argument("--mtq-am2", type=float, default=0.4)
    ap.add_argument("--control-limit-scale", type=float, default=1.0)
    # cost knobs (defaults = BC-2 pilot-validated recipe "V7", 2026-08-19:
    # 100% convergence over 1000 trials, 97% spike-free, final err p90 3.3 deg)
    ap.add_argument("--angle", type=float, default=1.0)
    ap.add_argument("--ang-vel", type=float, default=1e2)
    ap.add_argument("--mtq-weight", type=float, default=1e-1)
    ap.add_argument("--angle-N", type=float, default=100.0)
    ap.add_argument("--ang-vel-N", type=float, default=1000.0)
    ap.add_argument("--cost-type", type=int, default=5, choices=[0, 1, 3, 5])
    ap.add_argument("--huber-delta", type=float, default=0.35)
    ap.add_argument("--outer-iters", type=int, default=30)
    ap.add_argument("--ilqr-iters", type=int, default=20)
    ap.add_argument("--cost-tol", type=float, default=1e-5)
    ap.add_argument("--save-gains", action="store_true")
    args = ap.parse_args()

    ep_lo = centuries_since_j2000(jd_from_ymd(*map(int, args.epoch_start.split("-"))))
    ep_hi = centuries_since_j2000(jd_from_ymd(*map(int, args.epoch_end.split("-"))))
    cfg = {
        "durations": args.durations, "dt": args.dt, "alt_km": args.alt_km,
        "epoch_lo": ep_lo, "epoch_hi": ep_hi,
        "av_sigma_deg": args.av_sigma_deg, "mtq_am2": args.mtq_am2,
        "control_limit_scale": args.control_limit_scale,
        "inertia": BC2_INERTIA, "boresight": BC2_BORESIGHT,
        "angle": args.angle, "ang_vel": args.ang_vel, "mtq_weight": args.mtq_weight,
        "angle_N": args.angle_N, "ang_vel_N": args.ang_vel_N,
        "cost_type": args.cost_type, "huber_delta": args.huber_delta,
        "outer_iters": args.outer_iters, "ilqr_iters": args.ilqr_iters,
        "cost_tol": args.cost_tol, "save_gains": args.save_gains,
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({**cfg, "n_trials": args.n_trials,
                                                 "seed": args.seed}, indent=2))

    # seeds are a pure function of (--seed, trial index) so resume is exact
    seed_seq = np.random.SeedSequence(args.seed).spawn(args.n_trials)
    seeds = [int(s.generate_state(1)[0]) for s in seed_seq]

    done = load_done(out)
    todo = [i for i in range(args.n_trials) if i not in done]
    if done:
        print(f"resuming: {len(done)} trials already recorded, {len(todo)} to go")

    n_ok = n_fail = 0
    t0 = time.time()
    with (out / "index.jsonl").open("a") as idx, \
            ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_trial, i, seeds[i], cfg): i for i in todo}
        for fut in as_completed(futs):
            rec, arrays = fut.result()
            if arrays is not None:
                path = traj_path(out, rec["trial"])
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **arrays)
                n_ok += 1
            else:
                n_fail += 1
            idx.write(json.dumps(rec) + "\n")
            idx.flush()
            n_done = n_ok + n_fail
            if n_done % 200 == 0 or n_done == len(todo):
                rate = n_done / (time.time() - t0)
                print(f"{n_done}/{len(todo)}  ok={n_ok} fail={n_fail} "
                      f"({rate:.1f} trials/s, eta {(len(todo)-n_done)/max(rate,1e-9):.0f}s)",
                      flush=True)

    print(f"done: {n_ok} converged, {n_fail} failed this run "
          f"({100.0*n_ok/max(1,len(todo)):.1f}% success) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
