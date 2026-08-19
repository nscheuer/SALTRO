"""Quality analysis for a gen_dataset.py output directory.

Prints convergence, final-error, spike, and trajectory-quality stats, and
optionally writes a sample-trajectory plot. Trajectory-quality metrics are
computed from the stored .npz files:

- t_acq       first time the attitude error drops below --acq-deg
- dwell       fraction of post-acquisition time spent below --acq-deg
- excursion   max attitude error after first acquisition. Large values
              (> 45 deg) are mostly momentum tours: with a random initial
              rate the MTQ-only vehicle often cannot brake at first goal
              passage and coasts through before settling. Real physics,
              but cullable via this metric if unwanted for training.

Usage:
    python mc_datagen/analyze.py <outdir> [--acq-deg 10] [--plot out.png]
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load(outdir):
    recs = [json.loads(l) for l in (outdir / "index.jsonl").open()]
    return [r for r in recs if r["status"] == "converged"], recs


def traj_file(outdir, trial):
    return outdir / "trajs" / f"shard_{trial // 1000:04d}" / f"trial_{trial:06d}.npz"


def angle_err_deg(X, qgoal):
    q = X[3:7] / np.maximum(np.linalg.norm(X[3:7], axis=0), 1e-300)
    return np.rad2deg(2.0 * np.arccos(np.minimum(1.0, np.abs(qgoal @ q))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--acq-deg", type=float, default=10.0)
    ap.add_argument("--plot", type=str, default="")
    ap.add_argument("--max-load", type=int, default=2000,
                    help="max trajectories to load for quality metrics")
    args = ap.parse_args()

    ok, recs = load(args.outdir)
    n = len(recs)
    print(f"trials: {n}, converged: {len(ok)} ({100.0*len(ok)/max(1,n):.1f}%)")
    if len(ok) < n:
        from collections import Counter
        errs = Counter(r["error"].split(":")[0] for r in recs if r["status"] != "converged")
        for msg, cnt in errs.most_common():
            print(f"  fail[{cnt}]: {msg}")

    def pk(key, ps=(50, 90, 99, 100)):
        a = np.array([r[key] for r in ok])
        return "/".join(f"{v:.2f}" for v in np.percentile(a, ps))

    print(f"final_angle_deg p50/p90/p99/max: {pk('final_angle_deg')}")
    print(f"final_w_degs    p50/p90/p99/max: {pk('final_w_degs')}")
    print(f"solve_s         p50/p90/p99/max: {pk('solve_s')}")
    sp = np.array([r["isolated_spikes"] for r in ok])
    print(f"isolated_spikes: {100.0*(sp == 0).mean():.1f}% spike-free, "
          f"p99={np.percentile(sp, 99):.0f}, max={sp.max()}")

    sub = ok[: args.max_load]
    dwell, exc, t_acq = [], [], []
    for r in sub:
        d = np.load(traj_file(args.outdir, r["trial"]))
        ang = angle_err_deg(d["X"], d["qgoal"])
        acq = np.nonzero(ang < args.acq_deg)[0]
        if not len(acq):
            continue
        post = ang[acq[0]:]
        t_acq.append(acq[0] * float(d["dt"][0]))
        dwell.append(float((post < args.acq_deg).mean()))
        exc.append(float(post.max()))
    t_acq, dwell, exc = map(np.array, (t_acq, dwell, exc))
    print(f"acquired (<{args.acq_deg:.0f} deg at some point): "
          f"{100.0*len(t_acq)/max(1,len(sub)):.1f}% of {len(sub)} loaded")
    print(f"t_acq p50/p90: {np.percentile(t_acq, 50):.0f}/{np.percentile(t_acq, 90):.0f} s   "
          f"dwell p50: {np.percentile(dwell, 50):.2f}   "
          f"excursion>45deg (momentum tours): {100.0*(exc > 45).mean():.1f}%")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rng = np.random.default_rng(0)
        picks = list(rng.choice([r["trial"] for r in ok], 9, replace=False))
        picks += [r["trial"] for r in sorted(ok, key=lambda r: -r["final_angle_deg"])[:3]]
        fig, axes = plt.subplots(4, 3, figsize=(15, 14), sharex=True)
        for axc, tr in zip(axes.flat, picks):
            d = np.load(traj_file(args.outdir, tr))
            X, U = d["X"], d["U"]
            dt = float(d["dt"][0])
            t = np.arange(X.shape[1]) * dt
            ang = angle_err_deg(X, d["qgoal"])
            w = np.rad2deg(np.linalg.norm(X[:3], axis=0))
            axc.plot(t, ang, "C0")
            axc.plot(t, w * 10, "C1")
            ax2 = axc.twinx()
            for i in range(3):
                ax2.plot(t, U[i], lw=0.7, alpha=0.7)
            axc.set_title(f"trial {tr} (final {ang[-1]:.1f} deg)", fontsize=9)
        fig.suptitle("angle err (blue, deg), |w| x10 (orange, deg/s), MTQ dipoles (thin, right axis)")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=110)
        print(f"plot written to {args.plot}")


if __name__ == "__main__":
    main()
