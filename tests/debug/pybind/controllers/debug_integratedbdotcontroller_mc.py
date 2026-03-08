import sys
from pathlib import Path

import matplotlib
try:
	# Prefer Tk on WSL to avoid Qt/xcb plugin issues; require tkinter to be present.
	import tkinter  # type: ignore
	matplotlib.use("TkAgg")
except Exception as exc:
	print(f"[warn] Falling back to Agg backend ({exc})")
	matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _find_repo_root(start: Path) -> Path:
	for parent in [start] + list(start.parents):
		if (parent / "build").exists() and (parent / "tests").exists():
			return parent
	raise RuntimeError("Could not locate repository root from script path")


ROOT = _find_repo_root(Path(__file__).resolve())
sys.path.insert(0, str(ROOT / "build"))

import saltro_py


PI = np.pi
DEG2RAD = PI / 180.0
SEC_PER_CENTURY = 36525.0 * 86400.0


NUM_SAMPLES = 5
DT_SECONDS = 10.0
SIM_SECONDS = 1000.0
N = int(SIM_SECONDS / DT_SECONDS) + 1
TUMBLE_STOP_THRESHOLD = 0.5 * DEG2RAD
SEED = 20260302


def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
	v = rng.normal(size=3)
	n = np.linalg.norm(v)
	if n < 1e-12:
		return np.array([1.0, 0.0, 0.0])
	return v / n


def random_inertia(rng: np.random.Generator) -> np.ndarray:
	mass = rng.uniform(4.0, 60.0)
	lx = rng.uniform(0.10, 0.55)
	ly = rng.uniform(0.10, 0.55)
	lz = rng.uniform(0.10, 0.55)

	J = np.zeros((3, 3))
	J[0, 0] = (mass / 12.0) * (ly * ly + lz * lz)
	J[1, 1] = (mass / 12.0) * (lx * lx + lz * lz)
	J[2, 2] = (mass / 12.0) * (lx * lx + ly * ly)
	return J


def make_random_satellite_case(rng: np.random.Generator):
	settings = saltro_py.PlannerSettings()
	settings.init_traj.initcontroller = 2

	settings.disturbances.plan_for_aero = False
	settings.disturbances.plan_for_gg = False
	settings.disturbances.plan_for_srp = False
	settings.disturbances.plan_for_prop = False
	settings.disturbances.plan_for_gendist = False
	settings.disturbances.plan_for_resdipole = False

	settings.num_passes = 1
	settings.passes[0].dt = DT_SECONDS

	J = random_inertia(rng)
	satellite = saltro_py.Satellite(J, settings)

	Javg = max(1e-6, np.trace(J) / 3.0)
	n_mtq = int(rng.integers(2, 4))
	n_rw = int(rng.integers(1, 4))

	mtq_base = float(np.clip(0.8 * np.sqrt(Javg), 0.03, 0.35))
	rw_torque_base = float(np.clip(0.006 * Javg, 8e-5, 8e-3))
	rw_hmax_base = float(np.clip(0.35 * Javg, 0.005, 0.12))

	for _ in range(n_mtq):
		axis = random_unit_vector(rng)
		max_dipole = float(np.clip(mtq_base * rng.uniform(0.7, 1.4), 0.02, 0.40))
		satellite.addMTQ(axis, max_dipole)

	for _ in range(n_rw):
		axis = random_unit_vector(rng)
		max_torque = float(np.clip(rw_torque_base * rng.uniform(0.7, 1.4), 5e-5, 0.010))
		rw_J = float(np.clip(0.015 * Javg * rng.uniform(0.7, 1.4), 5e-6, 2e-3))
		h_max = float(np.clip(rw_hmax_base * rng.uniform(0.7, 1.4), 0.004, 0.16))
		satellite.addRW(axis, max_torque, rw_J, 0.0, h_max)

	return settings, satellite, J


def random_initial_state(satellite: "saltro_py.Satellite", rng: np.random.Generator) -> np.ndarray:
	x0 = np.zeros(satellite.stateDim)

	w_mag = rng.uniform(4.0 * DEG2RAD, 15.0 * DEG2RAD)
	x0[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3] = w_mag * random_unit_vector(rng)

	angle = rng.uniform(-PI, PI)
	axis = random_unit_vector(rng)
	q = np.zeros(4)
	q[0] = np.cos(0.5 * angle)
	q[1:] = axis * np.sin(0.5 * angle)
	q /= np.linalg.norm(q)
	x0[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = q

	return x0


def make_constant_environment():
	t_seconds = np.arange(N, dtype=float) * DT_SECONDS
	jtime = 0.25 + t_seconds / SEC_PER_CENTURY

	R = np.zeros((3, N), dtype=float)
	V = np.zeros((3, N), dtype=float)
	S = np.zeros((3, N), dtype=float)
	rho = np.zeros(N, dtype=float)

	B_const = np.array([2.2e-5, -1.6e-5, 3.1e-5], dtype=float)
	B = np.repeat(B_const[:, None], N, axis=1)

	q_goal = np.zeros((4, N), dtype=float)
	q_goal[0, :] = 1.0
	boresight = np.zeros((3, N), dtype=float)
	boresight[0, :] = 1.0

	return t_seconds, jtime, q_goal, boresight, R, V, B, S, rho


def plot_sample(sample_idx, t, X, U, satellite, initial_w, final_w, passed):
	av_idx = saltro_py.Satellite.AV_INDEX
	omega = X[av_idx:av_idx + 3, :]
	omega_norm = np.linalg.norm(omega, axis=0)

	n_mtq = satellite.numMTQ
	n_rw = satellite.numRW

	fig = plt.figure(figsize=(13, 9))
	status = "PASS" if passed else "FAIL"
	fig.suptitle(
		f"MC Sample {sample_idx:03d} | {status} | nMTQ={n_mtq}, nRW={n_rw} | "
		f"|w0|={initial_w:.4f} rad/s, |wf|={final_w:.4f} rad/s",
		fontsize=11,
	)

	ax_w = fig.add_subplot(2, 2, 1)
	ax_w.plot(t, omega[0, :], label="wx")
	ax_w.plot(t, omega[1, :], label="wy")
	ax_w.plot(t, omega[2, :], label="wz")
	ax_w.plot(t, omega_norm, "k--", lw=1.4, label="||w||")
	ax_w.hlines(TUMBLE_STOP_THRESHOLD, t[0], t[-1], colors="r", linestyles=":", linewidth=1.3, label="threshold")
	ax_w.set_title("Angular Rate")
	ax_w.set_xlabel("Time [s]")
	ax_w.set_ylabel("rad/s")
	ax_w.grid(True)
	ax_w.legend(fontsize=8, ncol=2)

	ax_mtq = fig.add_subplot(2, 2, 2)
	for i in range(n_mtq):
		(line,) = ax_mtq.plot(t, U[i, :], label=f"mtq{i}")
		ulim = float(satellite.getMTQ(i).u_max)
		color = line.get_color()
		ax_mtq.hlines(ulim, t[0], t[-1], colors=color, linestyles=":", linewidth=1.1)
		ax_mtq.hlines(-ulim, t[0], t[-1], colors=color, linestyles=":", linewidth=1.1)
	if n_mtq > 0:
		ax_mtq.plot(t, np.linalg.norm(U[:n_mtq, :], axis=0), "k--", lw=1.2, label="||u_mtq||")
	ax_mtq.set_title("MTQ Control")
	ax_mtq.set_xlabel("Time [s]")
	ax_mtq.set_ylabel("dipole")
	ax_mtq.grid(True)
	ax_mtq.legend(fontsize=8, ncol=2)

	ax_rw = fig.add_subplot(2, 2, 3)
	for i in range(n_rw):
		ui = n_mtq + i
		(line,) = ax_rw.plot(t, U[ui, :], label=f"rw{i}")
		ulim = float(satellite.getRW(i).u_max)
		color = line.get_color()
		ax_rw.hlines(ulim, t[0], t[-1], colors=color, linestyles=":", linewidth=1.1)
		ax_rw.hlines(-ulim, t[0], t[-1], colors=color, linestyles=":", linewidth=1.1)
	if n_rw > 0:
		ax_rw.plot(t, np.linalg.norm(U[n_mtq:, :], axis=0), "k--", lw=1.2, label="||u_rw||")
	ax_rw.set_title("RW Control")
	ax_rw.set_xlabel("Time [s]")
	ax_rw.set_ylabel("torque")
	ax_rw.grid(True)
	ax_rw.legend(fontsize=8, ncol=2)

	ax_text = fig.add_subplot(2, 2, 4)
	ax_text.axis("off")
	msg = (
		f"Result: {status}\n"
		f"Samples uses same criterion as C++ test\n"
		f"Pass iff final ||w|| <= {TUMBLE_STOP_THRESHOLD:.6f} rad/s\n"
		f"and final ||w|| < initial ||w||"
	)
	ax_text.text(0.05, 0.65, msg, fontsize=10, va="top")

	plt.tight_layout()


def main():
	rng = np.random.default_rng(SEED)
	t, jtime, q_goal, boresight, R, V, B, S, rho = make_constant_environment()

	n_pass = 0

	for sample in range(NUM_SAMPLES):
		settings, satellite, J = make_random_satellite_case(rng)
		x0 = random_initial_state(satellite, rng)

		try:
			ok, X, U = saltro_py.warm_start(
				settings,
				satellite,
				x0,
				jtime,
				q_goal,
				boresight,
				R,
				V,
				B,
				S,
				rho,
			)
		except RuntimeError as exc:
			print(
				f"[sample {sample:03d}] warm_start exception: {exc} | "
				f"nMTQ={satellite.numMTQ} nRW={satellite.numRW} DT={DT_SECONDS}"
			)
			continue

		if not ok:
			print(f"[sample {sample:03d}] warm_start failed")
			continue

		initial_w = float(np.linalg.norm(x0[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3]))
		final_w = float(np.linalg.norm(X[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3, -1]))
		passed = bool(np.isfinite(final_w) and final_w <= TUMBLE_STOP_THRESHOLD and final_w < initial_w)
		n_pass += int(passed)

		print(
			f"[sample {sample:03d}] {'PASS' if passed else 'FAIL'} "
			f"nMTQ={satellite.numMTQ} nRW={satellite.numRW} "
			f"|w0|={initial_w:.6f} |wf|={final_w:.6f} "
			f"Jdiag={np.diag(J)}"
		)

		plot_sample(sample, t, X, U, satellite, initial_w, final_w, passed)

	print(f"\nMonte Carlo summary: {n_pass}/{NUM_SAMPLES} passed")
	plt.show()


if __name__ == "__main__":
	main()

