import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _find_repo_root(start: Path) -> Path:
	for parent in [start] + list(start.parents):
		if (parent / "build").exists() and (parent / "tests").exists():
			return parent
	raise RuntimeError("Could not locate repository root from script path")


ROOT = _find_repo_root(Path(__file__).resolve())
sys.path.insert(0, str(ROOT / "build"))

import saltro_py


SEC_PER_CENTURY = 36525.0 * 86400.0


def make_satellite_and_settings():
	settings = saltro_py.PlannerSettings()
	settings.init_traj.initcontroller = 1  # ExcitationController

	settings.disturbances.plan_for_aero = False
	settings.disturbances.plan_for_gg = False
	settings.disturbances.plan_for_srp = False
	settings.disturbances.plan_for_prop = False
	settings.disturbances.plan_for_gendist = False
	settings.disturbances.plan_for_resdipole = False

	settings.num_passes = 1
	settings.passes[0].dt = 0.5

	J = np.diag([0.067, 0.071, 0.069]).astype(float)
	satellite = saltro_py.Satellite(J, settings)

	# MTQs
	satellite.addMTQ(np.array([1.0, 0.0, 0.0]), 0.2)
	satellite.addMTQ(np.array([0.0, 1.0, 0.0]), 0.2)
	satellite.addMTQ(np.array([0.0, 0.0, 1.0]), 0.2)

	# RWs
	satellite.addRW(np.array([1.0, 0.0, 0.0]), 0.001, 1e-5, 0.0, 0.02)
	satellite.addRW(np.array([0.0, 1.0, 0.0]), 0.001, 1e-5, 0.0, 0.02)
	satellite.addRW(np.array([0.0, 0.0, 1.0]), 0.001, 1e-5, 0.0, 0.02)

	return settings, satellite


def main():
	settings, satellite = make_satellite_and_settings()

	N = 200
	dt_seconds = 0.5
	dt_centuries = dt_seconds / SEC_PER_CENTURY
	t_seconds = np.arange(N, dtype=float) * dt_seconds
	jtime = 0.25 + np.arange(N, dtype=float) * dt_centuries

	r0 = np.array([7000e3, 0.0, 0.0], dtype=float)
	v0 = np.array([0.0, 7.5e3, 0.0], dtype=float)

	ok_orbit, R, V, B, S, rho = saltro_py.generate_orbit(
		r0,
		v0,
		jtime,
		0,
		0,
		0,
		0,
		0,
	)

	if not ok_orbit:
		raise RuntimeError("generate_orbit failed")

	x0 = np.zeros(satellite.stateDim)
	x0[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3] = np.array([0.05, -0.03, 0.02])
	x0[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1.0, 0.0, 0.0, 0.0])

	q_goal = np.zeros((4, N), dtype=float)
	q_goal[0, :] = 1.0
	boresight = np.zeros((3, N), dtype=float)
	boresight[0, :] = 1.0

	ok_warm, X, U = saltro_py.warm_start(
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

	if not ok_warm:
		raise RuntimeError("warm_start failed")

	av_idx = saltro_py.Satellite.AV_INDEX
	q_idx = saltro_py.Satellite.QUAT_INDEX
	omega = X[av_idx:av_idx + 3, :]
	quat = X[q_idx:q_idx + 4, :]

	fig = plt.figure(figsize=(14, 12))

	ax_q = fig.add_subplot(3, 2, 1)
	ax_q.plot(t_seconds, quat[0, :], label="q0")
	ax_q.plot(t_seconds, quat[1, :], label="q1")
	ax_q.plot(t_seconds, quat[2, :], label="q2")
	ax_q.plot(t_seconds, quat[3, :], label="q3")
	ax_q.set_title("Quaternion (ExcitationController)")
	ax_q.set_xlabel("Time [s]")
	ax_q.set_ylabel("q")
	ax_q.grid(True)
	ax_q.legend()

	ax_w = fig.add_subplot(3, 2, 2)
	ax_w.plot(t_seconds, omega[0, :], label="wx")
	ax_w.plot(t_seconds, omega[1, :], label="wy")
	ax_w.plot(t_seconds, omega[2, :], label="wz")
	ax_w.set_title("Angular Velocity")
	ax_w.set_xlabel("Time [s]")
	ax_w.set_ylabel("rad/s")
	ax_w.grid(True)
	ax_w.legend()

	ax_orb = fig.add_subplot(3, 2, 3, projection="3d")
	re = 6371e3
	u_sphere = np.linspace(0.0, 2.0 * np.pi, 60)
	v_sphere = np.linspace(0.0, np.pi, 30)
	xe = re * np.outer(np.cos(u_sphere), np.sin(v_sphere))
	ye = re * np.outer(np.sin(u_sphere), np.sin(v_sphere))
	ze = re * np.outer(np.ones_like(u_sphere), np.cos(v_sphere))
	ax_orb.plot_surface(xe, ye, ze, color="lightskyblue", alpha=0.35, linewidth=0.0, zorder=0)

	ax_orb.plot(R[0, :], R[1, :], R[2, :], lw=1.5)
	ax_orb.set_title("Orbit (ECI)")
	ax_orb.set_xlabel("X [m]")
	ax_orb.set_ylabel("Y [m]")
	ax_orb.set_zlabel("Z [m]")

	r_max = max(re, float(np.max(np.linalg.norm(R, axis=0))))
	lim = 1.05 * r_max
	ax_orb.set_xlim(-lim, lim)
	ax_orb.set_ylim(-lim, lim)
	ax_orb.set_zlim(-lim, lim)
	ax_orb.set_box_aspect((1.0, 1.0, 1.0))

	n_mtq = satellite.numMTQ
	n_rw = satellite.numRW

	ax_mtq = fig.add_subplot(3, 2, 4)
	for i in range(n_mtq):
		(line,) = ax_mtq.plot(t_seconds, U[i, :], label=f"mtq{i}")
		ulim = float(satellite.getMTQ(i).u_max)
		color = line.get_color()
		ax_mtq.hlines(ulim, t_seconds[0], t_seconds[-1], colors=color, linestyles=":", linewidth=1.2)
		ax_mtq.hlines(-ulim, t_seconds[0], t_seconds[-1], colors=color, linestyles=":", linewidth=1.2)
	ax_mtq.plot(t_seconds, np.linalg.norm(U[:n_mtq, :], axis=0), "k--", lw=1.2, label="||u_mtq||")
	ax_mtq.set_title("MTQ Control Effort")
	ax_mtq.set_xlabel("Time [s]")
	ax_mtq.set_ylabel("dipole")
	ax_mtq.grid(True)
	ax_mtq.legend(ncol=2, fontsize=8)

	ax_rw = fig.add_subplot(3, 2, 5)
	for i in range(n_rw):
		ui = n_mtq + i
		(line,) = ax_rw.plot(t_seconds, U[ui, :], label=f"rw{i}")
		ulim = float(satellite.getRW(i).u_max)
		color = line.get_color()
		ax_rw.hlines(ulim, t_seconds[0], t_seconds[-1], colors=color, linestyles=":", linewidth=1.2)
		ax_rw.hlines(-ulim, t_seconds[0], t_seconds[-1], colors=color, linestyles=":", linewidth=1.2)
	ax_rw.plot(t_seconds, np.linalg.norm(U[n_mtq:, :], axis=0), "k--", lw=1.2, label="||u_rw||")
	ax_rw.set_title("RW Control Effort")
	ax_rw.set_xlabel("Time [s]")
	ax_rw.set_ylabel("torque")
	ax_rw.grid(True)
	ax_rw.legend(ncol=2, fontsize=8)

	ax_blank = fig.add_subplot(3, 2, 6)
	ax_blank.axis("off")

	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	main()
