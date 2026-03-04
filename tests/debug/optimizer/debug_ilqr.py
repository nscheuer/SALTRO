import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "build"))
import saltro_py


SEC_PER_CENTURY = 36525.0 * 86400.0


class DebugILQR:
	"""Minimal Python reimplementation of the C++ iLQR loop using pybind bindings.

	This is intentionally verbose so you can set breakpoints and inspect every
	matrix, gain, and cost term while stepping in a debugger.
	"""

	def __init__(self, N: int = 8, dt: float = 0.5):
		self.N = N
		self.dt = dt
		self.settings = saltro_py.PlannerSettings()
		self.settings.num_passes = 1
		self.settings.passes[0].dt = dt
		# Disable disturbances for a clean tracking demo
		self.settings.disturbances.plan_for_aero = False
		self.settings.disturbances.plan_for_gg = False
		self.settings.disturbances.plan_for_srp = False
		self.settings.disturbances.plan_for_prop = False
		self.settings.disturbances.plan_for_gendist = False
		self.settings.disturbances.plan_for_resdipole = False

		J = np.diag([0.067, 0.071, 0.069])
		self.satellite = saltro_py.Satellite(J, self.settings)
		self._configure_actuators()
		self.nx = self.satellite.stateDim
		self.nu = self.satellite.controlDim

		self.jtime = self._make_time_grid()
		self.attitude_target_traj, self.boresight, self.attitude_target = self._make_targets()
		self.R, self.V, self.B, self.S, self.rho = self._make_environment()
		self.x0 = self._make_initial_state()

	def _configure_actuators(self):
		self.satellite.addMTQ(np.array([1.0, 0.0, 0.0]), 0.2)
		self.satellite.addMTQ(np.array([0.0, 1.0, 0.0]), 0.2)
		self.satellite.addMTQ(np.array([0.0, 0.0, 1.0]), 0.2)
		self.satellite.addRW(np.array([1.0, 0.0, 0.0]), 0.001, 1e-5, 0.0, 0.02)
		self.satellite.addRW(np.array([0.0, 1.0, 0.0]), 0.001, 1e-5, 0.0, 0.02)
		self.satellite.addRW(np.array([0.0, 0.0, 1.0]), 0.001, 1e-5, 0.0, 0.02)

	def _make_time_grid(self):
		jtime = np.zeros(self.N)
		dt_centuries = self.dt / SEC_PER_CENTURY
		for k in range(self.N):
			jtime[k] = 0.25 + k * dt_centuries
		return jtime

	def _make_targets(self):
		attitude_target_traj = np.zeros((4, self.N))
		attitude_target_traj[0, :] = 1.0
		boresight = np.zeros((3, self.N))
		boresight[0, :] = 1.0
		attitude_target = attitude_target_traj[:, -1]
		return attitude_target_traj, boresight, attitude_target

	def _make_environment(self):
		r0 = np.array([7000e3, 0.0, 0.0])
		v0 = np.array([0.0, 7500.0, 0.0])
		ok, R, V, B, S, rho = saltro_py.generate_orbit(
			r0,
			v0,
			self.jtime,
			0,
			0,
			0,
			0,
			0
		)
		if not ok:
			raise RuntimeError("generate_orbit failed")
		return R, V, B, S, rho.reshape(1, -1)

	def _make_initial_state(self):
		x0 = np.zeros(self.nx)
		x0[0:3] = np.array([0.02, -0.01, 0.015])
		x0[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
		return x0

	def warm_start(self):
		ok, X, U = saltro_py.warm_start(
			self.settings,
			self.satellite,
			self.x0,
			self.jtime,
			self.attitude_target_traj,
			self.boresight,
			self.R,
			self.V,
			self.B,
			self.S,
			self.rho
		)
		if not ok:
			raise RuntimeError("warm_start failed")
		return X, U

	def run_ilqr(self, max_iters: int = 50, cost_tol: float = 1e-4):
		X, U = self.warm_start()
		cost_cfg = self.settings.passes[0].cost
		J_prev = self.satellite.totalCost(
			X,
			U[:, : self.N - 1],
			self.B,
			self.boresight,
			self.attitude_target_traj,
			cost_cfg
		)

		iter_logs = []
		for it in range(max_iters):
			U_trim = U[:, : self.N - 1]
			ok, K_arr, d_arr, deltaV = saltro_py.backward_pass(
				self.satellite,
				X,
				U_trim,
				self.R,
				self.V,
				self.B,
				self.S,
				self.rho,
				self.boresight,
				self.attitude_target_traj,
				self.settings
			)
			if not ok:
				raise RuntimeError(f"backward_pass failed at iter {it}")

			K_list = [K_arr[k] for k in range(K_arr.shape[0])]
			d_list = [d_arr[:, k] for k in range(d_arr.shape[1])]

			ok, X_new, U_new, J_new = saltro_py.forward_pass(
				self.satellite,
				X,
				U,
				K_list,
				d_list,
				deltaV,
				self.B,
				self.R,
				self.V,
				self.S,
				self.rho,
				self.boresight,
				self.attitude_target_traj,
				self.settings,
				self.jtime,
				J_prev
			)
			if not ok:
				raise RuntimeError(f"forward_pass failed at iter {it}")

			improvement = J_prev - J_new
			iter_logs.append({
				"iter": it,
				"J_prev": J_prev,
				"J_new": J_new,
				"improvement": improvement,
				"deltaV": deltaV.copy(),
				"K_norm": max(np.linalg.norm(k) for k in K_list) if K_list else 0.0,
				"d_norm": max(np.linalg.norm(dk) for dk in d_list) if d_list else 0.0,
			})

			X, U, J_prev = X_new, U_new, J_new

			if abs(improvement) < cost_tol:
				break

		return X, U, J_prev, iter_logs


def main():
	ilqr = DebugILQR()
	X, U, J, logs = ilqr.run_ilqr()
	print("iLQR completed")
	print(f"Final cost: {J:.6e}")
	print(f"Iterations: {len(logs)}")
	if logs:
		print("First iteration:", logs[0])
		print("Last iteration:", logs[-1])


if __name__ == "__main__":
	main()
