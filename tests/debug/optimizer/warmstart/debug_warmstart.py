"""
Debug script for warm-start trajectory generation and cost analysis.

This script sets up the satellite, environment, and runs the warm-start
trajectory. Then it calls plot functions to visualize and analyze the results.

Easy to extend: just add new plot_* functions below and call them from main().
"""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "build"))
import saltro_py

from plot_warmstart import plot_warmstart
from plot_cost import plot_cost, plot_cost_component_evolution, plot_rw_cost_surfaces


SEC_PER_CENTURY = 36525.0 * 86400.0


def setup_satellite(N: int = 10):
	"""Create satellite with actuators and initial configuration."""
	settings = saltro_py.PlannerSettings()
	settings.num_passes = 1
	settings.passes[0].dt = 10.0
	settings.init_traj.initcontroller = 2  # IntegratedBdot
	
	# Disable disturbances for clean tracking demo
	settings.disturbances.plan_for_aero = False
	settings.disturbances.plan_for_gg = False
	settings.disturbances.plan_for_srp = False
	settings.disturbances.plan_for_prop = False
	settings.disturbances.plan_for_gendist = False
	settings.disturbances.plan_for_resdipole = False
	
	J = np.diag([0.067, 0.071, 0.069])
	satellite = saltro_py.Satellite(J, settings)
	
	# Add actuators
	satellite.addMTQ(np.array([1.0, 0.0, 0.0]), 0.2)
	satellite.addMTQ(np.array([0.0, 1.0, 0.0]), 0.2)
	satellite.addMTQ(np.array([0.0, 0.0, 1.0]), 0.2)
	satellite.addRW(np.array([1.0, 0.0, 0.0]), 0.001, 1e-5, 0.0, 0.02)
	satellite.addRW(np.array([0.0, 1.0, 0.0]), 0.001, 1e-5, 0.0, 0.02)
	satellite.addRW(np.array([0.0, 0.0, 1.0]), 0.001, 1e-5, 0.0, 0.02)
	
	return satellite, settings


def make_time_grid(N: int, dt: float):
	"""Create time grid in Julian centuries."""
	jtime = np.zeros(N)
	dt_centuries = dt / SEC_PER_CENTURY
	for k in range(N):
		jtime[k] = 0.25 + k * dt_centuries
	return jtime


def make_targets(N: int):
	"""Create attitude target trajectory."""
	attitude_target_traj = np.zeros((4, N))
	attitude_target_traj[0, :] = 1.0  # Identity quaternion
	boresight = np.zeros((3, N))
	boresight[0, :] = 1.0
	return attitude_target_traj, boresight


def make_environment(jtime: np.ndarray):
	"""Generate orbital environment."""
	r0 = np.array([7000e3, 0.0, 0.0])
	v0 = np.array([0.0, 7500.0, 0.0])
	ok, R, V, B, S, rho = saltro_py.generate_orbit(
		r0, v0, jtime, 0, 0, 0, 0, 0
	)
	if not ok:
		raise RuntimeError("generate_orbit failed")
	return R, V, B, S, rho.reshape(1, -1)


def make_initial_state(satellite):
	"""Create initial state with angular velocity."""
	x0 = np.zeros(satellite.stateDim)
	x0[0:3] = np.array([0.02, -0.01, 0.015])  # Angular velocity
	x0[3:7] = np.array([1.0, 0.0, 0.0, 0.0])   # Identity quaternion
	return x0


def generate_warmstart(satellite, settings, x0, jtime, attitude_target_traj, boresight, R, V, B, S, rho):
	"""Generate warm-start trajectory with timing."""
	N = len(jtime)
	
	start_time = time.time()
	ok, X, U = saltro_py.warm_start(
		settings, satellite, x0, jtime,
		attitude_target_traj, boresight,
		R, V, B, S, rho
	)
	elapsed = time.time() - start_time
	
	if not ok:
		raise RuntimeError("warm_start failed")
	
	return X, U, elapsed


def main():
	"""Main entry point: setup, generate, and analyze."""
	N = 10
	dt = 10.0
	
	# Setup
	satellite, settings = setup_satellite(N)
	jtime = make_time_grid(N, dt)
	attitude_target_traj, boresight = make_targets(N)
	R, V, B, S, rho = make_environment(jtime)
	x0 = make_initial_state(satellite)
	
	# Generate warm-start
	X, U, ws_time = generate_warmstart(
		satellite, settings, x0, jtime,
		attitude_target_traj, boresight,
		R, V, B, S, rho
	)
	print(f"Warm-start calculation time: {ws_time*1000:.2f} ms")
	print(f"Angular rates final: {X[0:3, -1]}")
	
	# Plot trajectory
	plot_warmstart(X, U, N, dt, attitude_target_traj)
	
	# Plot costs
	plot_cost(X, U, N, dt, satellite, attitude_target_traj, boresight, B, 
			  settings.passes[0].cost, x0)
	
	# Plot cost component evolution
	plot_cost_component_evolution(X, U, N, dt, satellite, attitude_target_traj, 
								   boresight, B, settings.passes[0].cost)
	
	# Plot 3D RW cost surfaces with trajectory
	plot_rw_cost_surfaces(X, U, N, dt, satellite, attitude_target_traj, 
						 boresight, B, settings.passes[0].cost, x0)


if __name__ == "__main__":
	main()
