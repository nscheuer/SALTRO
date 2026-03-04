"""
Cost analysis for warm-start trajectory.

This module provides functions to analyze and visualize trajectory costs,
including overall cost breakdown, component evolution, and 3D cost surfaces
for reaction wheel momentum motion.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import ScalarFormatter

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "build"))
import saltro_py


def _configure_sci_y(ax):
	"""Force scientific notation with 10^n multiplier on y-axis."""
	formatter = ScalarFormatter(useMathText=True)
	formatter.set_powerlimits((0, 0))
	ax.yaxis.set_major_formatter(formatter)
	ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))


def _is_eci_format(attitude_target_k):
	"""Match C++ convention: first element NaN means ECI-vector target."""
	return np.isnan(attitude_target_k[0])


def _normalize(v):
	n = np.linalg.norm(v)
	if n < 1e-12:
		return v
	return v / n


def _quat_from_two_vectors(v_from, v_to):
	"""Quaternion [w, x, y, z] rotating v_from to v_to."""
	a = _normalize(v_from)
	b = _normalize(v_to)
	dot = np.clip(np.dot(a, b), -1.0, 1.0)

	if dot > 1.0 - 1e-10:
		return np.array([1.0, 0.0, 0.0, 0.0])

	if dot < -1.0 + 1e-10:
		# 180 deg rotation: pick robust orthogonal axis
		axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
		if np.linalg.norm(axis) < 1e-10:
			axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
		axis = _normalize(axis)
		return np.array([0.0, axis[0], axis[1], axis[2]])

	c = np.cross(a, b)
	w = 1.0 + dot
	q = np.array([w, c[0], c[1], c[2]])
	return _normalize(q)


def _goal_quaternion_for_plot(attitude_target_k, boresight_k, q_current):
	"""Replicate processAttitudeTarget behavior used by stageCost."""
	if _is_eci_format(attitude_target_k):
		target_vec = attitude_target_k[1:4]
		if np.linalg.norm(target_vec) < 1e-9:
			return q_current.copy()
		return _quat_from_two_vectors(boresight_k, target_vec)
	return _normalize(attitude_target_k)


def _angle_error_deg(q, q_goal):
	qd = np.clip(np.abs(np.dot(_normalize(q), _normalize(q_goal))), -1.0, 1.0)
	return np.degrees(2.0 * np.arccos(qd))


def compute_trajectory_costs(X, U, N, satellite, attitude_target_traj, boresight, B, cost_cfg):
	"""
	Compute per-timestep costs for the trajectory.
	
	Args:
		X: State trajectory (nx x N)
		U: Control trajectory (nu x N-1)
		N: Number of timesteps
		satellite: Satellite object with stageCost method
		attitude_target_traj: Target quaternion trajectory (4 x N)
		boresight: Boresight vector for each timestep (3 x N)
		B: Magnetic field for each timestep (3 x N)
		cost_cfg: Cost configuration
	
	Returns:
		(total_cost, costs_per_step)
	"""
	costs_per_step = np.zeros(N)
	
	for k in range(N):
		x_k = X[:, k]
		u_k = U[:, k] if k < U.shape[1] else np.zeros(satellite.controlDim)
		B_k = B[:, k]
		
		costs_per_step[k] = satellite.stageCost(
			k, N, x_k, u_k,
			boresight[:, k],
			attitude_target_traj[:, k],
			B_k,
			cost_cfg
		)
	
	total_cost = np.sum(costs_per_step)
	return total_cost, costs_per_step


def compute_cost_components(X, U, N, satellite, attitude_target_traj, boresight, B, cost_cfg):
	"""
	Compute cost component breakdown by evaluating cost with each component enabled.
	
	Args:
		X: State trajectory (nx x N)
		U: Control trajectory (nu x N-1)
		N: Number of timesteps
		satellite: Satellite object
		attitude_target_traj: Target quaternion trajectory (4 x N)
		boresight: Boresight vector for each timestep (3 x N)
		B: Magnetic field for each timestep (3 x N)
		cost_cfg: Base cost configuration
	
	Returns:
		components: Dict with keys {attitude, angular_velocity, control, rw_momentum}
				   each with array of shape (N,)
	"""
	components = {
		'attitude': np.zeros(N),
		'angular_velocity': np.zeros(N),
		'control': np.zeros(N),
		'rw_momentum': np.zeros(N),
	}
	
	# For each component, create a cost config with only that component enabled
	for comp_name in components.keys():
		# Create a modified cost config
		cfg = saltro_py.CostConfig()
		cfg.angle = cost_cfg.angle if comp_name == 'attitude' else 0.0
		cfg.angle_N = cost_cfg.angle_N if comp_name == 'attitude' else 0.0
		cfg.ang_vel = cost_cfg.ang_vel if comp_name == 'angular_velocity' else 0.0
		cfg.ang_vel_N = cost_cfg.ang_vel_N if comp_name == 'angular_velocity' else 0.0
		cfg.ang_vel_mag = cost_cfg.ang_vel_mag if comp_name == 'angular_velocity' else 0.0
		cfg.ang_vel_mag_N = cost_cfg.ang_vel_mag_N if comp_name == 'angular_velocity' else 0.0
		cfg.ang_vel_err_dir = cost_cfg.ang_vel_err_dir if comp_name == 'angular_velocity' else 0.0
		cfg.ang_vel_err_dir_N = cost_cfg.ang_vel_err_dir_N if comp_name == 'angular_velocity' else 0.0
		cfg.control_mult = cost_cfg.control_mult if comp_name == 'control' else 0.0
		cfg.mtq_control_weight = cost_cfg.mtq_control_weight if comp_name == 'control' else 0.0
		cfg.rw_control_weight = cost_cfg.rw_control_weight if comp_name == 'control' else 0.0
		cfg.magic_control_weight = cost_cfg.magic_control_weight if comp_name == 'control' else 0.0
		cfg.rw_AM_weight = cost_cfg.rw_AM_weight if comp_name == 'rw_momentum' else 0.0
		cfg.rw_stic_weight = cost_cfg.rw_stic_weight if comp_name == 'rw_momentum' else 0.0
		cfg.RWh_max_mult = cost_cfg.RWh_max_mult if comp_name == 'rw_momentum' else 0.0
		cfg.RWh_stiction_mult = cost_cfg.RWh_stiction_mult if comp_name == 'rw_momentum' else 0.0
		cfg.RWh_ok_mult = cost_cfg.RWh_ok_mult if comp_name == 'rw_momentum' else 0.0
		cfg.ang_cost_func_type = cost_cfg.ang_cost_func_type
		cfg.use_cost_hess = False
		
		# Evaluate cost for each timestep with this component only
		for k in range(N):
			x_k = X[:, k]
			u_k = U[:, k] if k < U.shape[1] else np.zeros(satellite.controlDim)
			B_k = B[:, k]
			
			cost = satellite.stageCost(
				k, N, x_k, u_k,
				boresight[:, k],
				attitude_target_traj[:, k],
				B_k,
				cfg
			)
			components[comp_name][k] = max(0, cost)
	
	return components


def plot_cost_component_evolution(X, U, N, dt, satellite, attitude_target_traj, boresight, B, cost_cfg):
	"""
	Plot the evolution of each cost component over time.
	
	Creates a multi-panel figure showing how each cost component develops during the trajectory.
	
	Args:
		X: State trajectory (nx x N)
		U: Control trajectory (nu x N-1)
		N: Number of timesteps
		dt: Timestep duration
		satellite: Satellite object
		attitude_target_traj: Target quaternion trajectory (4 x N)
		boresight: Boresight vector for each timestep (3 x N)
		B: Magnetic field for each timestep (3 x N)
		cost_cfg: Cost configuration
	"""
	components = compute_cost_components(
		X, U, N, satellite, attitude_target_traj, boresight, B, cost_cfg
	)
	t_state = np.arange(N) * dt
	w = X[0:3, :]
	q = X[3:7, :]
	h_rw = X[7:10, :]
	t_control = (np.arange(N - 1) * dt if U.shape[1] == N - 1 else np.arange(N) * dt)

	fig = plt.figure(figsize=(16, 14), constrained_layout=True)
	gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.28)

	# Row 1: attitude cost + pointing context
	ax = fig.add_subplot(gs[0, 0])
	ax.plot(t_state, components['attitude'], 'o-', color='C0', linewidth=2, markersize=4)
	ax.set_title('Pointing Cost')
	ax.set_xlabel('Time [s]')
	ax.set_ylabel('Attitude Cost')
	ax.grid(True, alpha=0.3)
	_configure_sci_y(ax)

	q_goal = np.zeros_like(q)
	angle_err_deg = np.zeros(N)
	for k in range(N):
		q_goal[:, k] = _goal_quaternion_for_plot(attitude_target_traj[:, k], boresight[:, k], q[:, k])
		angle_err_deg[k] = _angle_error_deg(q[:, k], q_goal[:, k])

	ax = fig.add_subplot(gs[0, 1])
	ax.plot(t_state, q[0, :], label='q0', linewidth=1.6)
	ax.plot(t_state, q[1, :], label='q1', linewidth=1.6)
	ax.plot(t_state, q[2, :], label='q2', linewidth=1.6)
	ax.plot(t_state, q[3, :], label='q3', linewidth=1.6)
	ax.plot(t_state, q_goal[0, :], '--', label='q0 goal', alpha=0.8)
	ax.plot(t_state, q_goal[1, :], '--', label='q1 goal', alpha=0.8)
	ax.plot(t_state, q_goal[2, :], '--', label='q2 goal', alpha=0.8)
	ax.plot(t_state, q_goal[3, :], '--', label='q3 goal', alpha=0.8)
	ax2 = ax.twinx()
	ax2.plot(t_state, angle_err_deg, 'k-', linewidth=2.0, label='angle err [deg]')
	ax.set_title('Quaternion / Goal Quaternion / Pointing Error')
	ax.set_xlabel('Time [s]')
	ax.set_ylabel('Quaternion')
	ax2.set_ylabel('Angle Error [deg]')
	ax.grid(True, alpha=0.3)
	h1, l1 = ax.get_legend_handles_labels()
	h2, l2 = ax2.get_legend_handles_labels()
	ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=7, ncol=3)

	# Row 2: angular velocity cost + rates
	ax = fig.add_subplot(gs[1, 0])
	ax.plot(t_state, components['angular_velocity'], 'o-', color='C1', linewidth=2, markersize=4)
	ax.set_title('Angular Velocity Cost')
	ax.set_xlabel('Time [s]')
	ax.set_ylabel('Angular Velocity Cost')
	ax.grid(True, alpha=0.3)
	_configure_sci_y(ax)

	ax = fig.add_subplot(gs[1, 1])
	ax.plot(t_state, w[0, :], label='ωx', linewidth=1.8)
	ax.plot(t_state, w[1, :], label='ωy', linewidth=1.8)
	ax.plot(t_state, w[2, :], label='ωz', linewidth=1.8)
	ax.plot(t_state, np.linalg.norm(w, axis=0), 'k--', label='‖ω‖', linewidth=1.8)
	ax.set_title('Angular Rates')
	ax.set_xlabel('Time [s]')
	ax.set_ylabel('Rate [rad/s]')
	ax.grid(True, alpha=0.3)
	ax.legend(loc='upper right', fontsize=8)

	# Row 3: control cost + controls
	ax = fig.add_subplot(gs[2, 0])
	ax.plot(t_state, components['control'], 'o-', color='C2', linewidth=2, markersize=4)
	ax.set_title('Control Cost')
	ax.set_xlabel('Time [s]')
	ax.set_ylabel('Control Cost')
	ax.grid(True, alpha=0.3)
	_configure_sci_y(ax)

	ax = fig.add_subplot(gs[2, 1])
	ax.plot(t_control, U[0, :], label='MTQ x', linewidth=1.6)
	ax.plot(t_control, U[1, :], label='MTQ y', linewidth=1.6)
	ax.plot(t_control, U[2, :], label='MTQ z', linewidth=1.6)
	ax.plot(t_control, U[3, :], '--', label='RW τx', linewidth=1.6)
	ax.plot(t_control, U[4, :], '--', label='RW τy', linewidth=1.6)
	ax.plot(t_control, U[5, :], '--', label='RW τz', linewidth=1.6)
	ax.set_title('Control Inputs')
	ax.set_xlabel('Time [s]')
	ax.set_ylabel('Command')
	ax.grid(True, alpha=0.3)
	ax.legend(loc='upper right', fontsize=7, ncol=2)

	# Row 4: rw cost + rw momentum states
	ax = fig.add_subplot(gs[3, 0])
	ax.plot(t_state, components['rw_momentum'], 'o-', color='C3', linewidth=2, markersize=4)
	ax.set_title('RW Momentum Cost')
	ax.set_xlabel('Time [s]')
	ax.set_ylabel('RW Cost')
	ax.grid(True, alpha=0.3)
	_configure_sci_y(ax)

	ax = fig.add_subplot(gs[3, 1])
	ax.plot(t_state, h_rw[0, :], label='h_rw0', linewidth=1.8)
	ax.plot(t_state, h_rw[1, :], label='h_rw1', linewidth=1.8)
	ax.plot(t_state, h_rw[2, :], label='h_rw2', linewidth=1.8)
	ax.set_title('RW Momentum States')
	ax.set_xlabel('Time [s]')
	ax.set_ylabel('Momentum [N·m·s]')
	ax.grid(True, alpha=0.3)
	ax.legend(loc='upper right', fontsize=8)

	fig.suptitle('Individual Cost Terms - Mega Window', fontsize=14, fontweight='bold')
	fig.show()
	return fig


def plot_rw_cost_surfaces(X, U, N, dt, satellite, attitude_target_traj, boresight, B, cost_cfg, x0):
	"""
	Create 3D cost surface plots for each reaction wheel.
	
	For each RW, creates a 3D surface showing cost as a function of RW momentum,
	with the actual trajectory overlaid in 3D.
	
	Args:
		X: State trajectory (nx x N)
		U: Control trajectory (nu x N-1)
		N: Number of timesteps
		dt: Timestep duration
		satellite: Satellite object
		attitude_target_traj: Target quaternion trajectory (4 x N)
		boresight: Boresight vector for each timestep (3 x N)
		B: Magnetic field for each timestep (3 x N)
		cost_cfg: Cost configuration
		x0: Initial state
	"""
	# Use RW-only cost configuration for meaningful RW momentum surface
	rw_cfg = saltro_py.CostConfig()
	rw_cfg.angle = 0.0
	rw_cfg.angle_N = 0.0
	rw_cfg.ang_vel = 0.0
	rw_cfg.ang_vel_N = 0.0
	rw_cfg.ang_vel_mag = 0.0
	rw_cfg.ang_vel_mag_N = 0.0
	rw_cfg.ang_vel_err_dir = 0.0
	rw_cfg.ang_vel_err_dir_N = 0.0
	rw_cfg.control_mult = 0.0
	rw_cfg.mtq_control_weight = 0.0
	rw_cfg.rw_control_weight = 0.0
	rw_cfg.magic_control_weight = 0.0
	rw_cfg.rw_AM_weight = cost_cfg.rw_AM_weight
	rw_cfg.rw_stic_weight = cost_cfg.rw_stic_weight
	rw_cfg.RWh_max_mult = cost_cfg.RWh_max_mult
	rw_cfg.RWh_stiction_mult = cost_cfg.RWh_stiction_mult
	rw_cfg.RWh_ok_mult = cost_cfg.RWh_ok_mult
	rw_cfg.ang_cost_func_type = cost_cfg.ang_cost_func_type
	rw_cfg.use_cost_hess = False

	t_state = np.arange(N) * dt
	h_rw_trajectory = X[7:10, :]
	h_range = np.linspace(-0.02, 0.02, 70)

	fig = plt.figure(figsize=(18, 6))
	for wheel_idx in range(3):
		ax = fig.add_subplot(1, 3, wheel_idx + 1, projection='3d')

		# Build surface with axes as requested (flipped from previous version):
		# x: RW momentum, y: time, z: cost
		H, T = np.meshgrid(h_range, t_state)
		C = np.zeros_like(H)

		for ti in range(N):
			x_base = X[:, ti].copy()
			u_k = U[:, ti] if ti < U.shape[1] else np.zeros(satellite.controlDim)
			for hi, h_val in enumerate(h_range):
				x_test = x_base.copy()
				x_test[7 + wheel_idx] = h_val
				C[ti, hi] = satellite.stageCost(
					ti, N,
					x_test,
					u_k,
					boresight[:, ti],
					attitude_target_traj[:, ti],
					B[:, ti],
					rw_cfg,
				)

		ax.plot_surface(H, T, C, cmap='viridis', alpha=0.72, edgecolor='none', antialiased=True)

		# Trajectory line on the surface
		h_line = h_rw_trajectory[wheel_idx, :]
		c_line = np.zeros(N)
		for k in range(N):
			u_k = U[:, k] if k < U.shape[1] else np.zeros(satellite.controlDim)
			c_line[k] = satellite.stageCost(
				k, N,
				X[:, k],
				u_k,
				boresight[:, k],
				attitude_target_traj[:, k],
				B[:, k],
				rw_cfg,
			)

		ax.plot(h_line, t_state, c_line, 'r-', linewidth=2.8, label='RW trajectory over time')
		ax.scatter(h_line, t_state, c_line, c=t_state, cmap='coolwarm', s=30, edgecolors='black', linewidth=0.4)

		ax.set_xlabel(f'h_rw{wheel_idx} [N·m·s]', labelpad=10)
		ax.set_ylabel('Time [s]', labelpad=10)
		ax.set_zlabel('RW Cost', labelpad=10)
		ax.set_title(f'RW {wheel_idx}: Momentum-Cost-Time Surface', fontsize=12, fontweight='bold', pad=14)
		ax.view_init(elev=22, azim=-58)
		ax.legend(loc='upper left')

	plt.tight_layout()
	plt.show()

	return fig


def plot_cost(X, U, N, dt, satellite, attitude_target_traj, boresight, B, cost_cfg, x0):
	"""
	Create overall cost analysis plots.
	
	Generates a clean summary of the trajectory cost including total cost,
	per-timestep breakdown, accumulation, and component breakdown.
	
	Args:
		X: State trajectory (nx x N)
		U: Control trajectory (nu x N-1)
		N: Number of timesteps
		dt: Timestep duration
		satellite: Satellite object
		attitude_target_traj: Target quaternion trajectory (4 x N)
		boresight: Boresight vector for each timestep (3 x N)
		B: Magnetic field for each timestep (3 x N)
		cost_cfg: Cost configuration
		x0: Initial state (for RW cost function)
	"""
	start_time = time.time()
	total_cost, costs_per_step = compute_trajectory_costs(
		X, U, N, satellite, attitude_target_traj, boresight, B, cost_cfg
	)
	elapsed_cost = time.time() - start_time
	
	start_time = time.time()
	components = compute_cost_components(
		X, U, N, satellite, attitude_target_traj, boresight, B, cost_cfg
	)
	elapsed_components = time.time() - start_time
	
	# Time vectors
	t_state = np.arange(N) * dt
	
	# Define component styling
	component_names = list(components.keys())
	colors = ['#FF6B6B', '#2A9D8F', '#6D597A', '#FFA07A']
	component_sums = [np.sum(components[name]) for name in component_names]
	
	# Create cleaner 2x2 figure for overall cost analysis
	fig = plt.figure(figsize=(14, 10), constrained_layout=True)
	gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
	
	# (1) Total Cost - Single Value (larger, prominent)
	ax1 = fig.add_subplot(gs[0, 0])
	ax1.text(0.5, 0.5, f'Total Cost\n{total_cost:.4e}', 
			ha='center', va='center', fontsize=18, fontweight='bold',
			bbox=dict(boxstyle='round', facecolor='#E8F4F8', alpha=0.9, 
					 edgecolor='#4ECDC4', linewidth=2))
	ax1.set_xlim(0, 1)
	ax1.set_ylim(0, 1)
	ax1.axis('off')
	
	# (2) Cost per Timestep - Stacked by component
	ax2 = fig.add_subplot(gs[0, 1])
	
	# Create stacked bar chart
	bottom = np.zeros(N)
	for idx, comp_name in enumerate(component_names):
		ax2.bar(t_state, components[comp_name], width=dt*0.8, 
				bottom=bottom, label=comp_name.replace('_', ' ').title(), alpha=0.85, 
				color=colors[idx % len(colors)], edgecolor='black', linewidth=0.5)
		bottom += components[comp_name]
	
	ax2.set_xlabel('Time [s]', fontsize=10)
	ax2.set_ylabel('Cost per Timestep', fontsize=10)
	ax2.set_title('Cost Distribution Across Trajectory', fontsize=11, fontweight='bold')
	ax2.legend(loc='upper right', fontsize=8)
	ax2.grid(True, alpha=0.3, axis='y')
	_configure_sci_y(ax2)
	
	# (3) Accumulated Cost
	ax3 = fig.add_subplot(gs[1, 0])
	accumulated_cost = np.cumsum(costs_per_step)
	ax3.plot(t_state, accumulated_cost, 'o-', linewidth=2.5, markersize=6, color='#2C3E50')
	ax3.fill_between(t_state, 0, accumulated_cost, alpha=0.25, color='#3498DB')
	ax3.set_xlabel('Time [s]', fontsize=10)
	ax3.set_ylabel('Accumulated Cost', fontsize=10)
	ax3.set_title('Cumulative Cost Growth', fontsize=11, fontweight='bold')
	ax3.grid(True, alpha=0.3)
	_configure_sci_y(ax3)
	
	# (4) Cost Breakdown Pie Chart with Legend
	ax4 = fig.add_subplot(gs[1, 1])
	
	# Only plot non-zero components
	nonzero_components = [(name, value) for name, value in 
						 zip(component_names, component_sums) 
						 if value > 1e-10]
	if nonzero_components:
		names, values = zip(*nonzero_components)
		# Get colors matching the non-zero components
		nonzero_indices = [component_names.index(name) for name, _ in nonzero_components]
		nonzero_colors = [colors[idx % len(colors)] for idx in nonzero_indices]
		
		wedges, texts, autotexts = ax4.pie(values, autopct='%1.1f%%', 
										   colors=nonzero_colors, startangle=90)
		for autotext in autotexts:
			autotext.set_color('white')
			autotext.set_fontweight('bold')
			autotext.set_fontsize(9)
		
		# Create legend with component names and values
		legend_labels = [f'{name.replace("_", " ").title()}: {value:.2e}' 
						for name, value in zip(names, values)]
		ax4.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1), 
				  fontsize=8, framealpha=0.95)
	
	ax4.set_title('Component Contribution', fontsize=11, fontweight='bold')
	
	plt.suptitle('Warm-Start Trajectory: Overall Cost Analysis', 
				fontsize=13, fontweight='bold', y=0.98)
	plt.show()
	
	print(f"Cost calculation time: {elapsed_cost*1000:.2f} ms")
	print(f"Total cost: {total_cost:.4e}")
	
	return fig


def plot_ilqr_iteration_debug(cost_history, predicted_reductions, actual_reductions, alphas):
	"""
	Plot Python-side iLQR loop diagnostics.

	Args:
		cost_history: list of trajectory costs [J0, J1, ...]
		predicted_reductions: list of predicted reductions per iter
		actual_reductions: list of actual reductions per iter
		alphas: list of accepted line-search alpha per iter
	"""
	if len(cost_history) == 0:
		return None

	iters = np.arange(len(cost_history))
	iters_step = np.arange(1, len(cost_history))
	pred = np.asarray(predicted_reductions) if len(predicted_reductions) else np.array([])
	act = np.asarray(actual_reductions) if len(actual_reductions) else np.array([])
	alpha_arr = np.asarray(alphas) if len(alphas) else np.array([])

	fig = plt.figure(figsize=(14, 9), constrained_layout=True)
	gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

	# Cost by iteration
	ax1 = fig.add_subplot(gs[0, 0])
	ax1.plot(iters, cost_history, 'o-', linewidth=2.2, markersize=5, color='C0')
	ax1.set_xlabel('Iteration')
	ax1.set_ylabel('Cost')
	ax1.set_title('iLQR Cost per Iteration')
	ax1.grid(True, alpha=0.3)
	_configure_sci_y(ax1)

	# Reduction: predicted vs actual
	ax2 = fig.add_subplot(gs[0, 1])
	if len(pred) and len(act):
		ax2.plot(iters_step, pred, 's--', linewidth=1.8, markersize=5, label='Predicted ΔJ', color='C2')
		ax2.plot(iters_step, act, 'o-', linewidth=2.0, markersize=5, label='Actual ΔJ', color='C1')
	ax2.axhline(0.0, color='k', linestyle='--', alpha=0.35)
	ax2.set_xlabel('Iteration')
	ax2.set_ylabel('Cost Reduction')
	ax2.set_title('Predicted vs Actual Reduction')
	ax2.grid(True, alpha=0.3)
	_configure_sci_y(ax2)
	ax2.legend(loc='best', fontsize=8)

	# Line-search alpha
	ax3 = fig.add_subplot(gs[1, 0])
	if len(alpha_arr):
		ax3.step(iters_step, alpha_arr, where='mid', linewidth=2.0, color='C3')
		ax3.plot(iters_step, alpha_arr, 'o', color='C3', markersize=4)
	ax3.set_xlabel('Iteration')
	ax3.set_ylabel('Accepted alpha')
	ax3.set_title('Line Search Step Size')
	ax3.set_ylim(0.0, 1.05)
	ax3.grid(True, alpha=0.3)

	# Reduction ratio (actual/predicted)
	ax4 = fig.add_subplot(gs[1, 1])
	if len(pred) and len(act):
		ratio = np.divide(act, pred, out=np.zeros_like(act), where=np.abs(pred) > 1e-16)
		ax4.plot(iters_step, ratio, 'o-', linewidth=2.0, markersize=5, color='C4')
	ax4.axhline(1.0, color='k', linestyle='--', alpha=0.35)
	ax4.set_xlabel('Iteration')
	ax4.set_ylabel('Actual / Predicted')
	ax4.set_title('Reduction Agreement')
	ax4.grid(True, alpha=0.3)

	fig.suptitle('Python iLQR Loop Debug Diagnostics', fontsize=13, fontweight='bold')
	plt.show()
	return fig
