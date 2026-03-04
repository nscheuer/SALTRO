"""
Warm-start trajectory visualization.

This module provides a function to plot warm-start trajectory data.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def plot_warmstart(X, U, N, dt, attitude_target_traj):
	"""
	Plot warm-start trajectory.
	
	Args:
		X: State trajectory (nx x N)
		U: Control trajectory (nu x N-1)
		N: Number of timesteps
		dt: Timestep duration
		attitude_target_traj: Target quaternion trajectory (4 x N)
	"""
	
	# Time vectors
	t_state = np.arange(N) * dt
	t_control = (np.arange(N - 1) * dt 
				if U.shape[1] == N - 1 
				else np.arange(N) * dt)
	
	# Extract state components [w(3), q(4), h_rw(3)]
	w = X[0:3, :]
	q = X[3:7, :]
	h_rw = X[7:10, :]
	
	# Extract control components [m_mtq(3), tau_rw(3)]
	m_mtq = U[0:3, :]
	tau_rw = U[3:6, :]
	
	fig, axes = plt.subplots(5, 1, figsize=(12, 16))
	
	# Quaternion trajectory
	ax = axes[0]
	ax.plot(t_state, q[0, :], 'o-', label='q0 (scalar)', markersize=4)
	ax.plot(t_state, q[1, :], 's-', label='q1 (x)', markersize=4)
	ax.plot(t_state, q[2, :], '^-', label='q2 (y)', markersize=4)
	ax.plot(t_state, q[3, :], 'd-', label='q3 (z)', markersize=4)
	
	q_goal = attitude_target_traj
	ax.plot(t_state, q_goal[0, :], '--', color='C0', alpha=0.5, label='q0 goal')
	ax.plot(t_state, q_goal[1, :], '--', color='C1', alpha=0.5, label='q1 goal')
	ax.plot(t_state, q_goal[2, :], '--', color='C2', alpha=0.5, label='q2 goal')
	ax.plot(t_state, q_goal[3, :], '--', color='C3', alpha=0.5, label='q3 goal')
	
	ax.set_xlabel('Time [s]')
	ax.set_ylabel('Quaternion')
	ax.set_title('Quaternion Trajectory (Warm Start)')
	ax.legend(ncol=2, fontsize=8)
	ax.grid(True, alpha=0.3)
	
	# Angular velocity
	ax = axes[1]
	ax.plot(t_state, w[0, :], 'o-', label='ωx', markersize=4)
	ax.plot(t_state, w[1, :], 's-', label='ωy', markersize=4)
	ax.plot(t_state, w[2, :], '^-', label='ωz', markersize=4)
	ax.axhline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)
	ax.set_xlabel('Time [s]')
	ax.set_ylabel('Angular Rate [rad/s]')
	ax.set_title('Angular Velocity Trajectory')
	ax.legend()
	ax.grid(True, alpha=0.3)
	
	# Reaction wheel momentum
	ax = axes[2]
	ax.plot(t_state, h_rw[0, :], 'o-', label='h_rw1', markersize=4)
	ax.plot(t_state, h_rw[1, :], 's-', label='h_rw2', markersize=4)
	ax.plot(t_state, h_rw[2, :], '^-', label='h_rw3', markersize=4)
	ax.axhline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)
	ax.set_xlabel('Time [s]')
	ax.set_ylabel('Momentum [N·m·s]')
	ax.set_title('Reaction Wheel Momentum')
	ax.legend()
	ax.grid(True, alpha=0.3)
	
	# MTQ dipole commands
	ax = axes[3]
	ax.plot(t_control, m_mtq[0, :], 'o-', label='m_x', markersize=4)
	ax.plot(t_control, m_mtq[1, :], 's-', label='m_y', markersize=4)
	ax.plot(t_control, m_mtq[2, :], '^-', label='m_z', markersize=4)
	ax.axhline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)
	ax.set_xlabel('Time [s]')
	ax.set_ylabel('Dipole Moment [A·m²]')
	ax.set_title('MTQ Control Commands')
	ax.legend()
	ax.grid(True, alpha=0.3)
	
	# RW torque commands
	ax = axes[4]
	ax.plot(t_control, tau_rw[0, :], 'o-', label='τ_rw1', markersize=4)
	ax.plot(t_control, tau_rw[1, :], 's-', label='τ_rw2', markersize=4)
	ax.plot(t_control, tau_rw[2, :], '^-', label='τ_rw3', markersize=4)
	ax.axhline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)
	ax.set_xlabel('Time [s]')
	ax.set_ylabel('Torque [N·m]')
	ax.set_title('Reaction Wheel Torque Commands')
	ax.legend()
	ax.grid(True, alpha=0.3)
	
	plt.tight_layout()
	plt.suptitle('Warm-Start Trajectory Visualization', fontsize=14, fontweight='bold', y=0.995)
	plt.show()
	
	return fig
