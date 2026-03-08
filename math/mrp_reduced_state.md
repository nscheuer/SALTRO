# MRP Reduced State Representation

## Overview

The iLQR optimizer uses a **reduced state** representation for the backward and forward
passes to avoid Riccati instability caused by the 4-parameter quaternion's normalization
constraint.

## State Definitions

**Full state** (used for dynamics integration):
$$
\mathbf{x} = \begin{bmatrix} \boldsymbol{\omega} \\ \mathbf{q} \\ \mathbf{h}_{rw} \end{bmatrix}
\quad \text{dim} = 7 + n_{RW}
$$

**Reduced state** (used for iLQR Q-matrices):
$$
\delta\mathbf{z} = \begin{bmatrix} \delta\boldsymbol{\omega} \\ \delta\boldsymbol{\theta} \\ \delta\mathbf{h}_{rw} \end{bmatrix}
\quad \text{dim} = 6 + n_{RW}
$$

where $\delta\boldsymbol{\theta}$ is a 3-parameter attitude error (Modified Rodrigues Parameters).

## G-Matrix Projection

The G matrix projects full-state Jacobians to reduced-state:

$$
G(\mathbf{q}) = \begin{bmatrix}
I_3 & 0 & 0 \\
0 & W^T(\mathbf{q}) & 0 \\
0 & 0 & I_{n_{RW}}
\end{bmatrix}
\quad G \in \mathbb{R}^{(6+n_{RW}) \times (7+n_{RW})}
$$

where $W(\mathbf{q})$ is the 4×3 quaternion kinematics matrix from $\dot{\mathbf{q}} = \frac{1}{2}W\boldsymbol{\omega}$.

## Backward Pass Projection

The discrete-time Jacobians from RK4 are in full state space:
- $A_k \in \mathbb{R}^{(7+n_{RW}) \times (7+n_{RW})}$
- $B_k \in \mathbb{R}^{(7+n_{RW}) \times n_u}$

We project them to reduced state:
$$
\bar{A}_k = G(\mathbf{q}_{k+1}) \, A_k \, G^T(\mathbf{q}_k)
\quad \in \mathbb{R}^{(6+n_{RW}) \times (6+n_{RW})}
$$
$$
\bar{B}_k = G(\mathbf{q}_{k+1}) \, B_k
\quad \in \mathbb{R}^{(6+n_{RW}) \times n_u}
$$

The cost Jacobians and Hessians are similarly projected:
$$
\bar{l}_x = G_k \, l_x, \quad
\bar{l}_{xx} = G_k \, l_{xx} \, G_k^T, \quad
\bar{l}_{ux} = l_{ux} \, G_k^T
$$

All Q-matrices and the Riccati recursion then operate in the reduced (6+$n_{RW}$)-dimensional space.

## Forward Pass: MRP Error

In the forward pass, the state error between the rolled-out trajectory $\bar{x}_k$ and
the nominal $x_k$ is computed in reduced state:

$$
\delta\mathbf{z}_k = \begin{bmatrix}
\bar{\boldsymbol{\omega}}_k - \boldsymbol{\omega}_k \\
\text{MRP}(\mathbf{q}_{ref,k}^{-1} \otimes \bar{\mathbf{q}}_k) \\
\bar{\mathbf{h}}_{rw,k} - \mathbf{h}_{rw,k}
\end{bmatrix}
$$

where the MRP of a quaternion error $\mathbf{q}_e = [q_0, \mathbf{q}_v]^T$ is:

$$
\boldsymbol{\sigma} = \frac{2\,\mathbf{q}_v}{1 + q_0}
$$

## Riccati Update Fix

The value function propagation uses **unregularized** $Q_{uu}$:

$$
P_k = Q_{xx} + K_k^T Q_{uu} K_k + K_k^T Q_{ux} + Q_{ux}^T K_k
$$

Only the gain/feedforward computation uses the regularized version:

$$
K_k = -(Q_{uu} + \rho I)^{-1} Q_{ux}, \quad
d_k = -(Q_{uu} + \rho I)^{-1} Q_u
$$

This prevents regularization from inflating $P_k$ and creating positive feedback in the Riccati recursion.
