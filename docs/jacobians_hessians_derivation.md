# SALTRO Satellite Dynamics: Jacobian & Hessian Derivations

This document derives and validates all analytical expressions used in
`Satellite::dynamicsJacobians` and `Satellite::dynamicsHessians`.

---

## 1. State and Dynamics Overview

### State vector

$$
\mathbf{x} = \begin{bmatrix} \boldsymbol{\omega} \\ \mathbf{q} \\ \mathbf{h} \end{bmatrix}
\in \mathbb{R}^{n_x}, \quad n_x = 7 + n_{rw}
$$

- $\boldsymbol{\omega} \in \mathbb{R}^3$ – body angular velocity (indices 0–2)  
- $\mathbf{q} = [q_0, q_1, q_2, q_3]^\top \in \mathbb{R}^4$ – unit quaternion (indices 3–6)  
- $\mathbf{h} \in \mathbb{R}^{n_{rw}}$ – reaction-wheel angular momenta (indices 7+)

### Control vector

$$
\mathbf{u} = \begin{bmatrix} \mathbf{m} \\ \boldsymbol{\tau}_{rw} \end{bmatrix} \in \mathbb{R}^{n_u},
\quad n_u = n_{mtq} + n_{rw}
$$

### Equations of motion

**Angular velocity:**

$$
\dot{\boldsymbol{\omega}} = J_\text{eff}^{-1}\!\left(
  \boldsymbol{\tau}_\text{act} + \boldsymbol{\tau}_\text{dist}
  - \boldsymbol{\omega} \times (J\,\boldsymbol{\omega} + \mathbf{h}_{rw})
\right)
$$

where $J_\text{eff} = J - \sum_i J_{rw,i}\, \hat{a}_i \hat{a}_i^\top$ and $\mathbf{h}_{rw} = \sum_i h_i \hat{a}_i$.

**Quaternion kinematics:**

$$
\dot{\mathbf{q}} = \tfrac{1}{2}\, W(\mathbf{q})\,\boldsymbol{\omega}
$$

with

$$
W(\mathbf{q}) =
\begin{bmatrix}
-q_1 & -q_2 & -q_3 \\
 q_0 & -q_3 &  q_2 \\
 q_3 &  q_0 & -q_1 \\
-q_2 &  q_1 &  q_0
\end{bmatrix}
$$

**Reaction-wheel momentum:**

$$
\dot{h}_i = -\tau_{rw,i} - J_{rw,i}\, \hat{a}_i^\top \dot{\boldsymbol{\omega}}
$$

---

## 2. Actuator Torques

### Magnetorquer (MTQ)

$$
\boldsymbol{\tau}_{mtq,i} = -(B_\text{body} \times \hat{a}_i)\, m_i
= -[B_\text{body}]_\times \hat{a}_i\, m_i
$$

where $B_\text{body} = R(\mathbf{q})^\top B_\text{eci}$, $[v]_\times$ is the skew-symmetric matrix of $v$, and $m_i$ is the dipole input.

### Reaction Wheel (RW)

$$
\boldsymbol{\tau}_{rw,i} = \hat{a}_i\, \tau_i
$$

---

## 3. Jacobian of Angular Velocity: $\partial\dot{\boldsymbol{\omega}}/\partial\mathbf{x}$

### 3.1  $\partial\dot{\boldsymbol{\omega}}/\partial\boldsymbol{\omega}$

$$
\dot{\boldsymbol{\omega}} = J_\text{eff}^{-1}(\ldots - \boldsymbol{\omega} \times \boldsymbol{p}),
\quad \boldsymbol{p} = J\boldsymbol{\omega} + \mathbf{h}_{rw}
$$

Using the product rule on $\boldsymbol{\omega} \times \boldsymbol{p}$ where $\boldsymbol{p}$ also depends on $\boldsymbol{\omega}$:

$$
\frac{\partial(\boldsymbol{\omega}\times\boldsymbol{p})}{\partial\boldsymbol{\omega}}
= -[\boldsymbol{p}]_\times + [\boldsymbol{\omega}]_\times J
$$

Therefore:

$$
\boxed{
\frac{\partial\dot{\boldsymbol{\omega}}}{\partial\boldsymbol{\omega}}
= J_\text{eff}^{-1}\bigl([\boldsymbol{p}]_\times - [\boldsymbol{\omega}]_\times J\bigr)
}
$$

*Implementation:* `invJcom_noRW_ * (skew(angular_mom) - skew(w) * Jcom_)` ✓

### 3.2  $\partial\dot{\boldsymbol{\omega}}/\partial\mathbf{q}$

Torques depend on $\mathbf{q}$ through $B_\text{body} = R^\top B_\text{eci}$ (MTQs) and through the disturbance models (GG, drag, SRP).

$$
\frac{\partial\dot{\boldsymbol{\omega}}}{\partial\mathbf{q}}
= J_\text{eff}^{-1}
\left(
  \sum_i \frac{\partial\boldsymbol{\tau}_{mtq,i}}{\partial\mathbf{q}}
  + \frac{\partial\boldsymbol{\tau}_\text{dist}}{\partial\mathbf{q}}
\right)
$$

Each MTQ contribution uses the chain rule through $B_\text{body}$:

$$
\frac{\partial\boldsymbol{\tau}_{mtq,i}}{\partial\mathbf{q}}
= -[B_\text{body}]_\times \hat{a}_i \cdot m_i \qquad \Rightarrow
\quad \frac{\partial\boldsymbol{\tau}_{mtq,i}}{\partial q_j}
= -\left[\frac{\partial B_\text{body}}{\partial q_j}\right]_\times \hat{a}_i\, m_i
$$

with $\partial B_\text{body}/\partial\mathbf{q}$ provided by `drotmatTvecdq`. *Implementation* calls `MTQ::dtorq_dbasestate`, extracting the quaternion rows (3–6) of the 7×3 Jacobian, transposing to get a 3×4 block. ✓

### 3.3  $\partial\dot{\boldsymbol{\omega}}/\partial\mathbf{h}$

Only gyroscopic cross-coupling: $-\boldsymbol{\omega}\times\mathbf{h}_{rw}$ where $\mathbf{h}_{rw} = \sum_k h_k \hat{a}_k$.

$$
\frac{\partial\dot{\boldsymbol{\omega}}}{\partial h_k}
= -J_\text{eff}^{-1}\,(\boldsymbol{\omega}\times\hat{a}_k)
$$

*Implementation:* `-invJcom_noRW_ * skew(w) * axis_k` ✓

### 3.4  $\partial\dot{\boldsymbol{\omega}}/\partial\mathbf{u}$

MTQ: $\partial\boldsymbol{\tau}_{mtq,i}/\partial m_i = -(B_\text{body}\times\hat{a}_i)$

RW: $\partial\boldsymbol{\tau}_{rw,i}/\partial\tau_i = \hat{a}_i$

$$
\boxed{
\frac{\partial\dot{\boldsymbol{\omega}}}{\partial u_i}
= J_\text{eff}^{-1} \cdot
\begin{cases}
-(B_\text{body}\times\hat{a}_i) & \text{(MTQ)} \\
\hat{a}_i & \text{(RW)}
\end{cases}
}
$$

*Implementation* ✓

---

## 4. Jacobian of Quaternion Kinematics: $\partial\dot{\mathbf{q}}/\partial\mathbf{x}$

### 4.1  $\partial\dot{\mathbf{q}}/\partial\boldsymbol{\omega}$

$$
\dot{\mathbf{q}} = \tfrac{1}{2} W(\mathbf{q})\,\boldsymbol{\omega}
\implies
\frac{\partial\dot{\mathbf{q}}}{\partial\boldsymbol{\omega}} = \tfrac{1}{2} W(\mathbf{q})
$$

*Implementation* ✓

### 4.2  $\partial\dot{\mathbf{q}}/\partial\mathbf{q}$

Since $W(\mathbf{q})$ is linear in $\mathbf{q}$:

$$
\frac{\partial\dot{q}_i}{\partial q_j} = \tfrac{1}{2}\,
\frac{\partial [W(\mathbf{q})\,\boldsymbol{\omega}]_i}{\partial q_j}
= \tfrac{1}{2}\, \left[\frac{\partial W}{\partial q_j}\,\boldsymbol{\omega}\right]_i
$$

Expanding $W\,\boldsymbol{\omega}$ element-by-element:

$$
\begin{aligned}
\dot{q}_0 &= \tfrac{1}{2}(-q_1\omega_0 - q_2\omega_1 - q_3\omega_2) \\
\dot{q}_1 &= \tfrac{1}{2}(\phantom{-}q_0\omega_0 - q_3\omega_1 + q_2\omega_2) \\
\dot{q}_2 &= \tfrac{1}{2}(\phantom{-}q_3\omega_0 + q_0\omega_1 - q_1\omega_2) \\
\dot{q}_3 &= \tfrac{1}{2}(-q_2\omega_0 + q_1\omega_1 + q_0\omega_2)
\end{aligned}
$$

Differentiating with respect to each quaternion component:

| | $\partial/\partial q_0$ | $\partial/\partial q_1$ | $\partial/\partial q_2$ | $\partial/\partial q_3$ |
|---|---|---|---|---|
| $\partial\dot{q}_0$ | $0$ | $-\omega_0$ | $-\omega_1$ | $-\omega_2$ |
| $\partial\dot{q}_1$ | $+\omega_0$ | $0$ | $+\omega_2$ | $-\omega_1$ |
| $\partial\dot{q}_2$ | $+\omega_1$ | $-\omega_2$ | $0$ | $+\omega_0$ |
| $\partial\dot{q}_3$ | $+\omega_2$ | $+\omega_1$ | $\mathbf{-\omega_0}$ | $0$ |

(The factor of $\tfrac{1}{2}$ is applied in the implementation.)

---

### ⚠️ Bug 1 — Sign error in $\partial\dot{q}_3/\partial q_2$

**Correct:** $\partial\dot{q}_3/\partial q_2 = \tfrac{1}{2}(-\omega_0)$

**Code (before fix):**

```cpp
} else if (j == 2) {
    dW_col(3) = w(0);   // ← WRONG: should be -w(0)
```

Row 3 of $W\boldsymbol{\omega}$ is $(-q_2\omega_0 + q_1\omega_1 + q_0\omega_2)$, so $\partial/\partial q_2 = -\omega_0$.

**Fix:** change `dW_col(3) = w(0)` → `dW_col(3) = -w(0)`.

---

## 5. Jacobian of RW Momentum: $\partial\dot{\mathbf{h}}/\partial\mathbf{x}$

$$
\dot{h}_i = -\tau_{rw,i} - J_{rw,i}\,\hat{a}_i^\top\dot{\boldsymbol{\omega}}
$$

All derivatives follow from the chain rule through $\dot{\boldsymbol{\omega}}$:

$$
\frac{\partial\dot{h}_i}{\partial x_j}
= -J_{rw,i}\,\hat{a}_i^\top \frac{\partial\dot{\boldsymbol{\omega}}}{\partial x_j}
\quad (x_j \in \{\boldsymbol{\omega}, \mathbf{q}, \mathbf{h}\})
$$

For the control:
$$
\frac{\partial\dot{h}_i}{\partial u_j}
= -\delta_{j, n_{mtq}+i} - J_{rw,i}\,\hat{a}_i^\top \frac{\partial\dot{\boldsymbol{\omega}}}{\partial u_j}
$$

*Implementation* ✓

---

## 6. Hessian of Angular Velocity: $\partial^2\dot{\boldsymbol{\omega}}/\partial\mathbf{x}^2$

### 6.1  $\partial^2\dot{\boldsymbol{\omega}}/\partial\boldsymbol{\omega}^2$

The gyroscopic term $\boldsymbol{\omega}\times(J\boldsymbol{\omega}+\mathbf{h})$ is quadratic in $\boldsymbol{\omega}$. Its Hessian w.r.t. $\boldsymbol{\omega}$ is nonzero. For output component $i$, and indices $j,k$:

$$
\frac{\partial^2\dot{\omega}_i}{\partial\omega_j\partial\omega_k}
= -[J_\text{eff}^{-1}]_{i\cdot}\left(\frac{\partial^2(\boldsymbol{\omega}\times J\boldsymbol{\omega})}{\partial\omega_j\partial\omega_k}\right)
$$

*Implementation* ✓ (computed via nested loop over skew-symmetric terms)

### 6.2  $\partial^2\dot{\boldsymbol{\omega}}/\partial\boldsymbol{\omega}\partial\mathbf{h}$

$$
\frac{\partial^2\dot{\omega}_i}{\partial\omega_j\partial h_k}
= -[J_\text{eff}^{-1}]_{i\cdot}\,\frac{\partial(\boldsymbol{\omega}\times\hat{a}_k)}{\partial\omega_j}
= -[J_\text{eff}^{-1}]_{i\cdot}\,[\hat{a}_k]_\times^\top \mathbf{e}_j
$$

(symmetric in $j \leftrightarrow k$). *Implementation* ✓

### 6.3  $\partial^2\dot{\boldsymbol{\omega}}/\partial\mathbf{q}^2$

Comes from second derivatives of disturbance torques. Implemented by calling
`GGDisturbance::ddtorque_dqdq`, `DragDisturbance::ddtorque_dqdq`,
`SRPDisturbance::ddtorque_dqdq`. *Implementation* ✓

### 6.4  Mixed Hessian $\partial^2\dot{\boldsymbol{\omega}}/\partial\mathbf{u}\partial\mathbf{q}$ — MTQ contribution

The MTQ torque is:

$$
\boldsymbol{\tau}_{mtq,i} = -(B_\text{body}\times\hat{a}_i)\,m_i
$$

The mixed partial $\partial^2\boldsymbol{\tau}_{mtq,i}/\partial m_i\partial q_j$ is computed by `MTQ::ddtorq_dudbasestate`, which returns a tensor of type `T173 = Tensor3<1, 7, 3>`. Each slice (indexed by torque output component $k=0,1,2$) is a **$1\times 7$** matrix — the single row corresponds to the unit MTQ control $m_i$, and the 7 columns correspond to the 7 base-state components (3 angular velocity + 4 quaternion).

---

### ⚠️ Bug 2 — Out-of-bounds row access in MTQ Hessian

```cpp
// Inside a loop: for (int j = 0; j < num_mtq_; ++j) { ...
auto H_mtq_ux = mtq.ddtorq_dudbasestate(...);   // Tensor3<1,7,3>
double val = H_mtq_ux.slice(i)(j, QUAT_INDEX + 0);  // ← WRONG
```

`H_mtq_ux.slice(i)` is a **$1\times 7$** Eigen matrix. Accessing row `j` (which equals 1 or 2 for the second and third MTQ) triggers an Eigen bounds assertion → **SIGABRT**.

Each per-actuator call to `ddtorq_dudbasestate` returns the $1\times 7$ matrix for *that* actuator's control input — row index is always **0**.

**Fix:** replace `(j, QUAT_INDEX + k)` with `(0, QUAT_INDEX + k)` everywhere inside the inner MTQ loop, and extend to cover all 4 quaternion components.

---

## 7. Quaternion Hessian: $\partial^2\dot{\mathbf{q}}/\partial\mathbf{x}^2$

$\dot{\mathbf{q}} = \tfrac{1}{2}W(\mathbf{q})\boldsymbol{\omega}$ is **bilinear** in $(\mathbf{q}, \boldsymbol{\omega})$.

$$
\frac{\partial^2\dot{q}_i}{\partial q_j\,\partial\omega_k}
= \tfrac{1}{2}\frac{\partial W_{ik}}{\partial q_j}
$$

These are $\pm 1/2$ constants (independent of state/control). The code currently sets these to zero, which is **an approximation** that is valid when the optimizer handles the quaternion bilinear coupling separately. All purely quadratic entries ($\partial^2\dot{\mathbf{q}}/\partial\mathbf{q}^2$ etc.) are zero by linearity.

---

## 8. RW Momentum Hessian

Follows by chain rule through $\dot{\boldsymbol{\omega}}$:

$$
\frac{\partial^2\dot{h}_i}{\partial x_j\partial x_k}
= -J_{rw,i}\,\hat{a}_i^\top\frac{\partial^2\dot{\boldsymbol{\omega}}}{\partial x_j\partial x_k}
$$

*Implementation* ✓

---

## 9. Summary of Bugs Found

| # | Location | Issue | Fix |
|---|---|---|---|
| 1 | `satellite.cpp`, W-matrix Jacobian, `j==2` case | `dW_col(3) = w(0)` should be `dW_col(3) = -w(0)` (sign error in $\partial\dot{q}_3/\partial q_2$) | Negate the sign |
| 2 | `satellite.cpp`, MTQ Hessian accumulation | `H_mtq_ux.slice(i)(j, ...)` uses MTQ loop index `j` as row; `Tensor3<1,7,3>` has only 1 row → SIGABRT | Use row `0` instead of row `j` |
