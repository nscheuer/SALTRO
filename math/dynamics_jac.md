# Satellite Attitude Dynamics: Full Jacobian Derivations

## State and Input Structure

**State:** $\mathbf{x} = \begin{bmatrix} \boldsymbol{\omega} \\ \mathbf{q} \\ \mathbf{h} \end{bmatrix} \in \mathbb{R}^{3+4+N_{\text{rw}}}$

**Control Input:** $\mathbf{u} = \begin{bmatrix} \mathbf{m} \\ \mathbf{u}_{\text{rw}} \end{bmatrix} \in \mathbb{R}^{N_{\text{mtq}}+N_{\text{rw}}}$

**Dynamics:** $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, \mathbf{u}, \boldsymbol{\tau}_{\text{dist}})$ where $\boldsymbol{\tau}_{\text{dist}}$ is an external disturbance torque input.

---

## Compact Dynamics Representation

Breaking the full dynamics into three blocks:

$$
\begin{align}
\dot{\boldsymbol{\omega}} &= \mathbf{J}_{\text{com}}^{-1} \left[ -[B_{\text{body}}]_\times A_{\text{mtq}} \mathbf{m} + A_{\text{rw}} \mathbf{u}_{\text{rw}} + \boldsymbol{\tau}_{\text{dist}} - \boldsymbol{\omega} \times \left( \mathbf{J}_{\text{com}} \boldsymbol{\omega} + A_{\text{rw}} \mathbf{h} \right) \right] \\
\dot{\mathbf{q}} &= \frac{1}{2} \mathbf{W}(\mathbf{q}) \boldsymbol{\omega} \\
\dot{\mathbf{h}} &= -\mathbf{u}_{\text{rw}} - D_{\text{rw}} A_{\text{rw}}^T \dot{\boldsymbol{\omega}}
\end{align}
$$

where:
- $B_{\text{body}} = \mathbf{R}^T(\mathbf{q}) \mathbf{B}_{\text{eci}}$ = magnetic field in body frame
- $[v]_\times$ = skew-symmetric matrix for cross product (notation used throughout)
- $D_{\text{rw}} = \text{diag}(J_1, J_2, \ldots, J_{N_{\text{rw}}})$ = RW inertias
- $\mathbf{W}(\mathbf{q})$ = quaternion kinematics matrix

---

## Part 1: Jacobian $\frac{\partial \dot{\mathbf{x}}}{\partial \mathbf{x}} = \frac{\partial \mathbf{f}}{\partial \mathbf{x}}$

The state Jacobian is a block matrix:

$$
\frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{bmatrix}
\frac{\partial \dot{\boldsymbol{\omega}}}{\partial \boldsymbol{\omega}} & \frac{\partial \dot{\boldsymbol{\omega}}}{\partial \mathbf{q}} & \frac{\partial \dot{\boldsymbol{\omega}}}{\partial \mathbf{h}} \\[6pt]
\frac{\partial \dot{\mathbf{q}}}{\partial \boldsymbol{\omega}} & \frac{\partial \dot{\mathbf{q}}}{\partial \mathbf{q}} & \frac{\partial \dot{\mathbf{q}}}{\partial \mathbf{h}} \\[6pt]
\frac{\partial \dot{\mathbf{h}}}{\partial \boldsymbol{\omega}} & \frac{\partial \dot{\mathbf{h}}}{\partial \mathbf{q}} & \frac{\partial \dot{\mathbf{h}}}{\partial \mathbf{h}}
\end{bmatrix} \in \mathbb{R}^{(3+4+N_{\text{rw}}) \times (3+4+N_{\text{rw}})}$$

### 1.1 Block: $\frac{\partial \dot{\boldsymbol{\omega}}}{\partial \boldsymbol{\omega}}$ (3×3)

**Derivation:**

From:
$$\dot{\boldsymbol{\omega}} = \mathbf{J}_{\text{com}}^{-1} \left[ \cdots - \boldsymbol{\omega} \times \left( \mathbf{J}_{\text{com}} \boldsymbol{\omega} + A_{\text{rw}} \mathbf{h} \right) \right]$$

Taking the derivative with respect to $\boldsymbol{\omega}$:

$$\frac{\partial \dot{\boldsymbol{\omega}}}{\partial \boldsymbol{\omega}} = \mathbf{J}_{\text{com}}^{-1} \frac{\partial}{\partial \boldsymbol{\omega}} \left( -\boldsymbol{\omega} \times \left( \mathbf{J}_{\text{com}} \boldsymbol{\omega} + A_{\text{rw}} \mathbf{h} \right) \right)$$

Let $\mathbf{v} = \mathbf{J}_{\text{com}} \boldsymbol{\omega} + A_{\text{rw}} \mathbf{h}$. Using the product rule and $\frac{\partial (\mathbf{a} \times \mathbf{b})}{\partial \mathbf{a}} = -[\mathbf{b}]_\times$:

$$\frac{\partial}{\partial \boldsymbol{\omega}} \left( -\boldsymbol{\omega} \times \mathbf{v} \right) = -\left( -[\mathbf{v}]_\times + [\boldsymbol{\omega}]_\times \mathbf{J}_{\text{com}} \right) = \left[\mathbf{J}_{\text{com}} \boldsymbol{\omega} + A_{\text{rw}} \mathbf{h}\right]_\times - [\boldsymbol{\omega}]_\times \mathbf{J}_{\text{com}}$$

**Result:**

$$\boxed{\frac{\partial \dot{\boldsymbol{\omega}}}{\partial \boldsymbol{\omega}} = \mathbf{J}_{\text{com}}^{-1} \left( \left[\mathbf{J}_{\text{com}} \boldsymbol{\omega} + A_{\text{rw}} \mathbf{h}\right]_\times - [\boldsymbol{\omega}]_\times \mathbf{J}_{\text{com}} \right)}$$

---

### 1.2 Block: $\frac{\partial \dot{\boldsymbol{\omega}}}{\partial \mathbf{q}}$ (3×4)

**Derivation:**

The $\mathbf{q}$-dependence enters through $B_{\text{body}} = \mathbf{R}^T(\mathbf{q}) \mathbf{B}_{\text{eci}}$ in the magnetorquer torque:

$$\boldsymbol{\tau}_{\text{mtq}} = -[B_{\text{body}}]_\times A_{\text{mtq}} \mathbf{m} = -[\mathbf{R}^T(\mathbf{q}) \mathbf{B}_{\text{eci}}]_\times A_{\text{mtq}} \mathbf{m}$$

Therefore:

$$\frac{\partial \dot{\boldsymbol{\omega}}}{\partial \mathbf{q}} = -\mathbf{J}_{\text{com}}^{-1} \frac{\partial}{\partial \mathbf{q}} \left( [\mathbf{R}^T(\mathbf{q}) \mathbf{B}_{\text{eci}}]_\times A_{\text{mtq}} \mathbf{m} \right)$$

For the skew-matrix derivative, using the property that $\frac{\partial [v]_\times}{\partial v} = -I_3$ (in the sense of its action on vectors):

$$\frac{\partial [B_{\text{body}}]_\times}{\partial B_{\text{body}}} : \text{yields vectorized form}$$

In component form, if $C = [v]_\times w$, then:

$$\frac{\partial C_i}{\partial v_j} = (I \otimes w)_{\text{relevant rows}} - (w \otimes I)_{\text{relevant rows}}$$

More practically, using $[v]_\times w = v \times w$:

$$\frac{\partial (B_{\text{body}} \times (A_{\text{mtq}} \mathbf{m}))}{\partial B_{\text{body}}} = -[A_{\text{mtq}} \mathbf{m}]_\times$$

Chaining with the rotation:

$$\frac{\partial B_{\text{body}}}{\partial \mathbf{q}} = \frac{\partial \mathbf{R}^T(\mathbf{q}) \mathbf{B}_{\text{eci}}}{\partial \mathbf{q}}$$

The derivative of the rotation matrix with respect to quaternion is:

$$\frac{\partial \mathbf{R}^T(\mathbf{q})}{\partial \mathbf{q}} : \text{yields } \frac{d}{d\mathbf{q}} \left[\mathbf{R}^T(\mathbf{q})\right] \mathbf{B}_{\text{eci}}$$

**Result (in practical form):**

$$\boxed{\frac{\partial \dot{\boldsymbol{\omega}}}{\partial \mathbf{q}} = -\mathbf{J}_{\text{com}}^{-1} [A_{\text{mtq}} \mathbf{m}]_\times \frac{\partial \mathbf{R}^T(\mathbf{q})}{\partial \mathbf{q}} \mathbf{B}_{\text{eci}}}$$

**Note:** The quaternion derivative of the rotation matrix is:

$$\frac{d\mathbf{R}^T(\mathbf{q})}{d\mathbf{q}}$$

is a $(9 \times 4)$ matrix computed row-wise. In code, use automatic differentiation or the explicit quaternion-to-rotation-matrix derivative formula.

---

### 1.3 Block: $\frac{\partial \dot{\boldsymbol{\omega}}}{\partial \mathbf{h}}$ (3×$N_{\text{rw}}$)

**Derivation:**

From the gyroscopic term:

$$\dot{\boldsymbol{\omega}} = \mathbf{J}_{\text{com}}^{-1} \left[ \cdots - \boldsymbol{\omega} \times \left( \mathbf{J}_{\text{com}} \boldsymbol{\omega} + A_{\text{rw}} \mathbf{h} \right) \right]$$

Taking the derivative with respect to $\mathbf{h}$:

$$\frac{\partial \dot{\boldsymbol{\omega}}}{\partial \mathbf{h}} = \mathbf{J}_{\text{com}}^{-1} \frac{\partial}{\partial \mathbf{h}} \left( -\boldsymbol{\omega} \times A_{\text{rw}} \mathbf{h} \right)$$

$$= -\mathbf{J}_{\text{com}}^{-1} [\boldsymbol{\omega}]_\times A_{\text{rw}}$$

**Result:**

$$\boxed{\frac{\partial \dot{\boldsymbol{\omega}}}{\partial \mathbf{h}} = -\mathbf{J}_{\text{com}}^{-1} [\boldsymbol{\omega}]_\times A_{\text{rw}}}$$

This is a $(3 \times N_{\text{rw}})$ matrix.

---

### 1.4 Block: $\frac{\partial \dot{\mathbf{q}}}{\partial \boldsymbol{\omega}}$ (4×3)

**Derivation:**

From quaternion kinematics:

$$\dot{\mathbf{q}} = \frac{1}{2} \mathbf{W}(\mathbf{q}) \boldsymbol{\omega}$$

where $\mathbf{W}(\mathbf{q})$ depends only on $\mathbf{q}$:

$$\frac{\partial \dot{\mathbf{q}}}{\partial \boldsymbol{\omega}} = \frac{1}{2} \mathbf{W}(\mathbf{q})$$

**Result:**

$$\boxed{\frac{\partial \dot{\mathbf{q}}}{\partial \boldsymbol{\omega}} = \frac{1}{2} \mathbf{W}(\mathbf{q})}$$

where

$$\mathbf{W}(\mathbf{q}) = \frac{1}{2} \begin{bmatrix}
-q_1 & -q_2 & -q_3 \\
 q_0 & -q_3 &  q_2 \\
 q_3 &  q_0 & -q_1 \\
-q_2 &  q_1 &  q_0
\end{bmatrix}$$

and $\mathbf{q} = [q_0, q_1, q_2, q_3]^T$ (scalar-first convention).

---

### 1.5 Block: $\frac{\partial \dot{\mathbf{q}}}{\partial \mathbf{q}}$ (4×4)

**Derivation:**

From:

$$\dot{\mathbf{q}} = \frac{1}{2} \mathbf{W}(\mathbf{q}) \boldsymbol{\omega}$$

Taking the derivative with respect to $\mathbf{q}$:

$$\frac{\partial \dot{\mathbf{q}}}{\partial \mathbf{q}} = \frac{1}{2} \frac{\partial \mathbf{W}(\mathbf{q})}{\partial \mathbf{q}} \boldsymbol{\omega}$$

The matrix $\mathbf{W}(\mathbf{q})$ is linear in the components of $\mathbf{q}$. Taking derivatives of each row with respect to each component:

For the first row of $\mathbf{W}(\mathbf{q})$: $[-q_1, -q_2, -q_3]$

$$\frac{\partial}{\partial \mathbf{q}} [-q_1, -q_2, -q_3] \boldsymbol{\omega} = -\begin{bmatrix} 0 & \omega_1 & \omega_2 & \omega_3 \\ \vdots & \vdots & \vdots & \vdots \end{bmatrix}$$

Computing all four rows systematically:

$$\frac{\partial \mathbf{W}(\mathbf{q})}{\partial \mathbf{q}} = \frac{1}{2} \begin{bmatrix}
0 & -\omega_1 & -\omega_2 & -\omega_3 \\
\omega_1 & 0 & \omega_3 & -\omega_2 \\
\omega_2 & -\omega_3 & 0 & \omega_1 \\
\omega_3 & \omega_2 & -\omega_1 & 0
\end{bmatrix}$$

This is the skew-symmetric matrix for the cross product in quaternion space:

**Result:**

$$\boxed{\frac{\partial \dot{\mathbf{q}}}{\partial \mathbf{q}} = \frac{1}{2} \begin{bmatrix}
0 & -\omega_1 & -\omega_2 & -\omega_3 \\
\omega_1 & 0 & \omega_3 & -\omega_2 \\
\omega_2 & -\omega_3 & 0 & \omega_1 \\
\omega_3 & \omega_2 & -\omega_1 & 0
\end{bmatrix}}$$

Alternatively, this can be written as:

$$\boxed{\frac{\partial \dot{\mathbf{q}}}{\partial \mathbf{q}} = \frac{1}{2} [\boldsymbol{\omega}]_\text{quat}}$$

where $[\boldsymbol{\omega}]_\text{quat}$ is the quaternion-space cross-product matrix.

---

### 1.6 Block: $\frac{\partial \dot{\mathbf{q}}}{\partial \mathbf{h}}$ (4×$N_{\text{rw}}$)

**Result:**

$$\boxed{\frac{\partial \dot{\mathbf{q}}}{\partial \mathbf{h}} = \mathbf{0}_{4 \times N_{\text{rw}}}}$$

Quaternion kinematics depend only on $\boldsymbol{\omega}$ and $\mathbf{q}$, not on $\mathbf{h}$.

---

### 1.7 Block: $\frac{\partial \dot{\mathbf{h}}}{\partial \boldsymbol{\omega}}$ ($N_{\text{rw}}$ ×3)

**Derivation:**

From the RW momentum equation:

$$\dot{\mathbf{h}} = -\mathbf{u}_{\text{rw}} - D_{\text{rw}} A_{\text{rw}}^T \dot{\boldsymbol{\omega}}$$

Taking the derivative with respect to $\boldsymbol{\omega}$:

$$\frac{\partial \dot{\mathbf{h}}}{\partial \boldsymbol{\omega}} = -D_{\text{rw}} A_{\text{rw}}^T \frac{\partial \dot{\boldsymbol{\omega}}}{\partial \boldsymbol{\omega}}$$

Substituting the result from Section 1.1:

$$\frac{\partial \dot{\boldsymbol{\omega}}}{\partial \boldsymbol{\omega}} = \mathbf{J}_{\text{com}}^{-1} \left( \left[\mathbf{J}_{\text{com}} \boldsymbol{\omega} + A_{\text{rw}} \mathbf{h}\right]_\times - [\boldsymbol{\omega}]_\times \mathbf{J}_{\text{com}} \right)$$

**Result:**

$$\boxed{\frac{\partial \dot{\mathbf{h}}}{\partial \boldsymbol{\omega}} = -D_{\text{rw}} A_{\text{rw}}^T \mathbf{J}_{\text{com}}^{-1} \left( \left[\mathbf{J}_{\text{com}} \boldsymbol{\omega} + A_{\text{rw}} \mathbf{h}\right]_\times - [\boldsymbol{\omega}]_\times \mathbf{J}_{\text{com}} \right)}$$

Simplifying:

$$\boxed{\frac{\partial \dot{\mathbf{h}}}{\partial \boldsymbol{\omega}} = D_{\text{rw}} A_{\text{rw}}^T \mathbf{J}_{\text{com}}^{-1} \left( \left[\mathbf{J}_{\text{com}} \boldsymbol{\omega} + A_{\text{rw}} \mathbf{h}\right]_\times + [\boldsymbol{\omega}]_\times \mathbf{J}_{\text{com}} \right)}$$

---

### 1.8 Block: $\frac{\partial \dot{\mathbf{h}}}{\partial \mathbf{q}}$ ($N_{\text{rw}}$ ×4)

**Derivation:**

From:

$$\frac{\partial \dot{\mathbf{h}}}{\partial \mathbf{q}} = -D_{\text{rw}} A_{\text{rw}}^T \frac{\partial \dot{\boldsymbol{\omega}}}{\partial \mathbf{q}}$$

Using the result from Section 1.2:

$$\frac{\partial \dot{\boldsymbol{\omega}}}{\partial \mathbf{q}} = -\mathbf{J}_{\text{com}}^{-1} [A_{\text{mtq}} \mathbf{m}]_\times \frac{\partial \mathbf{R}^T(\mathbf{q})}{\partial \mathbf{q}} \mathbf{B}_{\text{eci}}$$

**Result:**

$$\boxed{\frac{\partial \dot{\mathbf{h}}}{\partial \mathbf{q}} = D_{\text{rw}} A_{\text{rw}}^T \mathbf{J}_{\text{com}}^{-1} [A_{\text{mtq}} \mathbf{m}]_\times \frac{\partial \mathbf{R}^T(\mathbf{q})}{\partial \mathbf{q}} \mathbf{B}_{\text{eci}}}$$

---

### 1.9 Block: $\frac{\partial \dot{\mathbf{h}}}{\partial \mathbf{h}}$ ($N_{\text{rw}}$ ×$N_{\text{rw}}$)

**Derivation:**

From:

$$\dot{\mathbf{h}} = -\mathbf{u}_{\text{rw}} - D_{\text{rw}} A_{\text{rw}}^T \dot{\boldsymbol{\omega}}$$

Taking the derivative with respect to $\mathbf{h}$:

$$\frac{\partial \dot{\mathbf{h}}}{\partial \mathbf{h}} = -D_{\text{rw}} A_{\text{rw}}^T \frac{\partial \dot{\boldsymbol{\omega}}}{\partial \mathbf{h}}$$

Using the result from Section 1.3:

$$\frac{\partial \dot{\boldsymbol{\omega}}}{\partial \mathbf{h}} = -\mathbf{J}_{\text{com}}^{-1} [\boldsymbol{\omega}]_\times A_{\text{rw}}$$

**Result:**

$$\boxed{\frac{\partial \dot{\mathbf{h}}}{\partial \mathbf{h}} = D_{\text{rw}} A_{\text{rw}}^T \mathbf{J}_{\text{com}}^{-1} [\boldsymbol{\omega}]_\times A_{\text{rw}}}$$

This is ($N_{\text{rw}}$ × $N_{\text{rw}}$) and typically small (RW count is usually 3–4).

---

## Part 2: Jacobian $\frac{\partial \dot{\mathbf{x}}}{\partial \mathbf{u}} = \frac{\partial \mathbf{f}}{\partial \mathbf{u}}$

The input Jacobian is a block matrix:

$$
\frac{\partial \mathbf{f}}{\partial \mathbf{u}} = \begin{bmatrix}
\frac{\partial \dot{\boldsymbol{\omega}}}{\partial \mathbf{m}} & \frac{\partial \dot{\boldsymbol{\omega}}}{\partial \mathbf{u}_{\text{rw}}} \\[6pt]
\frac{\partial \dot{\mathbf{q}}}{\partial \mathbf{m}} & \frac{\partial \dot{\mathbf{q}}}{\partial \mathbf{u}_{\text{rw}}} \\[6pt]
\frac{\partial \dot{\mathbf{h}}}{\partial \mathbf{m}} & \frac{\partial \dot{\mathbf{h}}}{\partial \mathbf{u}_{\text{rw}}}
\end{bmatrix}$$

where $\mathbf{u} = [\mathbf{m}, \mathbf{u}_{\text{rw}}]^T$ with $\mathbf{m} \in \mathbb{R}^{N_{\text{mtq}}}$ and $\mathbf{u}_{\text{rw}} \in \mathbb{R}^{N_{\text{rw}}}$.

### 2.1 Block: $\frac{\partial \dot{\boldsymbol{\omega}}}{\partial \mathbf{m}}$ (3×$N_{\text{mtq}}$)

**Derivation:**

From the magnetorquer torque:

$$\boldsymbol{\tau}_{\text{mtq}} = -[B_{\text{body}}]_\times A_{\text{mtq}} \mathbf{m}$$

The angular acceleration depends on this linearly:

$$\dot{\boldsymbol{\omega}} = \mathbf{J}_{\text{com}}^{-1} \left[ -[B_{\text{body}}]_\times A_{\text{mtq}} \mathbf{m} + \cdots \right]$$

Therefore:

$$\frac{\partial \dot{\boldsymbol{\omega}}}{\partial \mathbf{m}} = -\mathbf{J}_{\text{com}}^{-1} [B_{\text{body}}]_\times A_{\text{mtq}}$$

**Result:**

$$\boxed{\frac{\partial \dot{\boldsymbol{\omega}}}{\partial \mathbf{m}} = -\mathbf{J}_{\text{com}}^{-1} [B_{\text{body}}]_\times A_{\text{mtq}}}$$

where $B_{\text{body}} = \mathbf{R}^T(\mathbf{q}) \mathbf{B}_{\text{eci}}$ is evaluated at the current state.

---

### 2.2 Block: $\frac{\partial \dot{\boldsymbol{\omega}}}{\partial \mathbf{u}_{\text{rw}}}$ (3×$N_{\text{rw}}$)

**Derivation:**

From the reaction wheel torque:

$$A_{\text{rw}} \mathbf{u}_{\text{rw}}$$

appears linearly in the $\dot{\boldsymbol{\omega}}$ equation:

$$\dot{\boldsymbol{\omega}} = \mathbf{J}_{\text{com}}^{-1} \left[ \cdots + A_{\text{rw}} \mathbf{u}_{\text{rw}} + \cdots \right]$$

**Result:**

$$\boxed{\frac{\partial \dot{\boldsymbol{\omega}}}{\partial \mathbf{u}_{\text{rw}}} = \mathbf{J}_{\text{com}}^{-1} A_{\text{rw}}}$$

This is a $(3 \times N_{\text{rw}})$ matrix.

---

### 2.3 Block: $\frac{\partial \dot{\mathbf{q}}}{\partial \mathbf{m}}$ (4×$N_{\text{mtq}}$)

**Result:**

$$\boxed{\frac{\partial \dot{\mathbf{q}}}{\partial \mathbf{m}} = \mathbf{0}_{4 \times N_{\text{mtq}}}}$$

Quaternion kinematics do not depend on control inputs directly, only on $\boldsymbol{\omega}$.

---

### 2.4 Block: $\frac{\partial \dot{\mathbf{q}}}{\partial \mathbf{u}_{\text{rw}}}$ (4×$N_{\text{rw}}$)

**Result:**

$$\boxed{\frac{\partial \dot{\mathbf{q}}}{\partial \mathbf{u}_{\text{rw}}} = \mathbf{0}_{4 \times N_{\text{rw}}}}$$

Same reasoning as Section 2.3.

---

### 2.5 Block: $\frac{\partial \dot{\mathbf{h}}}{\partial \mathbf{m}}$ ($N_{\text{rw}}$ ×$N_{\text{mtq}}$)

**Derivation:**

From:

$$\dot{\mathbf{h}} = -\mathbf{u}_{\text{rw}} - D_{\text{rw}} A_{\text{rw}}^T \dot{\boldsymbol{\omega}}$$

Taking the derivative with respect to $\mathbf{m}$:

$$\frac{\partial \dot{\mathbf{h}}}{\partial \mathbf{m}} = -D_{\text{rw}} A_{\text{rw}}^T \frac{\partial \dot{\boldsymbol{\omega}}}{\partial \mathbf{m}}$$

Using the result from Section 2.1:

$$\frac{\partial \dot{\boldsymbol{\omega}}}{\partial \mathbf{m}} = -\mathbf{J}_{\text{com}}^{-1} [B_{\text{body}}]_\times A_{\text{mtq}}$$

**Result:**

$$\boxed{\frac{\partial \dot{\mathbf{h}}}{\partial \mathbf{m}} = D_{\text{rw}} A_{\text{rw}}^T \mathbf{J}_{\text{com}}^{-1} [B_{\text{body}}]_\times A_{\text{mtq}}}$$

---

### 2.6 Block: $\frac{\partial \dot{\mathbf{h}}}{\partial \mathbf{u}_{\text{rw}}}$ ($N_{\text{rw}}$ ×$N_{\text{rw}}$)

**Derivation:**

From:

$$\dot{\mathbf{h}} = -\mathbf{u}_{\text{rw}} - D_{\text{rw}} A_{\text{rw}}^T \dot{\boldsymbol{\omega}}$$

Taking the derivative with respect to $\mathbf{u}_{\text{rw}}$:

$$\frac{\partial \dot{\mathbf{h}}}{\partial \mathbf{u}_{\text{rw}}} = -I_{N_{\text{rw}}} - D_{\text{rw}} A_{\text{rw}}^T \frac{\partial \dot{\boldsymbol{\omega}}}{\partial \mathbf{u}_{\text{rw}}}$$

Using the result from Section 2.2:

$$\frac{\partial \dot{\boldsymbol{\omega}}}{\partial \mathbf{u}_{\text{rw}}} = \mathbf{J}_{\text{com}}^{-1} A_{\text{rw}}$$

**Result:**

$$\boxed{\frac{\partial \dot{\mathbf{h}}}{\partial \mathbf{u}_{\text{rw}}} = -I_{N_{\text{rw}}} - D_{\text{rw}} A_{\text{rw}}^T \mathbf{J}_{\text{com}}^{-1} A_{\text{rw}}}$$

or equivalently:

$$\boxed{\frac{\partial \dot{\mathbf{h}}}{\partial \mathbf{u}_{\text{rw}}} = -(I_{N_{\text{rw}}} + D_{\text{rw}} A_{\text{rw}}^T \mathbf{J}_{\text{com}}^{-1} A_{\text{rw}})}$$

---

## Part 3: Jacobian with respect to Disturbance Torque

The disturbance torque $\boldsymbol{\tau}_{\text{dist}}$ appears as an external additive input to the $\dot{\boldsymbol{\omega}}$ equation:

$$\dot{\boldsymbol{\omega}} = \mathbf{J}_{\text{com}}^{-1} \left[ \cdots + \boldsymbol{\tau}_{\text{dist}} - \cdots \right]$$

### 3.1 Block: $\frac{\partial \dot{\boldsymbol{\omega}}}{\partial \boldsymbol{\tau}_{\text{dist}}}$ (3×3)

**Derivation:**

The disturbance torque appears linearly:

$$\frac{\partial \dot{\boldsymbol{\omega}}}{\partial \boldsymbol{\tau}_{\text{dist}}} = \mathbf{J}_{\text{com}}^{-1}$$

**Result:**

$$\boxed{\frac{\partial \dot{\boldsymbol{\omega}}}{\partial \boldsymbol{\tau}_{\text{dist}}} = \mathbf{J}_{\text{com}}^{-1}}$$

---

### 3.2 Block: $\frac{\partial \dot{\mathbf{q}}}{\partial \boldsymbol{\tau}_{\text{dist}}}$ (4×3)

**Result:**

$$\boxed{\frac{\partial \dot{\mathbf{q}}}{\partial \boldsymbol{\tau}_{\text{dist}}} = \mathbf{0}_{4 \times 3}}$$

Quaternion kinematics do not depend on disturbance torques.

---

### 3.3 Block: $\frac{\partial \dot{\mathbf{h}}}{\partial \boldsymbol{\tau}_{\text{dist}}}$ ($N_{\text{rw}}$ ×3)

**Derivation:**

From:

$$\dot{\mathbf{h}} = -\mathbf{u}_{\text{rw}} - D_{\text{rw}} A_{\text{rw}}^T \dot{\boldsymbol{\omega}}$$

Taking the derivative with respect to $\boldsymbol{\tau}_{\text{dist}}$:

$$\frac{\partial \dot{\mathbf{h}}}{\partial \boldsymbol{\tau}_{\text{dist}}} = -D_{\text{rw}} A_{\text{rw}}^T \frac{\partial \dot{\boldsymbol{\omega}}}{\partial \boldsymbol{\tau}_{\text{dist}}}$$

Using the result from Section 3.1:

**Result:**

$$\boxed{\frac{\partial \dot{\mathbf{h}}}{\partial \boldsymbol{\tau}_{\text{dist}}} = -D_{\text{rw}} A_{\text{rw}}^T \mathbf{J}_{\text{com}}^{-1}}$$

---

## Summary Table

| Block | Dimensions | Expression |
|-------|------------|-----------|
| $\frac{\partial \dot{\boldsymbol{\omega}}}{\partial \boldsymbol{\omega}}$ | 3×3 | $\mathbf{J}_{\text{com}}^{-1} \left( -\left[\mathbf{J}_{\text{com}} \boldsymbol{\omega} + A_{\text{rw}} \mathbf{h}\right]_\times - [\boldsymbol{\omega}]_\times \mathbf{J}_{\text{com}} \right)$ |
| $\frac{\partial \dot{\boldsymbol{\omega}}}{\partial \mathbf{q}}$ | 3×4 | $-\mathbf{J}_{\text{com}}^{-1} [A_{\text{mtq}} \mathbf{m}]_\times \frac{\partial \mathbf{R}^T}{\partial \mathbf{q}} \mathbf{B}_{\text{eci}}$ |
| $\frac{\partial \dot{\boldsymbol{\omega}}}{\partial \mathbf{h}}$ | 3×$N_{\text{rw}}$ | $-\mathbf{J}_{\text{com}}^{-1} [\boldsymbol{\omega}]_\times A_{\text{rw}}$ |
| $\frac{\partial \dot{\mathbf{q}}}{\partial \boldsymbol{\omega}}$ | 4×3 | $\frac{1}{2} \mathbf{W}(\mathbf{q})$ |
| $\frac{\partial \dot{\mathbf{q}}}{\partial \mathbf{q}}$ | 4×4 | $\frac{1}{2} \begin{bmatrix} 0 & -\omega_1 & -\omega_2 & -\omega_3 \\ \omega_1 & 0 & \omega_3 & -\omega_2 \\ \omega_2 & -\omega_3 & 0 & \omega_1 \\ \omega_3 & \omega_2 & -\omega_1 & 0 \end{bmatrix}$ |
| $\frac{\partial \dot{\mathbf{q}}}{\partial \mathbf{h}}$ | 4×$N_{\text{rw}}$ | $\mathbf{0}$ |
| $\frac{\partial \dot{\mathbf{h}}}{\partial \boldsymbol{\omega}}$ | $N_{\text{rw}}$×3 | $D_{\text{rw}} A_{\text{rw}}^T \mathbf{J}_{\text{com}}^{-1} \left( \left[\mathbf{J}_{\text{com}} \boldsymbol{\omega} + A_{\text{rw}} \mathbf{h}\right]_\times + [\boldsymbol{\omega}]_\times \mathbf{J}_{\text{com}} \right)$ |
| $\frac{\partial \dot{\mathbf{h}}}{\partial \mathbf{q}}$ | $N_{\text{rw}}$×4 | $D_{\text{rw}} A_{\text{rw}}^T \mathbf{J}_{\text{com}}^{-1} [A_{\text{mtq}} \mathbf{m}]_\times \frac{\partial \mathbf{R}^T}{\partial \mathbf{q}} \mathbf{B}_{\text{eci}}$ |
| $\frac{\partial \dot{\mathbf{h}}}{\partial \mathbf{h}}$ | $N_{\text{rw}}$×$N_{\text{rw}}$ | $D_{\text{rw}} A_{\text{rw}}^T \mathbf{J}_{\text{com}}^{-1} [\boldsymbol{\omega}]_\times A_{\text{rw}}$ |
| $\frac{\partial \dot{\boldsymbol{\omega}}}{\partial \mathbf{m}}$ | 3×$N_{\text{mtq}}$ | $-\mathbf{J}_{\text{com}}^{-1} [B_{\text{body}}]_\times A_{\text{mtq}}$ |
| $\frac{\partial \dot{\boldsymbol{\omega}}}{\partial \mathbf{u}_{\text{rw}}}$ | 3×$N_{\text{rw}}$ | $\mathbf{J}_{\text{com}}^{-1} A_{\text{rw}}$ |
| $\frac{\partial \dot{\mathbf{q}}}{\partial \mathbf{m}}$ | 4×$N_{\text{mtq}}$ | $\mathbf{0}$ |
| $\frac{\partial \dot{\mathbf{q}}}{\partial \mathbf{u}_{\text{rw}}}$ | 4×$N_{\text{rw}}$ | $\mathbf{0}$ |
| $\frac{\partial \dot{\mathbf{h}}}{\partial \mathbf{m}}$ | $N_{\text{rw}}$×$N_{\text{mtq}}$ | $D_{\text{rw}} A_{\text{rw}}^T \mathbf{J}_{\text{com}}^{-1} [B_{\text{body}}]_\times A_{\text{mtq}}$ |
| $\frac{\partial \dot{\mathbf{h}}}{\partial \mathbf{u}_{\text{rw}}}$ | $N_{\text{rw}}$×$N_{\text{rw}}$ | $-(I_{N_{\text{rw}}} + D_{\text{rw}} A_{\text{rw}}^T \mathbf{J}_{\text{com}}^{-1} A_{\text{rw}})$ |
| $\frac{\partial \dot{\boldsymbol{\omega}}}{\partial \boldsymbol{\tau}_{\text{dist}}}$ | 3×3 | $\mathbf{J}_{\text{com}}^{-1}$ |
| $\frac{\partial \dot{\mathbf{q}}}{\partial \boldsymbol{\tau}_{\text{dist}}}$ | 4×3 | $\mathbf{0}$ |
| $\frac{\partial \dot{\mathbf{h}}}{\partial \boldsymbol{\tau}_{\text{dist}}}$ | $N_{\text{rw}}$×3 | $-D_{\text{rw}} A_{\text{rw}}^T \mathbf{J}_{\text{com}}^{-1}$ |

---

## Notes on Implementation

### Notation: Skew-Symmetric Matrix $[v]_\times$

The skew-symmetric (or "cross-product") matrix for vector $\mathbf{v} = [v_1, v_2, v_3]^T$ is:

$$[v]_\times = \begin{bmatrix}
0 & -v_3 & v_2 \\
v_3 & 0 & -v_1 \\
-v_2 & v_1 & 0
\end{bmatrix}$$

with the property: $[v]_\times w = v \times w$.

Derivatives of cross products:
- $\frac{\partial (v \times w)}{\partial v} = -[w]_\times$
- $\frac{\partial (v \times w)}{\partial w} = [v]_\times$

### Numerical Stability

1. **Skew-matrix operations:** Use `skew()` helper functions in code rather than explicitly forming $[v]_\times$.

2. **Matrix inversions:** Pre-compute $\mathbf{J}_{\text{com}}^{-1}$ once per time step, not for each Jacobian block.

3. **Vector norms:** Always check $\|v\| > \epsilon$ before computing $v / \|v\|$.

4. **Quaternion normalization:** Ensure $\|\mathbf{q}\| = 1$ throughout, as perturbations in quaternion space can violate the unit-norm constraint.

### Quaternion Derivative $\frac{\partial \mathbf{R}^T(\mathbf{q})}{\partial \mathbf{q}}$

For the scalar-first convention $\mathbf{q} = [q_0, q_1, q_2, q_3]^T$, the rotation matrix is:

$$\mathbf{R}(\mathbf{q}) = \begin{bmatrix}
1 - 2(q_2^2 + q_3^2) & 2(q_1 q_2 - q_0 q_3) & 2(q_1 q_3 + q_0 q_2) \\
2(q_1 q_2 + q_0 q_3) & 1 - 2(q_1^2 + q_3^2) & 2(q_2 q_3 - q_0 q_1) \\
2(q_1 q_3 - q_0 q_2) & 2(q_2 q_3 + q_0 q_1) & 1 - 2(q_1^2 + q_2^2)
\end{bmatrix}$$

The transpose derivative is computed element-wise using the chain rule:

$$\frac{\partial R_{ij}}{\partial q_k}, \quad i,j \in \{0,1,2\}, \quad k \in \{0,1,2,3\}$$

In code, use automatic differentiation (e.g., Eigen's autodiff module, PyTorch, JAX, or CasADi) to avoid manual expression of this $(9 \times 4)$ Jacobian.

### Automatic Differentiation in Code

For most implementations, use AD tools instead of hand-coding:
- **C++:** Eigen autodiff, CasADi, or custom tape-based AD
- **Python:** JAX, PyTorch, autograd, or sympy.diff
- **MATLAB:** symbolic derivatives or built-in jacobian()

This avoids transcription errors and ensures consistency with the actual dynamics code.
