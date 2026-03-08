# Satellite Attitude Dynamics

## State, Inputs, and Actuator Configuration

**State:**
$$\mathbf{x} = \begin{bmatrix} \boldsymbol{\omega} \\ \mathbf{q} \\ \mathbf{h} \end{bmatrix} \in \mathbb{R}^{3 + 4 + N_{\text{rw}}}$$
where $\boldsymbol{\omega}$ = angular velocity, $\mathbf{q}$ = unit quaternion, $\mathbf{h}$ = RW momenta.

**Control:**
$$\mathbf{u} = \begin{bmatrix} \mathbf{m} \\ \boldsymbol{\tau}_{\text{rw}} \end{bmatrix} \in \mathbb{R}^{N_{\text{mtq}} + N_{\text{rw}}}$$
where $\mathbf{m}$ = MTQ dipole moments, $\boldsymbol{\tau}_{\text{rw}}$ = RW torque commands.

**Actuator Axes:**
$$A_{\text{mtq}} = [\mathbf{a}_1^{\text{mtq}} \; \cdots \; \mathbf{a}_{N_{\text{mtq}}}^{\text{mtq}}] \in \mathbb{R}^{3 \times N_{\text{mtq}}}$$
$$A_{\text{rw}} = [\mathbf{a}_1^{\text{rw}} \; \cdots \; \mathbf{a}_{N_{\text{rw}}}^{\text{rw}}] \in \mathbb{R}^{3 \times N_{\text{rw}}}$$

**Environment:**
$\mathbf{R}_{\text{eci}}, \mathbf{B}_{\text{eci}}, \mathbf{S}_{\text{eci}}, \mathbf{V}_{\text{eci}}, \rho$ = position, magnetic field, sun direction, velocity (ECI), atmospheric density.

---

## Full Dynamics Equations

$$
\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, \mathbf{u}, \mathbf{R}_{\text{eci}}, \mathbf{B}_{\text{eci}}, \mathbf{S}_{\text{eci}}, \mathbf{V}_{\text{eci}}, \rho)
$$

### 1. Angular Velocity Rate (Euler's Rotational Equation)

$$
\dot{\boldsymbol{\omega}} = \mathbf{J}_{\text{com}}^{-1} \left[ \boldsymbol{\tau}_{\text{act}} + \boldsymbol{\tau}_{\text{dist}} - \boldsymbol{\omega} \times \left( \mathbf{J}_{\text{com}} \boldsymbol{\omega} + \mathbf{h}_{\text{rw,total}} \right) \right]
$$

where:
- $\mathbf{J}_{\text{com}}$ = spacecraft inertia matrix (symmetric, positive-definite)
- $\mathbf{J}_{\text{com}}^{-1}$ = inverse inertia (precomputed as `invJcom_noRW_` excluding RW inertias)
- $\boldsymbol{\tau}_{\text{act}}$ = total actuator torque (MTQ + RW)
- $\boldsymbol{\tau}_{\text{dist}}$ = total disturbance torque (GG + drag + SRP)
 - $\mathbf{h}_{\text{rw,total}} = A_{\text{rw}} \mathbf{h}$ = total RW angular momentum vector (columns of $A_{\text{rw}}$ are RW spin axes)
 - $\mathbf{a}_i$ = i-th reaction wheel spin axis (column $i$ of $A_{\text{rw}}$)
- $\times$ = cross product operator
- $\boldsymbol{\omega} \times (\mathbf{J}_{\text{com}} \boldsymbol{\omega} + \mathbf{h}_{\text{rw,total}})$ = gyroscopic coupling term

#### 1.1 Actuator Torque

Vectorized actuator torque (MTQ + RW):

$$
\boldsymbol{\tau}_{\text{act}} = \boldsymbol{\tau}_{\text{mtq}} + \boldsymbol{\tau}_{\text{rw}}
= -\operatorname{skew}(\mathbf{B}_{\text{body}})\,A_{\text{mtq}}\,\mathbf{m} + A_{\text{rw}}\,\mathbf{u}_{\text{rw}}
$$

where $\mathbf{B}_{\text{body}} = \mathbf{R}^T(\mathbf{q})\,\mathbf{B}_{\text{eci}}$ and $\operatorname{skew}(\mathbf{v})$ is the skew-symmetric matrix for cross products ($\mathbf{v}\times\mathbf{w}=\operatorname{skew}(\mathbf{v})\,\mathbf{w}$).

- Magnetorquer (vector form):

$$
\boldsymbol{\tau}_{\text{mtq}} = -\operatorname{skew}(\mathbf{B}_{\text{body}})\,A_{\text{mtq}}\,\mathbf{m}
$$

This is equivalent to $-\sum_i (\mathbf{B}_{\text{body}} \times \mathbf{a}^{\text{mtq}}_i) m_i$.

- Reaction wheels (vector form):

$$
\boldsymbol{\tau}_{\text{rw}} = A_{\text{rw}}\,\mathbf{u}_{\text{rw}}
$$

Here $A_{\text{mtq}}$ and $A_{\text{rw}}$ collect actuator axes as column vectors.

#### 1.2 Disturbance Torque

$$
\boldsymbol{\tau}_{\text{dist}} = \boldsymbol{\tau}_{\text{gg}} + \boldsymbol{\tau}_{\text{drag}} + \boldsymbol{\tau}_{\text{srp}}
$$

**Gravity Gradient Torque:**

$$
\boldsymbol{\tau}_{\text{gg}} = \frac{3\mu}{r^5} (\mathbf{r}_{\text{body}} \times \mathbf{J}_{\text{com}} \mathbf{r}_{\text{body}})
$$

where:
- $\mu$ = gravitational parameter of central body
- $\mathbf{r}_{\text{body}} = \mathbf{R}^T(\mathbf{q}) \mathbf{R}_{\text{eci}}$ = position vector in body frame
- $r = \|\mathbf{R}_{\text{eci}}\|$ = orbital radius
- Assumption: only second-order gravity gradient term (quadrupole effect)

**Aerodynamic Drag Torque:**

$$
\boldsymbol{\tau}_{\text{drag}} = -\frac{1}{2} \rho C_D A_{\text{ref}} \|\mathbf{V}_{\text{body}}\|^2 \left( \frac{\mathbf{V}_{\text{body}}}{\|\mathbf{V}_{\text{body}}\|} \times \mathbf{r}_{\text{cp}} \right)
$$

where:
- $\rho$ = atmospheric density
- $C_D$ = drag coefficient (configurable)
- $A_{\text{ref}}$ = reference area (configurable)
- $\mathbf{V}_{\text{body}} = \mathbf{R}^T(\mathbf{q}) \mathbf{V}_{\text{eci}}$ = velocity in body frame
- $\mathbf{r}_{\text{cp}}$ = center of pressure offset from center of mass (configurable)
- Assumes velocity direction dominates drag force direction

**Solar Radiation Pressure Torque:**

$$
\boldsymbol{\tau}_{\text{srp}} = -P_{\text{srp}} A_{\text{ref}} (1 + \epsilon) \left( \frac{\mathbf{S}_{\text{body}}}{\|\mathbf{S}_{\text{body}}\|} \times \mathbf{r}_{\text{cp}} \right) \cdot f_{\text{eclipse}}
$$

where:
- $P_{\text{srp}}$ = solar radiation pressure at 1 AU (constant: $\approx 4.56 \times 10^{-6}$ N/m²)
- $A_{\text{ref}}$ = reference area (same as drag)
- $\epsilon$ = surface reflectivity (0 = absorbing, 1 = perfectly reflecting)
- $\mathbf{S}_{\text{body}} = \mathbf{R}^T(\mathbf{q}) \mathbf{S}_{\text{eci}}$ = sun direction in body frame
- $f_{\text{eclipse}} \in \{0, 1\}$ = eclipse shadow function (0 when in Earth's shadow)
- Assumes specular reflection with single reflection coefficient

---

### 2. Quaternion Kinematics

$$
\dot{\mathbf{q}} = \frac{1}{2} \mathbf{W}(\mathbf{q}) \boldsymbol{\omega}
$$

where $\mathbf{W}(\mathbf{q})$ is the quaternion kinematics matrix:

$$
\mathbf{W}(\mathbf{q}) = \frac{1}{2} \begin{bmatrix}
-q_1 & -q_2 & -q_3 \\
 q_0 & -q_3 &  q_2 \\
 q_3 &  q_0 & -q_1 \\
-q_2 &  q_1 &  q_0
\end{bmatrix}
$$

Convention: $\mathbf{q} = \begin{bmatrix} q_0 & q_1 & q_2 & q_3 \end{bmatrix}^T$ where $q_0$ is the scalar part.

Constraint: $\|\mathbf{q}\| = 1$ (maintained by internal normalization in `dynamics()`)

---

### 3. Reaction Wheel Momentum Dynamics

$$
\dot{\mathbf{h}} = -\mathbf{u}_{\text{rw}} - D_{\text{rw}} A_{\text{rw}}^T \dot{\boldsymbol{\omega}}
$$

**Component-wise / matrix form:**

$$
\dot{\mathbf{h}} = -\mathbf{u}_{\text{rw}} - D_{\text{rw}}\,A_{\text{rw}}^T\,\dot{\boldsymbol{\omega}},
$$

or component-wise

$$
\dot{h}_i = -u_{\text{rw},i} - J_i\,\mathbf{a}_i^{\,T}\,\dot{\boldsymbol{\omega}},\quad i=1,\dots,N_{\text{rw}},
$$

with $D_{\text{rw}}=\mathrm{diag}(J_i)$ and $\mathbf{a}_i$ the columns of $A_{\text{rw}}$.

**Physical Interpretation:**
- **First term** ($-u_i$): Direct control torque reduces RW angular momentum (Newton's 3rd law)
- **Second term** ($-J_i \mathbf{a}_i^T \dot{\boldsymbol{\omega}}$): **RW inertial coupling** — spacecraft angular acceleration applies torque back to RW, changing stored momentum

**Constraints:**
$$
|h_i| \leq h_{\text{max}, i} \quad \text{(saturation)}
$$

**Implementation Note:** This coupling is **essential** for correct dynamics. When the spacecraft rotates, the RW must absorb inertial torque, which changes its stored angular momentum even without control input.

---

## Summary in Compact Form

$$
\boxed{\begin{aligned}
\dot{\boldsymbol{\omega}} &= \mathbf{J}_{\text{com}}^{-1}\Big[ -\operatorname{skew}(\mathbf{R}^T(\mathbf{q})\,\mathbf{B}_{\text{eci}})\,A_{\text{mtq}}\,\mathbf{m} + A_{\text{rw}}\,\mathbf{u}_{\text{rw}} \\
&\quad + \dfrac{3\mu}{r^5}\big(\mathbf{R}^T(\mathbf{q})\mathbf{R}_{\text{eci}}\big)\times\big(\mathbf{J}_{\text{com}}\,\mathbf{R}^T(\mathbf{q})\mathbf{R}_{\text{eci}}\big) \\
&\quad - \dfrac{1}{2}\rho C_D A_{\text{ref}} \|\mathbf{R}^T(\mathbf{q})\mathbf{V}_{\text{eci}}\|^2 \left( \dfrac{\mathbf{R}^T(\mathbf{q})\mathbf{V}_{\text{eci}}}{\|\mathbf{R}^T(\mathbf{q})\mathbf{V}_{\text{eci}}\|} \times \mathbf{r}_{\text{cp}} \right) \\
&\quad - P_{\text{srp}} A_{\text{ref}} (1 + \epsilon) \left( \dfrac{\mathbf{R}^T(\mathbf{q})\mathbf{S}_{\text{eci}}}{\|\mathbf{R}^T(\mathbf{q})\mathbf{S}_{\text{eci}}\|} \times \mathbf{r}_{\text{cp}} \right) f_{\text{eclipse}} \\
&\quad - \boldsymbol{\omega} \times \big( \mathbf{J}_{\text{com}}\,\boldsymbol{\omega} + A_{\text{rw}}\,\mathbf{h} \big) \Big] \\
\\
\dot{\mathbf{q}} &= \tfrac{1}{2}\,\mathbf{W}(\mathbf{q})\,\boldsymbol{\omega} \\
\\
\dot{\mathbf{h}} &= -\mathbf{u}_{\text{rw}} - D_{\text{rw}}\,A_{\text{rw}}^T\,\dot{\boldsymbol{\omega}}
\end{aligned}}
$$


---

## Implementation Notes

1. **Body Frame:** All vectors internally computed in spacecraft body-fixed frame via rotation matrix $\mathbf{R}(\mathbf{q})$.

2. **Cross Product:** In code, `a.cross(b)` or `skew(a) * b` implements the cross product.

3. **Quaternion Normalization:** The `dynamics()` function normalizes $\mathbf{q}$ at the start to ensure unit magnitude.

4. **RW Momentum Coupling:** The gyroscopic term $\boldsymbol{\omega} \times \mathbf{h}_{\text{rw,total}}$ couples RW momentum back into angular acceleration.

5. **Flight-Safe Disturbances:** GG, drag, and SRP computations include:
   - Norm checks before dividing (prevent NaN)
   - Try-catch blocks (graceful failure)
   - Configurable enable/disable flags

6. **Constants:**
   - $\mu = 3.986 \times 10^{14}$ m³/s² (Earth)
   - $P_{\text{srp}} = 4.56 \times 10^{-6}$ N/m² (solar constant effect)
