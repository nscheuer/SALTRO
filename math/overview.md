# Satellite Equations of Motion - Jacobian and Hessian Reference

## Overview
This document catalogs all Jacobian and Hessian computations used in the SALTRO satellite trajectory optimization system.

---

## Table of Contents
1. [Dynamics Jacobians & Hessians](#dynamics-jacobians--hessians)
2. [Cost Jacobians & Hessians](#cost-jacobians--hessians)
3. [Constraint Jacobians & Hessians](#constraint-jacobians--hessians)
4. [Actuator Jacobians & Hessians](#actuator-jacobians--hessians)
5. [Disturbance Jacobians & Hessians](#disturbance-jacobians--hessians)
6. [Mathematical Utility Functions](#mathematical-utility-functions)

---

## Dynamics Jacobians & Hessians

### 1. `dynamicsJacobians()`
**Location**: `src/pybind/satellite.cpp:301-576`

**Purpose**: Compute first-order derivatives of satellite dynamics with respect to state and control.

**Equations of Motion**:
```
ẋ = f(x, u, dist)
where:
  x = [w, q, h_rw]  (angular velocity, quaternion, RW momenta)
  u = [u_mtq, u_rw]  (MTQ dipoles, RW torques)
```

**Jacobians Computed**:

| Derivative | Symbol | Mathematical Definition | Dimensions | Notes |
|------------|--------|------------------------|------------|-------|
| State Jacobian | `jac_x` | ∂f/∂x = [∂ẋ/∂x] | (nx × nx) | Main state dynamics Jacobian |
| Control Jacobian | `jac_u` | ∂f/∂u = [∂ẋ/∂u] | (nx × nu) | Control input Jacobian |
| Disturbance Jacobian | `jac_dist` | ∂f/∂τ_dist | (nx × 3) | Disturbance torque sensitivity (unused) |

**Components of State Jacobian `jac_x`**:

| Block | Symbol | Derivative | Dimensions | Description |
|-------|--------|-----------|------------|-------------|
| (0:3, 0:3) | ∂ẇ/∂w | d/dw[J⁻¹(τ - w × (Jw + h_rw))] | (3 × 3) | Angular acceleration w.r.t. angular velocity |
| (0:3, 3:7) | ∂ẇ/∂q | d/dq[J⁻¹(τ_act + τ_dist)] | (3 × 4) | Angular acceleration w.r.t. quaternion (rotation coupling) |
| (0:3, 7:7+nrw) | ∂ẇ/∂h | d/dh[J⁻¹(-w × h_rw)] | (3 × nrw) | Angular acceleration w.r.t. RW momentum |
| (3:7, 0:3) | ∂q̇/∂w | 0.5 · W(q) | (4 × 3) | Quaternion rate w.r.t. angular velocity |
| (3:7, 3:7) | ∂q̇/∂q | 0.5 · ∂W/∂q · w | (4 × 4) | Quaternion rate w.r.t. quaternion (kinematic) |
| (7:7+nrw, 0:3) | ∂ḣ/∂w | -J_rw · axis · ∂ẇ/∂w | (nrw × 3) | RW momentum rate w.r.t. angular velocity |
| (7:7+nrw, 3:7) | ∂ḣ/∂q | -J_rw · axis · ∂ẇ/∂q | (nrw × 4) | RW momentum rate w.r.t. quaternion |
| (7:7+nrw, 7:7+nrw) | ∂ḣ/∂h | -J_rw · axis · ∂ẇ/∂h | (nrw × nrw) | RW momentum rate w.r.t. RW momentum |

**Components of Control Jacobian `jac_u`**:

| Block | Symbol | Derivative | Dimensions | Description |
|-------|--------|-----------|------------|-------------|
| (0:3, 0:nmtq) | ∂ẇ/∂u_mtq | J⁻¹ · ∂τ_mtq/∂u_mtq | (3 × nmtq) | Angular acceleration w.r.t. MTQ control |
| (0:3, nmtq:nu) | ∂ẇ/∂u_rw | J⁻¹ · axis_i | (3 × nrw) | Angular acceleration w.r.t. RW control |
| (3:7, :) | ∂q̇/∂u | 0 | (4 × nu) | Quaternion rate independent of control |
| (7:7+nrw, 0:nmtq) | ∂ḣ/∂u_mtq | -J_rw · axis · ∂ẇ/∂u_mtq | (nrw × nmtq) | RW momentum rate w.r.t. MTQ control (coupling) |
| (7:7+nrw, nmtq:nu) | ∂ḣ/∂u_rw | -δ_ij - J_rw · axis_i · ∂ẇ/∂u_rw,j | (nrw × nrw) | RW momentum rate w.r.t. RW control |

**Special Handling**:
- Quaternion normalization projection: `jac_x[:, 3:7] *= (I - q·q^T)` applied at end
- MTQ torque depends on magnetic field: τ_mtq = -(B_body × axis) · u
- RW torque is direct: τ_rw = axis · u
- Gyroscopic coupling: -w × (Jw + h_rw) introduces nonlinear w-dependence

---

### 2. `dynamicsHessians()`
**Location**: `src/pybind/satellite.cpp:578-966`

**Purpose**: Compute second-order derivatives of satellite dynamics for SQP/iLQR optimization.

**Hessians Computed**:

| Hessian | Symbol | Mathematical Definition | Type | Dimensions | Notes |
|---------|--------|------------------------|------|------------|-------|
| State-State | `hess_xx` | ∂²f/∂x∂x | Tensor3 | (nx, nx, nx) | Indexed by output equation |
| Control-State | `hess_ux` | ∂²f/∂u∂x | Tensor3 | (nu, nx, nx) | Mixed control-state derivatives |
| Control-Control | `hess_uu` | ∂²f/∂u∂u | Tensor3 | (nu, nu, nx) | Control-control coupling (mostly zero) |

**Tensor Indexing Convention**: 
- `hess_xx.slice(i)` = ∂²f_i/∂x∂x (i-th output equation)
- Each slice is an (nx × nx) matrix

**Components of `hess_xx` (State-State Hessian)**:

| Output Eqn | Slice Index | Non-zero Blocks | Description |
|------------|-------------|-----------------|-------------|
| ẇ₀, ẇ₁, ẇ₂ | 0, 1, 2 | ∂²ẇ/∂w∂w | Gyroscopic term w × (Jw) is quadratic in w |
| ẇ₀, ẇ₁, ẇ₂ | 0, 1, 2 | ∂²ẇ/∂w∂h | Cross product w × h_rw has mixed derivatives |
| ẇ₀, ẇ₁, ẇ₂ | 0, 1, 2 | ∂²ẇ/∂q∂q | Disturbance torques (GG, drag, SRP) depend on R(q) |
| q̇₀, q̇₁, q̇₂, q̇₃ | 3, 4, 5, 6 | ∂²q̇/∂w∂q | Quaternion kinematics: W(q) is bilinear in (q, w) |
| ḣ₀, ..., ḣₙ | 7+i | All blocks via chain rule | RW dynamics coupled through ẇ Hessians |

**Specific Second Derivatives**:

| Derivative | Formula | Notes |
|------------|---------|-------|
| ∂²ẇ/∂w∂w | J⁻¹ · ∂²/∂w∂w[-w × (Jw)] | Quadratic gyroscopic term |
| ∂²ẇ/∂w∂h_k | J⁻¹ · (-skew(e_j) · axis_k) | Linear in each variable separately |
| ∂²ẇ/∂q∂q | J⁻¹ · ∑ ∂²τ_dist/∂q∂q | From GG, drag, SRP Hessians |
| ∂²q̇/∂q∂w | 0.5 · ∂W/∂q (bilinear structure) | W matrix derivatives |
| ∂²ḣ_i/∂x∂x | -J_rw · axis_i · ∂²ẇ/∂x∂x | Chain rule through ẇ |

**Components of `hess_ux` (Control-State Hessian)**:

| Output Eqn | Non-zero Blocks | Description |
|------------|-----------------|-------------|
| ẇ₀, ẇ₁, ẇ₂ | ∂²ẇ/∂u_mtq∂q | MTQ torque coupling through B_body(q) |
| ḣ₀, ..., ḣₙ | ∂²ḣ/∂u∂q | RW momentum coupling via ẇ mixed derivatives |

**Components of `hess_uu` (Control-Control Hessian)**:
- **Mostly zero**: Actuator torques are linear in control inputs
- MTQ: τ = -(B_body × axis) · u (linear in u)
- RW: τ = axis · u (linear in u)

---

## Cost Jacobians & Hessians

### 3. `stageCostJacobians()`
**Location**: `src/pybind/satellite.cpp:1126-1372`

**Purpose**: Compute first-order derivatives of stage cost function.

**Cost Function Structure**:
```
L(x, u) = w_ang · L_ang(q) 
        + w_av · ||w||²
        + w_avmag · |w·b_body|
        + w_avang · cross_cost(q, w)
        + w_u · ||u||²_weighted
        + RW_momentum_penalties(h)
```

**Jacobians Computed**:

| Derivative | Symbol | Mathematical Definition | Dimensions | Description |
|------------|--------|------------------------|------------|-------------|
| Cost gradient (state) | `lx` | ∂L/∂x | (nx × 1) | Gradient w.r.t. state |
| Cost gradient (control) | `lu` | ∂L/∂u | (nu × 1) | Gradient w.r.t. control |
| Mixed cost Hessian | `lux` | ∂²L/∂u∂x | (nu × nx) | Mixed control-state (always zero) |

**Components of `lx` (State Gradient)**:

| Block | Symbol | Derivative | Formula | Notes |
|-------|--------|-----------|---------|-------|
| (0:3) | ∂L/∂w | Angular velocity cost | 2·w_av·w - sign(q̇)·w_avang·(W^T q_goal) + w_avmag·sign(w·b)·b_body | Three cost terms |
| (3:7) | ∂L/∂q | Attitude & kinematic cost | w_ang·(∂L_ang/∂q̇)·q_goal - w_avang·∂(q_goal^T W w)/∂q + w_avmag·∂(w·b_body)/∂q | Quaternion double-cover handled |
| (7:7+nrw) | ∂L/∂h | RW momentum penalty | ∂/∂h[saturation + stiction penalties] | Piecewise smooth penalties |

**Attitude Cost Derivatives** (∂L_ang/∂q̇ where q̇ = q·q_goal):

| Cost Type | `ang_cost_func_type` | Formula L_ang(q̇) | Derivative ∂L_ang/∂q̇ |
|-----------|---------------------|------------------|---------------------|
| Linear | 0 | 1 - \|q̇\| | -1 |
| Quadratic | 1 | 0.5·(1 - \|q̇\|)² | -(1 - \|q̇\|) |
| Quadratic Angular | 3 | 0.5·acos²(\|q̇\|) | -acos(\|q̇\|)/√(1 - q̇²) |
| Pseudo-Huber Angular | 5 | δ²(√(1+(θ/δ)²) − 1), θ = acos(\|q̇\|), δ = `ang_cost_huber_delta` | -(θ/√(1+(θ/δ)²))/√(1 - q̇²) |

Implemented set: `{0, 1, 3, 5}`, default **3**. Type 5 matches type 3's ½θ² near the goal (to O((θ/δ)²)) but its angle-gradient saturates at δ for θ ≫ δ (bounded urgency on large slews); retired ids 2/4 are not reused. Type 2 (raw `acos(|q̇|)`, derivative `-1/√(1 - q̇²)`) was **removed**: it is concave (anti-PSD under Gauss-Newton) and singular at both poles, including perfect alignment (q̇ = +1). Migrate to type 3 (same acos family, Taylor-protected at the aligned pole) or type 0. Type 4 (`1 - |q̇|²`) was also removed — it is type 1 with a doubled angle weight.

**Special Handling**:
- **Quaternion Alignment**: If q·q_goal < 0, flip q_goal → -q_goal to avoid sign discontinuities
- **Normalization Projection**: `lx[3:7] = (I - q·q^T) · lx[3:7]` to project gradient onto tangent space
- **Numerical Differentiation**: Cross-cost gradient ∂(q_goal^T W w)/∂q computed via finite differences for robustness

---

### 4. `stageCostHessians()`
**Location**: `src/pybind/satellite.cpp:1381-1590`

**Purpose**: Compute second-order derivatives of stage cost (optional, can be disabled).

**Hessians Computed**:

| Hessian | Symbol | Mathematical Definition | Dimensions | Conditional |
|---------|--------|------------------------|------------|-------------|
| State-State | `lxx` | ∂²L/∂x∂x | (nx × nx) | Only if `cost_cfg.use_cost_hess == true` |
| Control-Control | `luu` | ∂²L/∂u∂u | (nu × nu) | Always computed (diagonal) |
| Control-State | `lux` | ∂²L/∂u∂x | (nu × nx) | Always zero (no coupling) |

**Components of `lxx` (State-State Hessian)** (when enabled):

| Block | Symbol | Derivative | Description |
|-------|--------|-----------|-------------|
| (0:3, 0:3) | ∂²L/∂w∂w | 2·w_av·I + ∂²(cross_cost)/∂w∂w + ∂²(mag_cost)/∂w∂w | Angular velocity Hessian |
| (0:3, 3:7) | ∂²L/∂w∂q | ∂²(cross_cost)/∂w∂q + ∂²(mag_cost)/∂w∂q | Mixed w-q derivatives |
| (3:7, 3:7) | ∂²L/∂q∂q | w_ang·∂²L_ang/∂q∂q + ∂²(cross_cost)/∂q∂q + ∂²(mag_cost)/∂q∂q | Attitude Hessian (complex) |
| (7:7+nrw, 7:7+nrw) | ∂²L/∂h∂h | ∂²/∂h²[RW penalties] | Diagonal RW Hessian |

**Attitude Cost Second Derivatives** (∂²L_ang/∂q̇²):

| Cost Type | Second Derivative ∂²L_ang/∂q̇² | Notes |
|-----------|-------------------------------|-------|
| Linear (0) | 0 | Piecewise linear, Hessian = 0 (except at q̇=0) |
| Quadratic (1) | 1 | Constant curvature |
| Angular (2) | -q̇/(1 - q̇²)^(3/2) | Singular at q̇=±1 |
| Quadratic Angular (3) | [1 - acos(q̇)·q̇]/[(1 - q̇²)^(3/2)] | Complex curvature |
| Quadratic Dot (4) | -2 | Constant negative curvature |

**Components of `luu` (Control-Control Hessian)**:

| Block | Diagonal Elements | Formula |
|-------|------------------|---------|
| (0:nmtq, 0:nmtq) | ∂²L/∂u²_mtq,i | w_u · w_mtq / (u_max,i)² |
| (nmtq:nu, nmtq:nu) | ∂²L/∂u²_rw,i | w_u · w_rw / (u_max,i)² |

**Note**: Off-diagonal elements of `luu` are zero (no control coupling).

**Special Handling**:
- If `use_cost_hess == false`, only `luu` control curvature is computed (state Hessian set to zero)
- This improves optimizer stability by avoiding potentially unstable quaternion second derivatives
- Quaternion Hessian projection is complex and may be disabled for robustness

---

### 5. `terminalCostJacobians()` & `terminalCostHessians()`
**Location**: `src/pybind/satellite.cpp:1374-1379, 1593-1597`

**Implementation**: 
```cpp
VecX lx_N, MatX lu_N, MatX lux_N;
std::tie(lx_N, lu_N, lux_N) = stageCostJacobians(N-1, N, x, u, ..., cost_cfg);
return {lx_N, lu_N, lux_N};
```

**Note**: Terminal cost uses same formulas as stage cost, but with different weights:
- `w_ang_N` instead of `w_ang`
- `w_av_N` instead of `w_av`
- Control cost weight set to 0 (no control at terminal time)

---

## Constraint Jacobians & Hessians

### 6. `constraints()`
**Location**: `src/pybind/satellite.cpp:1682-1765`

**Purpose**: Compute inequality constraint values for optimization.

**Constraint Types**:

| Constraint | Description | Formula | Active When |
|------------|-------------|---------|-------------|
| Quaternion norm | Unit quaternion enforcement | c = q^T q - 1 | Always (if enabled) |
| Sun exclusion | Keep sun out of sensor FOV | c = cos(θ) - cos(FOV_half) | `plan_for_keep_out_zone == true` |
| Sun pointing | Point toward sun | c = -cos(θ) + cos(tolerance) | `plan_for_sun_points == true` |

### 7. `constraintJacobians()`
**Location**: `src/pybind/satellite.cpp:1770-1896`

**Purpose**: Compute first-order derivatives of constraints.

**Jacobians Computed**:

| Derivative | Symbol | Dimensions | Description |
|------------|--------|------------|-------------|
| Constraint-State | `jac_x` | (nc × nx) | ∂c/∂x where c = constraints vector |
| Constraint-Control | `jac_u` | (nc × nu) | ∂c/∂u (always zero - constraints don't depend on u) |

**Components of `jac_x`**:

| Constraint | Derivative | Formula | Notes |
|------------|-----------|---------|-------|
| Quaternion norm | ∂(q^T q - 1)/∂q | 2q^T | Linear in q |
| Sun exclusion | ∂cos(θ)/∂q | ∂/∂q[b_body · s_body] | Both vectors depend on q via rotation |
| Sun pointing | ∂cos(θ)/∂q | Same as above | Same computation, different bounds |

**Sun Angle Derivatives**:
```
cos(θ) = b_body · s_body = (R^T · b) · (R^T · s)
∂cos(θ)/∂q = (∂R^T/∂q · b)^T · (R^T · s) + (R^T · b)^T · (∂R^T/∂q · s)
           = b^T · (∂R/∂q · s_body) + s^T · (∂R/∂q · b_body)
```

Uses: `drotmatTvecdq(q, s_eci)` to compute ∂(R^T · s)/∂q

---

### 8. `constraintHessians()`
**Location**: `src/pybind/satellite.cpp:1902-1992`

**Purpose**: Compute second-order derivatives of constraints for SQP methods.

**Hessians Computed** (per constraint i):

| Hessian | Symbol | Dimensions | Type | Description |
|---------|--------|------------|------|-------------|
| Control-Control | `hess_uu[i]` | (nu × nu) | Matrix | ∂²c_i/∂u∂u (always zero) |
| Control-State | `hess_ux[i]` | (nu × nx) | Matrix | ∂²c_i/∂u∂x (always zero) |
| State-State | `hess_xx[i]` | (nx × nx) | Matrix | ∂²c_i/∂x∂x |

**Components of `hess_xx`** (State-State Constraint Hessian):

| Constraint Type | Non-zero Block | Derivative | Formula |
|----------------|----------------|-----------|---------|
| Quaternion norm | (3:7, 3:7) | ∂²(q^T q)/∂q∂q | 2·I₄ |
| Sun exclusion | (3:7, 3:7) | ∂²(b_body · s_body)/∂q∂q | Uses `ddrotmatTvecdqdq()` |
| Sun pointing | (3:7, 3:7) | ∂²(b_body · s_body)/∂q∂q | Same as exclusion |

**Sun Angle Second Derivatives**:
```
∂²(b·s)/∂q_i∂q_j = b^T · ∂²R/∂q_i∂q_j · s + s^T · ∂²R/∂q_i∂q_j · b
                  + (∂R/∂q_i · b)^T · (∂R/∂q_j · s) + (∂R/∂q_j · b)^T · (∂R/∂q_i · s)
```

Uses: `ddrotmatTvecdqdq(q, vec)` to compute ∂²(R^T · vec)/∂q∂q

---

## Actuator Jacobians & Hessians

### 9. MTQ (Magnetorquer) Derivatives

**File**: `src/pybind/actuators/MTQ.cpp`

#### 9.1 `MTQ::torque()`
**Formula**: τ = -(B_body × axis) · u

#### 9.2 `MTQ::dtorq_du()`
**Location**: Lines 13-17  
**Derivative**: ∂τ/∂u  
**Formula**: -(B_body × axis)  
**Dimensions**: (1 × 3) → (3 × 1)^T  
**Type**: `Mat13`

#### 9.3 `MTQ::dtorq_dbasestate()`
**Location**: Lines 19-29  
**Derivative**: ∂τ/∂x (where x = [w, q])  
**Formula**: Only quaternion block is non-zero: ∂τ/∂q = -u · (∂B_body/∂q × axis)  
**Dimensions**: (7 × 3)  
**Type**: `Mat73`  
**Uses**: `dB_dq` = ∂(R^T B_eci)/∂q passed as argument

#### 9.4 `MTQ::ddtorq_dudu()`
**Location**: Lines 31-33  
**Derivative**: ∂²τ/∂u∂u  
**Formula**: 0 (torque is linear in u)  
**Dimensions**: (1, 1, 3) tensor  
**Type**: `T113`

#### 9.5 `MTQ::ddtorq_dudbasestate()`
**Location**: Lines 35-47  
**Derivative**: ∂²τ/∂u∂x  
**Formula**: ∂²τ/∂u∂q_i = -(∂B_body/∂q_i × axis) for each quaternion component  
**Dimensions**: (1, 7, 3) tensor  
**Type**: `T173`  
**Uses**: `dB_dq` = ∂(R^T B_eci)/∂q

#### 9.6 `MTQ::ddtorq_dbasestatedbasestate()`
**Location**: Lines 49-61  
**Derivative**: ∂²τ/∂x∂x  
**Formula**: ∂²τ/∂q_i∂q_j = -u · ∂²(B_body × axis)/∂q_i∂q_j  
**Dimensions**: (7, 7, 3) tensor (only quaternion blocks non-zero)  
**Type**: `T773`  
**Uses**: `d2B_dq2` = ∂²(R^T B_eci)/∂q∂q (array of 3 matrices, one per torque component)

**Note**: For MTQ second derivatives w.r.t. quaternion:
- Cross product rule: ∂²(a × b)/∂q_i∂q_j when a depends on q
- Implemented using `d2B_dq2[k](i,j)` which gives ∂²B_k/∂q_i∂q_j for k-th component of B_body

---

### 10. RW (Reaction Wheel) Derivatives

**File**: `src/pybind/actuators/RW.cpp`

#### 10.1 `RW::torque()`
**Formula**: τ = axis · u

#### 10.2 `RW::storageTorque()`
**Location**: Lines 41-45  
**Purpose**: Compute momentum storage rate  
**Formula**: ḣ = -u (momentum increases opposite to torque on spacecraft)  
**Returns**: (1 × 1) matrix

#### 10.3 `RW::dtorq_du()`
**Location**: Lines 47-52  
**Derivative**: ∂τ/∂u  
**Formula**: axis  
**Dimensions**: (1 × 3) → (3 × 1)^T  
**Type**: `Mat13`

#### 10.4 `RW::dtorq_dbasestate()`
**Location**: Lines 54-58  
**Derivative**: ∂τ/∂x  
**Formula**: 0 (RW torque independent of state)  
**Dimensions**: (7 × 3)  
**Type**: `Mat73`

#### 10.5 `RW::dstor_torq_du()`
**Location**: Lines 60-65  
**Derivative**: ∂ḣ/∂u  
**Formula**: -1  
**Dimensions**: (1 × 1)  
**Type**: `Mat11`

#### 10.6 `RW::dstor_torq_dbasestate()`
**Location**: Lines 67-71  
**Derivative**: ∂ḣ/∂x  
**Formula**: 0 (storage rate independent of state, only depends on control)  
**Dimensions**: (7 × 1)  
**Type**: `Mat71`

**Note**: RW has no Hessians because torque is linear in both control and state.

---

## Disturbance Jacobians & Hessians

### 11. Gravity Gradient Disturbance

**File**: `src/pybind/disturbances/ggdisturbance.h` & `.cpp`

#### 11.1 `GGDisturbance::torque()`
**Formula**: τ_gg = (3μ/r³) · (r_body × J · r_body)  
**Where**: μ = gravitational parameter, r = position vector, J = inertia matrix

#### 11.2 `GGDisturbance::dtorque_dq()`
**Location**: Header line 100  
**Derivative**: ∂τ_gg/∂q  
**Purpose**: Compute how GG torque changes with spacecraft orientation  
**Formula**: ∂τ_gg/∂q = (3μ/r³) · ∂(r_body × J · r_body)/∂q  
**Dimensions**: (3 × 4)  
**Type**: `Mat34`  
**Uses**: `dr_body/dq` passed as argument

#### 11.3 `GGDisturbance::ddtorque_dqdq()`
**Location**: Header line 114  
**Derivative**: ∂²τ_gg/∂q∂q  
**Purpose**: Second-order sensitivity of GG torque to orientation  
**Dimensions**: Tensor (4, 4, 3) - one (4×4) matrix per torque component  
**Type**: `T443`  
**Uses**: Both `dr_body/dq` and `d²r_body/dq²` passed as arguments

---

### 12. Aerodynamic Drag Disturbance

**File**: `src/pybind/disturbances/dragdisturbance.h` & `.cpp`

#### 12.1 `DragDisturbance::torque()`
**Formula**: τ_drag = -0.5 · ρ · C_D · A · ||v||² · (v̂ × r_cp)  
**Where**: ρ = density, C_D = drag coefficient, A = area, v = velocity, r_cp = center of pressure offset

#### 12.2 `DragDisturbance::dtorque_dq()`
**Location**: Header line 96  
**Derivative**: ∂τ_drag/∂q  
**Formula**: Derivative of drag torque w.r.t. quaternion (via v_body rotation)  
**Dimensions**: (3 × 4)  
**Type**: `Mat34`  
**Uses**: `dv_body/dq` passed as argument

#### 12.3 `DragDisturbance::ddtorque_dqdq()`
**Location**: Header line 109  
**Derivative**: ∂²τ_drag/∂q∂q  
**Dimensions**: Tensor (4, 4, 3)  
**Type**: `T443`  
**Uses**: Both `dv_body/dq` and `d²v_body/dq²`

---

### 13. Solar Radiation Pressure Disturbance

**File**: `src/pybind/disturbances/srpdisturbance.h` & `.cpp`

#### 13.1 `SRPDisturbance::torque()`
**Formula**: τ_srp = -P_srp · A · (s̄ × r_cp) · (1 + ε) where eclipse handled externally  
**Where**: P_srp = solar pressure, A = area, s̄ = sun direction unit vector, ε = reflectivity

#### 13.2 `SRPDisturbance::dtorque_dq()`
**Location**: Header line 102  
**Derivative**: ∂τ_srp/∂q  
**Formula**: Derivative of SRP torque w.r.t. quaternion (via s_body rotation)  
**Dimensions**: (3 × 4)  
**Type**: `Mat34`  
**Uses**: `ds_body/dq` passed as argument

#### 13.3 `SRPDisturbance::ddtorque_dqdq()`
**Location**: Header line 115  
**Derivative**: ∂²τ_srp/∂q∂q  
**Dimensions**: Tensor (4, 4, 3)  
**Type**: `T443`  
**Uses**: Both `ds_body/dq` and `d²s_body/dq²`

---

## Mathematical Utility Functions

**File**: `include/saltro/math/quaternion.h`

### 14. `rotationMatrix(q)`
**Purpose**: Convert quaternion to rotation matrix  
**Formula**: R(q) ∈ SO(3)  
**Returns**: (3 × 3) rotation matrix

### 15. `findWMat(q)`
**Location**: Header line 55  
**Purpose**: Compute quaternion kinematics matrix  
**Formula**: 
```
W(q) = [[-q₁, -q₂, -q₃],
        [ q₀, -q₃,  q₂],
        [ q₃,  q₀, -q₁],
        [-q₂,  q₁,  q₀]]
```
**Use**: q̇ = 0.5 · W(q) · w  
**Returns**: (4 × 3) matrix  
**Type**: `Mat43`

### 16. `skewSymmetric(v)`
**Location**: Header line 90  
**Purpose**: Compute skew-symmetric matrix from vector  
**Formula**: 
```
skew(v) = [  0  -v₃  v₂]
          [ v₃   0  -v₁]
          [-v₂  v₁   0 ]
```
**Use**: Cross product a × b = skew(a) · b  
**Returns**: (3 × 3) matrix  
**Type**: `Mat33`

### 17. `drotmatTvecdq(q, v)`
**Location**: Header line 107  
**Purpose**: Compute Jacobian of rotated vector w.r.t. quaternion  
**Derivative**: ∂(R^T · v)/∂q where R = R(q)  
**Formula**: Derivatives of rotation matrix times constant vector  
**Returns**: (4 × 3) matrix - each row is ∂(R^T · v)/∂q_i  
**Type**: `Mat43`  
**Use**: Essential for all orientation-dependent derivatives

### 18. `ddrotmatTvecdqdq(q, v)`
**Location**: Header line 125  
**Purpose**: Compute Hessian of rotated vector w.r.t. quaternion  
**Derivative**: ∂²(R^T · v)/∂q∂q  
**Returns**: Array of 3 matrices (4 × 4), one per component of (R^T · v)  
**Type**: `std::array<Mat44, 3>`  
**Use**: 
- `result[k]` = ∂²[(R^T · v)_k]/∂q∂q for k = 0, 1, 2
- `result[k](i, j)` = ∂²[(R^T · v)_k]/∂q_i∂q_j

---

## Summary Statistics

### Total Derivative Computations

| Category | First Derivatives (Jacobians) | Second Derivatives (Hessians) |
|----------|------------------------------|------------------------------|
| Dynamics | 3 (∂f/∂x, ∂f/∂u, ∂f/∂τ_dist) | 3 (∂²f/∂x², ∂²f/∂u∂x, ∂²f/∂u²) |
| Cost | 2 (∂L/∂x, ∂L/∂u) | 3 (∂²L/∂x², ∂²L/∂u², ∂²L/∂u∂x) |
| Constraints | 2 (∂c/∂x, ∂c/∂u) | 3 (∂²c/∂x², ∂²c/∂u∂x, ∂²c/∂u²) |
| MTQ Actuator | 2 (∂τ/∂u, ∂τ/∂x) | 3 (∂²τ/∂u², ∂²τ/∂u∂x, ∂²τ/∂x²) |
| RW Actuator | 4 (∂τ/∂u, ∂τ/∂x, ∂ḣ/∂u, ∂ḣ/∂x) | 0 (all linear) |
| GG Disturbance | 1 (∂τ/∂q) | 1 (∂²τ/∂q²) |
| Drag Disturbance | 1 (∂τ/∂q) | 1 (∂²τ/∂q²) |
| SRP Disturbance | 1 (∂τ/∂q) | 1 (∂²τ/∂q²) |
| Math Utilities | 2 (∂(R^T v)/∂q, W matrix) | 1 (∂²(R^T v)/∂q²) |
| **TOTAL** | **17 Jacobians** | **13 Hessians** |

### Derivative Dependencies (Call Graph)

```
dynamicsJacobians()
├─ MTQ::dtorq_du()
├─ MTQ::dtorq_dbasestate() → drotmatTvecdq()
├─ GGDisturbance::dtorque_dq() → drotmatTvecdq()
├─ DragDisturbance::dtorque_dq() → drotmatTvecdq()
├─ SRPDisturbance::dtorque_dq() → drotmatTvecdq()
├─ findWMat()
└─ skewSymmetric()

dynamicsHessians()
├─ MTQ::ddtorq_dudbasestate() → drotmatTvecdq()
├─ MTQ::ddtorq_dbasestatedbasestate() → ddrotmatTvecdqdq()
├─ GGDisturbance::ddtorque_dqdq() → drotmatTvecdq(), ddrotmatTvecdqdq()
├─ DragDisturbance::ddtorque_dqdq() → drotmatTvecdq(), ddrotmatTvecdqdq()
├─ SRPDisturbance::ddtorque_dqdq() → drotmatTvecdq(), ddrotmatTvecdqdq()
└─ skewSymmetric()

stageCostJacobians()
├─ findWMat()
├─ drotmatTvecdq()
└─ numerical differentiation for ∂(q_goal^T W w)/∂q

stageCostHessians()
└─ analytical formulas (or disabled based on use_cost_hess flag)

constraintJacobians()
└─ drotmatTvecdq()

constraintHessians()
└─ ddrotmatTvecdqdq()
```

---

## Notes on Implementation

### Quaternion Normalization
- All Jacobians w.r.t. quaternion are projected onto tangent space: `jac[:, 3:7] *= (I - q·q^T)`
- This accounts for the fact that `dynamics()` and cost functions normalize q internally
- Without projection, gradients would be inconsistent with the constraint manifold

### Tensor Indexing Convention
- Hessians are stored as **output-indexed tensors**
- `hess_xx.slice(i)` = ∂²f_i/∂x∂x (the Hessian of the i-th output equation)
- Each slice is a symmetric (nx × nx) matrix
- This convention differs from standard "Hessian of scalar function" (which would be a single matrix)

### Numerical Stability
- All disturbance derivatives wrapped in try-catch blocks (flight-safe)
- Magnetic field, position, velocity norms checked before computing derivatives
- `safeAbs()`, `safeSign()`, `clampUnit()` functions prevent NaN propagation
- Finite difference used for complex cross-cost gradient ∂(q_goal^T W w)/∂q

### Optimization Performance
- Hessians are **optional** via `cost_cfg.use_cost_hess` flag
- Disabling state Hessians can improve stability (quaternion second derivatives are complex)
- Control Hessians (`luu`) always computed for backward pass regularization
- Some Hessians are analytically zero (control-control for MTQ/RW, control-state for cost)

---

## Potential Issues to Check

Based on the structure, here are areas where Jacobian/Hessian errors commonly occur:

1. **Quaternion Double-Cover**: 
   - ✅ Handled in `stageCostJacobians()` via q_goal alignment
   - ⚠️ Check if alignment is also done in Hessians

2. **Quaternion Normalization Projection**:
   - ✅ Applied to `dynamicsJacobians()` state Jacobian
   - ✅ Applied to `stageCostJacobians()` gradient
   - ⚠️ Check if applied correctly to Hessians (second-order projection)

3. **W Matrix Derivatives**:
   - ∂W/∂q implemented analytically in `dynamicsJacobians()`
   - ⚠️ Verify correctness of quaternion-angular velocity coupling

4. **Disturbance Hessian Sign**:
   - ∂²τ/∂q² for GG, drag, SRP computed separately
   - ⚠️ Verify signs when accumulating into `hess_xx`

5. **RW Coupling**:
   - RW momentum dynamics coupled through ẇ via chain rule
   - ⚠️ Verify all coupling terms in `dynamicsHessians()` for RW equations

6. **Cross-Cost Gradient**:
   - Uses numerical differentiation in `stageCostJacobians()`
   - ⚠️ Could be source of inaccuracy if ε too large/small

7. **Constraint Hessians**:
   - Sun angle constraints use complex rotation Hessians
   - ⚠️ Verify product rule application for (b_body · s_body)

---

## Testing Recommendations

To verify Jacobian/Hessian correctness:

1. **Finite Difference Check**: Compare analytical derivatives against numerical finite differences
2. **Symmetry Check**: Verify Hessian symmetry (∂²f/∂x_i∂x_j == ∂²f/∂x_j∂x_i)
3. **Quaternion Projection**: Verify gradient lies in tangent space (q^T · ∇_q L ≈ 0)
4. **Zero Blocks**: Confirm expected zero blocks (e.g., ∂ḣ/∂u for MTQ control)
5. **Unit Tests**: Test each actuator/disturbance derivative method independently
6. **Optimizer Convergence**: If optimizer fails, suspect Hessian sign errors or missing terms

---

*End of Jacobian/Hessian Reference*
