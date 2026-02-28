// PYBIND_DEPENDS: actuator MTQ RW plannersettings geometryconfig

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <saltro/pybind/satellite.h>
#include "tensor_py.h"  // Include tensor type caster for Hessian return types

namespace py = pybind11;

void bind_satellite(py::module_& m) {
    py::class_<Satellite>(m, "Satellite")
        .def(py::init<>(),
             R"doc(
Construct a Satellite with default settings.

The satellite is initialized with:
- Identity inertia matrix
- No actuators (MTQs or RWs)
- Default planner settings
)doc")
        .def(py::init<const Satellite::Mat33&, const PlannerSettings&>(),
             py::arg("Jcom"),
             py::arg("settings"),
             R"doc(
Construct a Satellite with specified inertia and settings.

Parameters
----------
Jcom : ndarray (3, 3)
    Inertia matrix (must be finite and invertible)
settings : PlannerSettings
    Planning configuration settings

Raises
------
ValueError
    If inertia matrix is singular or has non-finite entries
)doc")
        // Inertia management
        .def("setInertia", &Satellite::setInertia,
             py::arg("Jcom"),
             R"doc(
Set the satellite inertia matrix.

Parameters
----------
Jcom : ndarray (3, 3)
    New inertia matrix (must be finite and invertible)

Raises
------
ValueError
    If inertia matrix is singular or has non-finite entries
)doc")
        .def_property_readonly("inertia", &Satellite::inertia,
                               "Get the total inertia matrix (including RWs)")
        .def_property_readonly("invInertia", &Satellite::invInertia,
                               "Get the inverse of the total inertia matrix")
        .def_property_readonly("inertiaNoRW", &Satellite::inertiaNoRW,
                               "Get the inertia matrix excluding RW contributions")
        .def_property_readonly("invInertiaNoRW", &Satellite::invInertiaNoRW,
                               "Get the inverse inertia matrix excluding RW contributions")
        
        // Geometry configuration
        .def("setGeometryConfig", &Satellite::setGeometryConfig,
             py::arg("config"),
             "Set the geometry configuration for disturbance torque calculations")
        .def_property("geometryConfig", 
                     py::overload_cast<>(&Satellite::geometryConfig, py::const_),
                     py::overload_cast<>(&Satellite::geometryConfig),
                     "Get or set the geometry configuration")
        
        // Actuator management
        .def("addMTQ", &Satellite::addMTQ,
             py::arg("axis"),
             py::arg("max_dipole"),
             R"doc(
Add a Magnetic Torquer (MTQ) to the satellite.

Parameters
----------
axis : ndarray (3,)
    MTQ axis direction (will be normalized)
max_dipole : float
    Maximum magnetic dipole moment

Raises
------
RuntimeError
    If maximum number of MTQs already added
)doc")
        .def("addRW", &Satellite::addRW,
             py::arg("axis"),
             py::arg("max_torque"),
             py::arg("J"),
             py::arg("h0"),
             py::arg("h_max"),
             R"doc(
Add a Reaction Wheel (RW) to the satellite.

Parameters
----------
axis : ndarray (3,)
    Wheel spin axis direction (will be normalized)
max_torque : float
    Maximum torque output
J : float
    Wheel moment of inertia (must be > 0)
h0 : float
    Initial angular momentum
h_max : float
    Maximum angular momentum (must be >= 0)

Raises
------
RuntimeError
    If maximum number of RWs already added
)doc")
        .def_property_readonly("numMTQ", &Satellite::numMTQ,
                               "Get the number of MTQs")
        .def_property_readonly("numRW", &Satellite::numRW,
                               "Get the number of reaction wheels")
        
        // Actuator access
        .def("getMTQ",
             py::overload_cast<int>(&Satellite::getMTQ, py::const_),
             py::arg("i"),
             py::return_value_policy::reference_internal,
             R"doc(
Get the MTQ at the specified index.

Parameters
----------
i : int
    Index of the MTQ (0-based)

Returns
-------
MTQ
    Reference to the MTQ

Raises
------
IndexError
    If index is out of range
)doc")
        .def("getRW",
             py::overload_cast<int>(&Satellite::getRW, py::const_),
             py::arg("i"),
             py::return_value_policy::reference_internal,
             R"doc(
Get the reaction wheel at the specified index.

Parameters
----------
i : int
    Index of the RW (0-based)

Returns
-------
RW
    Reference to the reaction wheel

Raises
------
IndexError
    If index is out of range
)doc")
        
        // Dimensions
        .def_property_readonly("controlDim", &Satellite::controlDim,
                               "Get the control dimension (numMTQ + numRW)")
        .def_property_readonly("stateDim", &Satellite::stateDim,
                               "Get the state dimension (7 + numRW)")
        .def_property_readonly("reducedStateDim", &Satellite::reducedStateDim,
                               "Get the reduced state dimension (6 + numRW)")
        
        // Settings
        .def("setSettings", &Satellite::setSettings,
             py::arg("settings"),
             "Set the planner settings")
        .def_property_readonly("settings", &Satellite::settings,
                               "Get the planner settings")
        
        // Dynamics and torques
        .def("actuatorTorque", &Satellite::actuatorTorque,
             py::arg("x"),
             py::arg("u"),
             py::arg("B_eci"),
             R"doc(
Compute total actuator torque.

Parameters
----------
x : ndarray
    State vector (size: stateDim)
u : ndarray
    Control vector (size: controlDim)
B_eci : ndarray (3,)
    Magnetic field in ECI frame

Returns
-------
ndarray (3,)
    Total actuator torque in body frame
)doc")
        .def("disturbanceTorque", &Satellite::disturbanceTorque,
             py::arg("x"),
             py::arg("dist"),
             py::arg("R_eci"),
             py::arg("B_eci"),
             py::arg("S_eci"),
             py::arg("V_eci"),
             py::arg("rho"),
             R"doc(
Compute total disturbance torque.

Parameters
----------
x : ndarray
    State vector (size: stateDim)
dist : DisturbanceConfig
    Disturbance configuration
R_eci : ndarray (3,)
    Position in ECI frame
B_eci : ndarray (3,)
    Magnetic field in ECI frame
S_eci : ndarray (3,)
    Sun direction in ECI frame
V_eci : ndarray (3,)
    Velocity in ECI frame
rho : int
    Atmospheric density

Returns
-------
ndarray (3,)
    Total disturbance torque in body frame
)doc")
        .def("dynamics", &Satellite::dynamics,
             py::arg("x"),
             py::arg("u"),
             py::arg("dist"),
             py::arg("R_eci"),
             py::arg("B_eci"),
             py::arg("S_eci"),
             py::arg("V_eci"),
             py::arg("rho"),
             R"doc(
Compute satellite dynamics (state derivative).

Parameters
----------
x : ndarray
    State vector: [angular_velocity (3), quaternion (4), RW_momenta (numRW)]
u : ndarray
    Control vector: [MTQ_controls (numMTQ), RW_controls (numRW)]
dist : DisturbanceConfig
    Disturbance configuration
R_eci : ndarray (3,)
    Position in ECI frame
B_eci : ndarray (3,)
    Magnetic field in ECI frame
S_eci : ndarray (3,)
    Sun direction in ECI frame
V_eci : ndarray (3,)
    Velocity in ECI frame
rho : int
    Atmospheric density

Returns
-------
ndarray
    State derivative (size: stateDim)
)doc")
        .def("dynamicsJacobians", &Satellite::dynamicsJacobians,
             py::arg("x"),
             py::arg("u"),
             py::arg("dist"),
             py::arg("R_eci"),
             py::arg("B_eci"),
             py::arg("S_eci"),
             py::arg("V_eci"),
             R"doc(
Compute dynamics Jacobians (first-order partial derivatives).

Computes the Jacobians of the dynamics function f(x, u) with respect to
the state x, control u, and disturbance parameters.

Parameters
----------
x : ndarray
    State vector: [angular_velocity (3), quaternion (4), RW_momenta (numRW)]
u : ndarray
    Control vector: [MTQ_controls (numMTQ), RW_controls (numRW)]
dist : DisturbanceConfig
    Disturbance configuration
R_eci : ndarray (3,)
    Position in ECI frame
B_eci : ndarray (3,)
    Magnetic field in ECI frame
S_eci : ndarray (3,)
    Sun direction in ECI frame
V_eci : ndarray (3,)
    Velocity in ECI frame

Returns
-------
tuple[ndarray, ndarray, ndarray]
    (jac_x, jac_u, jac_dist) where:
    - jac_x: Jacobian w.r.t. state (stateDim x stateDim)
    - jac_u: Jacobian w.r.t. control (stateDim x controlDim)
    - jac_dist: Jacobian w.r.t. disturbance effects (stateDim x 3)
)doc")
        .def("dynamicsHessians", &Satellite::dynamicsHessians,
             py::arg("x"),
             py::arg("u"),
             py::arg("dist"),
             py::arg("R_eci"),
             py::arg("B_eci"),
             py::arg("S_eci"),
             py::arg("V_eci"),
             R"doc(
Compute dynamics Hessians (second-order partial derivatives).

Computes the Hessian tensors of the dynamics function f(x, u) with respect to
state-state, control-state, and control-control pairs. Each Hessian is a 3D
tensor where slice i corresponds to the Hessian of the i-th output component.

Parameters
----------
x : ndarray
    State vector: [angular_velocity (3), quaternion (4), RW_momenta (numRW)]
u : ndarray
    Control vector: [MTQ_controls (numMTQ), RW_controls (numRW)]
dist : DisturbanceConfig
    Disturbance configuration
R_eci : ndarray (3,)
    Position in ECI frame
B_eci : ndarray (3,)
    Magnetic field in ECI frame
S_eci : ndarray (3,)
    Sun direction in ECI frame
V_eci : ndarray (3,)
    Velocity in ECI frame

Returns
-------
tuple[Tensor3, Tensor3, Tensor3]
    (hess_xx, hess_ux, hess_uu) where each is a 3D tensor:
    - hess_xx: ∂²f/∂x² - indexed by output equation (stateDim slices of stateDim x stateDim)
    - hess_ux: ∂²f/∂u∂x - indexed by output equation (stateDim slices of controlDim x stateDim)
    - hess_uu: ∂²f/∂u² - indexed by output equation (stateDim slices of controlDim x controlDim)
)doc")
        .def("constraints", &Satellite::constraints,
             py::arg("k"),
             py::arg("N"),
             py::arg("x"),
             py::arg("u"),
             py::arg("sun_eci"),
             py::arg("cnst_cfg"),
             R"doc(
Evaluate inequality constraints c(x,u) <= 0.

Constraint ordering:
1) Angular velocity magnitude limit
2) Sun avoidance limit using body +X boresight
3) (k < N-1 only) MTQ and RW bounds, RW momentum bounds, RW stiction proxy

Parameters
----------
k : int
    Current time step
N : int
    Total number of time steps
x : ndarray
    State vector (size: stateDim)
u : ndarray
    Control vector (size: controlDim)
sun_eci : ndarray (3,)
    Sun direction vector in ECI frame
cnst_cfg : ConstraintConfig
    Constraint configuration

Returns
-------
ndarray
    Constraint vector c where each entry should satisfy c_i <= 0
)doc")
        .def("constraintJacobians", &Satellite::constraintJacobians,
             py::arg("k"),
             py::arg("N"),
             py::arg("x"),
             py::arg("u"),
             py::arg("sun_eci"),
             py::arg("cnst_cfg"),
             R"doc(
Compute constraint Jacobians ∂c/∂u and ∂c/∂x.

Parameters
----------
k : int
    Current time step
N : int
    Total number of time steps
x : ndarray
    State vector (size: stateDim)
u : ndarray
    Control vector (size: controlDim)
sun_eci : ndarray (3,)
    Sun direction vector in ECI frame
cnst_cfg : ConstraintConfig
    Constraint configuration

Returns
-------
tuple[ndarray, ndarray]
    (c_u, c_x) where c_u is (n_constraints, controlDim) and c_x is (n_constraints, stateDim)
)doc")
        .def("constraintHessians", &Satellite::constraintHessians,
             py::arg("k"),
             py::arg("N"),
             py::arg("x"),
             py::arg("u"),
             py::arg("sun_eci"),
             py::arg("cnst_cfg"),
             R"doc(
Compute constraint Hessians (second derivatives).

Returns tensors H_uu, H_ux, H_xx where each slice k contains the Hessian
of constraint k with respect to the corresponding variables.

Parameters
----------
k : int
    Current time step
N : int
    Total number of time steps
x : ndarray
    State vector (size: stateDim)
u : ndarray
    Control vector (size: controlDim)
sun_eci : ndarray (3,)
    Sun direction vector in ECI frame
cnst_cfg : ConstraintConfig
    Constraint configuration

Returns
-------
tuple[Tensor3, Tensor3, Tensor3]
    (H_uu, H_ux, H_xx) Hessian tensors for each constraint
)doc")
        
        // State index constants
        .def_readonly_static("AV_INDEX", &Satellite::AV_INDEX,
                             "Index of angular velocity in state vector")
        .def_readonly_static("QUAT_INDEX", &Satellite::QUAT_INDEX,
                             "Index of quaternion in state vector")
        .def_readonly_static("RW_MOMENTUM_INDEX", &Satellite::RW_MOMENTUM_INDEX,
                             "Index of first RW momentum in state vector");
}
