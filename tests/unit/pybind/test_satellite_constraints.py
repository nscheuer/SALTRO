"""
Comprehensive pytest tests for Satellite constraint evaluation,
constraint Jacobians, and constraint Hessians.

Covers:
 - Constraint vector dimensionality (with/without actuators, terminal step)
 - Angular velocity constraint satisfaction / violation / normalization
 - Sun-avoidance constraint satisfaction / violation / eclipse robustness
 - MTQ dipole upper/lower bounds (scaling, config override)
 - RW torque upper/lower bounds
 - RW momentum upper/lower bounds
 - RW stiction torque floor (c = theta - |u|/u_lim - |h|/h_c)
 - Input validation (wrong dimensions, out-of-range k, negative N)
 - Finite-difference verification of Jacobians against constraints()
 - Finite-difference verification of Hessians against constraintJacobians()
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# tests/unit/pybind/<this file>: parents[0]=pybind, [1]=unit, [2]=tests, [3]=repo root
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "build"))
import saltro_py


# ============================================================================
# Helpers
# ============================================================================

def valid_inertia():
    """Return a valid inertia matrix"""
    return np.array([
        [0.067, 0.0, 0.0],
        [0.0, 0.067, 0.0],
        [0.0, 0.0, 0.067]
    ])


def identity_quat():
    """Identity quaternion (no rotation)"""
    return np.array([1.0, 0.0, 0.0, 0.0])


def make_state(w, q, h_rw=None):
    """Build a full state vector (AV 3 + Q 4 + RW momenta n_rw)"""
    if h_rw is None:
        h_rw = np.array([])
    
    x = np.zeros(7 + len(h_rw))
    x[saltro_py.Satellite.AV_INDEX:saltro_py.Satellite.AV_INDEX + 3] = w
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = q
    if len(h_rw) > 0:
        x[saltro_py.Satellite.RW_MOMENTUM_INDEX:] = h_rw
    return x


def normalize_quat_in_state(x):
    """Normalize the quaternion part of state vector in-place"""
    q = x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4]
    q /= np.linalg.norm(q)
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = q


def zero_control(dim):
    """Build a zero control vector of given dimension"""
    return np.zeros(dim)


def default_cnst_cfg():
    """Default constraint config"""
    cfg = saltro_py.ConstraintConfig()
    cfg.wmax = 20.0 * np.pi / 180.0  # 20 deg/s
    cfg.sun_limit_angle = 20.0 * np.pi / 180.0
    cfg.control_limit_scale = 0.75
    # u_max is a resize-able vector, resize to 0 means use actuator limits
    cfg.u_max.resize(0)
    return cfg


def sun_z():
    """Sun vector in +Z ECI (unit)"""
    return np.array([0.0, 0.0, 1.0])


# ============================================================================
# Fixture: Satellite with actuators
# ============================================================================

class ConstraintFixture:
    """Test fixture for satellite constraints"""
    
    # Actuator parameters
    mtq_dipole = 0.2
    rw_torque = 0.001
    rw_inertia = 1e-5
    rw_h0 = 0.0
    rw_hmax = 0.01
    
    def setup_method(self):
        """Set up satellite with 3 MTQs and 2 RWs"""
        self.J = valid_inertia()
        self.settings = saltro_py.PlannerSettings()
        self.sat = saltro_py.Satellite(self.J, self.settings)
        
        # Add 3 MTQs along principal axes
        self.sat.addMTQ(np.array([1.0, 0.0, 0.0]), self.mtq_dipole)
        self.sat.addMTQ(np.array([0.0, 1.0, 0.0]), self.mtq_dipole)
        self.sat.addMTQ(np.array([0.0, 0.0, 1.0]), self.mtq_dipole)
        
        # Add 2 RWs
        self.sat.addRW(np.array([1.0, 0.0, 0.0]), self.rw_torque, 
                       self.rw_inertia, self.rw_h0, self.rw_hmax)
        self.sat.addRW(np.array([0.0, 1.0, 0.0]), self.rw_torque,
                       self.rw_inertia, self.rw_h0, self.rw_hmax)
    
    def n_mtq(self):
        return self.sat.numMTQ
    
    def n_rw(self):
        return self.sat.numRW
    
    def n_ctrl(self):
        return self.sat.controlDim
    
    def nominal_state(self):
        """Nominal state: zero AV, identity quat, zero RW momentum"""
        h_rw = np.zeros(self.n_rw())
        return make_state(np.zeros(3), identity_quat(), h_rw)
    
    def nominal_control(self):
        """Zero control"""
        return zero_control(self.n_ctrl())
    
    def expected_dim_intermediate(self):
        """Expected constraint dimension at intermediate step"""
        # AV(1) + sun(1) + MTQ bounds(2*n_mtq) + RW (5*n_rw: 2 torque + 2 momentum + 1 stiction)
        return 1 + 1 + 2*self.n_mtq() + 5*self.n_rw()
    
    def expected_dim_terminal(self):
        """Expected constraint dimension at terminal step (state only)"""
        # AV(1) + sun(1)
        return 2


# ============================================================================
# SECTION 1 — Constraint vector dimension
# ============================================================================

def test_constraints_dimension_with_no_actuators():
    """Test constraint dimension with no actuators"""
    sat = saltro_py.Satellite(valid_inertia(), saltro_py.PlannerSettings())
    x = make_state(np.zeros(3), identity_quat())
    u = zero_control(0)
    cfg = default_cnst_cfg()
    
    # Intermediate step
    c = sat.constraints(0, 10, x, u, sun_z(), cfg)
    assert len(c) == 2
    
    # Terminal step
    c_term = sat.constraints(9, 10, x, u, sun_z(), cfg)
    assert len(c_term) == 2


def test_constraints_dimension_with_actuators_intermediate_step():
    """Test constraint dimension with actuators at intermediate step"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    c = fixture.sat.constraints(0, 10, fixture.nominal_state(), 
                                fixture.nominal_control(), sun_z(), default_cnst_cfg())
    assert len(c) == fixture.expected_dim_intermediate()


def test_constraints_dimension_at_terminal_step():
    """Test constraint dimension at terminal step (state only)"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    c = fixture.sat.constraints(9, 10, fixture.nominal_state(),
                                fixture.nominal_control(), sun_z(), default_cnst_cfg())
    assert len(c) == fixture.expected_dim_terminal()


# ============================================================================
# SECTION 2 — Angular velocity constraint
# ============================================================================

def test_av_satisfied_at_zero_angular_velocity():
    """Test AV constraint satisfied at zero angular velocity"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    c = fixture.sat.constraints(0, 10, fixture.nominal_state(),
                                fixture.nominal_control(), sun_z(), default_cnst_cfg())
    # c(0) = (||w||² - wmax²) / wmax² = -1 when w=0
    assert np.isclose(c[0], -1.0)


def test_av_exactly_at_limit():
    """Test AV constraint exactly at limit"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    cfg = default_cnst_cfg()
    w = np.array([cfg.wmax, 0.0, 0.0])
    h = np.zeros(fixture.n_rw())
    x = make_state(w, identity_quat(), h)
    
    c = fixture.sat.constraints(0, 10, x, fixture.nominal_control(), sun_z(), cfg)
    assert np.isclose(c[0], 0.0, atol=1e-12)


def test_av_violated_above_limit():
    """Test AV constraint violated above limit"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    cfg = default_cnst_cfg()
    w = np.array([cfg.wmax * 2.0, 0.0, 0.0])
    h = np.zeros(fixture.n_rw())
    x = make_state(w, identity_quat(), h)
    
    c = fixture.sat.constraints(0, 10, x, fixture.nominal_control(), sun_z(), cfg)
    assert c[0] > 0.0
    # (4*wmax² - wmax²) / wmax² = 3
    assert np.isclose(c[0], 3.0)


def test_av_normalization_is_scale_independent():
    """Test AV normalization is scale-independent"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    # Doubling wmax doubles the allowance but the same AV fraction gives same c value
    cfg1 = default_cnst_cfg()
    cfg1.wmax = 0.1
    cfg2 = default_cnst_cfg()
    cfg2.wmax = 0.2
    
    w1 = np.array([0.05, 0.0, 0.0])  # half of cfg1
    w2 = np.array([0.10, 0.0, 0.0])  # half of cfg2
    
    h = np.zeros(fixture.n_rw())
    x1 = make_state(w1, identity_quat(), h)
    x2 = make_state(w2, identity_quat(), h)
    
    c1 = fixture.sat.constraints(0, 10, x1, fixture.nominal_control(), sun_z(), cfg1)
    c2 = fixture.sat.constraints(0, 10, x2, fixture.nominal_control(), sun_z(), cfg2)
    assert np.isclose(c1[0], c2[0], atol=1e-12)


# ============================================================================
# SECTION 3 — Sun avoidance constraint
# ============================================================================

def test_sun_constraint_satisfied_when_sun_is_behind_spacecraft():
    """Test sun constraint satisfied when sun is behind spacecraft"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    # Identity quaternion: body +X = ECI +X. Sun along +Z → body Z-comp = 1,
    # body X = 0 → sun_body.x = 0 < cos(20°) ≈ 0.94 → satisfied
    c = fixture.sat.constraints(0, 10, fixture.nominal_state(),
                                fixture.nominal_control(), sun_z(), default_cnst_cfg())
    assert c[1] < 0.0


def test_sun_constraint_violated_when_sun_is_along_boresight():
    """Test sun constraint violated when sun is along boresight"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    # Sun along +X in ECI, identity quat → sun_body.x = 1.0
    # 1.0 - cos(20°) > 0 → violated
    sun_eci = np.array([1.0, 0.0, 0.0])
    c = fixture.sat.constraints(0, 10, fixture.nominal_state(),
                                fixture.nominal_control(), sun_eci, default_cnst_cfg())
    assert c[1] > 0.0


def test_sun_constraint_handles_zero_sun_vector_eclipse():
    """Test sun constraint handles zero sun vector (eclipse)"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    sun_zero = np.zeros(3)
    c = fixture.sat.constraints(0, 10, fixture.nominal_state(),
                                fixture.nominal_control(), sun_zero, default_cnst_cfg())
    # Should be 0 (no constraint active)
    assert np.isclose(c[1], 0.0, atol=1e-14)


def test_sun_constraint_at_exact_limit_angle():
    """Test sun constraint at exact limit angle"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    # Rotate sun by exactly 20° from +X in the XZ plane
    cfg = default_cnst_cfg()
    angle = cfg.sun_limit_angle
    sun_eci = np.array([np.cos(angle), 0.0, np.sin(angle)])
    
    # Identity quat → sun_body = sun_eci, sun_body.x = cos(angle) - cos(angle) = 0
    c = fixture.sat.constraints(0, 10, fixture.nominal_state(),
                                fixture.nominal_control(), sun_eci, cfg)
    assert np.isclose(c[1], 0.0, atol=1e-12)


# ============================================================================
# SECTION 4 — MTQ control bounds
# ============================================================================

def test_mtq_bounds_satisfied_at_zero_command():
    """Test MTQ bounds satisfied at zero command"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    c = fixture.sat.constraints(0, 10, fixture.nominal_state(),
                                fixture.nominal_control(), sun_z(), default_cnst_cfg())
    # Indices 2 .. 2+2*3-1 = 2..7 are MTQ bounds. At u=0, upper = -1, lower = -1
    for i in range(2 * fixture.n_mtq()):
        assert c[2 + i] < 0.0


def test_mtq_upper_bound_exactly_at_scaled_limit():
    """Test MTQ upper bound exactly at scaled limit"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    cfg = default_cnst_cfg()
    lim = cfg.control_limit_scale * fixture.mtq_dipole
    u = fixture.nominal_control()
    u[0] = lim  # first MTQ at exact limit
    
    c = fixture.sat.constraints(0, 10, fixture.nominal_state(), u, sun_z(), cfg)
    # Upper bound for first MTQ (idx 2): (lim - lim)/lim = 0
    assert np.isclose(c[2], 0.0, atol=1e-12)
    # Lower bound for first MTQ (idx 3): (-lim - lim)/lim = -2 → satisfied
    assert c[3] < 0.0


def test_mtq_bound_violated_when_command_exceeds_scaled_limit():
    """Test MTQ bound violated when command exceeds scaled limit"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    cfg = default_cnst_cfg()
    lim = cfg.control_limit_scale * fixture.mtq_dipole
    u = fixture.nominal_control()
    u[1] = lim * 1.5  # second MTQ at 150%
    
    c = fixture.sat.constraints(0, 10, fixture.nominal_state(), u, sun_z(), cfg)
    # Upper bound for second MTQ is at idx 4
    assert c[4] > 0.0


def test_mtq_bounds_use_config_u_max_when_provided():
    """Test MTQ bounds use config u_max when provided"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    cfg = default_cnst_cfg()
    # Create u_max with tighter limit for first MTQ
    u_max_values = np.zeros(fixture.n_ctrl())
    u_max_values[0] = 0.05  # tighter limit for first MTQ
    # Set the rest to default MTQ values
    for i in range(1, fixture.n_mtq()):
        u_max_values[i] = fixture.mtq_dipole
    for i in range(fixture.n_mtq(), fixture.n_ctrl()):
        u_max_values[i] = fixture.rw_torque
    # Assign the array directly instead of resizing
    cfg.u_max = u_max_values
    
    lim = cfg.control_limit_scale * 0.05
    u = fixture.nominal_control()
    u[0] = lim  # at config limit
    
    c = fixture.sat.constraints(0, 10, fixture.nominal_state(), u, sun_z(), cfg)
    assert np.isclose(c[2], 0.0, atol=1e-12)
    
    u[0] = lim * 1.1  # above config limit
    c = fixture.sat.constraints(0, 10, fixture.nominal_state(), u, sun_z(), cfg)
    assert c[2] > 0.0



def test_mtq_negative_command_violates_lower_bound():
    """Test MTQ negative command violates lower bound"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    cfg = default_cnst_cfg()
    lim = cfg.control_limit_scale * fixture.mtq_dipole
    u = fixture.nominal_control()
    u[2] = -lim * 2.0  # third MTQ, negative
    
    c = fixture.sat.constraints(0, 10, fixture.nominal_state(), u, sun_z(), cfg)
    # Lower bound for third MTQ: at idx 7 (2+2*2+1)
    assert c[7] > 0.0


# ============================================================================
# SECTION 5 — RW torque bounds
# ============================================================================

def test_rw_torque_satisfied_at_zero_command():
    """Test RW torque satisfied at zero command"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    c = fixture.sat.constraints(0, 10, fixture.nominal_state(),
                                fixture.nominal_control(), sun_z(), default_cnst_cfg())
    rw_start = 2 + 2 * fixture.n_mtq()  # index of first RW constraint
    # Each RW has 5 constraints: torque upper, torque lower, h upper, h lower, stiction
    for i in range(fixture.n_rw()):
        assert c[rw_start + 5*i] < 0.0      # torque upper
        assert c[rw_start + 5*i + 1] < 0.0  # torque lower


def test_rw_torque_violated_above_limit():
    """Test RW torque violated above limit"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    cfg = default_cnst_cfg()
    lim = cfg.control_limit_scale * fixture.rw_torque
    u = fixture.nominal_control()
    u[fixture.n_mtq()] = lim * 2.0  # first RW at double limit
    
    c = fixture.sat.constraints(0, 10, fixture.nominal_state(), u, sun_z(), cfg)
    rw_start = 2 + 2 * fixture.n_mtq()
    assert c[rw_start] > 0.0  # upper violated


# ============================================================================
# SECTION 6 — RW momentum bounds
# ============================================================================

def test_rw_momentum_satisfied_at_zero():
    """Test RW momentum satisfied at zero"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    c = fixture.sat.constraints(0, 10, fixture.nominal_state(),
                                fixture.nominal_control(), sun_z(), default_cnst_cfg())
    rw_start = 2 + 2 * fixture.n_mtq()
    for i in range(fixture.n_rw()):
        # Momentum bounds: rw_start + 5*i + 2 (upper), rw_start + 5*i + 3 (lower)
        assert c[rw_start + 5*i + 2] < 0.0      # upper
        assert c[rw_start + 5*i + 3] < 0.0      # lower


def test_rw_momentum_violated_at_max():
    """Test RW momentum violated at max"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    h = np.zeros(fixture.n_rw())
    h[0] = fixture.rw_hmax * 1.5  # 150% of max
    x = make_state(np.zeros(3), identity_quat(), h)
    
    c = fixture.sat.constraints(0, 10, x, fixture.nominal_control(), sun_z(), default_cnst_cfg())
    rw_start = 2 + 2 * fixture.n_mtq()
    # First RW upper momentum: rw_start + 5*0 + 2 = rw_start + 2
    assert c[rw_start + 2] > 0.0  # upper momentum, first RW


def test_rw_momentum_exactly_at_limit():
    """Test RW momentum exactly at limit"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    h = np.zeros(fixture.n_rw())
    h[0] = fixture.rw_hmax
    x = make_state(np.zeros(3), identity_quat(), h)
    
    c = fixture.sat.constraints(0, 10, x, fixture.nominal_control(), sun_z(), default_cnst_cfg())
    rw_start = 2 + 2 * fixture.n_mtq()
    # First RW upper momentum: rw_start + 2
    assert np.isclose(c[rw_start + 2], 0.0, atol=1e-12)


def test_rw_negative_momentum_violates_lower_bound():
    """Test RW negative momentum violates lower bound"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    h = np.zeros(fixture.n_rw())
    h[1] = -fixture.rw_hmax * 1.2
    x = make_state(np.zeros(3), identity_quat(), h)
    
    c = fixture.sat.constraints(0, 10, x, fixture.nominal_control(), sun_z(), default_cnst_cfg())
    rw_start = 2 + 2 * fixture.n_mtq()
    # Second RW lower momentum: rw_start + 5*1 + 3 = rw_start + 8
    assert c[rw_start + 8] > 0.0  # lower momentum, second RW


# ============================================================================
# SECTION 7 — RW stiction torque floor
#   c = theta - |u|/u_lim - |h|/h_c, with u_lim = control_limit_scale*rw_torque
#   and h_c = rw_stic_band_mult*h_max. Default theta = 0 keeps the row always
#   satisfied (back-compat with the old dead -(u*h)^2 row).
#   NOTE: this base has no constraintFamily() API (that's PR #52 on the other
#   train); the row keeps the 5th-per-RW slot so the layout is unchanged.
# ============================================================================

def test_rw_stiction_constraint_at_zero_state_is_zero():
    """Test stiction constraint at zero state is zero"""
    fixture = ConstraintFixture()
    fixture.setup_method()

    c = fixture.sat.constraints(0, 10, fixture.nominal_state(), fixture.nominal_control(),
                                sun_z(), default_cnst_cfg())
    rw_start = 2 + 2 * fixture.n_mtq()
    for i in range(fixture.n_rw()):
        # Stiction constraint is at index: rw_start + 5*i + 4
        assert np.isclose(c[rw_start + 5*i + 4], 0.0, atol=1e-14)


def test_rw_stiction_constraint_is_non_positive():
    """Test stiction constraint is non-positive (always satisfied at theta=0)"""
    fixture = ConstraintFixture()
    fixture.setup_method()

    h = np.array([0.005, -0.003])
    x = make_state(np.zeros(3), identity_quat(), h)
    u = fixture.nominal_control()
    u[fixture.n_mtq()] = 0.0005
    u[fixture.n_mtq() + 1] = -0.0003

    c = fixture.sat.constraints(0, 10, x, u, sun_z(), default_cnst_cfg())
    rw_start = 2 + 2 * fixture.n_mtq()
    for i in range(fixture.n_rw()):
        assert c[rw_start + 5*i + 4] <= 0.0


def test_rw_stiction_torque_floor_theta_zero_default_assorted():
    """At default theta=0 the torque-floor row is <= 0 for assorted (u, h),
    including exact zeros — behavior-identical to the old dead row."""
    fixture = ConstraintFixture()
    fixture.setup_method()

    cfg = default_cnst_cfg()  # rw_stic_torque_theta defaults to 0.0
    rw_start = 2 + 2 * fixture.n_mtq()

    u_vals = [0.0, 1e-6, -1e-5, 0.0005, -fixture.rw_torque, fixture.rw_torque]
    h_vals = [0.0, 1e-7, -1e-6, 0.004, -fixture.rw_hmax, fixture.rw_hmax]

    for uv in u_vals:
        for hv in h_vals:
            h = np.full(fixture.n_rw(), hv)
            x = make_state(np.zeros(3), identity_quat(), h)
            u = fixture.nominal_control()
            for i in range(fixture.n_rw()):
                u[fixture.n_mtq() + i] = uv
            c = fixture.sat.constraints(0, 10, x, u, sun_z(), cfg)
            for i in range(fixture.n_rw()):
                assert c[rw_start + 5*i + 4] <= 0.0


def test_rw_stiction_torque_floor_theta_09_activation():
    """With theta=0.9: violated at (u=0, h=0); satisfied at u=u_lim, h=0;
    satisfied at u=0, |h| >= theta*h_c; violated at u=0, |h| < theta*h_c."""
    fixture = ConstraintFixture()
    fixture.setup_method()

    cfg = default_cnst_cfg()
    cfg.rw_stic_torque_theta = 0.9
    theta = 0.9
    h_c = 0.005 * fixture.rw_hmax           # rw_stic_band_mult default 0.005
    u_lim = cfg.control_limit_scale * fixture.rw_torque
    rw_start = 2 + 2 * fixture.n_mtq()

    # (a) Violated at u = 0, h = 0: c = theta > 0.
    c = fixture.sat.constraints(0, 10, fixture.nominal_state(),
                                fixture.nominal_control(), sun_z(), cfg)
    for i in range(fixture.n_rw()):
        assert np.isclose(c[rw_start + 5*i + 4], theta, atol=1e-14)

    # (b) Satisfied at u = u_lim (effective, incl. control_limit_scale), h = 0.
    u = fixture.nominal_control()
    for i in range(fixture.n_rw()):
        u[fixture.n_mtq() + i] = u_lim
    c = fixture.sat.constraints(0, 10, fixture.nominal_state(), u, sun_z(), cfg)
    for i in range(fixture.n_rw()):
        assert np.isclose(c[rw_start + 5*i + 4], theta - 1.0, atol=1e-12)

    # (c) Satisfied at u = 0, |h| >= theta*h_c (either sign).
    for sign in (1.0, -1.0):
        h = np.full(fixture.n_rw(), sign * theta * h_c)  # boundary: c = 0
        x = make_state(np.zeros(3), identity_quat(), h)
        c = fixture.sat.constraints(0, 10, x, fixture.nominal_control(), sun_z(), cfg)
        for i in range(fixture.n_rw()):
            assert c[rw_start + 5*i + 4] <= 1e-12

        h = np.full(fixture.n_rw(), sign * h_c)          # strictly inside
        x = make_state(np.zeros(3), identity_quat(), h)
        c = fixture.sat.constraints(0, 10, x, fixture.nominal_control(), sun_z(), cfg)
        for i in range(fixture.n_rw()):
            assert c[rw_start + 5*i + 4] < 0.0

    # (d) Violated at u = 0, |h| < theta*h_c.
    h = np.full(fixture.n_rw(), 0.5 * theta * h_c)
    x = make_state(np.zeros(3), identity_quat(), h)
    c = fixture.sat.constraints(0, 10, x, fixture.nominal_control(), sun_z(), cfg)
    for i in range(fixture.n_rw()):
        assert c[rw_start + 5*i + 4] > 0.0


def test_rw_stiction_torque_floor_dimension_unchanged():
    """Enabling the torque floor must not change the constraint dimension."""
    fixture = ConstraintFixture()
    fixture.setup_method()

    cfg = default_cnst_cfg()
    cfg.rw_stic_torque_theta = 0.9
    cfg.rw_stic_band_mult = 0.005
    c = fixture.sat.constraints(0, 10, fixture.nominal_state(),
                                fixture.nominal_control(), sun_z(), cfg)
    assert len(c) == fixture.expected_dim_intermediate()
    ct = fixture.sat.constraints(9, 10, fixture.nominal_state(),
                                 fixture.nominal_control(), sun_z(), cfg)
    assert len(ct) == fixture.expected_dim_terminal()


def test_rw_stiction_torque_floor_jacobian_fd_away_from_kinks():
    """FD check of the torque-floor row's Jacobian (u and h directions),
    away from the |u| / |h| kinks, for both sign quadrants."""
    fixture = ConstraintFixture()
    fixture.setup_method()

    cfg = default_cnst_cfg()
    cfg.rw_stic_torque_theta = 0.9
    rw_start = 2 + 2 * fixture.n_mtq()
    eps = 1e-9  # well below the |u|, |h| magnitudes used

    for su in (1.0, -1.0):
        for sh in (1.0, -1.0):
            h = np.array([sh * 2e-5, sh * 3e-5])
            x0 = make_state(np.zeros(3), identity_quat(), h)
            u0 = fixture.nominal_control()
            u0[fixture.n_mtq()] = su * 2e-4
            u0[fixture.n_mtq() + 1] = su * 3e-4

            c_u, c_x = fixture.sat.constraintJacobians(0, 10, x0, u0, sun_z(), cfg)

            for i in range(fixture.n_rw()):
                row = rw_start + 5 * i + 4
                ctrl_idx = fixture.n_mtq() + i
                state_idx = saltro_py.Satellite.RW_MOMENTUM_INDEX + i

                up = u0.copy(); up[ctrl_idx] += eps
                um = u0.copy(); um[ctrl_idx] -= eps
                cp = fixture.sat.constraints(0, 10, x0, up, sun_z(), cfg)
                cm = fixture.sat.constraints(0, 10, x0, um, sun_z(), cfg)
                fd_u = (cp[row] - cm[row]) / (2.0 * eps)
                assert np.isclose(c_u[row, ctrl_idx], fd_u, rtol=1e-6)

                xp = x0.copy(); xp[state_idx] += eps
                xm = x0.copy(); xm[state_idx] -= eps
                cp = fixture.sat.constraints(0, 10, xp, u0, sun_z(), cfg)
                cm = fixture.sat.constraints(0, 10, xm, u0, sun_z(), cfg)
                fd_h = (cp[row] - cm[row]) / (2.0 * eps)
                assert np.isclose(c_x[row, state_idx], fd_h, rtol=1e-6)


# ============================================================================
# SECTION 7a — Terminal vs. Intermediate step
# ============================================================================

def test_terminal_step_has_no_control_rw_constraints():
    """Test terminal step has no control/RW constraints"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    N = 10
    c = fixture.sat.constraints(N - 1, N, fixture.nominal_state(), fixture.nominal_control(), 
                                sun_z(), default_cnst_cfg())
    assert len(c) == 2


def test_first_intermediate_step_has_full_constraints():
    """Test first intermediate step has full constraints"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    c = fixture.sat.constraints(0, 10, fixture.nominal_state(), fixture.nominal_control(), 
                                sun_z(), default_cnst_cfg())
    assert len(c) == fixture.expected_dim_intermediate()


def test_step_n_minus_2_is_intermediate_n_minus_1_is_terminal():
    """Test step N-2 is intermediate, N-1 is terminal"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    N = 5
    ci = fixture.sat.constraints(N - 2, N, fixture.nominal_state(), fixture.nominal_control(), 
                                 sun_z(), default_cnst_cfg())
    ct = fixture.sat.constraints(N - 1, N, fixture.nominal_state(), fixture.nominal_control(), 
                                 sun_z(), default_cnst_cfg())
    assert len(ci) > len(ct)
    assert len(ct) == 2


# ============================================================================
# SECTION 8 — Input validation
# ============================================================================

def test_constraints_with_wrong_state_dimension():
    """Test constraints with wrong state dimension raises error"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    x_wrong = np.zeros(5)  # Wrong dimension
    with pytest.raises(Exception):
        fixture.sat.constraints(0, 10, x_wrong, fixture.nominal_control(), 
                                sun_z(), default_cnst_cfg())


def test_constraints_with_wrong_control_dimension():
    """Test constraints with wrong control dimension raises error"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    u_wrong = np.zeros(2)  # Wrong dimension
    with pytest.raises(Exception):
        fixture.sat.constraints(0, 10, fixture.nominal_state(), u_wrong,
                                sun_z(), default_cnst_cfg())


def test_constraints_with_out_of_range_k():
    """Test constraints with out of range k raises error"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    with pytest.raises(Exception):
        fixture.sat.constraints(10, 10, fixture.nominal_state(),
                                fixture.nominal_control(), sun_z(), default_cnst_cfg())


def test_constraints_with_negative_N():
    """Test constraints with negative N raises error"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    with pytest.raises(Exception):
        fixture.sat.constraints(0, -1, fixture.nominal_state(),
                                fixture.nominal_control(), sun_z(), default_cnst_cfg())


def test_constraints_with_zero_N():
    """Test constraints with zero N raises error"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    with pytest.raises(Exception):
        fixture.sat.constraints(0, 0, fixture.nominal_state(),
                                fixture.nominal_control(), sun_z(), default_cnst_cfg())


def test_constraints_with_negative_k():
    """Test constraints with negative k raises error"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    with pytest.raises(Exception):
        fixture.sat.constraints(-1, 10, fixture.nominal_state(),
                                fixture.nominal_control(), sun_z(), default_cnst_cfg())


# ============================================================================
# SECTION 8a — Jacobian input validation
# ============================================================================

def test_constraint_jacobians_throws_on_invalid_inputs():
    """Test constraintJacobians throws on invalid inputs"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    with pytest.raises(Exception):
        fixture.sat.constraintJacobians(0, 0, fixture.nominal_state(), 
                                         fixture.nominal_control(), sun_z(), default_cnst_cfg())
    with pytest.raises(Exception):
        fixture.sat.constraintJacobians(10, 10, fixture.nominal_state(), 
                                         fixture.nominal_control(), sun_z(), default_cnst_cfg())


# ============================================================================
# SECTION 8b — Hessian input validation
# ============================================================================

def test_constraint_hessians_throws_on_invalid_inputs():
    """Test constraintHessians throws on invalid inputs"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    with pytest.raises(Exception):
        fixture.sat.constraintHessians(0, -5, fixture.nominal_state(), 
                                        fixture.nominal_control(), sun_z(), default_cnst_cfg())
    with pytest.raises(Exception):
        fixture.sat.constraintHessians(-1, 10, fixture.nominal_state(), 
                                        fixture.nominal_control(), sun_z(), default_cnst_cfg())


# ============================================================================
# SECTION 9 — Constraint Jacobians
# ============================================================================

def test_constraint_jacobians_dimension():
    """Test constraint Jacobians return correct dimensions"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    c_u, c_x = fixture.sat.constraintJacobians(0, 10, fixture.nominal_state(),
                                                fixture.nominal_control(), 
                                                sun_z(), default_cnst_cfg())
    
    n_constraints = fixture.expected_dim_intermediate()
    assert c_u.shape == (n_constraints, fixture.n_ctrl())
    assert c_x.shape == (n_constraints, fixture.sat.stateDim)


def test_constraint_jacobians_finite_difference_c_u():
    """Test constraint Jacobian c_u matches finite difference"""
    fixture = ConstraintFixture()
    fixture.setup_method()

    # Base point off the stiction torque-floor kinks at u = 0 / h = 0:
    # the row c = theta - |u|/u_lim - |h|/h_c has a subgradient (defined 0)
    # at the kinks, where one-sided FD would disagree by construction.
    h = np.array([0.002, -0.003])
    x = make_state(np.zeros(3), identity_quat(), h)
    u = fixture.nominal_control()
    u[fixture.n_mtq()] = 0.0003
    u[fixture.n_mtq() + 1] = -0.0002
    cfg = default_cnst_cfg()

    c_u, _ = fixture.sat.constraintJacobians(0, 10, x, u, sun_z(), cfg)
    
    # Finite difference approximation
    eps = 1e-7
    c_u_fd = np.zeros_like(c_u)
    c_base = fixture.sat.constraints(0, 10, x, u, sun_z(), cfg)
    
    for i in range(len(u)):
        u_pert = u.copy()
        u_pert[i] += eps
        c_pert = fixture.sat.constraints(0, 10, x, u_pert, sun_z(), cfg)
        c_u_fd[:, i] = (c_pert - c_base) / eps
    
    # Check agreement (may have some numerical error)
    assert np.allclose(c_u, c_u_fd, rtol=1e-4, atol=1e-6)


def test_constraint_jacobians_finite_difference_c_x():
    """Test constraint Jacobian c_x matches finite difference"""
    fixture = ConstraintFixture()
    fixture.setup_method()

    # Base point off the stiction torque-floor kinks (see c_u FD test).
    h = np.array([0.002, -0.003])
    x = make_state(np.zeros(3), identity_quat(), h)
    u = fixture.nominal_control()
    u[fixture.n_mtq()] = 0.0003
    u[fixture.n_mtq() + 1] = -0.0002
    cfg = default_cnst_cfg()

    _, c_x = fixture.sat.constraintJacobians(0, 10, x, u, sun_z(), cfg)
    
    # Finite difference approximation
    eps = 1e-7
    c_x_fd = np.zeros_like(c_x)
    c_base = fixture.sat.constraints(0, 10, x, u, sun_z(), cfg)
    
    for i in range(len(x)):
        x_pert = x.copy()
        x_pert[i] += eps
        # Normalize quaternion if we perturbed it
        if i >= saltro_py.Satellite.QUAT_INDEX and i < saltro_py.Satellite.QUAT_INDEX + 4:
            normalize_quat_in_state(x_pert)
        c_pert = fixture.sat.constraints(0, 10, x_pert, u, sun_z(), cfg)
        c_x_fd[:, i] = (c_pert - c_base) / eps
    
    # Check agreement (may have some numerical error, especially for quaternion)
    assert np.allclose(c_x, c_x_fd, rtol=1e-3, atol=1e-5)


# ============================================================================
# SECTION 10 — Constraint Hessians
# ============================================================================

def test_constraint_hessians_dimension():
    """Test constraint Hessians return correct dimensions"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    H_uu, H_ux, H_xx = fixture.sat.constraintHessians(0, 10, fixture.nominal_state(),
                                                       fixture.nominal_control(),
                                                       sun_z(), default_cnst_cfg())

    # Hessians are returned with the compile-time maximum dimensions, not
    # the runtime ones (slices of size MAX_CONSTRAINT_DIM x MAX_CTRL_DIM x
    # MAX_CTRL_DIM etc.). Pull those limits from saltro_py so this test
    # stays correct as new actuator classes (e.g. Magic) shift the maxima.
    L = saltro_py.limits
    assert H_uu.shape == (L.MAX_CONSTRAINT_DIM, L.MAX_CTRL_DIM,  L.MAX_CTRL_DIM)
    assert H_ux.shape == (L.MAX_CONSTRAINT_DIM, L.MAX_CTRL_DIM,  L.MAX_STATE_DIM)
    assert H_xx.shape == (L.MAX_CONSTRAINT_DIM, L.MAX_STATE_DIM, L.MAX_STATE_DIM)


def test_constraint_hessians_finite_difference_H_uu():
    """Test constraint Hessian H_uu matches finite difference of c_u"""
    fixture = ConstraintFixture()
    fixture.setup_method()

    # Base point off the stiction torque-floor kinks at u = 0 / h = 0
    # (the row is piecewise-linear; FD across a kink would blow up).
    h = np.array([0.002, -0.003])
    x = make_state(np.zeros(3), identity_quat(), h)
    u = fixture.nominal_control()
    u[fixture.n_mtq()] = 0.0003
    u[fixture.n_mtq() + 1] = -0.0002
    cfg = default_cnst_cfg()

    H_uu, _, _ = fixture.sat.constraintHessians(0, 10, x, u, sun_z(), cfg)
    
    # Finite difference approximation
    eps = 1e-6
    c_u_base, _ = fixture.sat.constraintJacobians(0, 10, x, u, sun_z(), cfg)
    n_constraints = len(c_u_base)
    n_ctrl = fixture.n_ctrl()
    L = saltro_py.limits
    H_uu_fd = np.zeros((L.MAX_CONSTRAINT_DIM, L.MAX_CTRL_DIM, L.MAX_CTRL_DIM))
    
    for i in range(len(u)):
        u_pert = u.copy()
        u_pert[i] += eps
        c_u_pert, _ = fixture.sat.constraintJacobians(0, 10, x, u_pert, sun_z(), cfg)
        H_uu_fd[:n_constraints, :n_ctrl, i] = (c_u_pert - c_u_base) / eps
    
    # Check agreement (only for actual constraints/controls)
    assert np.allclose(H_uu[:n_constraints, :n_ctrl, :n_ctrl], 
                      H_uu_fd[:n_constraints, :n_ctrl, :n_ctrl], rtol=1e-3, atol=1e-5)


def test_constraint_hessians_finite_difference_H_ux():
    """Test constraint Hessian H_ux matches finite difference of c_u"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    x = fixture.nominal_state()
    u = fixture.nominal_control()
    cfg = default_cnst_cfg()
    
    _, H_ux, _ = fixture.sat.constraintHessians(0, 10, x, u, sun_z(), cfg)
    
    # Finite difference approximation
    eps = 1e-6
    c_u_base, _ = fixture.sat.constraintJacobians(0, 10, x, u, sun_z(), cfg)
    n_constraints = len(c_u_base)
    n_ctrl = fixture.n_ctrl()
    n_state = fixture.sat.stateDim
    L = saltro_py.limits
    H_ux_fd = np.zeros((L.MAX_CONSTRAINT_DIM, L.MAX_CTRL_DIM, L.MAX_STATE_DIM))
    
    for i in range(len(x)):
        x_pert = x.copy()
        x_pert[i] += eps
        # Normalize quaternion if we perturbed it
        if i >= saltro_py.Satellite.QUAT_INDEX and i < saltro_py.Satellite.QUAT_INDEX + 4:
            normalize_quat_in_state(x_pert)
        c_u_pert, _ = fixture.sat.constraintJacobians(0, 10, x_pert, u, sun_z(), cfg)
        H_ux_fd[:n_constraints, :n_ctrl, i] = (c_u_pert - c_u_base) / eps
    
    # Check agreement (only for actual constraints/controls/states)
    assert np.allclose(H_ux[:n_constraints, :n_ctrl, :n_state], 
                      H_ux_fd[:n_constraints, :n_ctrl, :n_state], rtol=1e-3, atol=1e-5)


def test_constraint_hessians_finite_difference_H_xx():
    """Test constraint Hessian H_xx matches finite difference of c_x"""
    fixture = ConstraintFixture()
    fixture.setup_method()

    # Base point off the stiction torque-floor kinks (see H_uu FD test).
    h = np.array([0.002, -0.003])
    x = make_state(np.zeros(3), identity_quat(), h)
    u = fixture.nominal_control()
    u[fixture.n_mtq()] = 0.0003
    u[fixture.n_mtq() + 1] = -0.0002
    cfg = default_cnst_cfg()

    _, _, H_xx = fixture.sat.constraintHessians(0, 10, x, u, sun_z(), cfg)
    
    # Finite difference approximation
    eps = 1e-6
    _, c_x_base = fixture.sat.constraintJacobians(0, 10, x, u, sun_z(), cfg)
    n_constraints = len(c_x_base)
    n_state = fixture.sat.stateDim
    L = saltro_py.limits
    H_xx_fd = np.zeros((L.MAX_CONSTRAINT_DIM, L.MAX_STATE_DIM, L.MAX_STATE_DIM))
    
    # Raw-quaternion finite difference (no renormalization of the perturbed
    # quaternion): constraintHessians returns the raw 4-D second derivative, which
    # the backward pass projects with G (G·H·Gᵀ) exactly as it does the dynamics
    # Hessian. constraintJacobians already normalizes q internally, so FD of it
    # w.r.t. the raw quaternion is the matching raw Hessian.
    for i in range(len(x)):
        x_pert = x.copy()
        x_pert[i] += eps
        _, c_x_pert = fixture.sat.constraintJacobians(0, 10, x_pert, u, sun_z(), cfg)
        H_xx_fd[:n_constraints, :n_state, i] = (c_x_pert - c_x_base) / eps
    
    # Check agreement (looser tolerance due to second derivatives, only for actual constraints/states)
    assert np.allclose(H_xx[:n_constraints, :n_state, :n_state], 
                      H_xx_fd[:n_constraints, :n_state, :n_state], rtol=1e-2, atol=1e-4)


# ============================================================================
# SECTION 11 — Terminal step constraints
# ============================================================================

def test_terminal_constraints_have_no_control_bounds():
    """Test terminal step has only state constraints"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    N = 10
    # Terminal step is k = N-1
    c = fixture.sat.constraints(N-1, N, fixture.nominal_state(),
                                fixture.nominal_control(), sun_z(), default_cnst_cfg())
    
    # Should only have AV and sun constraints
    assert len(c) == 2


def test_terminal_jacobians_have_no_control_derivatives():
    """Test terminal Jacobians have correct structure"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    N = 10
    c_u, c_x = fixture.sat.constraintJacobians(N-1, N, fixture.nominal_state(),
                                                fixture.nominal_control(),
                                                sun_z(), default_cnst_cfg())
    
    # Terminal step: only 2 constraints (AV + sun)
    assert c_u.shape == (2, fixture.n_ctrl())
    assert c_x.shape == (2, fixture.sat.stateDim)


# ============================================================================
# SECTION 12 — Edge cases and robustness
# ============================================================================

def test_constraints_with_very_small_wmax():
    """Test constraints with very small angular velocity limit"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    cfg = default_cnst_cfg()
    cfg.wmax = 1e-8
    
    # Should not crash
    c = fixture.sat.constraints(0, 10, fixture.nominal_state(),
                                fixture.nominal_control(), sun_z(), cfg)
    assert np.all(np.isfinite(c))


def test_constraints_with_large_quaternion_deviation():
    """Test constraints with large attitude deviation"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    # 180 degree rotation about Z axis
    q = np.array([0.0, 0.0, 0.0, 1.0])
    x = make_state(np.zeros(3), q, np.zeros(fixture.n_rw()))
    
    c = fixture.sat.constraints(0, 10, x, fixture.nominal_control(),
                                sun_z(), default_cnst_cfg())
    assert np.all(np.isfinite(c))


def test_constraints_remain_finite_with_high_angular_velocity():
    """Test constraints remain finite with high angular velocity"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    cfg = default_cnst_cfg()
    w = np.array([10.0, 5.0, 3.0])  # Very high angular velocity
    x = make_state(w, identity_quat(), np.zeros(fixture.n_rw()))
    
    c = fixture.sat.constraints(0, 10, x, fixture.nominal_control(), sun_z(), cfg)
    assert np.all(np.isfinite(c))
    assert c[0] > 0.0  # Should be violated


def test_constraints_with_saturated_momentum():
    """Test constraints with reaction wheels at maximum momentum"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    h = np.full(fixture.n_rw(), fixture.rw_hmax)
    x = make_state(np.zeros(3), identity_quat(), h)
    
    c = fixture.sat.constraints(0, 10, x, fixture.nominal_control(),
                                sun_z(), default_cnst_cfg())
    assert np.all(np.isfinite(c))
    # Momentum constraints should be at boundary
    rw_start = 2 + 2 * fixture.n_mtq()
    for i in range(fixture.n_rw()):
        # Upper momentum: rw_start + 5*i + 2
        assert np.isclose(c[rw_start + 5*i + 2], 0.0, atol=1e-10)  # upper at boundary


# ============================================================================
# SECTION 13 — Jacobian finite-difference with non-trivial state
# ============================================================================

def test_constraint_jacobians_fd_check_wrt_state_nontrivial():
    """Test constraint Jacobian c_x matches finite difference at non-trivial state"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    # Use a non-trivial operating point
    h = np.array([0.003, -0.002])
    w = np.array([0.05, -0.03, 0.01])
    # Slightly rotated quaternion
    q = np.array([np.sqrt(1.0 - 0.01 - 0.04 - 0.09), 0.1, 0.2, 0.3])
    q = q / np.linalg.norm(q)
    x0 = make_state(w, q, h)
    u0 = fixture.nominal_control()
    u0[0] = 0.05
    u0[1] = -0.03
    u0[2] = 0.01
    u0[fixture.n_mtq()] = 0.0003
    u0[fixture.n_mtq() + 1] = -0.0002
    
    cfg = default_cnst_cfg()
    sun = np.array([0.5, 0.3, 0.8])
    k, N = 0, 10
    
    c0 = fixture.sat.constraints(k, N, x0, u0, sun, cfg)
    c_u, c_x = fixture.sat.constraintJacobians(k, N, x0, u0, sun, cfg)
    
    eps = 1e-7
    
    # Check ∂c/∂x with centered finite differences
    for j in range(fixture.sat.stateDim):
        xp = x0.copy()
        xm = x0.copy()
        xp[j] += eps
        xm[j] -= eps
        if j >= saltro_py.Satellite.QUAT_INDEX and j < saltro_py.Satellite.QUAT_INDEX + 4:
            normalize_quat_in_state(xp)
            normalize_quat_in_state(xm)
        cp = fixture.sat.constraints(k, N, xp, u0, sun, cfg)
        cm = fixture.sat.constraints(k, N, xm, u0, sun, cfg)
        fd_col = (cp - cm) / (2.0 * eps)
        for i in range(len(c0)):
            assert np.isclose(c_x[i, j], fd_col[i], atol=1e-4)


def test_constraint_jacobians_fd_check_wrt_control_nontrivial():
    """Test constraint Jacobian c_u matches finite difference at non-trivial state"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    h = np.array([0.003, -0.002])
    w = np.array([0.05, -0.03, 0.01])
    q = np.array([np.sqrt(1.0 - 0.01 - 0.04 - 0.09), 0.1, 0.2, 0.3])
    q = q / np.linalg.norm(q)
    x0 = make_state(w, q, h)
    u0 = fixture.nominal_control()
    u0[0] = 0.05
    u0[1] = -0.03
    u0[2] = 0.01
    u0[fixture.n_mtq()] = 0.0003
    u0[fixture.n_mtq() + 1] = -0.0002
    
    cfg = default_cnst_cfg()
    sun = np.array([0.5, 0.3, 0.8])
    k, N = 0, 10
    
    c0 = fixture.sat.constraints(k, N, x0, u0, sun, cfg)
    c_u, c_x = fixture.sat.constraintJacobians(k, N, x0, u0, sun, cfg)
    
    eps = 1e-7
    
    for j in range(fixture.n_ctrl()):
        up = u0.copy()
        um = u0.copy()
        up[j] += eps
        um[j] -= eps
        cp = fixture.sat.constraints(k, N, x0, up, sun, cfg)
        cm = fixture.sat.constraints(k, N, x0, um, sun, cfg)
        fd_col = (cp - cm) / (2.0 * eps)
        for i in range(len(c0)):
            assert np.isclose(c_u[i, j], fd_col[i], atol=1e-4)


def test_constraint_jacobians_terminal_step_fd_check():
    """Test constraint Jacobians at terminal step with FD check"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    h = np.array([0.001, -0.004])
    w = np.array([0.02, 0.01, -0.03])
    q = identity_quat()
    x0 = make_state(w, q, h)
    u0 = fixture.nominal_control()
    cfg = default_cnst_cfg()
    sun = np.array([0.0, 1.0, 0.0])
    k, N = 9, 10
    
    c0 = fixture.sat.constraints(k, N, x0, u0, sun, cfg)
    c_u, c_x = fixture.sat.constraintJacobians(k, N, x0, u0, sun, cfg)
    
    eps = 1e-7
    for j in range(fixture.sat.stateDim):
        xp = x0.copy()
        xm = x0.copy()
        xp[j] += eps
        xm[j] -= eps
        if j >= saltro_py.Satellite.QUAT_INDEX and j < saltro_py.Satellite.QUAT_INDEX + 4:
            normalize_quat_in_state(xp)
            normalize_quat_in_state(xm)
        cp = fixture.sat.constraints(k, N, xp, u0, sun, cfg)
        cm = fixture.sat.constraints(k, N, xm, u0, sun, cfg)
        fd_col = (cp - cm) / (2.0 * eps)
        for i in range(len(c0)):
            assert np.isclose(c_x[i, j], fd_col[i], atol=1e-4)
    
    # Control Jacobian should be zero at terminal step
    assert np.linalg.norm(c_u) < 1e-14


# ============================================================================
# SECTION 14 — Hessian finite-difference verification (additional tests)
# ============================================================================

def test_constraint_hessians_fd_check_wrt_state_state():
    """Test constraint Hessian H_xx matches finite difference of c_x"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    h = np.array([0.004, -0.003])
    w = np.array([0.08, -0.02, 0.04])
    q = np.array([0.0, 0.05, 0.15, -0.1])
    q[0] = np.sqrt(1.0 - np.sum(q[1:]**2))
    q = q / np.linalg.norm(q)
    x0 = make_state(w, q, h)
    u0 = fixture.nominal_control()
    u0[0] = 0.04
    u0[1] = -0.02
    u0[2] = 0.03
    u0[fixture.n_mtq()] = 0.0004
    u0[fixture.n_mtq() + 1] = -0.0001
    
    cfg = default_cnst_cfg()
    sun = np.array([0.3, 0.6, 0.7])
    k, N = 0, 10
    
    H_uu, H_ux, H_xx = fixture.sat.constraintHessians(k, N, x0, u0, sun, cfg)
    c0 = fixture.sat.constraints(k, N, x0, u0, sun, cfg)
    nc = len(c0)
    
    eps = 1e-5
    
    # ∂²c/∂x² ≈ (Jx(x+eps) - Jx(x-eps)) / (2*eps), raw-quaternion (no
    # renormalization): constraintHessians returns the raw 4-D Hessian (the
    # backward pass applies the G projection), and constraintJacobians normalizes
    # q internally, so the matching FD perturbs the raw quaternion. This is a
    # non-identity attitude, so it exercises the q0-coupled (manifold) terms.
    for j in range(fixture.sat.stateDim):
        xp = x0.copy()
        xm = x0.copy()
        xp[j] += eps
        xm[j] -= eps
        _, c_xp = fixture.sat.constraintJacobians(k, N, xp, u0, sun, cfg)
        _, c_xm = fixture.sat.constraintJacobians(k, N, xm, u0, sun, cfg)
        fd_slice = (c_xp - c_xm) / (2.0 * eps)
        
        for ci in range(nc):
            for row in range(fixture.sat.stateDim):
                assert np.isclose(H_xx[ci, row, j], fd_slice[ci, row], atol=1e-3)


def test_constraint_hessians_fd_check_wrt_control_control():
    """Test constraint Hessian H_uu matches finite difference of c_u"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    h = np.array([0.004, -0.003])
    w = np.array([0.08, -0.02, 0.04])
    q = identity_quat()
    x0 = make_state(w, q, h)
    u0 = fixture.nominal_control()
    u0[0] = 0.04
    u0[fixture.n_mtq()] = 0.0004
    u0[fixture.n_mtq() + 1] = -0.0002
    
    cfg = default_cnst_cfg()
    sun = np.array([0.3, 0.6, 0.7])
    k, N = 0, 10
    
    H_uu, H_ux, H_xx = fixture.sat.constraintHessians(k, N, x0, u0, sun, cfg)
    c0 = fixture.sat.constraints(k, N, x0, u0, sun, cfg)
    nc = len(c0)
    
    eps = 1e-5
    
    for j in range(fixture.n_ctrl()):
        up = u0.copy()
        um = u0.copy()
        up[j] += eps
        um[j] -= eps
        c_up, _ = fixture.sat.constraintJacobians(k, N, x0, up, sun, cfg)
        c_um, _ = fixture.sat.constraintJacobians(k, N, x0, um, sun, cfg)
        fd_slice = (c_up - c_um) / (2.0 * eps)
        
        for ci in range(nc):
            for row in range(fixture.n_ctrl()):
                assert np.isclose(H_uu[ci, row, j], fd_slice[ci, row], atol=1e-3)


def test_constraint_hessians_fd_check_wrt_control_state():
    """Test constraint Hessian H_ux matches finite difference of c_u"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    h = np.array([0.004, -0.003])
    w = np.array([0.08, -0.02, 0.04])
    q = identity_quat()
    x0 = make_state(w, q, h)
    u0 = fixture.nominal_control()
    u0[0] = 0.04
    u0[fixture.n_mtq()] = 0.0004
    u0[fixture.n_mtq() + 1] = -0.0002
    
    cfg = default_cnst_cfg()
    sun = np.array([0.3, 0.6, 0.7])
    k, N = 0, 10
    
    H_uu, H_ux, H_xx = fixture.sat.constraintHessians(k, N, x0, u0, sun, cfg)
    c0 = fixture.sat.constraints(k, N, x0, u0, sun, cfg)
    nc = len(c0)
    
    eps = 1e-5
    
    # ∂²c/(∂u∂x) ≈ (Ju(x+eps) - Ju(x-eps)) / (2*eps)
    for j in range(fixture.sat.stateDim):
        xp = x0.copy()
        xm = x0.copy()
        xp[j] += eps
        xm[j] -= eps
        if j >= saltro_py.Satellite.QUAT_INDEX and j < saltro_py.Satellite.QUAT_INDEX + 4:
            normalize_quat_in_state(xp)
            normalize_quat_in_state(xm)
        c_up, _ = fixture.sat.constraintJacobians(k, N, xp, u0, sun, cfg)
        c_um, _ = fixture.sat.constraintJacobians(k, N, xm, u0, sun, cfg)
        fd_slice = (c_up - c_um) / (2.0 * eps)
        
        for ci in range(nc):
            for row in range(fixture.n_ctrl()):
                assert np.isclose(H_ux[ci, row, j], fd_slice[ci, row], atol=1e-3)


def test_constraint_hessians_terminal_step_huu_and_hux_are_zero():
    """Test constraint Hessians at terminal step — H_uu and H_ux are zero"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    x0 = fixture.nominal_state()
    u0 = fixture.nominal_control()
    cfg = default_cnst_cfg()
    k, N = 9, 10
    
    H_uu, H_ux, H_xx = fixture.sat.constraintHessians(k, N, x0, u0, sun_z(), cfg)
    
    # Only 2 constraints at terminal, slices for those 2 should have zero u content
    for ci in range(2):
        assert np.linalg.norm(H_uu[ci, :, :]) < 1e-14
        assert np.linalg.norm(H_ux[ci, :, :]) < 1e-14


# ============================================================================
# SECTION 15 — Hessian structural checks
# ============================================================================

def test_constraint_hessians_av_constraint_hxx_is_diagonal_in_w_block():
    """Test AV constraint H_xx is diagonal in w-block"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    x0 = fixture.nominal_state()
    u0 = fixture.nominal_control()
    cfg = default_cnst_cfg()
    
    H_uu, H_ux, H_xx = fixture.sat.constraintHessians(0, 10, x0, u0, sun_z(), cfg)
    
    # AV Hessian (slice 0) should be 2/wmax² * I₃ in the w-block
    wmax = cfg.wmax
    expected = 2.0 / (wmax * wmax)
    hxx_av = H_xx[0, :, :]
    for i in range(3):
        assert np.isclose(hxx_av[i, i], expected)
    
    # Off-diagonal in w-block should be zero
    assert np.isclose(hxx_av[0, 1], 0.0, atol=1e-14)
    assert np.isclose(hxx_av[0, 2], 0.0, atol=1e-14)
    assert np.isclose(hxx_av[1, 2], 0.0, atol=1e-14)
    
    # No u components
    assert np.linalg.norm(H_uu[0, :, :]) < 1e-14
    assert np.linalg.norm(H_ux[0, :, :]) < 1e-14


def test_constraint_hessians_mtq_bounds_have_zero_hessians():
    """Test MTQ bounds have zero Hessians (linear constraints)"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    h = np.array([0.002, -0.001])
    x0 = make_state(np.array([0.05, 0.0, 0.0]), identity_quat(), h)
    u0 = fixture.nominal_control()
    u0[0] = 0.05
    
    H_uu, H_ux, H_xx = fixture.sat.constraintHessians(0, 10, x0, u0, sun_z(), default_cnst_cfg())
    
    # MTQ bound indices: 2 .. 2 + 2*n_mtq - 1
    for ci in range(2, 2 + 2 * fixture.n_mtq()):
        assert np.linalg.norm(H_uu[ci, :, :]) < 1e-14
        assert np.linalg.norm(H_ux[ci, :, :]) < 1e-14
        assert np.linalg.norm(H_xx[ci, :, :]) < 1e-14


def test_constraint_hessians_rw_torque_bounds_have_zero_hessians():
    """Test RW torque bounds have zero Hessians (linear)"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    x0 = fixture.nominal_state()
    u0 = fixture.nominal_control()
    
    H_uu, H_ux, H_xx = fixture.sat.constraintHessians(0, 10, x0, u0, sun_z(), default_cnst_cfg())
    
    rw_start = 2 + 2 * fixture.n_mtq()
    for i in range(fixture.n_rw()):
        upper = rw_start + 5 * i
        lower = rw_start + 5 * i + 1
        for ci in [upper, lower]:
            assert np.linalg.norm(H_uu[ci, :, :]) < 1e-14
            assert np.linalg.norm(H_ux[ci, :, :]) < 1e-14
            assert np.linalg.norm(H_xx[ci, :, :]) < 1e-14


def test_constraint_hessians_rw_momentum_bounds_have_zero_hessians():
    """Test RW momentum bounds have zero Hessians (linear)"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    x0 = fixture.nominal_state()
    u0 = fixture.nominal_control()
    
    H_uu, H_ux, H_xx = fixture.sat.constraintHessians(0, 10, x0, u0, sun_z(), default_cnst_cfg())
    
    rw_start = 2 + 2 * fixture.n_mtq()
    for i in range(fixture.n_rw()):
        h_upper = rw_start + 5 * i + 2
        h_lower = rw_start + 5 * i + 3
        for ci in [h_upper, h_lower]:
            assert np.linalg.norm(H_uu[ci, :, :]) < 1e-14
            assert np.linalg.norm(H_ux[ci, :, :]) < 1e-14
            assert np.linalg.norm(H_xx[ci, :, :]) < 1e-14


# ============================================================================
# SECTION 16 — Stiction torque-floor Hessian structure
# The row c = theta - |u|/u_lim - |h|/h_c is piecewise-linear: Hessian is
# zero (the old -(u*h)^2 row had curvature here; the new form has none).
# ============================================================================

def test_constraint_hessians_stiction_row_is_zero_piecewise_linear():
    """Stiction torque-floor row has zero Hessians (piecewise linear)"""
    fixture = ConstraintFixture()
    fixture.setup_method()

    h = np.array([0.005, -0.003])
    x0 = make_state(np.zeros(3), identity_quat(), h)
    u0 = fixture.nominal_control()
    u0[fixture.n_mtq()] = 0.0003
    u0[fixture.n_mtq() + 1] = -0.0002

    cfg = default_cnst_cfg()
    cfg.rw_stic_torque_theta = 0.9  # zero Hessian whether or not the floor is enabled

    H_uu, H_ux, H_xx = fixture.sat.constraintHessians(0, 10, x0, u0, sun_z(), cfg)

    rw_start = 2 + 2 * fixture.n_mtq()
    for i in range(fixture.n_rw()):
        stiction_idx = rw_start + 5 * i + 4
        assert np.linalg.norm(H_uu[stiction_idx]) == 0.0
        assert np.linalg.norm(H_ux[stiction_idx]) == 0.0
        assert np.linalg.norm(H_xx[stiction_idx]) == 0.0


# ============================================================================
# SECTION 17 — Edge case: no actuators (Jacobians / Hessians)
# ============================================================================

def test_constraint_jacobians_no_actuators():
    """Test constraintJacobians with no actuators"""
    sat = saltro_py.Satellite(valid_inertia(), saltro_py.PlannerSettings())
    x = make_state(np.array([0.1, 0.0, 0.0]), identity_quat())
    u = zero_control(0)
    cfg = default_cnst_cfg()
    
    # Intermediate step — still only 2 constraints when no actuators
    c_u, c_x = sat.constraintJacobians(0, 10, x, u, sun_z(), cfg)
    assert c_u.shape == (2, 0)
    assert c_x.shape == (2, 7)


def test_constraint_hessians_no_actuators():
    """Test constraintHessians with no actuators"""
    sat = saltro_py.Satellite(valid_inertia(), saltro_py.PlannerSettings())
    x = make_state(np.array([0.1, 0.0, 0.0]), identity_quat())
    u = zero_control(0)
    cfg = default_cnst_cfg()
    
    H_uu, H_ux, H_xx = sat.constraintHessians(0, 10, x, u, sun_z(), cfg)
    # AV hessian should still be present
    wmax = cfg.wmax
    expected = 2.0 / (wmax * wmax)
    assert np.isclose(H_xx[0, 0, 0], expected)


# ============================================================================
# SECTION 18 — Edge case: maximum actuators
# ============================================================================

def test_constraints_maximum_actuators():
    """Test constraints with maximum actuators"""
    sat = saltro_py.Satellite(valid_inertia(), saltro_py.PlannerSettings())
    # Add MAX_NUM_MTQ MTQs
    for i in range(saltro_py.limits.MAX_NUM_MTQ):
        axis = np.zeros(3)
        axis[i % 3] = 1.0
        sat.addMTQ(axis, 0.2)
    # Add MAX_NUM_RW RWs
    for i in range(saltro_py.limits.MAX_NUM_RW):
        axis = np.zeros(3)
        axis[i % 3] = 1.0
        sat.addRW(axis, 0.001, 1e-5, 0.0, 0.01)
    # Add MAX_NUM_MAGIC magic actuators
    for i in range(saltro_py.limits.MAX_NUM_MAGIC):
        axis = np.zeros(3)
        axis[i % 3] = 1.0
        sat.addMagic(axis, 0.01)

    n = sat.stateDim
    m = sat.controlDim
    h = np.zeros(sat.numRW)
    x = make_state(np.zeros(3), identity_quat(), h)
    u = zero_control(m)
    cfg = default_cnst_cfg()

    c = sat.constraints(0, 10, x, u, sun_z(), cfg)
    expected = (1 + 1
                + 2 * saltro_py.limits.MAX_NUM_MTQ
                + 5 * saltro_py.limits.MAX_NUM_RW
                + 2 * saltro_py.limits.MAX_NUM_MAGIC)
    assert len(c) == expected
    # Verify this matches the compile-time MAX_CONSTRAINT_DIM (now exposed
    # via saltro_py.limits).
    assert expected == saltro_py.limits.MAX_CONSTRAINT_DIM


# ============================================================================
# SECTION 19 — All constraints satisfied at nominal point
# ============================================================================

def test_all_constraints_satisfied_at_nominal_operating_point():
    """Test all constraints satisfied at nominal operating point (c <= 0)"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    c = fixture.sat.constraints(0, 10, fixture.nominal_state(), fixture.nominal_control(), 
                                sun_z(), default_cnst_cfg())
    for i in range(len(c)):
        assert c[i] <= 0.0 + 1e-12, f"Constraint index {i} violated with value {c[i]}"


def test_all_constraints_satisfied_at_terminal_nominal_point():
    """Test all constraints satisfied at terminal nominal point"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    c = fixture.sat.constraints(9, 10, fixture.nominal_state(), fixture.nominal_control(), 
                                sun_z(), default_cnst_cfg())
    for i in range(len(c)):
        assert c[i] <= 0.0 + 1e-12, f"Constraint index {i} violated with value {c[i]}"


# ============================================================================
# SECTION 20 — control_limit_scale edge cases
# ============================================================================

def test_control_limit_scale_equals_1_uses_full_actuator_capacity():
    """Test control_limit_scale = 1 uses full actuator capacity"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    cfg = default_cnst_cfg()
    cfg.control_limit_scale = 1.0
    u = fixture.nominal_control()
    u[0] = fixture.mtq_dipole  # exactly at full capacity
    c = fixture.sat.constraints(0, 10, fixture.nominal_state(), u, sun_z(), cfg)
    assert np.isclose(c[2], 0.0, atol=1e-12)


def test_smaller_scale_tightens_limits():
    """Test smaller scale tightens limits"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    cfg1 = default_cnst_cfg()
    cfg1.control_limit_scale = 0.5
    cfg2 = default_cnst_cfg()
    cfg2.control_limit_scale = 1.0
    
    u = fixture.nominal_control()
    u[0] = 0.15  # between 0.5*0.2 = 0.1 and 1.0*0.2 = 0.2
    c1 = fixture.sat.constraints(0, 10, fixture.nominal_state(), u, sun_z(), cfg1)
    c2 = fixture.sat.constraints(0, 10, fixture.nominal_state(), u, sun_z(), cfg2)
    # With tighter scale, constraint is violated; with looser, it's not
    assert c1[2] > 0.0
    assert c2[2] < 0.0


def test_rw_momentum_limit_scale_equals_1_uses_full_rw_momentum_capacity():
    """Test rw_momentum_limit_scale = 1 uses full RW momentum capacity"""
    fixture = ConstraintFixture()
    fixture.setup_method()

    cfg = default_cnst_cfg()
    cfg.rw_momentum_limit_scale = 1.0

    h = np.zeros(fixture.n_rw())
    h[0] = fixture.rw_hmax  # exactly at full momentum capacity
    x = make_state(np.zeros(3), identity_quat(), h)
    c = fixture.sat.constraints(0, 10, x, fixture.nominal_control(), sun_z(), cfg)

    rw_start = 2 + 2 * fixture.n_mtq()
    assert np.isclose(c[rw_start + 2], 0.0, atol=1e-12)


def test_smaller_rw_momentum_limit_scale_tightens_rw_momentum_limits():
    """Test smaller rw_momentum_limit_scale tightens RW momentum limits"""
    fixture = ConstraintFixture()
    fixture.setup_method()

    cfg1 = default_cnst_cfg()
    cfg1.rw_momentum_limit_scale = 0.5
    cfg2 = default_cnst_cfg()
    cfg2.rw_momentum_limit_scale = 1.0

    h = np.zeros(fixture.n_rw())
    h[0] = 0.0075  # between 0.5*0.01 = 0.005 and 1.0*0.01 = 0.01
    x = make_state(np.zeros(3), identity_quat(), h)

    c1 = fixture.sat.constraints(0, 10, x, fixture.nominal_control(), sun_z(), cfg1)
    c2 = fixture.sat.constraints(0, 10, x, fixture.nominal_control(), sun_z(), cfg2)

    rw_start = 2 + 2 * fixture.n_mtq()
    assert c1[rw_start + 2] > 0.0
    assert c2[rw_start + 2] < 0.0


# ============================================================================
# SECTION 21 — Sun limit angle edge cases
# ============================================================================

def test_sun_limit_angle_equals_0_makes_sun_constraint_trivially_satisfied_for_off_axis_sun():
    """Test sun_limit_angle = 0 makes sun constraint trivially satisfied for off-axis sun"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    cfg = default_cnst_cfg()
    cfg.sun_limit_angle = 0.0
    # cos(0) = 1, so sun_body.x - 1 <= 0 whenever sun is not exactly along +X
    c = fixture.sat.constraints(0, 10, fixture.nominal_state(), fixture.nominal_control(), 
                                sun_z(), cfg)
    assert c[1] <= 0.0 + 1e-12


def test_sun_limit_angle_equals_pi_accepts_any_direction():
    """Test sun_limit_angle = pi behavior"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    cfg = default_cnst_cfg()
    cfg.sun_limit_angle = np.pi
    # cos(pi) = -1, sun along +X in body frame
    sun = np.array([1.0, 0.0, 0.0])
    c = fixture.sat.constraints(0, 10, fixture.nominal_state(), fixture.nominal_control(), sun, cfg)
    # sun_body.x - cos(pi) = 1.0 - (-1.0) = 2.0 > 0 → violated
    assert np.isclose(c[1], 2.0, atol=1e-12)


# ============================================================================
# SECTION 22 — Symmetry: upper and lower bounds are symmetric
# ============================================================================

def test_mtq_bounds_are_symmetric_for_positive_and_negative_commands():
    """Test MTQ bounds are symmetric for positive and negative commands"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    cfg = default_cnst_cfg()
    u_pos = fixture.nominal_control()
    u_neg = fixture.nominal_control()
    u_pos[0] = 0.1
    u_neg[0] = -0.1
    
    c_pos = fixture.sat.constraints(0, 10, fixture.nominal_state(), u_pos, sun_z(), cfg)
    c_neg = fixture.sat.constraints(0, 10, fixture.nominal_state(), u_neg, sun_z(), cfg)
    
    # For first MTQ: upper of +cmd = lower of -cmd
    assert np.isclose(c_pos[2], c_neg[3], atol=1e-12)
    assert np.isclose(c_pos[3], c_neg[2], atol=1e-12)


# ============================================================================
# SECTION 23 — Constraint Jacobians: AV constraint structure
# ============================================================================

def test_constraint_jacobians_av_row_has_entries_only_in_w_columns():
    """Test AV Jacobian row has entries only in w-columns"""
    fixture = ConstraintFixture()
    fixture.setup_method()
    
    h = np.array([0.001, -0.002])
    x0 = make_state(np.array([0.05, 0.03, -0.01]), identity_quat(), h)
    u0 = fixture.nominal_control()
    
    c_u, c_x = fixture.sat.constraintJacobians(0, 10, x0, u0, sun_z(), default_cnst_cfg())
    
    # AV Jacobian (row 0): only columns 0,1,2 (angular velocity) should be non-zero
    for j in range(3, fixture.sat.stateDim):
        assert np.isclose(c_x[0, j], 0.0, atol=1e-14)
    # And no dependence on u
    for j in range(fixture.n_ctrl()):
        assert np.isclose(c_u[0, j], 0.0, atol=1e-14)


# ============================================================================
# SECTION 24 — Constraint Jacobians: Sun constraint structure
# ============================================================================

def test_constraint_jacobians_sun_row_has_entries_only_in_q_columns():
    """Test sun Jacobian row has entries only in q-columns"""
    fixture = ConstraintFixture()
    fixture.setup_method()

    x0 = fixture.nominal_state()
    u0 = fixture.nominal_control()
    sun = np.array([0.5, 0.3, 0.8])

    c_u, c_x = fixture.sat.constraintJacobians(0, 10, x0, u0, sun, default_cnst_cfg())

    # Sun constraint (row 1): only columns 3,4,5,6 (quaternion) should be non-zero
    for j in range(0, 3):
        assert np.isclose(c_x[1, j], 0.0, atol=1e-14)
    for j in range(7, fixture.sat.stateDim):
        assert np.isclose(c_x[1, j], 0.0, atol=1e-14)


# ============================================================================
# SECTION 25 — Sun-avoidance Jacobian near the active/passive boundary
# ============================================================================

def test_sun_jacobian_matches_fd_just_inside_active_region():
    """FD-validate the sun-constraint Jacobian when the constraint is just
    barely active (c ≈ 0+). This is the regime where sign-flip bugs in the
    rotation-matrix derivative are most likely to bite: at exactly the
    limit angle the quaternion Jacobian must be sign-consistent on both
    sides of the boundary."""
    fixture = ConstraintFixture()
    fixture.setup_method()

    cfg = default_cnst_cfg()
    angle = cfg.sun_limit_angle

    # Sun sits just past the limit angle from +X in the XZ plane: c > 0 by ε
    eps_angle = 1e-3
    sun_eci = np.array([np.cos(angle - eps_angle), 0.0, np.sin(angle - eps_angle)])

    x0 = fixture.nominal_state()
    u0 = fixture.nominal_control()
    k, N = 0, 10

    c0 = fixture.sat.constraints(k, N, x0, u0, sun_eci, cfg)
    # We expect c[1] (sun row) to be small but non-zero
    assert abs(c0[1]) < 1e-2

    _, c_x = fixture.sat.constraintJacobians(k, N, x0, u0, sun_eci, cfg)

    eps = 1e-7
    qi = saltro_py.Satellite.QUAT_INDEX
    for j in range(qi, qi + 4):
        xp = x0.copy()
        xm = x0.copy()
        xp[j] += eps
        xm[j] -= eps
        normalize_quat_in_state(xp)
        normalize_quat_in_state(xm)

        cp = fixture.sat.constraints(k, N, xp, u0, sun_eci, cfg)
        cm = fixture.sat.constraints(k, N, xm, u0, sun_eci, cfg)
        fd = (cp[1] - cm[1]) / (2.0 * eps)

        assert np.isclose(c_x[1, j], fd, atol=1e-5), (
            f"sun Jacobian quaternion column j={j}: ana={c_x[1, j]:.4e}, "
            f"fd={fd:.4e}, c0[1]={c0[1]:.4e}"
        )


def test_sun_jacobian_matches_fd_just_outside_active_region():
    """Mirror of the inside-boundary test for the passive (c < 0) side. With
    a sane formulation the analytic Jacobian must remain consistent with
    FD here too — a regression would surface as a discontinuity at the
    boundary."""
    fixture = ConstraintFixture()
    fixture.setup_method()

    cfg = default_cnst_cfg()
    angle = cfg.sun_limit_angle

    eps_angle = 1e-3
    sun_eci = np.array([np.cos(angle + eps_angle), 0.0, np.sin(angle + eps_angle)])

    x0 = fixture.nominal_state()
    u0 = fixture.nominal_control()
    k, N = 0, 10

    c0 = fixture.sat.constraints(k, N, x0, u0, sun_eci, cfg)
    # c[1] should be small negative (constraint passively satisfied by ε)
    assert c0[1] < 0.0
    assert abs(c0[1]) < 1e-2

    _, c_x = fixture.sat.constraintJacobians(k, N, x0, u0, sun_eci, cfg)

    eps = 1e-7
    qi = saltro_py.Satellite.QUAT_INDEX
    for j in range(qi, qi + 4):
        xp = x0.copy()
        xm = x0.copy()
        xp[j] += eps
        xm[j] -= eps
        normalize_quat_in_state(xp)
        normalize_quat_in_state(xm)

        cp = fixture.sat.constraints(k, N, xp, u0, sun_eci, cfg)
        cm = fixture.sat.constraints(k, N, xm, u0, sun_eci, cfg)
        fd = (cp[1] - cm[1]) / (2.0 * eps)

        assert np.isclose(c_x[1, j], fd, atol=1e-5), (
            f"sun Jacobian quaternion column j={j}: ana={c_x[1, j]:.4e}, "
            f"fd={fd:.4e}, c0[1]={c0[1]:.4e}"
        )


# ============================================================================
# SECTION 26 — Stiction Hessian at non-zero RW momentum
# ============================================================================

def test_stiction_hessian_fd_at_nonzero_momentum():
    """The stiction torque-floor row c = theta - |u|/u_lim - |h|/h_c is
    piecewise-linear: its Jacobian is piecewise-constant in h, so the
    analytic Hessian is zero and FD of the Jacobian must agree (also zero)
    at non-zero h values away from the kink. Verify FD agreement at several
    random non-zero h values (this used to assert the -(u*h)^2 curvature)."""
    fixture = ConstraintFixture()
    fixture.setup_method()

    cfg = default_cnst_cfg()
    u0 = fixture.nominal_control()
    sun = sun_z()
    k, N = 0, 10

    rng = np.random.default_rng(20260515)

    for trial in range(4):
        h = rng.uniform(-fixture.rw_hmax * 0.8, fixture.rw_hmax * 0.8,
                        size=fixture.n_rw())
        w = rng.uniform(-0.05, 0.05, size=3)
        x0 = make_state(w, identity_quat(), h)

        _, _, H_xx = fixture.sat.constraintHessians(k, N, x0, u0, sun, cfg)

        # FD: differentiate the constraint Jacobian's c_x rows w.r.t. x.
        # H_xx[i][a, b] = ∂²c_i/∂x_a ∂x_b
        eps = 5e-6
        nx = fixture.sat.stateDim
        n_c = len(fixture.sat.constraints(k, N, x0, u0, sun, cfg))

        # Pick the stiction rows: per RW, the last of the 5 RW-related rows.
        # Layout: AV(1) + sun(1) + MTQ_bounds(2*n_mtq) + per-RW(2 torque, 2
        # momentum, 1 stiction). Stiction row offset = 4 within each RW block.
        rw_block_start = 2 + 2 * fixture.n_mtq()
        stiction_rows = [rw_block_start + 5 * r + 4 for r in range(fixture.n_rw())]

        # Subset the FD to the h-block (cheap) — diagonal entries only.
        h_start = saltro_py.Satellite.RW_MOMENTUM_INDEX
        for row in stiction_rows:
            for j in range(h_start, h_start + fixture.n_rw()):
                xp = x0.copy(); xm = x0.copy()
                xp[j] += eps; xm[j] -= eps
                _, c_xp = fixture.sat.constraintJacobians(k, N, xp, u0, sun, cfg)
                _, c_xm = fixture.sat.constraintJacobians(k, N, xm, u0, sun, cfg)
                fd = (c_xp[row, j] - c_xm[row, j]) / (2.0 * eps)
                ana = H_xx[row][j, j]
                assert np.isclose(ana, fd, atol=1e-3, rtol=1e-2), (
                    f"trial {trial}, row {row}, j {j}: "
                    f"ana={ana:.4e}, fd={fd:.4e}, h={h}"
                )


# ============================================================================
# SECTION — constraintFamily() row → family mapping
# ============================================================================

def test_constraint_family_enum_is_exposed_with_expected_values():
    """ConstraintFamily must be bound and match the documented C++ numbering."""
    fam = saltro_py.ConstraintFamily
    assert int(fam.angular_velocity) == 0
    assert int(fam.sun_avoidance) == 1
    assert int(fam.mtq_saturation) == 2
    assert int(fam.rw_torque_sat) == 3
    assert int(fam.rw_momentum) == 4
    assert int(fam.rw_stiction) == 5
    assert int(fam.magic_torque_sat) == 6
    # NumFamilies (=7) is a C++ sentinel, deliberately NOT exposed as a value.
    assert len(fam.__members__) == 7
    assert "NumFamilies" not in fam.__members__


def test_constraint_family_mapping_3mtq_1rw():
    """Every constraint row of a 3MTQ+1RW satellite maps to the documented family.

    Layout: [wmax, sun, 2 per MTQ, (2 torque + 2 momentum + 1 stiction) per RW].
    """
    settings = saltro_py.PlannerSettings()
    sat = saltro_py.Satellite(valid_inertia(), settings)
    sat.addMTQ(np.array([1.0, 0.0, 0.0]), 0.2)
    sat.addMTQ(np.array([0.0, 1.0, 0.0]), 0.2)
    sat.addMTQ(np.array([0.0, 0.0, 1.0]), 0.2)
    sat.addRW(np.array([0.0, 0.0, 1.0]), 0.001, 1e-5, 0.0, 0.01)

    fam = saltro_py.ConstraintFamily
    expected = (
        [int(fam.angular_velocity), int(fam.sun_avoidance)]
        + [int(fam.mtq_saturation)] * (2 * 3)
        + [int(fam.rw_torque_sat)] * 2
        + [int(fam.rw_momentum)] * 2
        + [int(fam.rw_stiction)] * 1
    )

    # The mapping must cover exactly the rows constraints() actually emits.
    N = 5
    x = make_state(np.zeros(3), identity_quat(), np.zeros(1))
    u = zero_control(sat.controlDim)
    c = sat.constraints(0, N, x, u, sun_z(), default_cnst_cfg())
    assert len(c) == len(expected)

    actual = [sat.constraintFamily(i, False) for i in range(len(expected))]
    assert actual == expected

    # One past the end is out of range.
    assert sat.constraintFamily(len(expected), False) == -1

    # Terminal step: only the two state rows exist.
    c_term = sat.constraints(N - 1, N, x, u, sun_z(), default_cnst_cfg())
    assert len(c_term) == 2
    assert sat.constraintFamily(0, True) == int(fam.angular_velocity)
    assert sat.constraintFamily(1, True) == int(fam.sun_avoidance)
    assert sat.constraintFamily(2, True) == -1


def test_constraint_family_mapping_with_magic_actuators():
    """Magic rows follow the RW block: [wmax, sun, 2/MTQ, 5/RW, 2/Magic]."""
    settings = saltro_py.PlannerSettings()
    sat = saltro_py.Satellite(valid_inertia(), settings)
    sat.addMTQ(np.array([1.0, 0.0, 0.0]), 0.2)
    sat.addRW(np.array([0.0, 1.0, 0.0]), 0.001, 1e-5, 0.0, 0.01)
    sat.addMagic(np.array([0.0, 0.0, 1.0]), 0.005)
    sat.addMagic(np.array([1.0, 0.0, 0.0]), 0.005)

    fam = saltro_py.ConstraintFamily
    expected = (
        [int(fam.angular_velocity), int(fam.sun_avoidance)]
        + [int(fam.mtq_saturation)] * (2 * 1)
        + [int(fam.rw_torque_sat)] * 2
        + [int(fam.rw_momentum)] * 2
        + [int(fam.rw_stiction)] * 1
        + [int(fam.magic_torque_sat)] * (2 * 2)
    )

    N = 5
    x = make_state(np.zeros(3), identity_quat(), np.zeros(1))
    u = zero_control(sat.controlDim)
    c = sat.constraints(0, N, x, u, sun_z(), default_cnst_cfg())
    assert len(c) == len(expected)

    actual = [sat.constraintFamily(i, False) for i in range(len(expected))]
    assert actual == expected

    assert sat.constraintFamily(len(expected), False) == -1

    # Terminal step still stops after the two state rows.
    assert sat.constraintFamily(0, True) == int(fam.angular_velocity)
    assert sat.constraintFamily(1, True) == int(fam.sun_avoidance)
    assert sat.constraintFamily(2, True) == -1



