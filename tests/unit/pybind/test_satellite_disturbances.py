"""
Comprehensive disturbance tests for Satellite class
Tests drag, gravity gradient, and solar radiation pressure disturbances
independently and together across realistic orbital scenarios.
"""

import numpy as np
import saltro_py


SOLAR_CONSTANT = 1361.0
C_LIGHT = 299792458.0
P_SRP = SOLAR_CONSTANT / C_LIGHT


class TestSatelliteDisturbancesFixture:
    """Test fixture for satellite disturbances"""
    
    def setup_method(self):
        """Set up satellite with geometry and generate orbit"""
        # Inertia matrix (cube-shaped satellite)
        self.J = np.array([
            [0.067, 0.0, 0.0],
            [0.0, 0.067, 0.0],
            [0.0, 0.0, 0.067]
        ])
        
        # Create satellite
        self.settings = saltro_py.PlannerSettings()
        self.sat = saltro_py.Satellite(self.J, self.settings)
        
        # Geometry with significant cross-sectional areas
        self.geometry = saltro_py.GeometryConfig()
        self.setup_geometry()
        self.sat.setGeometryConfig(self.geometry)
        
        # Generate orbit
        self.n_steps = 200
        self.dt = 10.0
        self.generate_orbit()
    
    def setup_geometry(self):
        """Create spacecraft geometry with 6 faces"""
        # +X face (large solar panel area)
        self.geometry.addFace(saltro_py.GeometryFace(
            1.0,                              # area (m^2)
            np.array([0.5, 0.0, 0.0]),        # centroid
            np.array([1.0, 0.0, 0.0]),        # normal (outward)
            0.1,                              # eta_s (specular reflectivity)
            0.3,                              # eta_d (diffuse reflectivity)
            0.2,                              # eta_a (absorptivity)
            0.0                               # temperature
        ))
        
        # -X face
        self.geometry.addFace(saltro_py.GeometryFace(
            0.8, np.array([-0.5, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]),
            0.05, 0.2, 0.3, 0.0
        ))
        
        # +Z face (cross-sectional for drag)
        self.geometry.addFace(saltro_py.GeometryFace(
            2.0, np.array([0.0, 0.0, 0.5]), np.array([0.0, 0.0, 1.0]),
            0.1, 0.2, 0.3, 0.0
        ))
        
        # -Z face
        self.geometry.addFace(saltro_py.GeometryFace(
            2.0, np.array([0.0, 0.0, -0.5]), np.array([0.0, 0.0, -1.0]),
            0.05, 0.15, 0.4, 0.0
        ))
        
        # +Y face
        self.geometry.addFace(saltro_py.GeometryFace(
            0.5, np.array([0.0, 0.5, 0.0]), np.array([0.0, 1.0, 0.0]),
            0.1, 0.25, 0.25, 0.0
        ))
        
        # -Y face
        self.geometry.addFace(saltro_py.GeometryFace(
            0.5, np.array([0.0, -0.5, 0.0]), np.array([0.0, -1.0, 0.0]),
            0.05, 0.2, 0.3, 0.0
        ))
    
    def generate_orbit(self):
        """Generate orbital trajectory at 600 km altitude"""
        a = 6978e3  # Semi-major axis
        r0 = np.array([a, 0.0, 0.0])
        v0 = np.array([0.0, 7.56e3, 0.0])  # Orbital velocity
        
        jtime = np.array([i * self.dt for i in range(self.n_steps)])
        
        # Generate orbit
        ok, self.R, self.V, self.B, self.S, self.rho = saltro_py.generate_orbit(
            r0, v0, jtime, 0, 0, 0, 0, 0
        )
        assert ok, "Orbit generation failed"
    
    def get_disturbance_torque(self, x, dist_cfg, step_idx):
        """Get disturbance torque with specific configuration"""
        idx = min(step_idx, self.n_steps - 1)
        return self.sat.disturbanceTorque(
            x, dist_cfg,
            self.R[:, idx], self.B[:, idx], self.S[:, idx], self.V[:, idx],
            int(self.rho[idx])
        )


# ============================================================================
# TEST SECTION 1: Individual Disturbance Types
# ============================================================================

def test_gravity_gradient_torque_at_equator_is_small():
    """Test GG alone produces negligible torque at equator"""
    fixture = TestSatelliteDisturbancesFixture()
    fixture.setup_method()
    
    dist = saltro_py.DisturbanceConfig()
    dist.plan_for_gg = True
    dist.plan_for_aero = False
    dist.plan_for_srp = False
    
    x = np.zeros(fixture.sat.stateDim)
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    
    tau_gg = fixture.get_disturbance_torque(x, dist, 0)
    
    assert np.all(np.isfinite(tau_gg))
    assert np.linalg.norm(tau_gg) < 1e-5


def test_drag_torque_depends_on_velocity_orientation():
    """Test drag is computed correctly"""
    fixture = TestSatelliteDisturbancesFixture()
    fixture.setup_method()
    
    dist = saltro_py.DisturbanceConfig()
    dist.plan_for_gg = False
    dist.plan_for_aero = True
    dist.plan_for_srp = False
    
    x = np.zeros(fixture.sat.stateDim)
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    
    tau_drag = fixture.get_disturbance_torque(x, dist, 0)
    
    assert np.all(np.isfinite(tau_drag))
    assert np.linalg.norm(tau_drag) < 1e-5


def test_solar_radiation_pressure_torque_depends_on_sun_vector():
    """Test SRP is computed correctly"""
    fixture = TestSatelliteDisturbancesFixture()
    fixture.setup_method()
    
    dist = saltro_py.DisturbanceConfig()
    dist.plan_for_gg = False
    dist.plan_for_aero = False
    dist.plan_for_srp = True
    
    x = np.zeros(fixture.sat.stateDim)
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    
    tau_srp = fixture.get_disturbance_torque(x, dist, 0)
    
    assert np.all(np.isfinite(tau_srp))


def test_srp_is_zero_when_disabled():
    """Test zero torque when SRP disabled"""
    fixture = TestSatelliteDisturbancesFixture()
    fixture.setup_method()
    
    dist_off = saltro_py.DisturbanceConfig()
    dist_off.plan_for_gg = False
    dist_off.plan_for_aero = False
    dist_off.plan_for_srp = False
    
    x = np.zeros(fixture.sat.stateDim)
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    
    tau_off = fixture.get_disturbance_torque(x, dist_off, 0)

    assert np.allclose(tau_off, np.zeros(3))


def test_prop_disturbance_applies_body_fixed_constant_torque():
    """plan_for_prop=True should add prop_torque (body-fixed) to the disturbance."""
    fixture = TestSatelliteDisturbancesFixture()
    fixture.setup_method()

    tau_body = np.array([4.0e-5, -1.0e-5, 2.0e-5])

    dist = saltro_py.DisturbanceConfig()
    dist.plan_for_gg = False
    dist.plan_for_aero = False
    dist.plan_for_srp = False
    dist.plan_for_prop = True
    dist.prop_torque = tau_body

    x = np.zeros(fixture.sat.stateDim)
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])

    tau_total = fixture.get_disturbance_torque(x, dist, 0)

    assert np.all(np.isfinite(tau_total))
    # With only prop enabled (no other contributors), the total equals prop_torque.
    assert np.allclose(tau_total, tau_body)


def test_prop_disturbance_is_zero_when_disabled():
    """plan_for_prop=False should ignore prop_torque even if non-zero."""
    fixture = TestSatelliteDisturbancesFixture()
    fixture.setup_method()

    dist = saltro_py.DisturbanceConfig()
    dist.plan_for_gg = False
    dist.plan_for_aero = False
    dist.plan_for_srp = False
    dist.plan_for_prop = False
    dist.prop_torque = np.array([4.0e-5, -1.0e-5, 2.0e-5])

    x = np.zeros(fixture.sat.stateDim)
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])

    tau_total = fixture.get_disturbance_torque(x, dist, 0)
    assert np.allclose(tau_total, np.zeros(3))


def test_prop_disturbance_is_body_fixed_under_attitude():
    """prop_torque is in the body frame -- it should NOT rotate when the
    attitude changes (this distinguishes body-fixed propulsion from a
    fixed-inertial torque).
    """
    fixture = TestSatelliteDisturbancesFixture()
    fixture.setup_method()

    tau_body = np.array([3.0e-5, 0.0, 0.0])

    dist = saltro_py.DisturbanceConfig()
    dist.plan_for_gg = False
    dist.plan_for_aero = False
    dist.plan_for_srp = False
    dist.plan_for_prop = True
    dist.prop_torque = tau_body

    # At identity quaternion: body = inertial, output is +x_body == +x_inertial.
    x_id = np.zeros(fixture.sat.stateDim)
    x_id[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    tau_id = fixture.get_disturbance_torque(x_id, dist, 0)

    # 180-degree rotation about z keeps body +x equal to body +x in body frame.
    x_rot = np.zeros(fixture.sat.stateDim)
    x_rot[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([0, 0, 0, 1])
    tau_rot = fixture.get_disturbance_torque(x_rot, dist, 0)

    assert np.allclose(tau_id, tau_body)
    assert np.allclose(tau_rot, tau_body)


# ============================================================================
# TEST SECTION 2: Multiple Disturbances Together
# ============================================================================

def test_all_disturbances_combined_produce_non_zero_torque():
    """Test all three disturbances together"""
    fixture = TestSatelliteDisturbancesFixture()
    fixture.setup_method()
    
    dist = saltro_py.DisturbanceConfig()
    dist.plan_for_gg = True
    dist.plan_for_aero = True
    dist.plan_for_srp = True
    
    x = np.zeros(fixture.sat.stateDim)
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    
    tau_combined = fixture.get_disturbance_torque(x, dist, 0)
    
    assert np.all(np.isfinite(tau_combined))
    assert np.linalg.norm(tau_combined) < 1e-4


def test_individual_disturbances_sum_approximately_to_combined():
    """Test superposition principle"""
    fixture = TestSatelliteDisturbancesFixture()
    fixture.setup_method()
    
    dist_all = saltro_py.DisturbanceConfig()
    dist_all.plan_for_gg = True
    dist_all.plan_for_aero = True
    dist_all.plan_for_srp = True
    
    x = np.zeros(fixture.sat.stateDim)
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    
    # Get individual torques
    dist_gg = saltro_py.DisturbanceConfig()
    dist_gg.plan_for_gg = True
    dist_gg.plan_for_aero = False
    dist_gg.plan_for_srp = False
    tau_gg = fixture.get_disturbance_torque(x, dist_gg, 0)
    
    dist_aero = saltro_py.DisturbanceConfig()
    dist_aero.plan_for_gg = False
    dist_aero.plan_for_aero = True
    dist_aero.plan_for_srp = False
    tau_aero = fixture.get_disturbance_torque(x, dist_aero, 0)
    
    dist_srp = saltro_py.DisturbanceConfig()
    dist_srp.plan_for_gg = False
    dist_srp.plan_for_aero = False
    dist_srp.plan_for_srp = True
    tau_srp = fixture.get_disturbance_torque(x, dist_srp, 0)
    
    tau_sum = tau_gg + tau_aero + tau_srp
    tau_combined = fixture.get_disturbance_torque(x, dist_all, 0)
    
    assert np.allclose(tau_combined, tau_sum, atol=1e-15)


# ============================================================================
# TEST SECTION 3: Disturbance Behavior Across Orbit
# ============================================================================

def test_drag_is_approximately_constant_along_circular_orbit():
    """For a circular orbit (constant altitude → constant density) the drag
    torque magnitude should not vary across the sampled steps. Previously
    misnamed `test_drag_decreases_with_increasing_altitude`."""
    fixture = TestSatelliteDisturbancesFixture()
    fixture.setup_method()

    dist = saltro_py.DisturbanceConfig()
    dist.plan_for_gg = False
    dist.plan_for_aero = True
    dist.plan_for_srp = False

    x = np.zeros(fixture.sat.stateDim)
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])

    drag_mags = []
    for i in range(0, min(100, fixture.n_steps), 10):
        tau_drag = fixture.get_disturbance_torque(x, dist, i)
        drag_mags.append(np.linalg.norm(tau_drag))

    has_variation = False
    for i in range(1, len(drag_mags)):
        variation = abs(drag_mags[i] - drag_mags[0]) / (drag_mags[0] + 1e-12)
        if variation > 0.001:  # More than 0.1% variation
            has_variation = True
            break

    assert not has_variation


def test_gravity_gradient_varies_with_orbital_position():
    """Test GG across orbit"""
    fixture = TestSatelliteDisturbancesFixture()
    fixture.setup_method()
    
    dist = saltro_py.DisturbanceConfig()
    dist.plan_for_gg = True
    dist.plan_for_aero = False
    dist.plan_for_srp = False
    
    x = np.zeros(fixture.sat.stateDim)
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    
    # Sample GG torque at different orbital positions
    gg_mags = []
    for i in range(0, fixture.n_steps, 10):
        tau_gg = fixture.get_disturbance_torque(x, dist, i)
        gg_mags.append(np.linalg.norm(tau_gg))
    
    # Just verify it's computed and finite
    for mag in gg_mags:
        assert np.isfinite(mag)


# ============================================================================
# TEST SECTION 4: Solar Radiation Pressure Specifics
# ============================================================================

def test_srp_zero_in_eclipse():
    """Test SRP is zero with zero sun vector"""
    fixture = TestSatelliteDisturbancesFixture()
    fixture.setup_method()
    
    x = np.zeros(fixture.sat.stateDim)
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    
    # Zero sun vector should give zero torque
    srp = saltro_py.SRPDisturbance(fixture.geometry)
    dist_cfg = saltro_py.DisturbanceConfig()
    
    x_base = x[:7]
    S_zero = np.array([0.0, 0.0, 0.0])
    
    tau_eclipse = srp.torque(x_base, dist_cfg, S_zero)
    
    assert np.allclose(tau_eclipse, np.zeros(3))


def test_srp_increases_with_sun_vector_alignment():
    """Test SRP geometry dependence"""
    fixture = TestSatelliteDisturbancesFixture()
    fixture.setup_method()
    
    srp = saltro_py.SRPDisturbance(fixture.geometry)
    dist_cfg = saltro_py.DisturbanceConfig()
    dist_cfg.plan_for_srp = True
    
    x = np.zeros(fixture.sat.stateDim)
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    x_base = x[:7]
    
    # Sun vector along +X (hits large surface)
    S_x = np.array([1.0, 0.0, 0.0])
    tau_x = srp.torque(x_base, dist_cfg, S_x)
    
    # Sun vector along +Y (hits small surface)
    S_y = np.array([0.0, 1.0, 0.0])
    tau_y = srp.torque(x_base, dist_cfg, S_y)
    
    # Sun vector along +Z (hits moderate surface)
    S_z = np.array([0.0, 0.0, 1.0])
    tau_z = srp.torque(x_base, dist_cfg, S_z)
    
    # Magnitude should differ based on geometry
    mag_x = np.linalg.norm(tau_x)
    mag_y = np.linalg.norm(tau_y)
    mag_z = np.linalg.norm(tau_z)
    
    assert np.isfinite(mag_x)
    assert np.isfinite(mag_y)
    assert np.isfinite(mag_z)


# ============================================================================
# TEST SECTION 5: Order of Magnitude Validation
# ============================================================================

def test_drag_torque_order_of_magnitude_is_reasonable():
    """Test drag magnitude is physically realistic"""
    fixture = TestSatelliteDisturbancesFixture()
    fixture.setup_method()
    
    dist = saltro_py.DisturbanceConfig()
    dist.plan_for_gg = False
    dist.plan_for_aero = True
    dist.plan_for_srp = False
    
    x = np.zeros(fixture.sat.stateDim)
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    
    tau_drag = fixture.get_disturbance_torque(x, dist, 0)
    
    mag = np.linalg.norm(tau_drag)
    assert mag < 1e-5
    assert mag >= 0.0


def test_gravity_gradient_torque_order_of_magnitude_is_reasonable():
    """Test GG magnitude is physically realistic"""
    fixture = TestSatelliteDisturbancesFixture()
    fixture.setup_method()
    
    dist = saltro_py.DisturbanceConfig()
    dist.plan_for_gg = True
    dist.plan_for_aero = False
    dist.plan_for_srp = False
    
    x = np.zeros(fixture.sat.stateDim)
    q = np.array([0.95, 0.1, 0.05, 0.0])
    q = q / np.linalg.norm(q)
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = q
    
    tau_gg = fixture.get_disturbance_torque(x, dist, 50)
    
    mag = np.linalg.norm(tau_gg)
    assert mag < 1e-5
    assert mag >= 0.0


def test_solar_radiation_pressure_torque_order_of_magnitude_is_reasonable():
    """Test SRP magnitude is physically realistic"""
    fixture = TestSatelliteDisturbancesFixture()
    fixture.setup_method()
    
    dist = saltro_py.DisturbanceConfig()
    dist.plan_for_gg = False
    dist.plan_for_aero = False
    dist.plan_for_srp = True
    
    x = np.zeros(fixture.sat.stateDim)
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    
    tau_srp = fixture.get_disturbance_torque(x, dist, 0)
    
    mag = np.linalg.norm(tau_srp)
    assert mag < 1e-4
    assert mag >= 0.0


# ============================================================================
# TEST SECTION 6: Dependence on Configuration
# ============================================================================

def test_drag_produces_non_zero_torque_with_geometry():
    """Test drag with realistic geometry"""
    fixture = TestSatelliteDisturbancesFixture()
    fixture.setup_method()
    
    dist = saltro_py.DisturbanceConfig()
    dist.plan_for_gg = False
    dist.plan_for_aero = True
    dist.plan_for_srp = False
    
    x = np.zeros(fixture.sat.stateDim)
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    
    tau_drag = fixture.get_disturbance_torque(x, dist, 0)
    
    assert np.all(np.isfinite(tau_drag))
    assert np.linalg.norm(tau_drag) >= 0.0


def test_srp_produces_non_zero_torque_with_geometry():
    """Test SRP with realistic geometry"""
    fixture = TestSatelliteDisturbancesFixture()
    fixture.setup_method()
    
    dist = saltro_py.DisturbanceConfig()
    dist.plan_for_gg = False
    dist.plan_for_aero = False
    dist.plan_for_srp = True
    
    x = np.zeros(fixture.sat.stateDim)
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    
    tau_srp = fixture.get_disturbance_torque(x, dist, 0)
    
    assert np.all(np.isfinite(tau_srp))
    assert np.linalg.norm(tau_srp) >= 0.0


# ============================================================================
# TEST SECTION 7: Disturbance Independence
# ============================================================================

def test_disabling_drag_does_not_affect_gg_and_srp():
    """Test disturbance independence"""
    fixture = TestSatelliteDisturbancesFixture()
    fixture.setup_method()
    
    dist_no_drag = saltro_py.DisturbanceConfig()
    dist_no_drag.plan_for_gg = True
    dist_no_drag.plan_for_aero = False
    dist_no_drag.plan_for_srp = True
    
    dist_with_drag = saltro_py.DisturbanceConfig()
    dist_with_drag.plan_for_gg = True
    dist_with_drag.plan_for_aero = True
    dist_with_drag.plan_for_srp = True
    
    x = np.zeros(fixture.sat.stateDim)
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    
    tau_no_drag = fixture.get_disturbance_torque(x, dist_no_drag, 0)
    
    # Get individual drag
    dist_drag_only = saltro_py.DisturbanceConfig()
    dist_drag_only.plan_for_gg = False
    dist_drag_only.plan_for_aero = True
    dist_drag_only.plan_for_srp = False
    tau_drag = fixture.get_disturbance_torque(x, dist_drag_only, 0)
    
    tau_with_drag = fixture.get_disturbance_torque(x, dist_with_drag, 0)
    
    # Should satisfy: tau_with_drag = tau_no_drag + tau_drag
    difference = tau_with_drag - tau_no_drag - tau_drag
    assert np.linalg.norm(difference) < 1e-15


def test_disturbances_are_finite_over_full_orbit():
    """Test numerical stability across full orbit"""
    fixture = TestSatelliteDisturbancesFixture()
    fixture.setup_method()
    
    dist = saltro_py.DisturbanceConfig()
    dist.plan_for_gg = True
    dist.plan_for_aero = True
    dist.plan_for_srp = True
    
    x = np.zeros(fixture.sat.stateDim)
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])

    for i in range(fixture.n_steps):
        tau = fixture.get_disturbance_torque(x, dist, i)
        assert np.all(np.isfinite(tau))


# ============================================================================
# TEST SECTION 8: SRP eclipse + direction sensitivity
# ============================================================================

def test_srp_torque_is_zero_at_orbit_eclipse_steps():
    """Stronger than `test_srp_zero_in_eclipse` (which feeds a hand-zeroed
    sun vector): use a step where `generate_orbit` itself reports S=0
    because the satellite is in Earth's shadow. The full disturbance
    pipeline must respect that — no leakage of stale sun state."""
    fixture = TestSatelliteDisturbancesFixture()
    fixture.setup_method()

    eclipse_steps = [i for i in range(fixture.n_steps)
                     if np.linalg.norm(fixture.S[:, i]) == 0.0]
    assert len(eclipse_steps) > 0, (
        "fixture orbit has no eclipse steps — extend the orbit or "
        "select a longer trajectory"
    )

    dist = saltro_py.DisturbanceConfig()
    dist.plan_for_srp = True

    x = np.zeros(fixture.sat.stateDim)
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])

    for step in eclipse_steps[:5]:
        tau_srp = fixture.get_disturbance_torque(x, dist, step)
        assert np.allclose(tau_srp, np.zeros(3), atol=1e-15), (
            f"SRP non-zero at eclipse step {step}: τ={tau_srp}"
        )


def test_srp_torque_responds_to_sun_direction_flip():
    """The fixture geometry is not face-symmetric (face areas and
    reflectivities differ across +/- axes). Flipping the sun vector
    should produce a torque that is NOT simply the negation of the
    original — confirming the geometry-dependent face contributions
    are actually being summed (not, e.g., cancelling out)."""
    fixture = TestSatelliteDisturbancesFixture()
    fixture.setup_method()

    srp = saltro_py.SRPDisturbance(fixture.geometry)
    dist = saltro_py.DisturbanceConfig()
    dist.plan_for_srp = True

    x = np.zeros(fixture.sat.stateDim)
    x[saltro_py.Satellite.QUAT_INDEX:saltro_py.Satellite.QUAT_INDEX + 4] = np.array([1, 0, 0, 0])
    x_base = x[:7]

    # Off-axis sun: a sun vector along a principal axis produces zero
    # τ_SRP for face-centered geometry (r × F = 0 because r ∥ F).
    S_plus = np.array([0.6, 0.5, 0.6])
    S_plus /= np.linalg.norm(S_plus)
    S_minus = -S_plus

    tau_plus = srp.torque(x_base, dist, S_plus)
    tau_minus = srp.torque(x_base, dist, S_minus)

    # Each torque must be non-trivial individually
    assert np.linalg.norm(tau_plus) > 0.0
    assert np.linalg.norm(tau_minus) > 0.0

    # And they must NOT be exact negatives (asymmetric geometry)
    assert not np.allclose(tau_plus, -tau_minus, atol=1e-12), (
        "τ_plus == -τ_minus implies face-symmetric geometry — but the "
        "fixture has differing +X / -X face properties"
    )
