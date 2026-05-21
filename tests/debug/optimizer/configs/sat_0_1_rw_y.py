"""Configuration helper: 0 MTQ + 1 y-axis RW with isotropic body inertia.

Used by the property tests in ``tests/unit/optimizer/test_property_*.py``
that need a single-axis-actuator fixture for symmetry-preservation and
1-axis-LQR-equivalent comparison.
"""
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
import saltro_py


def create_satellite(plannersettings: saltro_py.PlannerSettings) -> saltro_py.Satellite:
    """0 MTQ + 1 y-axis RW satellite. Isotropic body inertia 0.1 kg m^2
    so the single-axis problem reduces cleanly to a scalar LQR.
    """
    J = np.diag([0.1, 0.1, 0.1])
    satellite = saltro_py.Satellite(J, plannersettings)
    # addRW signature: (axis, max_torque, J_wheel, h0, h_max).
    satellite.addRW(
        np.array([0.0, 1.0, 0.0]),
        1.0,        # max_torque
        1e-6,       # J_wheel
        0.0,        # h0
        10.0,       # h_max
    )
    return satellite
