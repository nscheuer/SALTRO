import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
import saltro_py

def create_satellite(plannersettings: saltro_py.PlannerSettings) -> saltro_py.Satellite:
    """3 MTQ + 1 RW hybrid satellite configuration"""
    J = np.array([[0.03136490806, 5.88304e-05, -0.00671361357],
                  [5.88304e-05, 0.03409127827, -0.00012334756],
                  [-0.00671361357, -0.00012334756, 0.01004091997]])

    satellite = saltro_py.Satellite(J, plannersettings)

    # Add magnetorquers
    satellite.addMTQ(np.array([1.0, 0.0, 0.0]), 0.2)
    satellite.addMTQ(np.array([0.0, 1.0, 0.0]), 0.2)
    satellite.addMTQ(np.array([0.0, 0.0, 1.0]), 0.2)

    # Add reaction wheel
    satellite.addRW(np.array([1.0, 0.0, 0.0]), 5.7e-6, 0.0023, 0.0, 0.0036)

    return satellite
