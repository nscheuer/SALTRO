import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
import saltro_py

def create_3mtq_satellite(plannersettings: saltro_py.PlannerSettings) -> saltro_py.Satellite:
    J = np.diag([0.067, 0.071, 0.069])

    satellite = saltro_py.Satellite(J, plannersettings)

    satellite.addMTQ(np.array([1.0, 0.0, 0.0]), 0.2)
    satellite.addMTQ(np.array([0.0, 1.0, 0.0]), 0.2)
    satellite.addMTQ(np.array([0.0, 0.0, 1.0]), 0.2)

    return satellite