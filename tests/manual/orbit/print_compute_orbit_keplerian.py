import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "build"))
import saltro_py

if __name__ == "__main__":
    MAX = 1000
    N = 100

    r0 = np.array([[7000e3],
                [0.0],
                [0.0]], dtype=np.float64)

    v0 = np.array([[0.0],
                [7500.0],
                [0.0]], dtype=np.float64)

    t = np.zeros((1, MAX), dtype=np.float64)

    SEC_PER_JULIAN_CENTURY = 36525.0 * 86400.0

    for i in range(N):
        t[0, i] = 0.2 + i * 10.0 / SEC_PER_JULIAN_CENTURY

    R = np.zeros((3, MAX), dtype=np.float64, order="F")
    V = np.zeros((3, MAX), dtype=np.float64, order="F")

    ok = saltro_py.compute_orbit_keplerian(
        r0,
        v0,
        t,
        N,
        R,
        V
    )
    
    print("R0:", R[:, 0])
    print("R10:", R[:, 10])