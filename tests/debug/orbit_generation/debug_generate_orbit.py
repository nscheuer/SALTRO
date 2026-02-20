import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "build"))
import saltro_py


def main():
    # Initial state (same as C++ test)
    r0 = np.array([7000e3, 0.0, 0.0], dtype=float)
    v0 = np.array([0.0, 7.5e3, 0.0], dtype=float)

    # Time array (10 points, 60 second step)
    jtime_length = 10
    jtime = np.array([i * 60.0 for i in range(jtime_length)], dtype=float)

    # Run with model 0 for everything
    ok, R, V, B, S, rho = saltro_py.generate_orbit(
        r0,
        v0,
        jtime,
        0,  # orbit_model
        0,  # magnetic_model
        0,  # sun_model
        0,  # density_model
    )

    if not ok:
        print("generate_orbit failed")
        return

    def print_series(name, M):
        print(f"{name}:")
        print("  initial:", M[:, 0])
        print("  at jtime_length-1:", M[:, jtime_length - 1])
        print()

    print_series("R", R)
    print_series("V", V)
    print_series("B", B)
    print_series("S", S)

    print("rho:")
    print("  initial:", rho[0])
    print("  at jtime_length-1:", rho[jtime_length - 1])


if __name__ == "__main__":
    main()