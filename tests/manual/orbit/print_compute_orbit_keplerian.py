import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "build"))
import saltro_py

if __name__ == "__main__":
    MAX = 10000
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
        t[0, i] = 0.22 + i * 10.0 / SEC_PER_JULIAN_CENTURY

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

    if not ok:
        raise RuntimeError("Propagation failed")

    # Only first N columns are valid
    R_valid = R[:, :N]

    # --- 3D orbit plot ---
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.plot(R_valid[0], R_valid[1], R_valid[2], label="Orbit")
    ax.scatter(R_valid[0,0], R_valid[1,0], R_valid[2,0], label="Start")

    # Draw Earth for scale
    RE = 6371e3
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    x = RE * np.outer(np.cos(u), np.sin(v))
    y = RE * np.outer(np.sin(u), np.sin(v))
    z = RE * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.2)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("Orbit position (ECI)")
    ax.legend()

    # Equal aspect ratio
    max_range = np.array([
        R_valid[0].max() - R_valid[0].min(),
        R_valid[1].max() - R_valid[1].min(),
        R_valid[2].max() - R_valid[2].min()
    ]).max() / 2.0

    mid = np.array([
        (R_valid[0].max() + R_valid[0].min()) * 0.5,
        (R_valid[1].max() + R_valid[1].min()) * 0.5,
        (R_valid[2].max() + R_valid[2].min()) * 0.5
    ])

    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.show()