import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "build"))
import saltro_py

if __name__ == "__main__":
    MAX = 10000
    N = 200

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
    B = np.zeros((3, MAX), dtype=np.float64, order="F")

    # --- propagate orbit ---
    ok = saltro_py.compute_orbit_keplerian(r0, v0, t, N, R, V)
    if not ok:
        raise RuntimeError("orbit failed")

    # --- compute magnetic field ---
    ok = saltro_py.compute_magnetic_dipole(R, t, N, B)
    if not ok:
        raise RuntimeError("B-field failed")

    Rv = R[:, :N]
    Bv = B[:, :N]

    print("First B:", Bv[:, 0])
    print("Last  B:", Bv[:, -1])

    Bmag = np.linalg.norm(Bv, axis=0)

    plt.figure()
    plt.plot(Bmag)
    plt.title("Magnetic field magnitude along orbit")
    plt.xlabel("step")
    plt.ylabel("|B| (Tesla)")
    plt.grid(True)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.plot(Rv[0], Rv[1], Rv[2], label="orbit")

    # draw Earth
    RE = 6371e3
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    x = RE * np.outer(np.cos(u), np.sin(v))
    y = RE * np.outer(np.sin(u), np.sin(v))
    z = RE * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.2)

    # plot some B vectors (downsample for clarity)
    step = max(1, N // 40)
    scale = 2e7  # adjust for visibility

    ax.quiver(
        Rv[0, ::step],
        Rv[1, ::step],
        Rv[2, ::step],
        Bv[0, ::step],
        Bv[1, ::step],
        Bv[2, ::step],
        length=scale,
        normalize=True
    )

    ax.set_title("Orbit with magnetic field vectors (ECI)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend()

    fig2, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 7))

    axs[0].plot(Bv[0])
    axs[0].set_ylabel("Bx (T)")
    axs[0].grid(True)

    axs[1].plot(Bv[1])
    axs[1].set_ylabel("By (T)")
    axs[1].grid(True)

    axs[2].plot(Bv[2])
    axs[2].set_ylabel("Bz (T)")
    axs[2].set_xlabel("step")
    axs[2].grid(True)

    fig2.suptitle("Magnetic field components (ECI)")

    plt.show()