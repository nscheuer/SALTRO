"""Compare Python vs C++ outputs for 3_0_slew90_dt60 case."""
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(ROOT / "tests" / "debug" / "optimizer" / "alilqr_python"))
sys.path.insert(0, str(ROOT / "tests" / "debug" / "optimizer" / "configs"))

import saltro_py
from sat_3_0_mtq import create_satellite
from trajOpt import trajOpt
from debug_3_0_slew90_dt60 import create_planner_settings


def main():
    plannersettings = create_planner_settings()
    satellite = create_satellite(plannersettings)

    jtime = np.array([0.22, 0.22 + 5400 / (36525 * 86400)])
    qgoal = np.array([
        [np.sqrt(2) / 2, np.sqrt(2) / 2],
        [0.0, 0.0],
        [0.0, 0.0],
        [np.sqrt(2) / 2, np.sqrt(2) / 2],
    ])
    boresight = np.array([
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ])

    w0 = np.array([0.0, 0.0, 0.0])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    x0 = np.hstack((w0, q0))

    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7.5e3, 0.0])

    print("Running Python trajOpt...")
    X_py, U_py, stop_py, _, _, _, _, _ = trajOpt(
        plannersettings, satellite, x0, r0, v0, jtime, qgoal, boresight, debug=True
    )

    print("Running C++ trajOpt...")
    ok, X_cpp, U_cpp, _K = saltro_py.trajOpt(
        plannersettings, satellite, x0, r0, v0, jtime, qgoal, boresight
    )

    if not ok:
        raise RuntimeError("C++ trajOpt failed")

    diff_X = np.abs(X_py - X_cpp)
    diff_U = np.abs(U_py - U_cpp)

    max_diff_X = float(np.max(diff_X))
    max_diff_U = float(np.max(diff_U))

    print("\n=== 3_0_slew90_dt60 Comparison ===")
    print(f"Python stop reason: {stop_py}")
    print(f"Shapes: X_py={X_py.shape}, X_cpp={X_cpp.shape}, U_py={U_py.shape}, U_cpp={U_cpp.shape}")
    print(f"max_abs_diff_X = {max_diff_X:.16e}")
    print(f"max_abs_diff_U = {max_diff_U:.16e}")

    tol = 1e-12
    if max_diff_X < tol and max_diff_U < tol:
        print(f"RESULT: MATCH (tol={tol:.0e})")
    else:
        ix = np.unravel_index(np.argmax(diff_X), diff_X.shape)
        iu = np.unravel_index(np.argmax(diff_U), diff_U.shape)
        print(f"RESULT: MISMATCH (tol={tol:.0e})")
        print(f"largest X diff at {ix}: {diff_X[ix]:.6e}")
        print(f"largest U diff at {iu}: {diff_U[iu]:.6e}")


if __name__ == "__main__":
    main()
