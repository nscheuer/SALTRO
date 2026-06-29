"""Python twin of tests/unit/math/test_matrix.cpp.

Note: the C++ psd_clip mutates its argument in place; the Python binding
returns the clipped matrix instead (see python/math/matrix_py.cpp).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "build"))

import numpy as np
import saltro_py as s


def _min_eig(M):
    M = np.asarray(M)
    sym = 0.5 * (M + M.T)
    return np.linalg.eigvalsh(sym).min()


def test_psd_clip_result_is_symmetric():
    M = np.array([[2.0, -1.0, 0.5],
                  [-1.0, 1.0, 0.3],
                  [0.5, 0.3, -2.0]])  # indefinite
    P = np.array(s.psd_clip(M))
    assert np.linalg.norm(P - P.T) < 1e-12


def test_psd_clip_clamps_negative_eigenvalues_to_zero():
    M = np.array([[1.0, 0.0, 0.0],
                  [0.0, -3.0, 0.0],
                  [0.0, 0.0, 2.0]])
    P = np.array(s.psd_clip(M))
    assert _min_eig(P) >= -1e-12
    expected = np.zeros((3, 3))
    expected[0, 0] = 1.0
    expected[1, 1] = 0.0
    expected[2, 2] = 2.0
    assert np.linalg.norm(P - expected) < 1e-12


def test_psd_clip_leaves_psd_matrix_unchanged():
    A = np.array([[1.0, 2.0, 0.0, -1.0],
                  [0.5, 1.0, 3.0, 0.2],
                  [-0.3, 0.4, 1.0, 0.7],
                  [0.1, -0.2, 0.5, 2.0]])
    P0 = A.T @ A  # PSD by construction
    P = np.array(s.psd_clip(P0))
    assert np.linalg.norm(P - P0) < 1e-10


def test_psd_clip_is_idempotent():
    M = np.array([[2.0, -1.0, 0.5],
                  [-1.0, -1.0, 0.3],
                  [0.5, 0.3, -2.0]])
    M1 = np.array(s.psd_clip(M))
    M2 = np.array(s.psd_clip(M1))
    assert np.linalg.norm(M2 - M1) < 1e-10


def test_psd_clip_symmetrizes_asymmetric_input():
    M = np.array([[3.0, 2.0],
                  [0.0, 3.0]])  # asymmetric; sym part eigvals 3 +/- 1 -> PSD
    Msym = 0.5 * (M + M.T)
    P = np.array(s.psd_clip(M))
    assert np.linalg.norm(P - P.T) < 1e-12
    assert np.linalg.norm(P - Msym) < 1e-10
