import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "build"))

try:
    import saltro_py
except ImportError as e:
    pytest.skip(f"saltro Python bindings not available: {e}", allow_module_level=True)


def min_eig(matrix):
    vals = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    return np.min(vals)


def test_psd_clip_result_is_symmetric():
    m = np.array([[2.0, -1.0, 0.5],
                  [-1.0, 1.0, 0.3],
                  [0.5, 0.3, -2.0]])
    clipped = saltro_py.psd_clip(m)
    assert np.linalg.norm(clipped - clipped.T) < 1e-12


def test_psd_clip_clamps_negative_eigenvalues_to_zero():
    m = np.diag([1.0, -3.0, 2.0])
    clipped = saltro_py.psd_clip(m)
    expected = np.diag([1.0, 0.0, 2.0])
    assert min_eig(clipped) >= -1e-12
    assert np.linalg.norm(clipped - expected) < 1e-12


def test_psd_clip_leaves_psd_matrix_unchanged():
    a = np.array([[1.0, 2.0, 0.0, -1.0],
                  [0.5, 1.0, 3.0, 0.2],
                  [-0.3, 0.4, 1.0, 0.7],
                  [0.1, -0.2, 0.5, 2.0]])
    p = a.T @ a
    clipped = saltro_py.psd_clip(p)
    assert np.linalg.norm(clipped - p) < 1e-10


def test_psd_clip_is_idempotent():
    m = np.array([[2.0, -1.0, 0.5],
                  [-1.0, -1.0, 0.3],
                  [0.5, 0.3, -2.0]])
    clipped_once = saltro_py.psd_clip(m)
    clipped_twice = saltro_py.psd_clip(clipped_once)
    assert np.linalg.norm(clipped_twice - clipped_once) < 1e-10


def test_psd_clip_symmetrizes_asymmetric_input():
    m = np.array([[3.0, 2.0],
                  [0.0, 3.0]])
    msym = 0.5 * (m + m.T)
    clipped = saltro_py.psd_clip(m)
    assert np.linalg.norm(clipped - clipped.T) < 1e-12
    assert np.linalg.norm(clipped - msym) < 1e-10
