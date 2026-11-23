"""speed study helpers: the batch-mode schedule and the tail quantile."""
import importlib.util
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _speed():
    if "lukin_speed" not in sys.modules:
        spec = importlib.util.spec_from_file_location("lukin_speed", os.path.join(ROOT, "scripts/tesseract_speed.py"))
        m = sys.modules["lukin_speed"] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
    return sys.modules["lukin_speed"]


def test_greedy_makespan():
    sp = _speed()
    T = np.array([[3.0, 1, 1, 1, 0], [5, 0, 0, 0, 0], [1, 2, 3, 4, 5]])
    assert np.allclose(sp.greedy_makespan(T, 1), [6, 5, 15])
    assert np.allclose(sp.greedy_makespan(T, 2), [3, 5, 9])        # [1, 3, 5 | 2, 4]: 1+3+5 = 9, 2+4 = 6
    assert np.allclose(sp.greedy_makespan(T, 64), [3, 5, 5])
    assert np.allclose(sp.greedy_makespan(np.zeros((4, 0)), 16), 0)


def test_batches_and_quantile():
    sp = _speed()
    T = sp.batches(np.array([0, 5, 9, 10, 15]), np.array([1.0, 2, 3, 4, 5]), 8, 3)
    assert T.tolist() == [[1, 2, 0], [3, 4, 5], [0, 0, 0]]
    v, w = np.array([1.0, 2.0, 10.0]), np.array([0.98, 0.015, 0.005])
    assert sp.wquantile(v, w, 0.5) == 1.0 and sp.wquantile(v, w, 0.99) == 2.0 and sp.wquantile(v, w, 0.999) == 10.0
