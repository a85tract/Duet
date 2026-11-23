"""Tests of scripts/paired_stats.py: a zero failure count adds 0.5
to both counts instead of giving an infinite or undefined ratio, a complete family of CNOT patterns redraws shots
counts."""
import argparse
import json

import numpy as np
import pytest

import paired_stats as E


def fail_vectors(n, counts, start=0):
    out = []
    for k in counts:
        x = np.zeros(n, bool)
        x[start:start + k] = True
        out.append(x)
    return out


def test_zero_count_adds_half():
    fx, fy = fail_vectors(100, [0, 0, 0, 0]), fail_vectors(100, [3, 0, 0, 0])
    r, logs = E.paired(E.table(fx, fy), 1, "circuits")
    assert r["zero_count"] and r["fails_x"] == 0 and r["fails_y"] == 3
    assert r["ratio"] == pytest.approx(0.5 / 3.5)
    assert np.isfinite(logs).all() and r["ratio95"][0] <= r["ratio"] <= r["ratio95"][1]
    r, logs = E.paired(E.table(fx, fx), 1, "shots")
    assert r["zero_count"] and r["ratio"] == 1.0 and r["ratio95"] == [1.0, 1.0]


def test_enumerated_patterns_redraw_shots():
    fx, fy = fail_vectors(1000, [20, 20]), fail_vectors(1000, [10, 10], start=20)
    r, _ = E.paired(E.table(fx, fy), 1, "circuits")
    assert r["ratio"] == pytest.approx(2.0) and r["ratio95"] == pytest.approx([2.0, 2.0])
    r, _ = E.paired(E.table(fx, fy), 1, "shots")
    assert r["ratio"] == pytest.approx(2.0) and not r["zero_count"]
    assert r["ratio95"][0] < 1.6 and r["ratio95"][1] > 2.5


def block(runs, P, q, lo, hi, obs, pred, optimal):
    np.savez(runs / P / f"scip_q{q:02d}_s{lo}_{hi}.npz", q=q, lo=lo, hi=hi, point=P, obs=np.array(obs, np.int64),
             m_pred=np.array(pred, np.int64), m_optimal=np.array(optimal, bool), m_ok=np.ones(hi - lo, bool))


