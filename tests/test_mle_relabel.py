"""The exact MLE under a relabelling of the blocks: with the
product basis carried to the relabelled circuit and the detectors moved by the map (block, cell, round), every shot
has the same optimal weight, proven, with syndrome-consistent solutions on both circuits (Lukin model)."""
import numpy as np
import pytest

pytest.importorskip("pyscipopt")

from duet import clifford_circuits as cc  # noqa: E402
from duet import mle  # noqa: E402

P = 0.003


def relabel(lay, prep, meas, perm):
    K = len(prep)
    out = []
    for tok in lay:
        new = [""] * K
        for b, ch in enumerate(tok):
            new[perm[b]] = str(perm[int(ch)]) if ch.isdigit() else ch
        out.append("".join(new))
    inv = [perm.index(b) for b in range(K)]
    return out, tuple(prep[i] for i in inv), tuple(meas[i] for i in inv)


@pytest.mark.parametrize("K", [3, 4])
def test_mle_weight_under_relabelling(K):
    d = 3
    rng = np.random.default_rng(71 + K)
    for _ in range(2):
        lay, prep, meas = cc.sample_circuit_k(rng, K, 4, 0.5)
        perm = [int(v) for v in rng.permutation(K)]
        lay2, prep2, meas2 = relabel(lay, prep, meas, perm)
        c1, m1 = cc.make_circuit(d, lay, P, prep, meas, meta=True, noise="lukin")
        basis = [sorted(perm[b] for b in sub) for sub, _ in m1["products"]]
        c2, m2 = cc.make_circuit(d, lay2, P, prep2, meas2, meta=True, noise="lukin", products=basis)
        key2 = {(int(b), int(x), int(y), int(t)): k
                for k, (b, (x, y), t) in enumerate(zip(m2["det_block"], m2["det_xy"], m2["det_t"]))}
        dmap = np.array([key2[(perm[int(b)], int(x), int(y), int(t))]
                         for b, (x, y), t in zip(m1["det_block"], m1["det_xy"], m1["det_t"])])
        D1, D2 = (mle.Dem(c.detector_error_model(decompose_errors=False)) for c in (c1, c2))
        s1, s2 = (mle.Scip(D, time_limit=120.0, params={"presolving/maxrestarts": 0}) for D in (D1, D2))
        for r in c1.compile_detector_sampler(seed=4).sample(30):
            r2 = np.zeros_like(r)
            r2[dmap] = r
            a, b = s1.solve(np.flatnonzero(r)), s2.solve(np.flatnonzero(r2))
            assert a["optimal"] and b["optimal"]
            assert np.array_equal(D1.syndrome(a["x"]), np.flatnonzero(r))
            assert np.array_equal(D2.syndrome(b["x"]), np.flatnonzero(r2))
            assert abs(a["weight"] - b["weight"]) <= 1e-9 * max(1.0, a["weight"]), (lay, perm)
