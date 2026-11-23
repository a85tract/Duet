"""Tests of duet/clifford_circuits.py: CNOT-only circuits equal duet.circuits.make_circuit's bit for bit;
Clifford circuits have deterministic detectors and observables; the observables' noiseless values equal the logical
tableau's; the fold round applies logical S in both frames; the circuit distance stays d."""
import itertools

import numpy as np
import pytest
import stim

from duet import circuits
from duet import clifford_circuits as cc

TOKENS = ["C0", "C1", "II", "IH", "HI", "IS", "SI", "HH", "SS", "HS", "SH"]


def rand_layers(rng, L, gates="IHSXYZ"):
    out = []
    for _ in range(L):
        if rng.random() < 0.4:
            out.append("C0" if rng.random() < 0.5 else "C1")
        else:
            out.append(rng.choice(list(gates)) + rng.choice(list(gates)))
    return out


def ref_observables(c):
    ref = c.reference_sample()
    n = c.num_measurements
    return [int(sum(ref[n + t.value] for t in inst.targets_copy()) % 2)
            for inst in c.flattened() if inst.name == "OBSERVABLE_INCLUDE"]


def no_pauli(layers):
    return [t if t[0] == "C" else "".join("I" if g in cc.PAULI else g for g in t) for t in layers]


@pytest.mark.parametrize("d", [3, 5, 7])
def test_cnot_only_identical(d):
    for L in (1, 4, 8):
        for seed in range(2):
            dirs = circuits.cnot_directions(L, seed)
            for prep, meas in ((("Z", "Z"), ("Z", "Z")), (("X", "X"), ("X", "X")), (("Z", "X"), ("X", "Z"))):
                for p in (0.0, 0.003):
                    a = circuits.make_circuit(d, L, p, directions=dirs, prep=prep, meas=meas)
                    b = cc.make_circuit(d, cc.cnot_layers(dirs), p, prep=prep, meas=meas)
                    assert str(a) == str(b), (d, L, seed, prep, meas, p)


def test_exhaustive_short_circuits_deterministic():
    for lay in itertools.product(TOKENS, repeat=2):
        for prep in itertools.product("ZX", repeat=2):
            for meas in itertools.product("ZX", repeat=2):
                c = cc.make_circuit(3, list(lay), 0.003, prep, meas)
                c.detector_error_model(decompose_errors=False)        # raises on a non-deterministic detector


@pytest.mark.parametrize("d,n", [(3, 80), (5, 12)])
def test_random_circuits_match_tableau(d, n):
    rng = np.random.default_rng(20261028 + d)
    seen = np.zeros(3, int)
    for _ in range(n):
        lay = rand_layers(rng, int(rng.integers(1, 9)))
        prep, meas = tuple(rng.choice(["Z", "X"], 2)), tuple(rng.choice(["Z", "X"], 2))
        c, meta = cc.make_circuit(d, lay, 0.003, prep, meas, meta=True)
        dem = c.detector_error_model(decompose_errors=False)
        assert dem.num_detectors == c.num_detectors == len(meta["det_t"])
        c0 = cc.make_circuit(d, no_pauli(lay), 0.0, prep, meas)
        prods = cc.logical_products(lay, prep, meas)
        assert [s for s, _ in prods] == [s for s, _ in cc.logical_products(no_pauli(lay), prep, meas)]
        assert ref_observables(c0) == [v for _, v in cc.logical_products(no_pauli(lay), prep, meas)]
        assert c.num_observables == len(prods)
        seen[len(prods)] += 1
        # as make_circuit: half the checks give detectors in round 0 and at the end, all of them in between
        assert c.num_detectors == 2 * len(lay) * (d * d - 1)
    assert seen[1] > 0 and (seen[2] > 0 or n < 50)


@pytest.mark.parametrize("d", [3, 5])
@pytest.mark.parametrize("f", [0, 1])
def test_fold_round_is_logical_s(d, f):
    """In a one-block round with the fold: X-bar -> Y-bar and Z-bar -> Z-bar with no measurement and sign +1 (logical S);
    every check has a flow through the round."""
    patch = circuits.rotated_patch(d)
    fl = cc.round_flows(d, f, True)
    assert set(fl) == {(t, s) for t in "XZ" for s in range(len(patch[t]))}
    idx = {q: k for k, q in enumerate(patch["data"])}
    anc = [(t, s) for t in "XZ" for s in range(len(patch[t]))]
    for k, (t, s) in enumerate(anc):
        idx[patch[t][s]] = len(patch["data"]) + k
    n = len(idx)
    xs, steps = cc._round_ops(patch, f)
    c = stim.Circuit()
    c.append("R", [idx[patch[t][s]] for t, s in anc])
    c.append("H", [idx[m] for m in xs])
    for k, st in enumerate(steps):
        c.append("CX", [v for m, dq, mc in st for v in ((idx[m], idx[dq]) if mc else (idx[dq], idx[m]))])
        if k == 1:
            prs, s1, s2 = cc.fold_gates(d)
            c.append("CZ", [idx[q] for pr in prs for q in pr])
            c.append("S", [idx[q] for q in s1])
            c.append("S_DAG", [idx[q] for q in s2])
    c.append("H", [idx[m] for m in xs])
    c.append("M", [idx[patch[t][s]] for t, s in anc])
    lx, lz = cc.logical_line(d, "X", f, d), cc.logical_line(d, "Z", f, d)
    X, Z, Y = (stim.PauliString(n) for _ in range(3))
    for q in lx:
        X[idx[q]] = Y[idx[q]] = "X"
    for q in lz:
        Z[idx[q]] = "Z"
        Y[idx[q]] = "Y" if q in lx else "Z"
    assert c.has_flow(stim.Flow(input=X, output=Y), unsigned=False)
    assert c.has_flow(stim.Flow(input=Z, output=Z), unsigned=False)


@pytest.mark.parametrize("d", [3, 5])
def test_main_diagonal_fold_loses_distance(d):
    """The S result: the same mid-cycle fold about the main diagonal is also a logical S, but for some bases the circuit
    distance drops below d (to 2 at d = 3 and 3 at d = 5 with preparation XZ and readout ZX here), while the
    anti-diagonal keeps d (test_circuit_distance)."""
    lay = ["SI", "C0", "IS", "C1"]
    kw = dict(dont_explore_detection_event_sets_with_size_above=4, dont_explore_edges_with_degree_above=4,
              dont_explore_edges_increasing_symptom_degree=False, canonicalize_circuit_errors=True)
    for prep in itertools.product("ZX", repeat=2):
        for meas in itertools.product("ZX", repeat=2):
            c = cc.make_circuit(d, lay, 0.003, prep, meas, kind="main")
            if c.num_observables and len(c.search_for_undetectable_logical_errors(**kw)) < d:
                print("main-diagonal fold below distance", d, prep, meas)
                return
    raise AssertionError("the main-diagonal fold kept distance d for every basis")


@pytest.mark.parametrize("d", [3, 5])
def test_circuit_distance(d):
    """Shortest undetectable logical error (stim's search, hyperedges included) of circuits with S, H and rotated
    CNOTs, for every basis choice that leaves an observable (the first two)."""
    cases = [["SI", "C0", "IS", "C1"], ["HS", "C0", "SH", "C1"], ["IH", "C0", "SS", "C1"], ["SH", "II", "SH", "C1"]]
    for lay in cases[: 4 if d == 3 else 2]:
        done = 0
        for prep in itertools.product("ZX", repeat=2):
            for meas in itertools.product("ZX", repeat=2):
                c = cc.make_circuit(d, lay, 0.003, prep, meas)
                if c.num_observables == 0 or done == 2:
                    continue
                done += 1
                err = c.search_for_undetectable_logical_errors(dont_explore_detection_event_sets_with_size_above=4,
                                                               dont_explore_edges_with_degree_above=4,
                                                               dont_explore_edges_increasing_symptom_degree=False,
                                                               canonicalize_circuit_errors=True)
                assert len(err) == d, (lay, prep, meas, len(err))
        assert done == 2, lay


def test_frames_and_metadata():
    c, meta = cc.make_circuit(3, ["IH", "C0", "HS", "C1"], 0.003, ("Z", "X"), ("X", "Z"), meta=True)
    assert meta["frames"] == [(0, 1), (0, 1), (1, 1), (1, 1)] and meta["final_frames"] == (1, 1)
    assert set(meta["det_t"]) == set(range(5)) and set(meta["det_type"]) == {0, 1}
    coords = c.get_detector_coordinates()
    for k in range(c.num_detectors):
        assert int(coords[k][2]) == meta["det_t"][k]
