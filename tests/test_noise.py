"""Tests of the Lukin group's noise model in duet/clifford_circuits.py: the default model's
circuits are unchanged bit for bit (hashes of the circuit text recorded with the code before the switch existed, commit
c929c12); with noise="lukin" each block starts in its exact code state from a noiseless encoding unitary, every
detector (the first round's of both check types included) and every observable is deterministic, the noise channels
are the model's, the circuit distance stays d, and the training pipeline runs on it."""
import hashlib
import itertools

import numpy as np
import pytest
import stim

from duet import clifford_circuits as cc
from duet.circuits import rotated_patch
from test_clifford_circuits import no_pauli, rand_layers, ref_observables

CASES = [(3, "C0 SI IH C1", "ZX", "XZ"), (5, "HS C0 SS C1 II", "XX", "ZZ"), (3, "SH II SH C1", "ZZ", "XX"),
         (5, "C1 C0", "ZZ", "ZZ"), (3, "HH SS", "XX", "XX"), (7, "IS C0 HI C1", "XZ", "ZX")]
UNIFORM_SHA = {(3, "C0 SI IH C1", "ZX", "XZ"): "66a1d327051e813b", (5, "HS C0 SS C1 II", "XX", "ZZ"): "c4decf586e27e2d8",
               (3, "SH II SH C1", "ZZ", "XX"): "8422876f6f2c10fc", (5, "C1 C0", "ZZ", "ZZ"): "eb37918b2c0e83f1",
               (3, "HH SS", "XX", "XX"): "2cf77c626a5b291a", (7, "IS C0 HI C1", "XZ", "ZX"): "35f18662d5fb5f5d"}
P = 0.003


def circuit_sha(c):
    return hashlib.sha256(str(c).encode()).hexdigest()[:16]


def test_uniform_circuits_unchanged():
    assert set(UNIFORM_SHA) == set(CASES)
    for (d, lay, prep, meas), sha in UNIFORM_SHA.items():
        c = cc.make_circuit(d, lay.split(), P, tuple(prep), tuple(meas))
        assert circuit_sha(c) == sha, (d, lay, prep, meas)
        assert str(cc.make_circuit(d, lay.split(), P, tuple(prep), tuple(meas), noise="uniform")) == str(c)


@pytest.mark.parametrize("d", [3, 5, 7])
def test_encoder_gives_the_code_state(d):
    patch = rotated_patch(d)
    data_k = {q: k for k, q in enumerate(patch["data"])}
    n = len(patch["data"])
    for basis in "ZX":
        piv, pairs = cc.encoder(d, basis)
        sim = stim.TableauSimulator()
        sim.set_num_qubits(n)
        if basis == "X":
            sim.h(*range(n))
        sim.h(*piv)
        for a, b in pairs:
            sim.cnot(a, b)
        for t in "XZ":
            for m in patch[t]:
                ps = stim.PauliString(n)
                for q in patch["supp"][m]:
                    ps[data_k[q]] = t
                assert sim.peek_observable_expectation(ps) == 1, (basis, t, m)
        logical = stim.PauliString(n)
        for q in cc.logical_line(d, basis, 0, d):
            logical[data_k[q]] = basis
        assert sim.peek_observable_expectation(logical) == 1, basis


@pytest.mark.parametrize("d,n", [(3, 60), (5, 10)])
def test_lukin_circuits_deterministic_and_match_tableau(d, n):
    rng = np.random.default_rng(20261029 + d)
    for _ in range(n):
        lay = rand_layers(rng, int(rng.integers(1, 9)))
        prep, meas = tuple(rng.choice(["Z", "X"], 2)), tuple(rng.choice(["Z", "X"], 2))
        c, meta = cc.make_circuit(d, lay, P, prep, meas, meta=True, noise="lukin")
        dem = c.detector_error_model(decompose_errors=False)      # raises on a non-deterministic detector or observable
        assert dem.num_detectors == c.num_detectors == len(meta["det_t"])
        # every check gives a detector in round 0 (the exact code state), all of them in between, half at the end
        assert c.num_detectors == (2 * len(lay) + 1) * (d * d - 1)
        assert (meta["det_t"] == 0).sum() == 2 * (d * d - 1)
        assert c.num_observables == len(cc.logical_products(lay, prep, meas))
        c0 = cc.make_circuit(d, no_pauli(lay), 0.0, prep, meas, noise="lukin")
        assert ref_observables(c0) == [v for _, v in cc.logical_products(no_pauli(lay), prep, meas)]
        dets, obs = c0.compile_detector_sampler(seed=1).sample(8, separate_observables=True)
        assert not dets.any() and not obs.any()


def test_exhaustive_short_lukin_circuits_deterministic():
    for lay in itertools.product(["C0", "C1", "II", "IH", "HS", "SI", "SS"], repeat=2):
        for prep in itertools.product("ZX", repeat=2):
            for meas in itertools.product("ZX", repeat=2):
                cc.make_circuit(3, list(lay), P, prep, meas, noise="lukin").detector_error_model(decompose_errors=False)


def test_lukin_noise_channels():
    c = cc.make_circuit(3, ["HS", "C0", "SH", "C1"], P, ("Z", "X"), ("X", "Z"), noise="lukin")
    one_q, two_q = {"H", "S", "S_DAG"}, {"CX", "CZ"}
    tick, gates, seen = 0, [], set()
    for inst in c.flattened():
        name, args, tg = inst.name, inst.gate_args_copy(), [t.value for t in inst.targets_copy()]
        if name == "TICK":
            tick, gates = tick + 1, []
            continue
        if tick == 0:                                    # the encoding: resets, H and CX only, no noise
            assert name in {"QUBIT_COORDS", "R", "RX", "H", "CX"}, name
            continue
        seen.add(name)
        if name == "DEPOLARIZE2":
            assert args == [P] and set(tg) <= {q for g, q in gates if g in two_q}
        elif name == "DEPOLARIZE1":                      # after single-qubit gates only: no idle noise
            assert args == [P / 10] and set(tg) <= {q for g, q in gates if g in one_q}
        elif name == "X_ERROR":
            assert args == [P / 10] and set(tg) == {q for g, q in gates if g == "R"}
        elif name in {"M", "MX"}:
            assert args == [P / 10]
        else:
            assert name in one_q | two_q | {"R", "DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS"}, name
        gates += [(name, q) for q in tg]
    assert {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "M", "MX"} <= seen and "Z_ERROR" not in seen


@pytest.mark.parametrize("d", [3, 5])
def test_lukin_circuit_distance(d):
    for lay, prep, meas in ((["SI", "C0", "IS", "C1"], "XZ", "ZX"), (["HS", "C0", "SH", "C1"], "ZX", "ZX")):
        c = cc.make_circuit(d, lay, P, tuple(prep), tuple(meas), noise="lukin")
        err = c.search_for_undetectable_logical_errors(dont_explore_detection_event_sets_with_size_above=4,
                                                       dont_explore_edges_with_degree_above=4,
                                                       dont_explore_edges_increasing_symptom_degree=False,
                                                       canonicalize_circuit_errors=True)
        assert len(err) == d, (lay, prep, meas, len(err))


def test_train_cl_runs_on_lukin_circuits(tmp_path):
    from duet.net import train_cl
    from test_train_resume import ARGS
    out = str(tmp_path / "lukin")
    assert train_cl.main(ARGS + ["--noise", "lukin", "--max-steps", "2", "--out", out]) == 3
    import json
    rows = [json.loads(x) for x in open(f"{out}/log.jsonl")]
    assert rows[-1]["step"] == 2 and set(rows[-1]["ema_per_depth"]) == {"mix", "cnot"}
