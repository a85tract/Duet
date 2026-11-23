import numpy as np
import pytest
import stim

from duet.lukin_bench import (GATES, N_BLOCKS, PAIRINGS, _Code, bench_circuits, logical_stabilizers, make_circuit,
                             random_logical)


def _noiseless_zero(circuit, shots=64):
    dets, obs = circuit.compile_detector_sampler(seed=1).sample(shots, separate_observables=True)
    return not dets.any() and not obs.any()


def _logical(layers, target):
    """A hand-made logical circuit in the format of ``random_logical``."""
    return {"layers": layers, "target": target, "bases": "".join(t if t != "I" else "Z" for t in target)}


def test_gate_set_option():
    c = random_logical(10, 0, gate_set="IXZH")
    assert set("".join(g for g, _ in c["layers"])) <= set("IXZH")
    assert c["target"] in logical_stabilizers(c["layers"])


def test_random_logical_seeded_and_well_formed():
    a, b = random_logical(6, 3), random_logical(6, 3)
    assert a["layers"] == b["layers"] and a["target"] == b["target"]
    circs = bench_circuits(10)
    assert len({str(c["layers"]) for c in circs}) == 20
    for c in circs:
        for gates, cnots in c["layers"]:
            assert len(gates) == N_BLOCKS and set(gates) <= set(GATES)
            assert tuple(sorted(tuple(sorted(p)) for p in cnots)) in PAIRINGS  # a perfect matching
        assert "Y" not in c["target"] and set(c["target"]) != {"I"}


@pytest.mark.parametrize("depth", [1, 4, 10])
def test_targets_are_stabilizers(depth):
    """Every candidate target has expectation +-1 on the logical output state (stim tableau of four qubits)."""
    for c in bench_circuits(depth):
        sim = stim.TableauSimulator()
        sim.h(*range(N_BLOCKS))
        for gates, cnots in c["layers"]:
            for b, g in enumerate(gates):
                if g != "I":
                    getattr(sim, g.lower())(b)
            for ct, tg in cnots:
                sim.cnot(ct, tg)
        for s in logical_stabilizers(c["layers"]):
            assert abs(sim.peek_observable_expectation(stim.PauliString(s))) == 1
        assert c["target"] in logical_stabilizers(c["layers"])


def test_rotation_maps_code_onto_itself():
    for d in (3, 5, 7):
        code = _Code(d)  # the constructor asserts that supports and logicals map onto the other type
        moved = sorted(k for step in code.swap_steps for pair in step for k in pair)
        assert len(set(moved)) == d * d - 1  # every data qubit but the centre moves


HAND = [  # (layers, every candidate target is tested)
    [("HIII", ((0, 1), (2, 3)))],
    [("HHII", ((1, 0), (2, 3))), ("HIIH", ((0, 2), (1, 3)))],
    [("IXYZ", ((0, 3), (2, 1))), ("HHHH", ((3, 0), (1, 2))), ("HZIH", ((0, 1), (3, 2)))],
    [("HIII", ((1, 0), (3, 2))), ("IIII", ((0, 2), (3, 1))), ("HIII", ((0, 1), (2, 3))), ("HIII", ((1, 0), (2, 3)))],
]


@pytest.mark.parametrize("d", [3, 5])
@pytest.mark.parametrize("k", range(len(HAND)))
def test_hand_circuits_every_target(d, k):
    layers = HAND[k]
    for target in logical_stabilizers(layers):
        for last, two in ((True, False), (False, False), (True, True)):
            kw = {"round_after_last_layer": last, "round_after_1q_layer": two}
            c0 = make_circuit(d, _logical(layers, target), 0.0, **kw)
            assert _noiseless_zero(c0), (layers, target, kw)
            rounds = len(layers) * (1 + two) - (not last)
            n = d * d - 1
            assert c0.num_detectors == N_BLOCKS * (n * rounds + n // 2)
            make_circuit(d, _logical(layers, target), 3e-3, **kw).detector_error_model()


@pytest.mark.parametrize("d", [3, 5])
@pytest.mark.parametrize("depth", [1, 2, 4, 10])
def test_noiseless_deterministic(d, depth):
    for logical in bench_circuits(depth)[:6]:
        c0 = make_circuit(d, logical, 0.0)
        assert c0.num_observables == 1
        assert _noiseless_zero(c0)
        n = d * d - 1  # stabilizers per block: every round gives one detector each, the readout half of them
        assert c0.num_detectors == N_BLOCKS * (n * depth + n // 2)
        c = make_circuit(d, logical, 3e-3)
        assert c.detector_error_model().num_detectors == c0.num_detectors


def test_noise_model():
    p = 3e-3
    q = round(p / 10, 12)
    allowed = {"DEPOLARIZE1": {(q,)}, "DEPOLARIZE2": {(p,)}, "X_ERROR": {(q,)}, "M": {(q,)}, "MX": {(q,)},
               "MPP": {()}}
    c = make_circuit(3, random_logical(3, 0), p)
    found = {}
    for inst in c.flattened():
        if inst.name in allowed or "ERROR" in inst.name or "CHANNEL" in inst.name or inst.name.startswith(("M", "E")):
            found.setdefault(inst.name, set()).add(tuple(round(a, 12) for a in inst.gate_args_copy()))
    assert set(allowed) - {"MX"} <= set(found) <= set(allowed)
    assert all(found[k] == allowed[k] for k in found)
    # the lattice rotation is noiseless: the SWAP steps are followed by the next layer's TICK only
    names = [inst.name for inst in c.flattened()]
    assert all(names[i + 1] in ("SWAP", "TICK") for i, nm in enumerate(names) if nm == "SWAP")


@pytest.mark.parametrize("d", [3, 5])
def test_circuit_distance(d):
    """No set of fewer than d faults flips the target without a detection event, also across H and CNOTs."""
    layers = [("HIHI", ((0, 1), (3, 2))), ("IHHI", ((1, 2), (0, 3))), ("HHII", ((2, 0), (1, 3)))]
    for target, two in [(t, False) for t in logical_stabilizers(layers)[:4]] + [(logical_stabilizers(layers)[0], True)]:
        c = make_circuit(d, _logical(layers, target), 3e-3, round_after_1q_layer=two)
        err = c.search_for_undetectable_logical_errors(
            dont_explore_detection_event_sets_with_size_above=4,
            dont_explore_edges_with_degree_above=4,
            dont_explore_edges_increasing_symptom_degree=False,
            canonicalize_circuit_errors=True)
        assert len(err) == d, (target, two, len(err))


def test_dem_sizes_recorded():
    """Size of the benchmark's detector error models (printed for the record; d = 3, 5, 7 at depths 4 and 10)."""
    for d in (3, 5, 7):
        for depth in (4, 10):
            dem = make_circuit(d, random_logical(depth, 0), 3e-3).detector_error_model(decompose_errors=False)
            n_err = sum(1 for inst in dem.flattened() if inst.type == "error")
            print(f"\nDEM d={d} depth={depth}: detectors {dem.num_detectors}, error mechanisms {n_err}")
            assert dem.num_detectors == N_BLOCKS * ((d * d - 1) * depth + (d * d - 1) // 2)
