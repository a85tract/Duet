import numpy as np
import pytest
import stim

from duet.circuits import cnot_directions, dem_counts, make_circuit, rotated_patch, stab_coords


@pytest.mark.parametrize("d", [3, 5, 7, 9, 11])
def test_patch(d):
    patch = rotated_patch(d)
    assert len(patch["data"]) == d * d
    assert len(patch["LX"]) == len(patch["LZ"]) == d


def _noiseless_zero(circuit, shots=64):
    dets, obs = circuit.compile_detector_sampler().sample(shots, separate_observables=True)
    return not dets.any() and not obs.any()


@pytest.mark.parametrize("d", [3, 5])
@pytest.mark.parametrize("basis", ["Z", "X"])
@pytest.mark.parametrize("L,rpl", [(4, 1), (6, 1), (8, 1), (4, "d")])
def test_noiseless_deterministic(d, basis, L, rpl):
    rpl = d if rpl == "d" else rpl
    c0 = make_circuit(d, L, 0.0, basis=basis, rounds_per_layer=rpl)
    assert c0.num_observables == 2
    assert _noiseless_zero(c0)
    # every stabilizer measurement after the first round gives a detector; the first round and the final
    # measurement give one per stabilizer of the preparation / measurement basis
    n = (d * d - 1) // 2
    assert c0.num_detectors == 2 * (n + 2 * n * (L * rpl - 1) + n)
    # the noisy circuit has a detector error model (stim raises on non-deterministic detectors or observables)
    c = make_circuit(d, L, 1e-3, basis=basis, rounds_per_layer=rpl)
    assert c.detector_error_model().num_detectors == c0.num_detectors


@pytest.mark.parametrize("directions,prep,meas,n_obs", [
    ([0], ("X", "Z"), ("X", "X"), 1),  # Bell pair: only X_A X_B is reliable
    ([0], ("X", "Z"), ("Z", "Z"), 1),  # Bell pair: only Z_A Z_B is reliable
    ([1], ("X", "Z"), ("X", "Z"), 2),  # CNOT B->A on |+0> leaves |+0>
    ([0, 1, 0], ("X", "Z"), ("Z", "X"), 2),  # three CNOTs make a SWAP: |0+>
])
def test_mixed_bases(directions, prep, meas, n_obs):
    c0 = make_circuit(3, len(directions), 0.0, directions=directions, prep=prep, meas=meas)
    assert c0.num_observables == n_obs
    assert _noiseless_zero(c0)
    make_circuit(3, len(directions), 1e-3, directions=directions, prep=prep, meas=meas).detector_error_model()


@pytest.mark.parametrize("d", [3, 5])
@pytest.mark.parametrize("basis", ["Z", "X"])
def test_circuit_distance(d, basis):
    """No set of fewer than d faults flips an observable without a detection event."""
    c = make_circuit(d, 3, 1e-3, basis=basis, directions=[0, 1, 1])
    err = c.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=4,
        dont_explore_edges_with_degree_above=4,
        dont_explore_edges_increasing_symptom_degree=False,
        canonicalize_circuit_errors=True)
    assert len(err) == d


def test_detector_coordinates_match_stab_coords():
    d = 5
    c = make_circuit(d, 4, 1e-3)
    known = {xy for block in stab_coords(d) for t in "XZ" for xy in block[t]}
    coords = c.get_detector_coordinates()
    assert all(tuple(v[:2]) in known for v in coords.values())
    assert max(v[2] for v in coords.values()) == 4  # 4 rounds, final measurement at t = 4


def test_directions_seeded():
    assert cnot_directions(8, 2026) == cnot_directions(8, 2026)
    assert set(cnot_directions(64, 1)) == {0, 1}


# L = 8, one round per layer, basis Z, seed 2026: detectors, error mechanisms, hyperedges, hyperedges of one
RECORDED_COUNTS = {3: (128, 2108, 1511, 772, 0), 5: (384, 7350, 5835, 3582, 905),
                   7: (768, 15776, 12831, 8344, 3092), 9: (1280, 27386, 22499, 15058, 6399),
                   11: (1920, 42180, 34839, 23724, 10826)}


@pytest.mark.parametrize("d", [3, 5, 7, 9, 11])
def test_dem_counts(d):
    counts = dem_counts(make_circuit(d, 8, 2e-3), d)
    print(f"\nDEM d={d} L=8 one round/layer basis Z: {counts}")
    assert counts["hyperedges"] >= counts["hyperedges_one_type"] > 0
    assert np.isclose(counts["detectors"], 2 * (d * d - 1) * 8)
    keys = ("detectors", "error_mechanisms", "hyperedges", "hyperedges_one_type", "undecomposable")
    assert tuple(counts[k] for k in keys) == RECORDED_COUNTS[d]
