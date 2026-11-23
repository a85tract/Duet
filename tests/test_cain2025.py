"""Tests for the Cain et al. 2025 decoder (duet/decoders/cain2025.py)."""
import numpy as np
import pymatching
import stim

from duet.decoders.cain2025 import CainDecoder, subgraphs_from_coords


def memory(basis="z", d=3, p=0.004, rounds=None):
    return stim.Circuit.generated(f"surface_code:rotated_memory_{basis}", distance=d, rounds=rounds or d,
                                  after_clifford_depolarization=p, before_measure_flip_probability=p,
                                  after_reset_flip_probability=p, before_round_data_depolarization=p)


def test_memory_equals_full_matching():
    # In a Z (X) memory the product is Z (X) bar, so its subgraph is all Z (X) checks, and matching it predicts the
    # same flips as matching the full decomposed model (whose X and Z check graphs are disconnected).
    for basis in ("z", "x"):
        c = memory(basis)
        dets, obs = c.compile_detector_sampler(seed=7).sample(3000, separate_observables=True)
        full = pymatching.Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True))
        dec = CainDecoder(c, subgraphs="auto")
        assert 0 < len(dec.subgraphs[0][0]) < c.num_detectors and not dec.subgraphs[0][1]
        assert np.array_equal(dec.decode_batch(dets), full.decode_batch(dets).astype(bool))
        pred_t, ns = dec.decode_timed(dets[:200])
        assert np.array_equal(pred_t, dec.decode_batch(dets[:200])) and ns.shape == (200, 1)


def test_noiseless_shots_decode_to_no_flip():
    c = memory()
    dec = CainDecoder(c)
    assert not dec.decode_batch(np.zeros((5, c.num_detectors), bool)).any()


def test_subgraphs_from_coords():
    c = stim.Circuit("""
        R 0 1
        M 0 1
        DETECTOR(0, 1) rec[-2]
        DETECTOR(1) rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-2]
        OBSERVABLE_INCLUDE(1) rec[-1]
    """)
    V = subgraphs_from_coords(c)
    assert [v.tolist() for v in V] == [[0], [0, 1]]



def test_pilot_circuit_subgraphs_are_matchable():
    from duet.circuits import make_circuit
    for basis in ("Z", "X"):
        dec = CainDecoder(make_circuit(3, 4, 0.003, basis=basis), with_stats=True)
        assert all(s["components_over_2"] == 0 and s["conflicting_edges"] == 0 for s in dec.stats)
        assert all(s["undetected_flip_prob"] == 0 for s in dec.stats)


def test_one_round_circuits():
    # One round before the data readout: the round's ancillas and the data are measured in the
    # first measurement layer, where the data qubits' initial resets count as resets of the round too; the data must
    # not be taken for ancillas (the subgraphs refused these circuits before).
    for basis in ("z", "x"):
        for d in (3, 5):
            c = memory(basis, d=d, rounds=1)
            dets, obs = c.compile_detector_sampler(seed=7).sample(3000, separate_observables=True)
            full = pymatching.Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True))
            dec = CainDecoder(c)
            assert not dec.subgraphs[0][1]
            assert np.array_equal(dec.decode_batch(dets), full.decode_batch(dets).astype(bool))
    from duet.circuits import make_circuit
    for basis in ("Z", "X"):
        for d in (3, 5):
            dec = CainDecoder(make_circuit(d, 1, 0.003, basis=basis), with_stats=True)
            assert dec.info["num_blocks"] == 2
            assert all(s["components_over_2"] == 0 and s["conflicting_edges"] == 0 for s in dec.stats)
            assert all(s["undetected_flip_prob"] == 0 for s in dec.stats)
