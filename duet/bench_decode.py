"""Decoders of the open benchmark on one circuit of a point, shared by
  lomr LoMatching 0.3.0 (MoMatching) on Definition-1 regions (duet.clifford_circuits.regions)
  cain our Cain et al. (duet.decoders.cain2025)
  bplomr BP+LoMatching (lomatching's BeliefMoMatching on the same regions; min-sum, 100 iterations, scaling 0.5)
  mle the exact MLE: duet.mle.Scip with presolving restarts off, settings
The three matching decoders are built with ignore_decomposition_failures=True: where
stim cannot split an error of a product's model into graphlike pieces, that error leaves the matching graphs (BP keeps
the full model) and `dropped` counts it per product; a circuit with no dropped error is in the buildable subset.
Every MLE result is checked (optimal, solution syndrome equal to the shot's detectors, weight equal to the dual bound),
and tally counts failures, target failures and shots over the checked results only."""
import time

import numpy as np

from duet import bench_sets as bs
from duet import clifford_circuits as cc

DECODERS = ("lomr", "cain", "bplomr", "mle")
MLE_PARAMS = {"presolving/maxrestarts": 0}


class Mle:
    """duet.mle.Scip on the circuit's full error model (decompose_errors=False)."""

    def __init__(self, circuit, time_limit=600.0):
        from duet import mle
        self.D = mle.Dem(circuit.detector_error_model(decompose_errors=False))
        self.sol = mle.Scip(self.D, time_limit=time_limit, params=MLE_PARAMS)
        self.k = circuit.num_observables
        self.dropped = [0] * self.k
        self.settings = {"solver": "duet.mle.Scip", "params": MLE_PARAMS, "time_limit": time_limit}

    def decode_checked(self, dets):
        """(predicted flips (shots, k) bool, certified (shots,) bool, CPU seconds per shot)."""
        pred = np.zeros((len(dets), self.k), bool)
        ok = np.zeros(len(dets), bool)
        sec = np.zeros(len(dets))
        self.xs = []                                    # each shot's solution
        for s, row in enumerate(dets):
            fired = np.flatnonzero(row)
            t0 = time.process_time()
            r = self.sol.solve(fired)
            sec[s] = time.process_time() - t0
            self.xs.append(r["x"])
            if r["x"] is not None:
                o = self.D.observables(r["x"])
                pred[s] = [(o >> j) & 1 for j in range(self.k)]
                ok[s] = (r["optimal"] and np.array_equal(self.D.syndrome(r["x"]), fired)
                         and abs(r["weight"] - r["bound"]) <= 1e-6 * max(1.0, abs(r["weight"])))
        return pred, ok, sec


def build(name, entry, c_, mle_limit=600.0, transform=None):
    """(stim circuit, decoder) for circuit c_ of point `entry` (a manifest entry or any dict with d, p, noise);
    transform(circuit) -> circuit changes the noise only (the pilot's GHZ variants)."""
    c, meta = bs.circuit(entry, c_, meta=True)
    if transform is not None:
        c = transform(c)
    lay, meas = c_["layers"].split(), tuple(c_["meas"])
    if name == "lomr":
        from duet.decoders.lomatching_dec import LoMatchingRegions
        return c, LoMatchingRegions(c, cc.regions(meta, lay, meas), ignore_decomposition_failures=True)
    if name == "cain":
        from duet.decoders.cain2025 import CainDecoder
        return c, CainDecoder(c, ignore_decomposition_failures=True)
    if name == "bplomr":
        from duet.decoders.lomatching_dec import BpLomRegions
        return c, BpLomRegions(c, cc.regions(meta, lay, meas), entry["d"], ignore_decomposition_failures=True)
    if name == "mle":
        return c, Mle(c, mle_limit)
    raise ValueError(name)


def decode(name, dec, dets):
    """(predicted flips (shots, k) bool, certified (shots,) bool, CPU seconds) of decoder `name` on bool dets."""
    t0 = time.process_time()
    if name == "mle":
        pred, ok, sec = dec.decode_checked(dets)
        return pred, ok, float(sec.sum())
    if name == "bplomr":
        pred = np.array([dec.decode(r) for r in dets], bool).reshape(len(dets), -1)
    else:
        pred = np.asarray(dec.decode_batch(dets), bool).reshape(len(dets), -1)
    return pred, np.ones(len(dets), bool), time.process_time() - t0


def tally(pred, obs, ok, target_slot=None):
    """Shots, failures (any product wrong) and target failures (slot target_slot wrong) over the certified shots."""
    fail = (pred != obs).any(1) & ok
    out = dict(shots=int(ok.sum()), fails=int(fail.sum()), uncertified=int((~ok).sum()))
    if target_slot is not None:
        out["target_fails"] = int(((pred[:, target_slot] != obs[:, target_slot]) & ok).sum())
    return out
