"""LoMatching (Serra-Peralta, Shaw, Terhal, arXiv 2505.13599): the released ``lomatching`` package.

For each reliable observable, LoMatching keeps the detectors of the (block, stabilizer type, round) slices that the
observable's single-detector boundary edges touch (its "observing region") and decodes the observable with PyMatching
on a detector error model over that region.

Two ways to build that per-observable model:
- ``LoMatching`` (released 0.3.0, ``MoMatching.from_circuit``): the circuit with only the region's detectors is
  re-analysed by stim with ``decompose_errors=True``.
- ``LoMatchingProj``: every error of the full model is projected onto the region's detectors, only the decoded
  observable is kept, and PyMatching ignores projections with more than two detectors. This is how lomatching 0.1.0
  (the release behind the paper's data) built its subgraphs (``decoder.get_subgraph``); the region is the 0.3.0 one.
"""
import lomatching
import numpy as np
import stim
from lomatching import MoMatching
from lomatching.util import get_detector_indices_for_subgraphs_from_dem
from pymatching import Matching

from ..circuits import stab_coords
from .cain2025 import graphlike


class LoMatching:
    def __init__(self, circuit, d: int):
        self.dec = MoMatching.from_circuit(circuit, stab_coords(d))
        self.settings = {"package": f"lomatching {lomatching.__version__}", "class": "MoMatching",
                         "subgraph_detectors": [len(x) for x in self.dec._det_inds_subgraphs]}

    def decode(self, dets):
        return self.dec.decode(dets)

    def decode_batch(self, dets):
        return self.dec.decode_batch(dets)


def projected_dem(dem: stim.DetectorErrorModel, det_inds, obs: int) -> stim.DetectorErrorModel:
    """Errors of ``dem`` restricted to the detectors ``det_inds`` (renumbered 0..n-1) and to observable ``obs``."""
    new_id = {int(d): i for i, d in enumerate(det_inds)}
    sub = stim.DetectorErrorModel()
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        dets = sorted({new_id[t.val] for t in inst.targets_copy() if t.is_relative_detector_id() and t.val in new_id})
        if not dets:
            continue
        flips = any(t.is_logical_observable_id() and t.val == obs for t in inst.targets_copy())
        targets = [stim.target_relative_detector_id(i) for i in dets]
        targets += [stim.target_logical_observable_id(0)] if flips else []
        sub.append("error", inst.args_copy(), targets)
    return sub


class LoMatchingProj:
    def __init__(self, circuit, d: int | None = None, stab=None, allow_gauge_detectors: bool = False):
        dem = circuit.detector_error_model(allow_gauge_detectors=allow_gauge_detectors)
        self.det_inds = get_detector_indices_for_subgraphs_from_dem(dem, stab if stab is not None else stab_coords(d))
        self.m = [Matching.from_detector_error_model(projected_dem(dem, inds, k))
                  for k, inds in enumerate(self.det_inds)]
        self.settings = {"package": f"lomatching {lomatching.__version__} regions, 0.1.0-style projection",
                         "subgraph_detectors": [len(x) for x in self.det_inds]}

    def decode(self, dets):
        return np.array([m.decode(dets[inds])[0] for m, inds in zip(self.m, self.det_inds)], dtype=bool)

    def decode_batch(self, dets):
        return np.stack([m.decode_batch(dets[:, inds])[:, 0] for m, inds in zip(self.m, self.det_inds)], axis=1)


class LoMatchingRegions:
    """LoMatching 0.3.0's decoder (MoMatching: per observable, PyMatching on the circuit re-analysed by stim with only
    the region's detectors, decompose_errors=True) on observing regions given by the caller, for circuits whose
    observables have no single-detector boundary edge in some slices (the centre-line logicals of the Clifford stage),
    where ``from_circuit``'s region search misses slices. The Clifford stage passes duet.clifford_circuits.regions
    (Definition 1 of the LoMatching paper, by back-propagation through the logical circuit)."""

    def __init__(self, circuit, regions, ignore_decomposition_failures=False):
        """ignore_decomposition_failures: build where stim cannot split an error of a region's model into graphlike
        pieces, without that error (duet.decoders.cain2025.graphlike; per observable counts in self.dropped)."""
        from lomatching.util import get_circuit_subgraph
        inds = [np.asarray(sorted(r), dtype=int) for r in regions]
        subs, self.dropped = [], []
        for r in inds:
            dem, n = graphlike(get_circuit_subgraph(circuit, r).detector_error_model(
                decompose_errors=True, ignore_decomposition_failures=ignore_decomposition_failures))
            subs.append(dem)
            self.dropped.append(n)
        self.dec = MoMatching(det_inds_subgraphs=inds, dem_subgraphs=subs, dem=circuit.detector_error_model())
        self.settings = {"package": f"lomatching {lomatching.__version__}", "class": "MoMatching, regions given",
                         "subgraph_detectors": [len(x) for x in inds], "dropped": self.dropped}

    def decode(self, dets):
        return self.dec.decode(dets)

    def decode_batch(self, dets):
        return self.dec.decode_batch(dets)


def missing_pieces(dem):
    """The pieces of decomposed errors that have no error of their own, as lists of targets."""
    have, need = set(), {}
    for ins in dem.flattened():
        if ins.type != "error":
            continue
        comps = [[]]
        for t in ins.targets_copy():
            if t.is_separator():
                comps.append([])
            else:
                comps[-1].append(t)
        dets = [frozenset(t.val for t in c if t.is_relative_detector_id()) for c in comps]
        if len(comps) == 1:
            have.add(dets[0])
        else:
            for k, c in zip(dets, comps):
                need.setdefault(k, c)
    return [c for k, c in need.items() if k not in have]


def _hld_graph(orig, pieces):
    """lomatching.util.dem_to_hld_graph for models where stim split an error into a piece that occurs nowhere else
    (an observable-only piece, or with ignore_decomposition_failures a new edge), which dem_to_hld_graph cannot look
    up. Each such piece becomes an edge (an error of probability 0, which the graph does not use) whose BP support is
    only the errors it is a piece of; their number per observable is appended to `pieces`."""
    def f(dem, det_inds_subgraph):
        miss = missing_pieces(dem)
        pieces.append(len(miss))
        if not miss:
            return orig(dem, det_inds_subgraph)
        dem = dem.flattened()
        for c in miss:
            dem.append("error", [0.0], c)
        added = {frozenset(t.val for t in c if t.is_relative_detector_id()) for c in miss}
        h, l, support = orig(dem, det_inds_subgraph)
        return h, l, [sup[1:] if sup[0] in added else sup for sup in support]
    return f


class _Graphlike:
    """A region's circuit inside BeliefMoMatching's constructor: its detector error model leaves the errors stim cannot
    decompose whole and drops them (counts appended to `dropped`)."""

    def __init__(self, circuit, dropped):
        self.circuit, self.dropped = circuit, dropped

    def detector_error_model(self, **kw):
        dem, n = graphlike(self.circuit.detector_error_model(ignore_decomposition_failures=True, **kw))
        self.dropped.append(n)
        return dem


class BpLomRegions:
    """BP+LoMatching (lomatching 0.3.0's BeliefMoMatching, min-sum BP) on observing regions given
    by the caller (see LoMatchingRegions): BeliefMoMatching computes its regions inside its constructor with
    lomatching.decoder.get_detector_indices_for_subgraphs_from_dem, which is swapped for the given regions while the
    decoder is built (and restored after). lomatching.decoder.dem_to_hld_graph is swapped too (_hld_graph): it fails
    on a model where stim split an error into a piece that has no error of its own; the wrapper adds such pieces and changes nothing elsewhere.

    ignore_decomposition_failures: BeliefMoMatching's matching stage takes each
    region's model from stim with decompose_errors=True and Matching.from_check_matrix refuses a column with more than
    two detectors. With the option, lomatching.decoder.get_circuit_subgraph is swapped too, for a wrapper whose model
    leaves such errors whole and drops them (cain2025.graphlike; per observable counts in self.dropped). The BP stage
    keeps the full model (decompose_errors=False), every error included."""

    def __init__(self, circuit, regions, d: int, max_iter: int = 100, ms_scaling_factor: float = 0.5,
                 ignore_decomposition_failures=False):
        import lomatching.decoder as LD
        from lomatching import BeliefMoMatching
        inds = [np.asarray(sorted(r), dtype=int) for r in regions]
        orig, orig_sub, orig_hld = (LD.get_detector_indices_for_subgraphs_from_dem, LD.get_circuit_subgraph,
                                    LD.dem_to_hld_graph)
        self.dropped, self.pieces = [], []
        LD.get_detector_indices_for_subgraphs_from_dem = lambda dem, sc, filter_dets=None: [x.copy() for x in inds]
        LD.dem_to_hld_graph = _hld_graph(orig_hld, self.pieces)
        if ignore_decomposition_failures:
            LD.get_circuit_subgraph = lambda *a, **k: _Graphlike(orig_sub(*a, **k), self.dropped)
        try:
            self.dec = BeliefMoMatching(circuit, stab_coords(d), bp_method="min_sum", max_iter=max_iter)
        finally:
            LD.get_detector_indices_for_subgraphs_from_dem, LD.get_circuit_subgraph = orig, orig_sub
            LD.dem_to_hld_graph = orig_hld
        if not ignore_decomposition_failures:
            self.dropped = [0] * len(inds)
        self.dec._bp_decoder.max_iter = max_iter
        self.dec._bp_decoder.ms_scaling_factor = ms_scaling_factor
        self.settings = {"package": f"lomatching {lomatching.__version__}", "class": "BeliefMoMatching, regions given",
                         "max_iter": max_iter, "ms_scaling_factor": ms_scaling_factor, "dropped": self.dropped,
                         "pieces_added": self.pieces}

    def decode(self, row):
        return np.asarray(self.dec.decode(row), bool)
