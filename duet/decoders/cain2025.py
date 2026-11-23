"""Correlated decoder of Cain et al. 2025 (arXiv 2505.13587): one matching problem per reliable logical product.

Each stim observable of the circuit is taken to be one reliable logical Pauli product. For observable k the decoder
keeps only the checks (detectors) on the back-propagation path of the product (its decoding subgraph V_k, the paper's
App. B): it builds the circuit with only those detectors and that observable, lets stim derive its detector error model
(every error mechanism restricted to V_k, equal restricted mechanisms merged, correlated errors decomposed into graphlike
components), and decodes it with PyMatching. Products are decoded independently.

V_k comes from one of three sources:
  "coords": detector coordinates list the products whose subgraph holds the detector (the format of the paper's
            Zenodo circuits, e.g. DETECTOR(0, 3, 9) belongs to products 0, 3 and 9);
  "auto": computed from the circuit with the paper's rule: at the start of each syndrome-extraction (SE) round, the
            check of a stabilizer of block j and basis B is kept iff the back-propagated product has a B component
            (B or Y) on block j. stim detecting regions supply both the stabilizer of each check and the product;
  "all": every product uses all detectors (plain matching on the full model, for memory circuits).

CLI:
  python cain2025.py --circuit c.stim --dets dets.b8 --pred pred.b8 [--times times.npy] [--subgraphs auto]
reads detection events (stim shot-data format), writes predicted observable flips in the same format and, with
--times, the per-shot single-thread decode time in nanoseconds (summed over products) as a.npy array.
"""

import argparse
import itertools
import time

import numpy as np
import pymatching
import stim

ANNOTATIONS = {"DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS", "TICK"}


# ----------------------------------------------------------------------------------------------- subgraphs (V_k)

def subgraphs_from_coords(circuit):
    """V_k from detector coordinates that list product indices (the Zenodo convention)."""
    coords = circuit.get_detector_coordinates()
    members = [[] for _ in range(circuit.num_observables)]
    for det, c in coords.items():
        for v in c:
            k = int(round(v))
            if k != v or not 0 <= k < circuit.num_observables:
                raise ValueError(f"detector {det} coordinate {v} is not a product index")
            members[k].append(det)
    return [np.array(sorted(m), dtype=np.int64) for m in members]


def _measured_qubits(instr):
    return {t.qubit_value for t in instr.targets_copy() if t.is_qubit_target or t.is_x_target
            or t.is_y_target or t.is_z_target}


def _num_measurements(instr):
    c = stim.Circuit()
    c.append(instr)
    return c.num_measurements


class _Block:
    """Union-find over qubits."""

    def __init__(self):
        self.parent = {}

    def find(self, q):
        self.parent.setdefault(q, q)
        while self.parent[q] != q:
            self.parent[q] = self.parent[self.parent[q]]
            q = self.parent[q]
        return q

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def subgraphs_auto(circuit):
    """V_k from the circuit by the paper's rule; returns (list of V_k, info dict).

    Round ticks: for every measurement instruction that ends a detector, a tick after the gate layer and before the
    round's first ancilla-data gate (for a measurement without ancillas, e.g. the final data readout: just before it and
    any basis change on the measured qubits). Init tick: before the first two-qubit gate. At a round tick, a detector's
    detecting region (ancillas removed) is the product of stabilizers it checks, and an observable's region is the
    back-propagated product P_k^t. A detector belongs to V_k iff at every tick where its region is non-empty, each block
    it touches is a block where P_k^t has a component of the region's type (X, or Z; Y counts for both). For ordinary
    checks only the tick of their latest round is non-empty, which is the paper's rule; first-round checks are also
    seen at the init tick, which keeps only those built from initial stabilizers of blocks the product terminates on.
    """
    flat = circuit.flattened()
    instrs = [ins for ins in flat if ins.name != "TICK"]
    gd = {name: stim.gate_data(name) for name in {ins.name for ins in instrs}}

    # Pass 1: measurement records, measurement layers (runs of measurements with no unitary gate between), resets.
    rec_instr, layer_of, layer_first, layer_last = [], {}, [], []
    last_reset_at, resets_before, det_recs = {}, {}, []
    unitary_since_meas, layer, first_2q = True, -1, None
    for i, ins in enumerate(instrs):
        g = gd[ins.name]
        if ins.name == "DETECTOR":
            n = len(rec_instr)
            det_recs.append([n + t.value for t in ins.targets_copy() if t.is_measurement_record_target])
            continue
        if ins.name in ANNOTATIONS:
            continue
        if g.produces_measurements:
            if unitary_since_meas:
                layer += 1
                layer_first.append(i)
                layer_last.append(i)
                unitary_since_meas = False
            layer_of[i] = layer
            layer_last[layer] = i
            resets_before[i] = dict(last_reset_at)
            rec_instr.extend([i] * _num_measurements(ins))
        elif g.is_unitary:
            unitary_since_meas = True
            if first_2q is None and g.is_two_qubit_gate:
                first_2q = i
        if g.is_reset:
            for q in _measured_qubits(ins):
                last_reset_at[q] = i

    # Pass 2: tick position and ancillas for every measurement instruction that ends a detector; blocks.
    det_instr = []
    for d, recs in enumerate(det_recs):
        if not recs:
            raise ValueError(f"detector {d} has no measurement records")
        det_instr.append(rec_instr[max(recs)])
    # A measured qubit is an ancilla of its instruction if it was reset in this round (since the previous measurement
    # layer began) and has not served as the data partner of an earlier instruction's ancillas since that reset. The
    # second condition matters only when one round precedes the data readout: both then sit in
    # the first measurement layer, where the data qubits' initial resets also count as resets of the round.
    tick_at, ancillas, data_at = {}, set(), {}
    blocks = _Block()
    for i in sorted(set(det_instr)):
        lay = layer_of[i]
        start = layer_last[lay - 1] + 1 if lay > 0 else 0
        prev_first = layer_first[lay - 1] if lay > 0 else 0
        measured = _measured_qubits(instrs[i])
        anc = {q for q in measured if resets_before[i].get(q, -1) >= prev_first
               and data_at.get(q, -1) < resets_before[i][q]}
        tick = None
        if anc:
            for j in range(start, i):
                ins = instrs[j]
                g = gd[ins.name]
                if ins.name in ANNOTATIONS or not g.is_unitary or not g.is_two_qubit_gate:
                    continue
                ts = ins.targets_copy()
                if not all(t.is_qubit_target for t in ts):
                    continue                      # classically controlled gate: not an ancilla-data interaction
                qs = [t.qubit_value for t in ts]
                hits = [(x, y) for x, y in zip(qs[::2], qs[1::2]) if (x in anc) != (y in anc)]
                if hits:
                    tick = j if tick is None else tick
                    for x, y in hits:
                        blocks.union(x, y)
                        data_at[y if x in anc else x] = j
            if tick is None:
                raise ValueError(f"measurement instruction {i}: ancillas never interact with data in its round")
        else:
            j = i - 1
            while j >= start:
                ins = instrs[j]
                g = gd[ins.name]
                basis_change = g.is_unitary and g.is_single_qubit_gate and _measured_qubits(ins) <= measured
                if ins.name in ANNOTATIONS or (g.is_noisy_gate and not g.produces_measurements) or basis_change:
                    j -= 1
                    continue
                break
            tick = j + 1
        tick_at[i] = tick
        ancillas |= anc

    # Pass 3: circuit with one TICK per tick position; detecting regions of all detectors and observables.
    positions = sorted(set(tick_at.values()) | ({first_2q} if first_2q is not None else set()))
    tick_of = {pos: n for n, pos in enumerate(positions)}
    ticked = stim.Circuit()
    for i, ins in enumerate(instrs):
        if i in tick_of:
            ticked.append("TICK")
        ticked.append(ins)
    nq, K, D = ticked.num_qubits, circuit.num_observables, len(det_instr)
    label = np.array([blocks.find(q) for q in range(nq)])
    keep = np.ones(nq, bool)
    keep[sorted(ancillas)] = False

    def xz(ps):
        xs, zs = ps.to_numpy()
        out_x, out_z = np.zeros(nq, bool), np.zeros(nq, bool)
        out_x[:len(xs)], out_z[:len(zs)] = xs, zs
        return out_x & keep, out_z & keep

    ticks = list(range(len(positions)))
    obs_regions = ticked.detecting_regions(targets=[stim.DemTarget.logical_observable_id(k) for k in range(K)],
                                           ticks=ticks)
    xblocks = [[set() for _ in ticks] for _ in range(K)]      # blocks where P_k^t has an X (Z) component
    zblocks = [[set() for _ in ticks] for _ in range(K)]
    for k in range(K):
        for tk, ps in obs_regions.get(stim.DemTarget.logical_observable_id(k), {}).items():
            ox, oz = xz(ps)
            xblocks[k][tk], zblocks[k][tk] = set(label[ox].tolist()), set(label[oz].tolist())
    det_regions = ticked.detecting_regions(targets=[stim.DemTarget.relative_detector_id(d) for d in range(D)],
                                           ticks=ticks)
    init_tk = tick_of.get(first_2q)
    regions = []          # per detector: {tick: {block: (X-support, Z-support)}}, ancillas removed
    for d in range(D):
        reg = {}
        for tk, ps in det_regions.get(stim.DemTarget.relative_detector_id(d), {}).items():
            x, z = xz(ps)
            if x.any() and z.any():
                raise ValueError(f"detector {d}: mixed X/Z region at tick {tk} (tick misplaced?)")
            per = {}
            for q in np.flatnonzero(x | z):
                per.setdefault(int(label[q]), set()).add(int(q))
            if per:
                reg[tk] = {bl: (frozenset(qs), frozenset()) if x.any() else (frozenset(), frozenset(qs))
                           for bl, qs in per.items()}
        if not reg or max(reg) != tick_of[tick_at[det_instr[d]]]:
            raise ValueError(f"detector {d}: no data region at the start of its latest round")
        regions.append(reg)

    def covered(reg, k):
        return all((not xs or bl in xblocks[k][tk]) and (not zs or bl in zblocks[k][tk])
                   for tk, per in reg.items() for bl, (xs, zs) in per.items())

    def xor(regs):
        out = {}
        for reg in regs:
            for tk, per in reg.items():
                o = out.setdefault(tk, {})
                for bl, (xs, zs) in per.items():
                    ox, oz = o.get(bl, (frozenset(), frozenset()))
                    o[bl] = (ox ^ xs, oz ^ zs)
        out = {tk: {bl: v for bl, v in per.items() if v[0] or v[1]} for tk, per in out.items()}
        return {tk: per for tk, per in out.items() if per}

    # First-round checks (sensitive at the init tick) are grouped by shared initial stabilizers; within a group the
    # sparsest XOR combinations that satisfy the rule are the product's first-round checks.
    first = [d for d in range(D) if init_tk in regions[d] and len(regions[d]) > 1]
    uf = _Block()
    for d in first:
        for bl, sup in regions[d][init_tk].items():
            uf.union(("D", d), ("S", bl, sup))
    groups = {}
    for d in first:
        groups.setdefault(uf.find(("D", d)), []).append(d)
    groups = [sorted(g) for g in groups.values()]
    if any(len(g) > 12 for g in groups):
        raise ValueError("a group of first-round checks is too large to enumerate")
    first_set = set(first)
    out, n_combos = [], []
    for k in range(K):
        singles = [d for d in range(D) if d not in first_set and covered(regions[d], k)]
        combos = []
        for g in groups:
            basis = []
            for size in range(1, len(g) + 1):
                for sub in itertools.combinations(range(len(g)), size):
                    reg = xor([regions[g[i]] for i in sub])
                    if not reg or not covered(reg, k):
                        continue
                    mask = sum(1 << i for i in sub)
                    for v in basis:
                        mask = min(mask, mask ^ v)
                    if mask == 0:
                        continue
                    basis.append(mask)
                    basis.sort(reverse=True)
                    if size == 1:
                        singles.append(g[sub[0]])
                    else:
                        combos.append(tuple(g[i] for i in sub))
        out.append((np.array(sorted(singles), dtype=np.int64), combos))
        n_combos.append(len(combos))
    info = {"num_ticks": len(positions), "num_blocks": len(set(label[keep].tolist())),
            "first_round_detectors": len(first), "combined_checks": n_combos}
    return out, info


# ------------------------------------------------------------------------------------------- error model, graphs

def product_circuit(instrs, singles, combos, k):
    """The circuit with only product k's checks and observable k (renumbered 0): the detectors in `singles` (kept in
    place, in order), then one detector per combination in `combos` (XOR of those detectors), appended at the end."""
    keep = set(singles.tolist())
    need = {d for c in combos for d in c}
    recs, out, d, n_meas = {}, stim.Circuit(), 0, 0
    for ins in instrs:
        if ins.name == "DETECTOR":
            if d in need:
                recs[d] = {n_meas + t.value for t in ins.targets_copy() if t.is_measurement_record_target}
            if d in keep:
                out.append(ins)
            d += 1
        elif ins.name == "OBSERVABLE_INCLUDE":
            if int(ins.gate_args_copy()[0]) == k:
                out.append("OBSERVABLE_INCLUDE", ins.targets_copy(), [0])
        else:
            out.append(ins)
            if ins.name not in ANNOTATIONS and stim.gate_data(ins.name).produces_measurements:
                n_meas += _num_measurements(ins)
    for c in combos:
        rs = set()
        for m in c:
            rs ^= recs[m]
        out.append("DETECTOR", [stim.target_rec(r - n_meas) for r in sorted(rs)])
    return out


def dem_stats(dem):
    """Error mechanisms, graphlike components, components with > 2 detectors, detector sets whose components differ in
    their effect on the product, and the total probability of undetected flips of the product."""
    errors = components = big = 0
    effect, undetected = {}, []
    for ins in dem.flattened():
        if ins.type != "error":
            continue
        errors += 1
        p = ins.args_copy()[0]
        comp = [[]]
        for t in ins.targets_copy():
            if t.is_separator():
                comp.append([])
            else:
                comp[-1].append(t)
        for c in comp:
            components += 1
            dets = tuple(sorted(t.val for t in c if t.is_relative_detector_id()))
            flip = any(t.is_logical_observable_id() for t in c)
            big += len(dets) > 2
            if not dets and flip:
                undetected.append(p)
            effect.setdefault(dets, set()).add(flip)
    return {"errors": errors, "components": components, "components_over_2": big,
            "conflicting_edges": sum(len(v) > 1 for v in effect.values()),
            "undetected_flip_prob": float(1 - np.prod(1 - 2 * np.array(undetected))) / 2 if undetected else 0.0}


def graphlike(dem):
    """(dem without its errors that have a component with more than two detectors, their count). Built with
    ignore_decomposition_failures=True, stim leaves an error it cannot split into graphlike pieces whole; PyMatching
    would drop it from its graph, and this drops it explicitly. dem itself when
    there is none; otherwise flattened, with its detector and observable counts kept."""
    out, n = stim.DetectorErrorModel(), 0
    for ins in dem.flattened():
        comp = [0]
        for t in ins.targets_copy() if ins.type == "error" else ():
            if t.is_separator():
                comp.append(0)
            else:
                comp[-1] += t.is_relative_detector_id()
        if max(comp) > 2:
            n += 1
        else:
            out.append(ins)
    if n == 0:
        return dem, 0
    if out.num_detectors < dem.num_detectors:
        out.append("detector", [], [stim.target_relative_detector_id(dem.num_detectors - 1)])
    if out.num_observables < dem.num_observables:
        out.append("logical_observable", [], [stim.target_logical_observable_id(dem.num_observables - 1)])
    return out, n


# --------------------------------------------------------------------------------------------------- the decoder

class CainDecoder:
    """Decode every observable of `circuit` as a reliable logical product on its own matchable subgraph."""

    def __init__(self, circuit, subgraphs="auto", with_stats=False, ignore_decomposition_failures=False):
        """ignore_decomposition_failures: build where stim cannot split an error of a product's model into graphlike
        pieces, without that error (graphlike; per product counts in self.dropped)."""
        D, K = circuit.num_detectors, circuit.num_observables
        self.num_detectors, self.num_observables = D, K
        self.info = {}
        if subgraphs == "coords":
            checks = [(V, []) for V in subgraphs_from_coords(circuit)]
        elif subgraphs == "auto":
            checks, self.info = subgraphs_auto(circuit)
        elif subgraphs == "all":
            checks = [(np.arange(D), [])] * K
        elif isinstance(subgraphs, str):
            raise ValueError(subgraphs)
        else:
            checks = [(np.asarray(x[0], np.int64), list(x[1])) if isinstance(x, tuple) else (np.asarray(x, np.int64), [])
                      for x in subgraphs]
        self.subgraphs = checks
        self.settings = {"package": f"cain2025 reimplementation, stim {stim.__version__}, "
                                    f"pymatching {pymatching.__version__}",
                         "subgraphs": subgraphs if isinstance(subgraphs, str) else "given",
                         "checks_per_product": [int(len(v) + len(c)) for v, c in checks],
                         "combined_checks": [len(c) for _, c in checks]}
        # syndrome of product k: its single checks, then its combined checks (XORs of consecutive gathered columns)
        self._single, self._combo = [], []
        for v, c in checks:
            self._single.append(np.asarray(v, np.int64))
            lens = [len(x) for x in c]
            self._combo.append((np.concatenate([np.asarray(x, np.int64) for x in c]),
                                np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)) if c else None)
        instrs = list(circuit.flattened())
        self.matchings, self.stats, self.dropped = [], [], []
        for k, (v, c) in enumerate(checks):
            if len(v) + len(c) == 0:
                raise ValueError(f"product {k} has an empty decoding subgraph")
            dem, n = graphlike(product_circuit(instrs, v, c, k).detector_error_model(
                decompose_errors=True, approximate_disjoint_errors=True,
                ignore_decomposition_failures=ignore_decomposition_failures))
            self.dropped.append(n)
            st = dem_stats(dem) if with_stats else {}
            st["checks"] = int(len(v) + len(c))
            self.stats.append(st)
            self.matchings.append(pymatching.Matching.from_detector_error_model(dem))

    def _syndrome(self, dets, k):
        """Product k's checks for one shot (1-D, numpy's fast single-index path) or a batch (2-D) of detection events."""
        single, combo = self._single[k], self._combo[k]
        if dets.ndim == 1:
            x = dets[single]
            return x if combo is None else np.concatenate((x, np.bitwise_xor.reduceat(dets[combo[0]], combo[1])))
        x = dets[:, single]
        return x if combo is None else np.concatenate(
            (x, np.bitwise_xor.reduceat(dets[:, combo[0]], combo[1], axis=1)), axis=1)

    def decode(self, dets):
        """One shot: bool or uint8 vector (num_detectors) in, bool vector of predicted observable flips out."""
        dets = np.asarray(dets)
        return np.array([m.decode(self._syndrome(dets, k))[0] for k, m in enumerate(self.matchings)], dtype=bool)

    def decode_batch(self, dets):
        """dets: bool or uint8 array (shots, num_detectors). Returns bool array (shots, num_observables)."""
        dets = np.asarray(dets, dtype=np.uint8)
        out = np.zeros((dets.shape[0], self.num_observables), dtype=bool)
        for k, m in enumerate(self.matchings):
            out[:, k] = m.decode_batch(np.ascontiguousarray(self._syndrome(dets, k)))[:, 0]
        return out

    def decode_timed(self, dets):
        """Per-shot decoding, single thread. Returns (predictions, ns per shot and product (shots, K))."""
        dets = np.asarray(dets)
        shots, K = dets.shape[0], self.num_observables
        out = np.zeros((shots, K), dtype=bool)
        ns = np.zeros((shots, K), dtype=np.int64)
        clock = time.perf_counter_ns
        for s in range(shots):
            row = dets[s]
            for k, m in enumerate(self.matchings):
                t0 = clock()
                out[s, k] = m.decode(self._syndrome(row, k))[0]
                ns[s, k] = clock() - t0
        return out, ns


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--circuit", required=True)
    ap.add_argument("--dets", required=True, help="detection events, stim shot-data file")
    ap.add_argument("--format", default="b8", help="stim shot-data format of --dets and --pred (default b8)")
    ap.add_argument("--pred", required=True, help="output: predicted observable flips")
    ap.add_argument("--times", help="output .npy: per-shot decode time in ns, summed over products (per-shot loop)")
    ap.add_argument("--subgraphs", default="auto", choices=["auto", "coords", "all"])
    a = ap.parse_args()
    circuit = stim.Circuit.from_file(a.circuit)
    dec = CainDecoder(circuit, subgraphs=a.subgraphs)
    dets = stim.read_shot_data_file(path=a.dets, format=a.format, num_detectors=circuit.num_detectors)
    if a.times:
        pred, ns = dec.decode_timed(dets)
        np.save(a.times, ns.sum(axis=1))
    else:
        pred = dec.decode_batch(dets)
    stim.write_shot_data_file(data=pred, path=a.pred, format=a.format, num_observables=circuit.num_observables)


if __name__ == "__main__":
    main()
