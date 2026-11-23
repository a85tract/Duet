"""Stim circuits: rotated surface-code blocks joined by layers of transversal CNOTs.

Circuit of one experiment (``make_circuit``)
--------------------------------------------
1. Transversal preparation. Every data qubit of block b is reset in basis ``prep[b]`` (``R`` for Z, ``RX`` for X), so
   block b holds logical |0> or |+>.
2. ``L`` layers. Layer l applies a transversal CNOT between block 0 (A) and block 1 (B): data qubit k of the control
   block with data qubit k of the target block, for every k. ``directions[l] = 0`` means A controls B, 1 means B
   controls A. Each CNOT layer is followed by ``rounds_per_layer`` rounds of syndrome extraction on both blocks at once
   (1 in the pilot; d in the control experiment). No round separates preparation and the first CNOT (as in Cain et al.
   2025 and Bonilla Ataides et al. 2025; LoMatching's paper has one).
3. Transversal measurement of every data qubit of block b in basis ``meas[b]`` (``M`` or ``MX``).

Default directions: drawn i.i.d. uniform from {0, 1} by ``numpy.random.default_rng(seed).integers(0, 2, size=L)``
with seed ``DEFAULT_SEED`` (``cnot_directions``). Stim circuits carry no comments, so result files record the
directions through ``describe``.

Code. Rotated surface code of odd distance d, laid out as in stim's generated circuits: data qubits at (2i+1, 2j+1),
i, j in [0, d); measure qubits at even coordinates; weight-2 X stabilizers on the boundaries y = 0 and y = 2d,
weight-2 Z stabilizers on x = 0 and x = 2d. Logical Z = Z on the row y = 1, logical X = X on the column x = 1.
Block b is shifted by ``b * (2d + 4)`` in x. Syndrome extraction: reset measure qubits, H on the X-type ones, four CNOT
layers, H, measure. CNOT orders (offsets from the measure qubit to its data qubit) are chosen so that the hook error of
an X (Z) stabilizer, the two data qubits of its last two CNOTs, lies perpendicular to the logical X (Z) operator, which
keeps the circuit distance at d (tests check this).

Noise model (uniform circuit-level depolarizing noise, one parameter p; "SD6"-style):
- after every single-qubit gate (the H of X-type syndrome extraction): DEPOLARIZE1(p) on its qubit;
- after every two-qubit gate (syndrome-extraction CNOTs and the transversal CNOTs): DEPOLARIZE2(p) on the pair;
- every qubit idle during a layer (between two TICKs) that contains a gate, reset or measurement on other qubits:
  DEPOLARIZE1(p);
- after every reset: X_ERROR(p) after R, Z_ERROR(p) after RX;
- every measurement result is flipped with probability p (``M(p)``, ``MX(p)``).
No other noise; in particular no extra noise for qubit motion and no leakage or loss. This is the uniform model of
LoMatching's circuit library (surface-sim ``CircuitNoiseModel``, all parameters p); Cain et al. 2025 use the same
two-qubit channel but depolarizing p at resets and measurements and no single-qubit-gate or idle noise.

Detectors. Each stabilizer measurement is compared with the value it must have given earlier measurements and the
CNOTs since then: a CNOT c->t maps Z_s(t) -> Z_s(c) Z_s(t) and X_s(c) -> X_s(c) X_s(t) (Heisenberg picture), so
after a CNOT A->B the next Z_s(B) result is compared with the XOR of the last Z_s(A) and Z_s(B) results. Stabilizers
that are random after preparation are tracked as random symbols; a detector is any XOR of results whose expected value
has no random symbol left (``_Frames``). Detector coordinates: (x, y, t) with (x, y) the measure qubit's coordinates and
t the round index (0, 1...; the final data measurement has t = number of rounds). This is the format
``lomatching.MoMatching`` needs.

Observables. The reliable logical measurement products (Cain et al. 2025): products of final logical measurement
results that are deterministic in the noiseless circuit, found by tracking the logical operators like the stabilizers.
One ``OBSERVABLE_INCLUDE`` per product of an independent generating set. With both blocks prepared and measured in the
same basis (the pilot), the transversal CNOTs map Z-type (X-type) logicals to Z-type (X-type) products, the logical
state stays |00> (|++>), and both single-block logicals are reliable: observable 0 is block A's, observable 1 is block
B's. A shot fails if any observable is predicted wrong.
"""

from __future__ import annotations

import itertools

import numpy as np
import stim

DEFAULT_SEED = 2026

# Offsets from a measure qubit to its data qubits, in CNOT order. The last two offsets of X_ORDER differ in x only
# (hook along x, perpendicular to logical X on a column); those of Z_ORDER differ in y only (perpendicular to logical
# Z on a row). Both orders touch every data qubit at most once per layer and give commuting X/Z measurements.
X_ORDER = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
Z_ORDER = [(1, 1), (1, -1), (-1, 1), (-1, -1)]


def cnot_directions(L: int, seed: int = DEFAULT_SEED) -> list[int]:
    """Directions of the L transversal CNOT layers: 0 means block A controls B, 1 means B controls A."""
    return [int(v) for v in np.random.default_rng(seed).integers(0, 2, size=L)]


def rotated_patch(d: int) -> dict:
    """Rotated surface code of distance d, unshifted: coordinates of data, X- and Z-measure qubits, logicals."""
    if d < 3 or d % 2 == 0:
        raise ValueError(f"d must be odd and >= 3, got {d}")
    data = [(2 * i + 1, 2 * j + 1) for j in range(d) for i in range(d)]
    xm, zm = [], []
    for j in range(d + 1):
        for i in range(d + 1):
            parity = (i % 2) != (j % 2)
            if parity and i in (0, d):
                continue
            if not parity and j in (0, d):
                continue
            (xm if parity else zm).append((2 * i, 2 * j))
    data_set = set(data)
    supp = {m: [(m[0] + dx, m[1] + dy) for dx in (-1, 1) for dy in (-1, 1) if (m[0] + dx, m[1] + dy) in data_set]
            for m in xm + zm}
    log_z = [q for q in data if q[1] == 1]
    log_x = [q for q in data if q[0] == 1]
    # sanity: stabilizers commute with the other type's logical, and the logicals anticommute
    assert all(len(set(supp[m]) & set(log_z)) % 2 == 0 for m in xm)
    assert all(len(set(supp[m]) & set(log_x)) % 2 == 0 for m in zm)
    assert len(set(log_x) & set(log_z)) % 2 == 1 and len(xm) == len(zm) == (d * d - 1) // 2
    return {"d": d, "data": data, "X": xm, "Z": zm, "supp": supp, "LX": log_x, "LZ": log_z}


def block_offset(d: int, b: int) -> int:
    return b * (2 * d + 4)


def stab_coords(d: int, n_blocks: int = 2) -> list[dict[str, list[tuple[float, float]]]]:
    """Measure-qubit coordinates per block and type, in the format of ``lomatching.MoMatching.from_circuit``."""
    patch = rotated_patch(d)
    return [{t: [(float(x + block_offset(d, b)), float(y)) for x, y in patch[t]] for t in "XZ"}
            for b in range(n_blocks)]


class _Frames:
    """Expected noiseless value of every tracked operator, as a set of terms whose XOR it equals.

    Terms >= 0 are absolute measurement-record indices; terms < 0 are random symbols (values fixed by preparation but
    unknown). Keys are (block, type, s) with s a measure-qubit index or "L" for the block's logical operator.
    """

    def __init__(self):
        self.e: dict[tuple, set[int]] = {}
        self._next_symbol = -1

    def prepare(self, b: int, basis: str, stabs: dict[str, int]):
        other = "X" if basis == "Z" else "Z"
        for s in list(range(stabs[basis])) + ["L"]:
            self.e[(b, basis, s)] = set()
        for s in list(range(stabs[other])) + ["L"]:
            self.e[(b, other, s)] = {self._next_symbol}
            self._next_symbol -= 1

    def cnot(self, c: int, t: int, stabs: dict[str, int]):
        for s in list(range(stabs["Z"])) + ["L"]:
            self.e[(t, "Z", s)] ^= self.e[(c, "Z", s)]
        for s in list(range(stabs["X"])) + ["L"]:
            self.e[(c, "X", s)] ^= self.e[(t, "X", s)]


def _deterministic_combinations(slots: list[tuple[set[int], set[int]]]) -> list[tuple[list[int], set[int]]]:
    """Independent subsets of measured slots whose expected value has no random symbol.

    ``slots[i] = (result_terms, expected_terms)``. Returns ``(subset, record_terms)`` per combination, singletons first;
    ``record_terms`` is the XOR of results and expectations, all measurement records.
    """
    basis: list[int] = []  # GF(2) basis of chosen subsets, as bitmasks, kept in echelon form by leading bit
    out = []
    n = len(slots)
    for size in range(1, n + 1):
        for subset in itertools.combinations(range(n), size):
            terms: set[int] = set()
            for i in subset:
                terms ^= slots[i][0] ^ slots[i][1]
            if any(t < 0 for t in terms):
                continue
            mask = sum(1 << i for i in subset)
            for v in basis:
                mask = min(mask, mask ^ v)
            if mask == 0:
                continue
            basis.append(mask)
            basis.sort(reverse=True)
            out.append((list(subset), terms))
    return out


def make_circuit(d: int, L: int, p: float, basis: str = "Z", rounds_per_layer: int = 1,
                 directions: list[int] | None = None, seed: int = DEFAULT_SEED,
                 prep: tuple[str, str] | None = None, meas: tuple[str, str] | None = None) -> stim.Circuit:
    """Two rotated distance-d blocks, L transversal CNOT layers, ``rounds_per_layer`` rounds after each layer.

    ``basis`` sets both ``prep`` and ``meas`` of both blocks unless they are given explicitly. Noise strength ``p``
    (0 gives the noiseless circuit). See the module docstring for the exact construction and noise model.
    """
    if directions is None:
        directions = cnot_directions(L, seed)
    if len(directions) != L or any(v not in (0, 1) for v in directions):
        raise ValueError(f"need {L} directions in {{0, 1}}, got {directions}")
    prep = prep or (basis, basis)
    meas = meas or (basis, basis)
    patch = rotated_patch(d)
    nb = 2
    stabs = {"X": len(patch["X"]), "Z": len(patch["Z"])}

    # qubit indices: per block, data then X-measure then Z-measure qubits
    index: dict[tuple, int] = {}
    coords: dict[int, tuple[int, int]] = {}
    for b in range(nb):
        dx = block_offset(d, b)
        for kind in ("data", "X", "Z"):
            for k, (x, y) in enumerate(patch[kind]):
                q = len(index)
                index[(b, kind, k)] = q
                coords[q] = (x + dx, y)
    data_q = [[index[(b, "data", k)] for k in range(len(patch["data"]))] for b in range(nb)]
    data_k = {q: k for k, q in enumerate(patch["data"])}
    anc = [(b, t, s) for b in range(nb) for t in "XZ" for s in range(stabs[t])]
    anc_q = [index[(b, t, s)] for b, t, s in anc]
    x_anc_q = [index[(b, "X", s)] for b in range(nb) for s in range(stabs["X"])]
    all_q = set(index.values())

    c = stim.Circuit()
    for q in sorted(coords):
        c.append("QUBIT_COORDS", [q], coords[q])

    def noise(name, targets, prob=p):
        if prob > 0 and targets:
            c.append(name, targets, prob)

    def end_layer(busy):
        noise("DEPOLARIZE1", sorted(all_q - set(busy)))
        c.append("TICK")

    def measure(name, targets):
        if p > 0:
            c.append(name, targets, p)
        else:
            c.append(name, targets)

    def cx_layer(pairs):
        flat = [q for pair in pairs for q in pair]
        c.append("CX", flat)
        noise("DEPOLARIZE2", flat)
        end_layer(flat)

    frames = _Frames()
    n_meas = 0
    rnd = 0

    def emit_detectors(groups):
        """groups: list of lists of (coord_q, result_terms, key) for slots that may combine."""
        for group in groups:
            slots = [(res, frames.e[key]) for _, res, key in group]
            for subset, terms in _deterministic_combinations(slots):
                x, y = coords[group[subset[-1]][0]]
                c.append("DETECTOR", [stim.target_rec(t - n_meas) for t in sorted(terms)], (x, y, rnd))

    # 1. transversal preparation
    for b in range(nb):
        if prep[b] == "Z":
            c.append("R", data_q[b])
            noise("X_ERROR", data_q[b])
        else:
            c.append("RX", data_q[b])
            noise("Z_ERROR", data_q[b])
        frames.prepare(b, prep[b], stabs)
    end_layer([q for b in range(nb) for q in data_q[b]])

    for layer in range(L):
        # 2a. transversal CNOT
        ctrl, targ = (0, 1) if directions[layer] == 0 else (1, 0)
        cx_layer(list(zip(data_q[ctrl], data_q[targ])))
        frames.cnot(ctrl, targ, stabs)
        # 2b. syndrome extraction rounds
        for _ in range(rounds_per_layer):
            c.append("R", anc_q)
            noise("X_ERROR", anc_q)
            end_layer(anc_q)
            c.append("H", x_anc_q)
            noise("DEPOLARIZE1", x_anc_q)
            end_layer(x_anc_q)
            for step in range(4):
                pairs = []
                for b, t, s in anc:
                    m = patch[t][s]
                    off = (X_ORDER if t == "X" else Z_ORDER)[step]
                    dq = (m[0] + off[0], m[1] + off[1])
                    if dq in data_k:
                        a, dqi = index[(b, t, s)], data_q[b][data_k[dq]]
                        pairs.append((a, dqi) if t == "X" else (dqi, a))
                cx_layer(pairs)
            c.append("H", x_anc_q)
            noise("DEPOLARIZE1", x_anc_q)
            end_layer(x_anc_q)
            measure("M", anc_q)
            rec = {key: n_meas + i for i, key in enumerate(anc)}
            n_meas += len(anc)
            emit_detectors([[(index[(b, t, s)], {rec[(b, t, s)]}, (b, t, s)) for b in range(nb)]
                            for t in "XZ" for s in range(stabs[t])])
            for key, r in rec.items():
                frames.e[key] = {r}
            end_layer(anc_q)
            rnd += 1

    # 3. transversal measurement
    data_rec = {}
    for b in range(nb):
        measure("M" if meas[b] == "Z" else "MX", data_q[b])
        for k in range(len(patch["data"])):
            data_rec[(b, k)] = n_meas + k
        n_meas += len(patch["data"])
    groups = []
    for t in "XZ":
        for s in range(stabs[t]):
            group = []
            for b in range(nb):
                if meas[b] == t:
                    res = {data_rec[(b, data_k[q])] for q in patch["supp"][patch[t][s]]}
                    group.append((index[(b, t, s)], res, (b, t, s)))
            if group:
                groups.append(group)
    emit_detectors(groups)
    logicals = [({data_rec[(b, data_k[q])] for q in patch["L" + meas[b]]}, frames.e[(b, meas[b], "L")])
                for b in range(nb)]
    for k, (subset, terms) in enumerate(_deterministic_combinations(logicals)):
        expected: set[int] = set()
        for i in subset:
            expected ^= logicals[i][1]
        assert not expected  # logical frames hold random symbols only, so a reliable product expects 0
        c.append("OBSERVABLE_INCLUDE", [stim.target_rec(t - n_meas) for t in sorted(terms)], k)
    return c


def describe(d: int, L: int, basis: str = "Z", rounds_per_layer: int = 1, seed: int = DEFAULT_SEED) -> dict:
    """Parameters that identify a pilot circuit, for result files."""
    return {"d": d, "L": L, "basis": basis, "rounds_per_layer": rounds_per_layer, "seed": seed,
            "directions": "".join(map(str, cnot_directions(L, seed)))}


def dem_counts(circuit: stim.Circuit, d: int) -> dict:
    """Size of the detector error model and of its non-graphlike part.

    ``hyperedges``: undecomposed errors with more than 2 detectors (most are Y-type errors that flip X- and Z-type
    detectors); ``hyperedges_one_type``: errors with more than 2 detectors of one stabilizer type (the hyperedges a
    matching decoder of one error sector meets); ``undecomposable``: errors stim's ``decompose_errors`` cannot split into
    graphlike parts (left whole with ``ignore_decomposition_failures``).
    """
    z_coords = {xy for block in stab_coords(d) for xy in block["Z"]}
    is_z = {k: tuple(v[:2]) in z_coords for k, v in circuit.get_detector_coordinates().items()}
    n_err = n_hyper = n_hyper_type = 0
    for inst in circuit.detector_error_model(decompose_errors=False).flattened():
        if inst.type == "error":
            dets = [t.val for t in inst.targets_copy() if t.is_relative_detector_id()]
            n_z = sum(is_z[k] for k in dets)
            n_err += 1
            n_hyper += len(dets) > 2
            n_hyper_type += n_z > 2 or len(dets) - n_z > 2
    n_undec = 0
    dem2 = circuit.detector_error_model(decompose_errors=True, ignore_decomposition_failures=True)
    for inst in dem2.flattened():
        if inst.type == "error":
            comp = 0
            for t in inst.targets_copy() + [stim.DemTarget.separator()]:
                if t.is_separator():
                    n_undec += comp > 2
                    if comp > 2:
                        break
                    comp = 0
                elif t.is_relative_detector_id():
                    comp += 1
    return {"detectors": circuit.num_detectors, "observables": circuit.num_observables, "error_mechanisms": n_err,
            "hyperedges": n_hyper, "hyperedges_one_type": n_hyper_type, "undecomposable": n_undec}


if __name__ == "__main__":
    import json
    import sys

    rows = []
    for d in (3, 5, 7, 9):
        for rpl in (1, d):
            info = describe(d, 8, "Z", rpl)
            rows.append({**info, "p": 0.002, **dem_counts(make_circuit(d, 8, 0.002, rounds_per_layer=rpl), d)})
            print(json.dumps(rows[-1]), flush=True)
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as f:
            json.dump(rows, f, indent=1)
