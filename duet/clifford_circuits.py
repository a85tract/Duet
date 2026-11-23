"""Stim circuits of the Clifford family: K rotated surface-code blocks under layers of transversal Clifford gates. Generalises
duet/circuits.py's make_circuit (same code, layout, syndrome extraction, noise model and detector rule), whose CNOT-only
circuits it reproduces bit for bit (tests/test_clifford_circuits.py); the two-block circuits are those of the two-block
code before the K-block change, bit for bit (tests/test_circuits_multi.py).

Layers. A circuit is a list of layer tokens, each followed by one round of syndrome extraction on every block. A token
has one character per block (K = len(prep) blocks, block b at x offset block_offset(d, b)):
  a digit j the block controls a transversal CNOT whose target is block j (DEPOLARIZE2(p) on every pair);
  "+" the block is the target of the CNOT whose control names it;
  a letter a single-block gate, one of
               I nothing;
               H transversal H on every data qubit of the block (DEPOLARIZE1(p) after each; one tick, before the
                      round, the other qubits idle). The block's code becomes the 90-degree-rotated code in place:
                      every measure qubit keeps its cell and its CNOT order and measures the other Pauli type (the
                      block's frame f flips 0 <-> 1);
               S fold-transversal S embedded in the layer's syndrome round (Chen, Chen, Lu, Pan, PRL 2025,
                      arXiv 2412.01391, and their code): after the round's second CNOT step the block's data and
                      measure qubits hold an unrotated surface code (the half-cycle state); CZ on every pair of its
                      qubits mirrored about the patch's anti-diagonal (data with data, bulk measure qubits with bulk
                      measure qubits; boundary measure qubits stay out), S on the fold line's data qubits and S_DAG on
                      its measure qubits, then the round goes on (one extra tick: DEPOLARIZE2(p) on the CZ pairs,
                      DEPOLARIZE1(p) on the S / S_DAG qubits, idle noise on every other qubit);
               X Y Z Paulis, applied in software (a Pauli frame): no gate and no noise, so the stim circuit is the
                      one of I; they change only the expected logical outcomes (``logical_products``). With
                      pauli="gate" they are physical
                      instead: the Pauli on every data qubit of the block in the layer's gate tick, then
                      DEPOLARIZE1 at the model's single-qubit rate; transversal Paulis commute with every check, so the
                      detectors are those of the frame version and the reference sample carries the Pauli.
The CNOT pairs of a layer are disjoint, and the CNOTs and H gates of a layer share one tick before its round. Two-block
tokens: "C0" (= "1+", block A controls) and "C1" (= "+0"); "gh" (g on A, h on B) is already of the K-character form.
At a CNOT the control's data qubit at position p meets the target's data qubit at pi(p): pi is the identity when both
blocks are in the same frame, otherwise the 90-degree rotation about the patch centre that maps the rotated code onto
the original one (the noiseless lattice rotation of MCCD and of Bonilla Ataides et al.; Chen et al. move the atoms with
two AOD reflections). Frames are tracked through the reference map phi_0 = identity, phi_1 = R90 (physical position ->
reference position): pi = phi_t^-1 phi_c, so compositions stay consistent.

Detectors. As make_circuit: every stabilizer measurement is compared with the value its check must have from earlier
results (random values after preparation are tracked as symbols; deterministic XOR combinations across the blocks at
the same reference cell are kept, singletons first, each placed on the highest block of its combination). The
relation of a check across a round comes from stim's flows: an ordinary round measures each check directly (in = out =
its own result); in a round with the fold, each check q has the results whose parity equals its value before the
round (``in``; X checks take the mirrored Z checks of the same round, as Chen et al.'s three-result detectors) and
those equal to its value after the round (``out``), solved by stim on a one-block round (``round_flows``).
Coordinates (x, y, t): the measure qubit and the round (the final data measurement has t = L).

Noise models (``noise``). "uniform" (the default, make_circuit's): every operation at p, i.e. DEPOLARIZE2(p) after
each two-qubit gate, DEPOLARIZE1(p) after each single-qubit gate, X_ERROR / Z_ERROR(p) after each reset, measurement
flips p, DEPOLARIZE1(p) on every qubit idle in a tick, and the data qubits prepared by noisy transversal resets.
"lukin": DEPOLARIZE2(p) after each
two-qubit gate (transversal CNOT, syndrome CNOT, fold CZ), DEPOLARIZE1(p/10) after each single-qubit gate (H, S,
S_DAG), X_ERROR(p/10) after each reset and measurement flips p/10, no idle noise; each block starts in its exact code
state |0_L> or |+_L> from a noiseless encoding unitary (``encoder``), so the first round's checks of both types have
fixed values and all of them are detectors. Two references for the matching decoders of the open benchmark, both with noisy transversal resets as "uniform": "si1000" (Gidney and Fowler's SI1000 as
LoMatching ran it through surface-sim, with the CNOT taken as a native two-qubit gate): DEPOLARIZE2(p) after each
two-qubit gate, DEPOLARIZE1(p/10) after each single-qubit gate and on every qubit idle in a tick, X_ERROR / Z_ERROR(2p)
after each reset, measurement flips 5p, and DEPOLARIZE1(2p) in addition to the p/10 on the qubits idle while the measure
qubits are reset or measured; "cain" (Cain et al., arXiv 2505.13587, App. E): DEPOLARIZE2(p) after each two-qubit gate, DEPOLARIZE1(p) after
each reset and before each measurement, written as its exact equivalent, a flip with probability 2p/3 (the third
Pauli leaves the reset state and the measured value alone), noiseless single-qubit gates, no idle noise.
Switches: noise "<model>/<switch>..." zeroes components of the model: dreset
(the data qubits' preparation), areset (the measure qubits' resets), dmeas (the final data measurement), ameas (the
measure qubits' measurements), idle, g1 (single-qubit gates), g2 (two-qubit gates); enc prepares each block by the
noiseless encoder with every first-round check a detector (as "lukin"), encu by the encoder with the detectors of the
product-state preparation only. Without a switch every circuit is the model's, bit for bit (tests/test_short_noise.py).

Observables. The reliable products (Cain et al. 2025) of the K final logical measurements, from the K-qubit logical
Clifford tableau (``logical_products``: products whose expectation is +-1 in the noiseless logical circuit, an
independent set, singletons first). Each one reads the final data records on the measured logical's representative
line of its blocks (``logical_line``): Z on a row in frame 0 and on a column in frame 1, X the other way round; the line
is make_circuit's (y = 1, x = 1) in CNOT-only circuits and the centre line (y = d, x = d) otherwise (``line0``).
"""

from __future__ import annotations

import functools
import itertools

import numpy as np
import stim

from duet.circuits import X_ORDER, Z_ORDER, _deterministic_combinations, block_offset, rotated_patch

ONE = "IHSXYZ"
PAULI = "XYZ"
FOLD = "anti"        # the fold of the data grid: "anti" (x, y) -> (2d - y, 2d - x), "main" (x, y) -> (y, x)
S_FIRST = "S"        # the gate on the fold's data qubits; its measure qubits take the inverse
SWITCHES = ("dreset", "areset", "dmeas", "ameas", "idle", "g1", "g2", "enc", "encu")   # make_circuit's noise switches


def parse_k(layers, K: int | None = None) -> list[tuple[tuple[tuple[int, int], ...], str]]:
    """Layer tokens of K blocks (module docstring; K from the first token when not given) -> [(pairs, gates)]: pairs
    ((control, target)...) in the order of the controls' blocks, gates the K single-block gate letters (I on the
    blocks of a CNOT)."""
    out = []
    for tok in layers:
        if K is None:
            K = 2 if tok in ("C0", "C1") else len(tok)
        if tok in ("C0", "C1") and K == 2:
            tok = "1+" if tok == "C0" else "+0"
        bad = f"bad layer token {tok!r} for {K} blocks: one of {ONE}, a target's block or + per block (C0, C1 for two)"
        if len(tok) != K or K > 10:
            raise ValueError(bad)
        pairs = []
        for b, ch in enumerate(tok):
            if ch.isdigit():
                t = int(ch)
                if t >= K or t == b or tok[t] != "+":
                    raise ValueError(bad)
                pairs.append((b, t))
            elif ch != "+" and ch not in ONE:
                raise ValueError(bad)
        if tok.count("+") != len(pairs) or len({t for _, t in pairs}) != len(pairs):
            raise ValueError(bad)
        out.append((tuple(pairs), "".join(ch if ch in ONE else "I" for ch in tok)))
    return out


def parse(layers) -> list[tuple]:
    """Two-block layer tokens -> [("C", control) | ("1", gate_A, gate_B)] (the two-block network's tables)."""
    return [("C", pairs[0][0]) if pairs else ("1", gates[0], gates[1]) for pairs, gates in parse_k(layers, 2)]


def rot90(d: int, xy):
    """R90 about the patch centre (d, d): maps data to data, measure qubits to measure qubits, the code onto its
    H-image (and back)."""
    x, y = xy
    return (2 * d - y, x)


def rot270(d: int, xy):
    x, y = xy
    return (y, 2 * d - x)


def fold_map(d: int, xy, kind: str = FOLD):
    x, y = xy
    return (2 * d - y, 2 * d - x) if kind == "anti" else (y, x)


def line0(d: int, ops) -> int:
    """The coordinate of the logicals' representative line: 1 (make_circuit's row y = 1 and column x = 1) in circuits of
    CNOT and idle layers only, d (the centre row and column) as soon as an H or S occurs on any block. The rotations
    R90, R270 and the fold map the centre row onto the centre column and back, so a gate never moves a logical off its
    representative (row / column 1 would pick up stabilizers whose values are random or recorded). ops: parse_k's
    (pairs, gates) per layer or parse's two-block tuples."""
    def gates(op):
        return op[1] if isinstance(op[0], tuple) else ("".join(op[1:]) if op[0] == "1" else "")
    return d if any(set(gates(op)) & set("HS") for op in ops) else 1


def logical_line(d: int, basis: str, f: int, c0: int):
    """Data coordinates of the representative of the block's logical `basis` in frame f: Z on the row y = c0 and X on
    the column x = c0 in frame 0, the other way round in frame 1."""
    row = (basis == "Z") == (f == 0)
    return [(2 * i + 1, c0) for i in range(d)] if row else [(c0, 2 * j + 1) for j in range(d)]


def cur_type(t_static: str, f: int) -> str:
    """The Pauli type a measure qubit of static type t_static measures in frame f."""
    return t_static if f == 0 else ("Z" if t_static == "X" else "X")


def fold_gates(d: int, kind: str = FOLD, s_first: str = S_FIRST):
    """(CZ pairs, S qubits, S_DAG qubits) of the fold as coordinates of one block: every data qubit and every bulk
    measure qubit (Chen et al.'s half-cycle code; the boundary measure qubits stay out) pairs with its mirror image; on
    the fold line the data qubits take s_first and the measure qubits the other one (Chen et al.: S on data, S_DAG on
    ancillas)."""
    patch = rotated_patch(d)
    qs = list(patch["data"]) + [m for t in "XZ" for m in patch[t] if 0 < m[0] < 2 * d and 0 < m[1] < 2 * d]
    pairs, fixed = [], []
    for q in qs:
        m = fold_map(d, q, kind)
        if m == q:
            fixed.append(q)
        elif q < m:
            pairs.append((q, m))
    other = "S_DAG" if s_first == "S" else "S"
    is_data = set(patch["data"])
    s_q = sorted(q for q in fixed if (q in is_data) == (s_first == "S"))
    sd_q = sorted(q for q in fixed if (q in is_data) != (s_first == "S"))
    assert len(pairs) * 2 + len(fixed) == len(qs) and other
    return pairs, s_q, sd_q


def _round_ops(patch, f: int):
    """One block's syndrome round in frame f: (current-X measure qubits, [4 lists of (measure qubit, data qubit,
    measure-is-control)]) in make_circuit's order (static type X then Z, index s)."""
    data_set = set(patch["data"])
    xs = [patch[t][s] for t in "XZ" for s in range(len(patch[t])) if cur_type(t, f) == "X"]
    steps = []
    for step in range(4):
        st = []
        for t in "XZ":
            for s, m in enumerate(patch[t]):
                off = (X_ORDER if t == "X" else Z_ORDER)[step]
                dq = (m[0] + off[0], m[1] + off[1])
                if dq in data_set:
                    st.append((m, dq, cur_type(t, f) == "X"))
        steps.append(st)
    return xs, steps


@functools.lru_cache(maxsize=None)
def round_flows(d: int, f: int, fold: bool, kind: str = FOLD, s_first: str = S_FIRST):
    """Per measure qubit (t_static, s) of one block in frame f: (in, out) as lists of measurement indices of the round
    (make_circuit's ancilla order: static X then Z). in: results whose parity equals the check's value before the
    round; out: those equal to its value after it. Solved with stim's flows on a noiseless one-block round."""
    patch = rotated_patch(d)
    idx = {q: k for k, q in enumerate(patch["data"])}
    anc = [(t, s) for t in "XZ" for s in range(len(patch[t]))]
    for k, (t, s) in enumerate(anc):
        idx[patch[t][s]] = len(patch["data"]) + k
    xs, steps = _round_ops(patch, f)
    c = stim.Circuit()
    a_q = [idx[patch[t][s]] for t, s in anc]
    c.append("R", a_q)
    c.append("H", [idx[m] for m in xs])
    for k, st in enumerate(steps):
        c.append("CX", [v for m, dq, mc in st for v in ((idx[m], idx[dq]) if mc else (idx[dq], idx[m]))])
        if fold and k == 1:
            pairs, s_q, sd_q = fold_gates(d, kind, s_first)
            c.append("CZ", [idx[q] for pr in pairs for q in pr])
            c.append("S", [idx[q] for q in s_q])
            c.append("S_DAG", [idx[q] for q in sd_q])
    c.append("H", [idx[m] for m in xs])
    c.append("M", a_q)
    n = len(idx)
    flows, keys = [], []
    for t, s in anc:
        ty = cur_type(t, f)
        ps = stim.PauliString(n)
        for q in patch["supp"][patch[t][s]]:
            ps[idx[q]] = ty
        flows += [stim.Flow(input=ps, output=stim.PauliString(n)), stim.Flow(input=stim.PauliString(n), output=ps)]
        keys.append((t, s))
    sol = c.solve_flow_measurements(flows)
    out = {}
    for k, key in enumerate(keys):
        a, b = sol[2 * k], sol[2 * k + 1]
        if a is None or b is None:
            raise ValueError(f"d={d} f={f} fold={fold}: check {key} has no flow through the round ({a}, {b})")
        out[key] = (sorted(a), sorted(b))
    return out


def logical_products(layers, prep, meas, s_gate: str = "S", basis=None):
    """The reliable products of the final logical measurements from the K-qubit logical tableau (K = len(prep)):
    [(blocks, value)], an independent set, singletons first (the enumeration of make_circuit). value is the noiseless
    outcome (0 or 1), Pauli frame included. s_gate: the logical gate the fold applies ("S" or "S_DAG"). basis: the
    products to use instead, in this order (block lists; each must be reliable and the set independent), e.g. a basis
    carried through a relabelling of the blocks."""
    K = len(prep)
    sim = stim.TableauSimulator()
    sim.set_num_qubits(K)
    for b in range(K):
        if prep[b] == "X":
            sim.h(b)
    for pairs, gates in parse_k(layers, K):
        for c, t in pairs:
            sim.cnot(c, t)
        for b, g in enumerate(gates):
            if g == "H":
                sim.h(b)
            elif g == "S":
                (sim.s if s_gate == "S" else sim.s_dag)(b)
            elif g in PAULI:
                getattr(sim, g.lower())(b)
    if basis is not None:
        return _given_products(sim, K, basis, meas)
    out, basis = [], []
    for size in range(1, K + 1):
        for sub in itertools.combinations(range(K), size):
            ps = stim.PauliString(K)
            for b in sub:
                ps[b] = meas[b]
            e = sim.peek_observable_expectation(ps)
            if e == 0:
                continue
            mask = sum(1 << b for b in sub)
            for v in basis:
                mask = min(mask, mask ^ v)
            if mask == 0:
                continue
            basis.append(mask)
            basis.sort(reverse=True)
            out.append((list(sub), int(e < 0)))
    return out


def _given_products(sim, K, given, meas):
    out, masks = [], []
    for sub in given:
        sub = sorted(int(b) for b in sub)
        ps = stim.PauliString(K)
        for b in sub:
            ps[b] = meas[b]
        e = sim.peek_observable_expectation(ps)
        mask = sum(1 << b for b in sub)
        for v in masks:
            mask = min(mask, mask ^ v)
        if e == 0 or mask == 0:
            raise ValueError(f"product {sub} is {'random' if e == 0 else 'dependent on the others'}")
        masks.append(mask)
        masks.sort(reverse=True)
        out.append((sub, int(e < 0)))
    return out


def frames_of(layers, K: int | None = None) -> list[tuple[int, ...]]:
    """The frames (f_0..., f_K-1) during round t = 0.. L - 1 (after layer t's gate)."""
    ops = parse_k(layers, K)
    f, out = [0] * (len(ops[0][1]) if ops else 0), []
    for _, gates in ops:
        for b, g in enumerate(gates):
            if g == "H":
                f[b] ^= 1
        out.append(tuple(f))
    return out


class _Frames:
    """make_circuit's frame tracker keyed by measure qubit (b, t_static, s): the expected value of the check that
    measure qubit measures (its type follows the block's frame), as a set of record indices (>= 0) and symbols (< 0)."""

    def __init__(self):
        self.e: dict[tuple, set[int]] = {}
        self._next = -1

    def prepare(self, b: int, basis: str, stabs: dict[str, int], code_state: bool = False):
        for t in "XZ":
            for s in range(stabs[t]):
                if t == basis or code_state:
                    self.e[(b, t, s)] = set()
                else:
                    self.e[(b, t, s)] = {self._next}
                    self._next -= 1


@functools.lru_cache(maxsize=None)
def encoder(d: int, basis: str) -> tuple[list[int], list[tuple[int, int]]]:
    """The noiseless encoding unitary of one block for noise="lukin": (pivots, CX pairs) as data-qubit indices.
    basis "Z": from |0...0> (R), H on the pivots, then CX pivot -> j for each row of the X checks' matrix in reduced
    row echelon form over GF(2), whose pivot columns appear in one row each; the state is prod_g (1 + X_g) |0...0> up to
    normalisation, which is |0_L> with every check and Z_L at +1 (the CX pairs commute: pivots only control, the other
    qubits only take). basis "X": the Hadamard dual, from |+...+> (RX), H on the pivots of the Z checks' matrix, then
    CX j -> pivot, giving |+_L>."""
    patch = rotated_patch(d)
    data_k = {q: k for k, q in enumerate(patch["data"])}
    other = "X" if basis == "Z" else "Z"
    rows = np.zeros((len(patch[other]), len(patch["data"])), np.uint8)
    for r, m in enumerate(patch[other]):
        for q in patch["supp"][m]:
            rows[r, data_k[q]] = 1
    piv, r = [], 0
    for col in range(rows.shape[1]):
        hit = [i for i in range(r, len(rows)) if rows[i, col]]
        if not hit:
            continue
        rows[[r, hit[0]]] = rows[[hit[0], r]]
        for i in range(len(rows)):
            if i != r and rows[i, col]:
                rows[i] ^= rows[r]
        piv.append(col)
        r += 1
        if r == len(rows):
            break
    assert r == len(rows), "dependent checks"
    pairs = [(pc, int(j)) for i, pc in enumerate(piv) for j in np.flatnonzero(rows[i]) if j != pc]
    return piv, (pairs if basis == "Z" else [(j, pc) for pc, j in pairs])


def make_circuit(d: int, layers, p: float, prep=("Z", "Z"), meas=("Z", "Z"), kind: str = FOLD,
                 s_first: str = S_FIRST, s_gate: str = "S", meta: bool = False, c0: int | None = None,
                 noise: str = "uniform", products=None, pauli: str = "frame"):
    """K = len(prep) rotated distance-d blocks, the given layers (module docstring), one syndrome round per layer,
    noise model ``noise`` ("uniform", "lukin", "si1000" or "cain", module docstring) at p; Paulis in software
    (pauli="frame") or as noisy physical gates (pauli="gate").
    Returns the stim circuit, or (circuit, meta) with meta = dict(frames, det_block, det_xy, det_t, det_type,
    products, mirror) when meta is True."""
    nb = len(prep)
    if len(meas) != nb:
        raise ValueError(f"{nb} preparation bases but {len(meas)} readout bases")
    ops = parse_k(layers, nb)
    L = len(ops)
    patch = rotated_patch(d)
    stabs = {"X": len(patch["X"]), "Z": len(patch["Z"])}
    cell = {patch[t][s]: (t, s) for t in "XZ" for s in range(stabs[t])}

    index: dict[tuple, int] = {}
    coords: dict[int, tuple[int, int]] = {}
    for b in range(nb):
        dx = block_offset(d, b)
        for kind_q in ("data", "X", "Z"):
            for k, (x, y) in enumerate(patch[kind_q]):
                q = len(index)
                index[(b, kind_q, k)] = q
                coords[q] = (x + dx, y)
    data_q = [[index[(b, "data", k)] for k in range(len(patch["data"]))] for b in range(nb)]
    data_k = {q: k for k, q in enumerate(patch["data"])}
    qat = {(b, patch[kq][k]): q for (b, kq, k), q in index.items()}     # (block, unshifted coordinates) -> qubit
    anc = [(b, t, s) for b in range(nb) for t in "XZ" for s in range(stabs[t])]
    anc_q = [index[(b, t, s)] for b, t, s in anc]
    all_q = set(index.values())

    c = stim.Circuit()
    for q in sorted(coords):
        c.append("QUBIT_COORDS", [q], coords[q])

    p_mr = 0.0                    # extra idle rate while the measure qubits reset or measure (si1000), on top of p_idle
    noise, _, sw = noise.partition("/")          # switches: "<model>/<switch>..." (docstring)
    switches = set(filter(None, sw.split(",")))
    if switches - set(SWITCHES):
        raise ValueError(f"noise switches {sorted(switches - set(SWITCHES))}: of {SWITCHES}")
    if noise == "uniform":
        p1 = p2 = p_reset = p_meas = p_idle = p
    elif noise == "lukin":
        p2, p1, p_reset, p_meas, p_idle = p, p / 10, p / 10, p / 10, 0.0
    elif noise == "si1000":
        p2, p1, p_reset, p_meas, p_idle, p_mr = p, p / 10, 2 * p, 5 * p, p / 10, 2 * p
    elif noise == "cain":
        p2, p1, p_reset, p_meas, p_idle = p, 0.0, 2 * p / 3, 2 * p / 3, 0.0
    else:
        raise ValueError(f"noise model {noise!r}")
    p_dreset = 0.0 if "dreset" in switches else p_reset
    p_reset = 0.0 if "areset" in switches else p_reset
    p_dmeas = 0.0 if "dmeas" in switches else p_meas
    p_meas = 0.0 if "ameas" in switches else p_meas
    if "idle" in switches:
        p_idle = p_mr = 0.0
    p1 = 0.0 if "g1" in switches else p1
    p2 = 0.0 if "g2" in switches else p2
    encode = noise == "lukin" or bool(switches & {"enc", "encu"})
    if pauli not in ("frame", "gate"):
        raise ValueError(f"pauli {pauli!r}: 'frame' or 'gate'")

    def add_noise(name, targets, prob):
        if prob > 0 and targets:
            c.append(name, targets, prob)

    def end_layer(busy, extra=0.0):
        idle = sorted(all_q - set(busy))
        add_noise("DEPOLARIZE1", idle, p_idle)
        add_noise("DEPOLARIZE1", idle, extra)
        c.append("TICK")

    def measure(name, targets, prob):
        if prob > 0:
            c.append(name, targets, prob)
        else:
            c.append(name, targets)

    def cx_layer(pairs):
        flat = [q for pair in pairs for q in pair]
        c.append("CX", flat)
        add_noise("DEPOLARIZE2", flat, p2)
        end_layer(flat)

    frames = _Frames()
    f = [0] * nb
    n_meas = 0
    rnd = 0
    det_info = []                                        # (block, x, y, t, type) per detector

    def phi(b, xy):                                      # physical -> reference position
        return rot90(d, xy) if f[b] else xy

    def phi_inv(b, xy):
        return rot270(d, xy) if f[b] else xy

    def pair_map(ctrl, targ, xy):                       # control position -> its partner's position in the target
        return xy if f[ctrl] == f[targ] else phi_inv(targ, phi(ctrl, xy))

    def pair_map_inv(ctrl, targ, xy):
        return xy if f[ctrl] == f[targ] else phi_inv(ctrl, phi(targ, xy))

    def emit_detectors(groups):
        """groups: lists of (block, coord_q, result_terms, key, type) for slots that may combine."""
        for group in groups:
            slots = [(res, frames.e[key]) for _, _, res, key, _ in group]
            for subset, terms in _deterministic_combinations(slots):
                b, cq, _, _, ty = group[subset[-1]]
                x, y = coords[cq]
                c.append("DETECTOR", [stim.target_rec(t - n_meas) for t in sorted(terms)], (x, y, rnd))
                det_info.append((b, x - block_offset(d, b), y, rnd, ty))

    def ref_groups(make_slot):
        """Slots grouped by reference cell, in make_circuit's order (static type, index), blocks 0 then 1."""
        groups = []
        for t in "XZ":
            for s in range(stabs[t]):
                group = []
                for b in range(nb):
                    slot = make_slot(b, cell[phi_inv(b, patch[t][s])], t)
                    if slot is not None:
                        group.append(slot)
                if group:
                    groups.append(group)
        return groups

    # 1. preparation: noisy transversal resets, or (lukin) the noiseless encoding of the exact code state
    for b in range(nb):
        if encode:
            piv, pairs = encoder(d, prep[b])
            c.append("R" if prep[b] == "Z" else "RX", data_q[b])
            c.append("H", [data_q[b][k] for k in piv])
            c.append("CX", [data_q[b][k] for pair in pairs for k in pair])
            frames.prepare(b, prep[b], stabs, code_state="encu" not in switches)
            continue
        if prep[b] == "Z":
            c.append("R", data_q[b])
            add_noise("X_ERROR", data_q[b], p_dreset)
        else:
            c.append("RX", data_q[b])
            add_noise("Z_ERROR", data_q[b], p_dreset)
        frames.prepare(b, prep[b], stabs)
    end_layer([q for b in range(nb) for q in data_q[b]])

    mirror = {}
    for layer, (cnots, gates) in enumerate(ops):
        # 2a. the layer's gates: its CNOT pairs and H gates in one tick (the S folds sit inside the round)
        cxq, new = [], dict(frames.e)
        for ctrl, targ in cnots:
            cxq += [(data_q[ctrl][k], data_q[targ][data_k[pair_map(ctrl, targ, q)]])
                    for k, q in enumerate(patch["data"])]
            for t in "XZ":
                for s in range(stabs[t]):
                    m = patch[t][s]
                    if cur_type(t, f[targ]) == "Z":
                        src = cell[pair_map_inv(ctrl, targ, m)]
                        new[(targ, t, s)] = frames.e[(targ, t, s)] ^ frames.e[(ctrl, *src)]
                    if cur_type(t, f[ctrl]) == "X":
                        new[(ctrl, t, s)] = frames.e[(ctrl, t, s)] ^ frames.e[(targ, *cell[pair_map(ctrl, targ, m)])]
        frames.e = new
        hs = [b for b in range(nb) if gates[b] == "H"]
        hq = [q for b in hs for q in data_q[b]]
        flat = [q for pair in cxq for q in pair]
        if flat:
            c.append("CX", flat)
            add_noise("DEPOLARIZE2", flat, p2)
        if hq:
            c.append("H", hq)
            add_noise("DEPOLARIZE1", hq, p1)
        pq = [q for b in range(nb) if pauli == "gate" and gates[b] in PAULI for q in data_q[b]]
        for g in PAULI if pq else "":
            gq = [q for b in range(nb) if gates[b] == g for q in data_q[b]]
            if gq:
                c.append(g, gq)
        add_noise("DEPOLARIZE1", pq, p1)
        if flat or hq or pq:
            end_layer(flat + hq + pq)
        for b in hs:
            f[b] ^= 1
        folds = [b for b in range(nb) if gates[b] == "S"]
        # 2b. one syndrome extraction round (the fold of an S inside it)
        c.append("R", anc_q)
        add_noise("X_ERROR", anc_q, p_reset)
        end_layer(anc_q, p_mr)
        x_anc_q = [index[(b, t, s)] for b in range(nb) for t in "XZ" for s in range(stabs[t]) if cur_type(t, f[b]) == "X"]
        c.append("H", x_anc_q)
        add_noise("DEPOLARIZE1", x_anc_q, p1)
        end_layer(x_anc_q)
        for step in range(4):
            pairs = []
            for b, t, s in anc:
                m = patch[t][s]
                off = (X_ORDER if t == "X" else Z_ORDER)[step]
                dq = (m[0] + off[0], m[1] + off[1])
                if dq in data_k:
                    a, dqi = index[(b, t, s)], data_q[b][data_k[dq]]
                    pairs.append((a, dqi) if cur_type(t, f[b]) == "X" else (dqi, a))
            cx_layer(pairs)
            if step == 1 and folds:
                cz, s_q, sd_q = [], [], []
                for b in folds:
                    prs, s1, s2 = fold_gates(d, kind, s_first)
                    cz += [qat[(b, q)] for pr in prs for q in pr]
                    s_q += [qat[(b, q)] for q in s1]
                    sd_q += [qat[(b, q)] for q in s2]
                c.append("CZ", cz)
                add_noise("DEPOLARIZE2", cz, p2)
                c.append("S", s_q)
                c.append("S_DAG", sd_q)
                add_noise("DEPOLARIZE1", s_q + sd_q, p1)
                end_layer(cz + s_q + sd_q)
        c.append("H", x_anc_q)
        add_noise("DEPOLARIZE1", x_anc_q, p1)
        end_layer(x_anc_q)
        measure("M", anc_q, p_meas)
        rec = {key: n_meas + i for i, key in enumerate(anc)}
        base = {b: n_meas + i0 for b, i0 in ((b, anc.index((b, "X", 0))) for b in range(nb))}
        n_meas += len(anc)
        flows = {b: round_flows(d, f[b], b in folds, kind, s_first) for b in range(nb)}
        for b in folds:
            mirror[(layer, b)] = {key: [k for k in io[0]] for key, io in flows[b].items()}

        def in_slot(b, key, t_ref):
            res = {base[b] + k for k in flows[b][key][0]}
            return (b, index[(b, *key)], res, (b, *key), t_ref)

        emit_detectors(ref_groups(in_slot))
        for b in range(nb):
            for key, (_, out) in flows[b].items():
                frames.e[(b, *key)] = {base[b] + k for k in out}
        end_layer(anc_q, p_mr)
        rnd += 1

    # 3. transversal measurement
    data_rec = {}
    for b in range(nb):
        measure("M" if meas[b] == "Z" else "MX", data_q[b], p_dmeas)
        for k in range(len(patch["data"])):
            data_rec[(b, k)] = n_meas + k
        n_meas += len(patch["data"])

    def fin_slot(b, key, t_ref):
        if meas[b] != t_ref:
            return None
        res = {data_rec[(b, data_k[q])] for q in patch["supp"][patch[key[0]][key[1]]]}
        return (b, index[(b, *key)], res, (b, *key), t_ref)

    emit_detectors(ref_groups(fin_slot))
    products = logical_products(layers, prep, meas, s_gate, products)
    c0 = line0(d, ops) if c0 is None else c0
    for k, (sub, _) in enumerate(products):
        terms: set[int] = set()
        for b in sub:
            terms ^= {data_rec[(b, data_k[q])] for q in logical_line(d, meas[b], f[b], c0)}
        c.append("OBSERVABLE_INCLUDE", [stim.target_rec(t - n_meas) for t in sorted(terms)], k)
    if not meta:
        return c
    return c, dict(frames=frames_of(layers, nb), det_block=np.array([r[0] for r in det_info], np.int64),
                   det_xy=np.array([(r[1], r[2]) for r in det_info], np.int64),
                   det_t=np.array([r[3] for r in det_info], np.int64),
                   det_type=np.array([0 if r[4] == "X" else 1 for r in det_info], np.int64),
                   products=products, mirror=mirror, final_frames=tuple(f))


def cnot_layers(directions) -> list[str]:
    """make_circuit's directions (0: A controls B) as layer tokens."""
    return ["C0" if v == 0 else "C1" for v in directions]


SEED_BASE, SEED_SPAN = 18_420_000_000_000_000_000, 10 ** 16   # stim seeds of these circuits: [1.842e19, 1.843e19) < 2^64
KBLOCK_SEED_BASE = 18_430_000_000_000_000_000                  # (K blocks): [1.843e19, 1.844e19)
P_CNOT = 1 / 3       # the prototype's layer mix: a CNOT (direction uniform) or one gate per block uniform in I H S


def sample_circuit(rng, L: int, p_cnot: float = P_CNOT, gates: str = "IHS", min_products: int = 1):
    """One random circuit of the prototype's family: (layers, prep, meas). Layers i.i.d.: a CNOT with probability
    p_cnot (either direction), otherwise one gate per block drawn uniformly from `gates`; preparation and readout bases
    uniform in Z, X per block. Circuits with fewer than min_products reliable products are drawn again."""
    while True:
        layers = []
        for _ in range(L):
            if rng.random() < p_cnot:
                layers.append("C0" if rng.random() < 0.5 else "C1")
            else:
                layers.append(gates[rng.integers(len(gates))] + gates[rng.integers(len(gates))])
        prep = tuple("ZX"[i] for i in rng.integers(2, size=2))
        meas = tuple("ZX"[i] for i in rng.integers(2, size=2))
        if len(logical_products(layers, prep, meas)) >= min_products:
            return layers, prep, meas


def sample_circuit_k(rng, K: int, L: int, p_pair: float = P_CNOT, gates: str = "IHS", min_products: int = 1,
                     prep=None, meas=None):
    """One random K-block circuit (layers, prep, meas) of the K-block family: per layer the
    blocks are shuffled and taken in consecutive pairs, each pair is a CNOT with probability p_pair in a uniform
    direction, and every block outside a CNOT gets a gate drawn uniformly from `gates`; bases uniform in Z, X per block;
    circuits with fewer than min_products reliable products are drawn again. For K = 2 the law of sample_circuit (a
    CNOT layer with probability p_pair, otherwise one gate per block) with other generator calls. prep, meas: fixed
    bases instead (e.g. all Z for the cnot family)."""
    while True:
        layers = []
        for _ in range(L):
            tok = [""] * K
            perm = rng.permutation(K)
            for i in range(0, K - 1, 2):
                if rng.random() < p_pair:
                    c, t = (int(perm[i]), int(perm[i + 1])) if rng.random() < 0.5 else (int(perm[i + 1]), int(perm[i]))
                    tok[c], tok[t] = str(t), "+"
            layers.append("".join(ch or gates[rng.integers(len(gates))] for ch in tok))
        pr = tuple("ZX"[i] for i in rng.integers(2, size=K)) if prep is None else tuple(prep)
        me = tuple("ZX"[i] for i in rng.integers(2, size=K)) if meas is None else tuple(meas)
        if len(logical_products(layers, pr, me)) >= min_products:
            return layers, pr, me


def _logical_ops(op):
    """The logical gate tick and the fold of one layer (parse_k's (pairs, gates)) as K-qubit stim circuits (Paulis
    dropped)."""
    tick, fold = stim.Circuit(), stim.Circuit()
    for c, t in op[0]:
        tick.append("CX", [c, t])
    for b, g in enumerate(op[1]):
        if g == "H":
            tick.append("H", [b])
        elif g == "S":
            fold.append("S", [b])
    return tick, fold


def observable_parts(layers, blocks, meas) -> np.ndarray:
    """(L + 1, K, 2) uint8 [t, block, X part / Z part], K = len(meas): the logical Pauli of the product of the final
    measurements of `blocks`, back-propagated to the start of round t (after layer t's H or CNOT, before its fold);
    t = L is the product itself at the final measurement. Over the K-qubit logical circuit (stim.PauliString.before)."""
    K = len(meas)
    ops = parse_k(layers, K)
    L = len(ops)
    ps = stim.PauliString(K)
    for b in blocks:
        ps[b] = meas[b]
    out = np.zeros((L + 1, K, 2), np.uint8)

    def put(t, q):
        for b in range(K):
            out[t, b] = (q[b] in (1, 2), q[b] in (2, 3))          # stim: 0 I, 1 X, 2 Y, 3 Z

    put(L, ps)
    q = ps
    for t in range(L - 1, -1, -1):
        tick, fold = _logical_ops(ops[t])
        if len(fold):
            q = q.before(fold)
        put(t, q)
        if len(tick):
            q = q.before(tick)
    return out


def regions(meta, layers, meas) -> list[np.ndarray]:
    """Per observable, its observing region in the sense of LoMatching's Definition 1 (and Cain et al.'s V_k): the
    detectors of round t (and of the final measurement) of block b whose check type is a part of the observable's
    logical back-propagated to the start of round t."""
    out = []
    for sub, _ in meta["products"]:
        parts = observable_parts(layers, sub, meas)
        keep = parts[meta["det_t"], meta["det_block"], meta["det_type"]] == 1
        out.append(np.flatnonzero(keep))
    return out
