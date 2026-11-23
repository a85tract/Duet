"""The Fig. 1b benchmark of Bonilla Ataides, Gu, Yelin, Lukin, arXiv 2509.11370 ("Neural decoders for universal
quantum algorithms"), rebuilt in stim. Their Methods ("Numerical simulation details") and Fig. 1a are the source; the
items they leave open are marked [choice] and listed in notes/LUKIN_SPEC.md.

Logical circuit (``random_logical``)
------------------------------------
Four logical qubits (surface-code blocks), all prepared in |+>. Depth D = D layers; layer l:
1. a single-qubit Clifford on every block, drawn uniformly from {I, X, Y, Z, H};
2. CNOTs between "randomly selected pairs... in random orientations": a uniformly random perfect matching of the
   four blocks (one of three pairings, so two CNOTs per layer, as drawn in their Fig. 1a) [choice], each pair with
   control and target drawn uniformly;
3. one syndrome-extraction round on every block (their Methods). Their published MLE values are reproduced only with
   a second round after the single-qubit layer (``round_after_1q_layer=True``), which is what their released compiler
   produces when the single-qubit gates and the CNOTs are two algorithm layers.
After the last layer every block is measured transversally. The target is one stabilizer of the generated logical
stabilizer state whose factors are all I, X or Z (no Y, as in their Methods), drawn uniformly from all such
non-identity stabilizers [choice: they say only "a specified target stabilizer"]. Block b is measured in the basis of
the target's factor on b; blocks outside the target's support are measured in Z [choice]. A shot fails when the
decoded parity of the target differs from the true one. Circuits are drawn from
``numpy.random.default_rng([SEED, D, i])``, i = 0..19; the same 20 logical circuits are used at every distance.

Physical circuit (``make_circuit``)
-----------------------------------
Code: rotated surface code of odd distance d in the layout of ``duet.circuits.rotated_patch`` (data qubits at
(2i+1, 2j+1); weight-2 X stabilizers on y = 0, 2d; weight-2 Z stabilizers on x = 0, 2d; logical X = X on the column
x = 1, logical Z = Z on the row y = 1). Block b is shifted by b(2d+4) in x. Why rotated: see LUKIN_SPEC.md.
- Preparation, noiseless (their "noiseless unitary circuit"): RX on every data qubit, then one noiseless MPP of every
  Z stabilizer. The MPP results are the noiseless references of the Z stabilizers; the resulting state is logical |+>
  up to a known Pauli frame, which gives the same detector and observable statistics as a unitary encoder.
- Single-qubit layer: the gate on every data qubit of the block, then DEPOLARIZE1(p/10) on those qubits, except for I,
  which is noiseless [choice: their released code (Zenodo 10.5281/zenodo.20735728, ``simulate/noise_model.py``) gives
  I the idle rate, which their p = 0.3 % recipe leaves at 0; ``noisy_identity=True`` gives I p/10 too]. H is
  "Hadamard gates to all physical qubits, followed by a noiseless lattice rotation to align the logical operators":
  after the H layer, noiseless SWAPs move the data qubit at (x, y) to (y, 2d - x), a 90 degree rotation of the patch
  that maps every X-stabilizer support onto a Z-stabilizer support and back (their code does the same with
  ``np.rot90``). The rotated patch is again the standard layout, so the next rounds use the standard measurement
  schedule (hook errors stay perpendicular to the logical of their type) and the transversal CNOTs pair equal positions.
- CNOT layer: CX between equal data positions of control and target block, then DEPOLARIZE2(p) on every pair.
- Syndrome round (all blocks at once; "the standard fault-tolerant pattern"): R on all measure qubits, X_ERROR(p/10);
  H on the X-type ones, DEPOLARIZE1(p/10); four CX layers in the hook-safe orders ``X_ORDER``/``Z_ORDER`` of
  ``duet.circuits``, DEPOLARIZE2(p) after each; H on the X-type ones, DEPOLARIZE1(p/10); M with result flip p/10.
- Final transversal measurement: M or MX of every data qubit with result flip p/10.
- No idle noise and no other noise (their Methods list two-qubit gates p, single-qubit gates p/10, resets and
  measurements p/10).

Detectors: every stabilizer measurement is compared with the value the previous measurements predict through the gates
since then (CNOT c->t: Z(t) <- Z(c) Z(t), X(c) <- X(c) X(t); H with the rotation: the X stabilizer at plaquette m
takes the previous Z stabilizer at plaquette rot^-1(m), and vice versa). So every round gives one detector per
stabilizer; the final measurement gives one per stabilizer of the measured type. Coordinates (x, y, t): measure-qubit
position and round t = 0, 1...; the final measurement has t = number of rounds. The observable is the product of the
canonical logical measurement results over the target's support, plus the noiseless references its expected value
needs.
"""

from __future__ import annotations

import numpy as np
import stim

from duet.circuits import X_ORDER, Z_ORDER, rotated_patch

N_BLOCKS = 4
GATES = "IXYZH"
PAIRINGS = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))
SEED = 250911370  # base seed of the benchmark circuits: default_rng([SEED, depth, index])
N_CIRCUITS = 20


# ---------------------------------------------------------------------------------------------- logical level

def random_layers(depth: int, rng: np.random.Generator, gate_set: str = GATES, pairing: str = "random"
                  ) -> list[tuple[str, tuple[tuple[int, int], ...]]]:
    """``depth`` layers of (gates, cnots): gates[b] in gate_set for block b; cnots = two (control, target) pairs.
    ``pairing``: "fixed" always pairs (0, 1) and (2, 3) (random
    orientations), "none" has no CNOTs; the random draws are the same in every mode."""
    assert pairing in ("random", "fixed", "none"), pairing
    layers = []
    for _ in range(depth):
        gates = "".join(gate_set[k] for k in rng.integers(0, len(gate_set), size=N_BLOCKS))
        pairs = PAIRINGS[rng.integers(0, len(PAIRINGS))]
        flips = rng.integers(0, 2, size=len(pairs))
        if pairing == "fixed":
            pairs = PAIRINGS[0]
        cnots = tuple((a, b) if f == 0 else (b, a) for (a, b), f in zip(pairs, flips))
        layers.append((gates, () if pairing == "none" else cnots))
    return layers


def logical_stabilizers(layers) -> list[str]:
    """Non-identity stabilizers of the logical output state with no Y factor, as strings over 'IXZ' (sign dropped),
    in a fixed order (by the bitmask of canonical generators that forms them)."""
    sim = stim.TableauSimulator()
    sim.h(*range(N_BLOCKS))
    for gates, cnots in layers:
        for b, g in enumerate(gates):
            if g != "I":
                getattr(sim, g.lower())(b)
        for c, t in cnots:
            sim.cnot(c, t)
    gens = sim.canonical_stabilizers()
    out = []
    for mask in range(1, 1 << N_BLOCKS):
        prod = stim.PauliString(N_BLOCKS)
        for i in range(N_BLOCKS):
            if mask >> i & 1:
                prod *= gens[i]
        s = "".join("IXYZ"[prod[k]] for k in range(N_BLOCKS))
        if "Y" not in s:
            out.append(s)
    return out


def random_logical(depth: int, index: int, seed: int = SEED, gate_set: str = GATES, pairing: str = "random") -> dict:
    """Circuit ``index`` of depth ``depth``: layers, target stabilizer and final measurement bases. ``gate_set``
    'IXZH' is the surface-code gate set of their released code (no Y), a variant; ``pairing``: random_layers."""
    rng = np.random.default_rng([seed, depth, index])
    layers = random_layers(depth, rng, gate_set, pairing)
    cands = logical_stabilizers(layers)
    if not cands:  # never seen: H, CNOT and Paulis keep the state real, so some Y-free stabilizer exists
        raise RuntimeError(f"no Y-free stabilizer for depth {depth} index {index}")
    target = cands[rng.integers(0, len(cands))]
    bases = "".join(t if t != "I" else "Z" for t in target)
    return {"depth": depth, "index": index, "seed": [seed, depth, index], "layers": layers, "target": target,
            "bases": bases, "n_candidates": len(cands)}


def layers_str(layers) -> str:
    """Compact text form, e.g. 'XYHY 0>1 2>3 | HZXH 3>0 1>2'."""
    return " | ".join(g + " " + " ".join(f"{c}>{t}" for c, t in cn) for g, cn in layers)


# --------------------------------------------------------------------------------------------- physical level

def rotation(d: int, xy, ccw: bool = False):
    """Noiseless lattice rotation after a transversal H: position (x, y) -> (y, 2d - x), the turn their released code
    makes (``np.rot90`` of the data grid in ``SurfaceCodeBase._apply_h``, read in our coordinates: identical DEMs in); ``ccw`` the opposite turn (x, y) -> (2d - y, x), kept only for that check."""
    return (2 * d - xy[1], xy[0]) if ccw else (xy[1], 2 * d - xy[0])


def _gf2_solve(rows: list[int], target: int) -> list[int] | None:
    """Indices of ``rows`` (bitmasks) whose XOR is ``target``, or None."""
    basis: list[tuple[int, int]] = []  # (vector, combination mask), reduced by leading bit
    for i, r in enumerate(rows):
        comb = 1 << i
        for v, c in basis:
            if r ^ v < r:
                r, comb = r ^ v, comb ^ c
        if r:
            basis.append((r, comb))
            basis.sort(reverse=True)
    comb = 0
    for v, c in basis:
        if target ^ v < target:
            target, comb = target ^ v, comb ^ c
    return None if target else [i for i in range(len(rows)) if comb >> i & 1]


class _Code:
    """The rotated patch plus the index maps that the H-with-rotation update needs."""

    def __init__(self, d: int, ccw: bool = False):
        self.d = d
        rot = lambda xy: rotation(d, xy, ccw)
        self.patch = patch = rotated_patch(d)
        self.data_k = {q: k for k, q in enumerate(patch["data"])}
        self.idx = {t: {m: s for s, m in enumerate(patch[t])} for t in "XZ"}
        mask = lambda qs: sum(1 << self.data_k[q] for q in qs)
        self.supp_mask = {t: [mask(patch["supp"][m]) for m in patch[t]] for t in "XZ"}
        self.L = {t: patch["L" + t] for t in "XZ"}
        # after H + rotation: stabilizer of type t at plaquette m <- stabilizer of the other type at rot^-1(m)
        inv = lambda xy: rotation(d, xy, not ccw)
        other = {"X": "Z", "Z": "X"}
        self.h_stab = {t: [self.idx[other[t]][inv(m)] for m in patch[t]] for t in "XZ"}
        # logical of type t after <- operator of the other type on rot^-1(L_t) before = L_other x (stabilizers)
        self.h_log = {}
        for t in "XZ":
            o = other[t]
            back = mask(inv(q) for q in self.L[t])
            sol = _gf2_solve(self.supp_mask[o], back ^ mask(self.L[o]))
            assert sol is not None, "rotation must map logical representatives onto logical representatives"
            self.h_log[t] = sol
        # the rotation maps stabilizer supports onto supports of the other type
        for t in "XZ":
            for s, m in enumerate(patch[t]):
                img = {rot(q) for q in patch["supp"][m]}
                assert img == set(patch["supp"][rot(m)]) and rot(m) in self.idx[other[t]]
        # SWAP steps that realise the rotation: cycles (q0 q1 q2 q3) -> SWAP(q0,q1), SWAP(q0,q2), SWAP(q0,q3)
        steps: list[list[tuple[int, int]]] = [[], [], []]
        seen = set()
        for q in patch["data"]:
            if q in seen:
                continue
            cyc = [q]
            while rot(cyc[-1]) != q:
                cyc.append(rot(cyc[-1]))
            seen.update(cyc)
            for i in range(1, len(cyc)):
                steps[i - 1].append((self.data_k[cyc[0]], self.data_k[cyc[i]]))
        self.swap_steps = [s for s in steps if s]


class _Frames:
    """Expected noiseless value of each tracked operator as a set of terms whose XOR it is: measurement-record indices
    (>= 0) and random symbols (< 0). Stabilizers carry two frames: 'det', the value the next measurement is compared
    with (the last result, carried through the gates), and 'val', the value in noiseless terms only (preparation
    references and symbols), which the logical frames need under H. Logicals carry 'val' only. Pauli gates change only
    signs, which stim's reference sample absorbs, so they are not tracked."""

    def __init__(self, code: _Code):
        self.code = code
        self.det: dict[tuple, set[int]] = {}
        self.val: dict[tuple, set[int]] = {}
        self.sym = -1

    def prepare_plus(self, b: int, z_refs: list[int]):
        n = len(self.code.patch["X"])
        for s in range(n):
            self.det[(b, "X", s)], self.val[(b, "X", s)] = set(), set()
            self.det[(b, "Z", s)], self.val[(b, "Z", s)] = {z_refs[s]}, {z_refs[s]}
        self.val[(b, "X", "L")] = set()
        self.val[(b, "Z", "L")] = {self.sym}
        self.sym -= 1

    def cnot(self, c: int, t: int):
        n = len(self.code.patch["X"])
        for fr in (self.det, self.val):
            for s in range(n):
                fr[(t, "Z", s)] = fr[(t, "Z", s)] ^ fr[(c, "Z", s)]
                fr[(c, "X", s)] = fr[(c, "X", s)] ^ fr[(t, "X", s)]
        self.val[(t, "Z", "L")] = self.val[(t, "Z", "L")] ^ self.val[(c, "Z", "L")]
        self.val[(c, "X", "L")] = self.val[(c, "X", "L")] ^ self.val[(t, "X", "L")]

    def hadamard(self, b: int):
        other = {"X": "Z", "Z": "X"}
        old = dict(self.val)  # logicals first, from the stabilizer frames before the gate
        for t in "XZ":
            v = set(old[(b, other[t], "L")])
            for s in self.code.h_log[t]:
                v ^= old[(b, other[t], s)]
            self.val[(b, t, "L")] = v
        for fr in (self.det, self.val):
            old = dict(fr)
            for t in "XZ":
                for s, so in enumerate(self.code.h_stab[t]):
                    fr[(b, t, s)] = old[(b, other[t], so)]


def make_circuit(d: int, logical: dict, p: float, noisy_identity: bool = False,
                 round_after_last_layer: bool = True, round_after_1q_layer: bool = False,
                 h_rotation: str = "cw") -> stim.Circuit:
    """Physical circuit of one logical circuit (``random_logical``) at distance d and noise strength p (0 gives the
    noiseless circuit). Variants for checking the reconstruction: ``round_after_last_layer=False`` drops the syndrome
    round of the last layer (their Fig. 1a draws none between the last CNOTs and the readout);
    ``round_after_1q_layer=True`` adds a round after every single-qubit layer, as their released compiler does when
    the single-qubit gates and the CNOTs are two algorithm layers (every layer gets a round on the blocks it acts on),
    so a depth-D circuit has 2D rounds. ``h_rotation="ccw"`` turns the patch the other way after an H (not what their
    released code does; ``rotation``)."""
    assert h_rotation in ("cw", "ccw"), h_rotation
    code = _Code(d, ccw=h_rotation == "ccw")
    patch = code.patch
    n_stab = len(patch["X"])
    layers, target, bases = logical["layers"], logical["target"], logical["bases"]
    off = lambda b: b * (2 * d + 4)

    index: dict[tuple, int] = {}
    c = stim.Circuit()
    for b in range(N_BLOCKS):
        for kind in ("data", "X", "Z"):
            for k, (x, y) in enumerate(patch[kind]):
                index[(b, kind, k)] = len(index)
                c.append("QUBIT_COORDS", [index[(b, kind, k)]], (x + off(b), y))
    data_q = [[index[(b, "data", k)] for k in range(d * d)] for b in range(N_BLOCKS)]
    anc = [(b, t, s) for b in range(N_BLOCKS) for t in "XZ" for s in range(n_stab)]
    anc_q = [index[a] for a in anc]
    x_anc_q = [index[(b, "X", s)] for b in range(N_BLOCKS) for s in range(n_stab)]
    p1 = p / 10

    def noise(name, targets, prob):
        if prob > 0 and targets:
            c.append(name, targets, prob)

    def measure(name, targets):
        c.append(name, targets, p1) if p1 > 0 else c.append(name, targets)

    frames = _Frames(code)
    n_meas = 0

    # noiseless preparation of logical |+> on every block
    c.append("RX", [q for b in range(N_BLOCKS) for q in data_q[b]])
    for b in range(N_BLOCKS):
        mpp = []
        for s, m in enumerate(patch["Z"]):
            qs = [data_q[b][code.data_k[q]] for q in patch["supp"][m]]
            for i, q in enumerate(qs):
                mpp += ([stim.target_combiner()] if i else []) + [stim.target_z(q)]
        c.append("MPP", mpp)
        frames.prepare_plus(b, list(range(n_meas, n_meas + n_stab)))
        n_meas += n_stab
    c.append("TICK")

    def detector(terms, xy, t):
        assert all(v >= 0 for v in terms)
        c.append("DETECTOR", [stim.target_rec(v - n_meas) for v in sorted(terms)], (xy[0], xy[1], t))

    def syndrome_round(rnd):
        nonlocal n_meas
        c.append("R", anc_q)
        noise("X_ERROR", anc_q, p1)
        c.append("TICK")
        c.append("H", x_anc_q)
        noise("DEPOLARIZE1", x_anc_q, p1)
        c.append("TICK")
        for step in range(4):
            flat = []
            for b, t, s in anc:
                m = patch[t][s]
                o = (X_ORDER if t == "X" else Z_ORDER)[step]
                dq = (m[0] + o[0], m[1] + o[1])
                if dq in code.data_k:
                    a, q = index[(b, t, s)], data_q[b][code.data_k[dq]]
                    flat += [a, q] if t == "X" else [q, a]
            c.append("CX", flat)
            noise("DEPOLARIZE2", flat, p)
            c.append("TICK")
        c.append("H", x_anc_q)
        noise("DEPOLARIZE1", x_anc_q, p1)
        c.append("TICK")
        measure("M", anc_q)
        first = n_meas
        n_meas += len(anc)
        for i, (b, t, s) in enumerate(anc):
            m = patch[t][s]
            detector({first + i} ^ frames.det[(b, t, s)], (m[0] + off(b), m[1]), rnd)
            frames.det[(b, t, s)] = {first + i}
        c.append("TICK")

    rnd = 0
    for layer, (gates, cnots) in enumerate(layers):
        for g in "XYZH":
            qs = [q for b in range(N_BLOCKS) if gates[b] == g for q in data_q[b]]
            if qs:
                c.append(g, qs)
        noise("DEPOLARIZE1", [q for b in range(N_BLOCKS) if noisy_identity or gates[b] != "I"
                              for q in data_q[b]], p1)
        h_blocks = [b for b in range(N_BLOCKS) if gates[b] == "H"]
        for step in code.swap_steps:  # noiseless lattice rotation of every block that got an H
            if h_blocks:
                c.append("SWAP", [data_q[b][k] for b in h_blocks for pair in step for k in pair])
        for b in h_blocks:
            frames.hadamard(b)
        c.append("TICK")
        if round_after_1q_layer:
            syndrome_round(rnd)
            rnd += 1
        flat = []
        for ctl, tgt in cnots:
            flat += [q for pair in zip(data_q[ctl], data_q[tgt]) for q in pair]
            frames.cnot(ctl, tgt)
        if flat:
            c.append("CX", flat)
        noise("DEPOLARIZE2", flat, p)
        c.append("TICK")
        if layer < len(layers) - 1 or round_after_last_layer:
            syndrome_round(rnd)
            rnd += 1

    # final transversal measurement
    first = n_meas
    for b in range(N_BLOCKS):
        measure("M" if bases[b] == "Z" else "MX", data_q[b])
    n_meas += N_BLOCKS * d * d
    rec = lambda b, q: first + b * d * d + code.data_k[q]
    for b in range(N_BLOCKS):
        t = bases[b]
        for s, m in enumerate(patch[t]):
            detector({rec(b, q) for q in patch["supp"][m]} ^ frames.det[(b, t, s)], (m[0] + off(b), m[1]), rnd)
    obs: set[int] = set()
    for b in range(N_BLOCKS):
        if target[b] != "I":
            assert target[b] == bases[b]
            obs ^= {rec(b, q) for q in code.L[target[b]]} ^ frames.val[(b, target[b], "L")]
    assert all(v >= 0 for v in obs), f"target {target} is not deterministic"
    c.append("OBSERVABLE_INCLUDE", [stim.target_rec(v - n_meas) for v in sorted(obs)], 0)
    return c


def bench_circuits(depth: int, n: int = N_CIRCUITS, seed: int = SEED) -> list[dict]:
    """The benchmark's logical circuits of one depth."""
    return [random_logical(depth, i, seed) for i in range(n)]


def describe(logical: dict) -> dict:
    """JSON-ready description of a logical circuit, for result files."""
    return {"depth": logical["depth"], "index": logical["index"], "seed": logical["seed"],
            "layers": layers_str(logical["layers"]), "target": logical["target"], "bases": logical["bases"],
            "n_candidates": logical["n_candidates"]}
