"""K-block circuit families (K = 2, 3, 4), as used by the development reads
target block of a CNOT whose control this block is, or "+" for its target; duet.clifford_circuits.parse_k).

  mix duet.clifford_circuits.sample_circuit_k(rng, K, L, q, "IHS"), q = P_SET[i mod 4] for circuit i, bases uniform
  cnot sample_circuit_k(rng, K, L, 1, "I", all Z): floor(K / 2) CNOT pairs per layer (K = 3: one block idles)
  f1b inspired by the Lukin group's Fig. 1b (K = 4; not their benchmark): per depth unit a gate layer (I, H or S per
          block) and a CNOT layer on a random perfect matching in random directions, so L = 2 units; preparation all
          X, readout uniform
  struct named circuits, one per (name, K, L): GHZ preparation (H on block 0, then a CNOT chain 0-1-2-3 or a tree, 0-1
          then 0-2 and 1-3), two Bell pairs (K = 4), each idle to depth L and read out all Z (_z) or all X (_x); one
          layer repeated to depth L (a CNOT ladder of alternating direction, H on all blocks, S on all blocks) with the
          open bases duet.cliff_sets.struct_circuit's rule picks (most reliable products, then most X letters)
  ent entangled, then idle: random GHZ groups, then
          idle layers to depth L (ent); the GHZ and Bell circuits of struct are among its draws
Circuits without a reliable product are drawn again (struct circuits without one are dropped)."""
import itertools

import numpy as np

from duet import clifford_circuits as cc

P_SET = (0.0, 1 / 3, 2 / 3, 1.0)
FAMILY = {"mix": 1, "cnot": 2, "f1b": 3, "struct": 4}


def tok(K, gates=None, pairs=()):
    """One layer: gates per block (default I) and CNOT pairs (control, target)."""
    t = list(gates) if gates else ["I"] * K
    for c, g in pairs:
        t[c], t[g] = str(g), "+"
    return "".join(t)


def mix(rng, K, L, q):
    return cc.sample_circuit_k(rng, K, L, q, "IHS")


def cnot(rng, K, L):
    return cc.sample_circuit_k(rng, K, L, 1.0, "I", prep=("Z",) * K, meas=("Z",) * K)


def f1b(rng, units, gates="IHS"):
    K = 4
    while True:
        lay = []
        for _ in range(units):
            lay.append("".join(gates[i] for i in rng.integers(len(gates), size=K)))
            p = [int(v) for v in rng.permutation(K)]
            lay.append(tok(K, pairs=[(p[i], p[i + 1]) if rng.random() < 0.5 else (p[i + 1], p[i]) for i in (0, 2)]))
        prep, meas = ("X",) * K, tuple("ZX"[i] for i in rng.integers(2, size=K))
        if cc.logical_products(lay, prep, meas):
            return lay, prep, meas


def ent(rng, K, L):
    """Entangled, then idle: the shuffled blocks are cut into groups (each gap cut with probability 1/2, at least one
    group of two or more); a group of two or more becomes a GHZ state: its first block prepared in X, or in Z with H in
    layer 0 (probability 1/2 each), then every further block the target of a CNOT from a block already in the state
    (the one added last, a chain, or a random one, a tree; 1/2 each), each CNOT in the earliest layer where both blocks
    are free; idle to depth L; readout all Z or all X per group (1/2 each). A single block: preparation and readout
    uniform in Z, X."""
    while True:
        perm = [int(b) for b in rng.permutation(K)]
        cuts = [i for i in range(1, K) if rng.random() < 0.5]
        groups = [perm[a:b] for a, b in zip([0] + cuts, cuts + [K])]
        if max(map(len, groups)) < 2:
            continue
        prep, meas = ["Z"] * K, ["Z"] * K
        gates, pairs = {}, {}                     # layer -> {block: gate}, layer -> [(control, target)]
        busy = set()                              # (layer, block)
        for g in groups:
            if len(g) == 1:
                prep[g[0]], meas[g[0]] = "ZX"[rng.integers(2)], "ZX"[rng.integers(2)]
                continue
            ro, chain = "ZX"[rng.integers(2)], rng.random() < 0.5
            ready = {}
            if rng.random() < 0.5:
                prep[g[0]], ready[g[0]] = "X", 0
            else:
                gates.setdefault(0, {})[g[0]] = "H"
                busy.add((0, g[0]))
                ready[g[0]] = 1
            for i, t in enumerate(g[1:], 1):
                c = g[i - 1] if chain else g[int(rng.integers(i))]
                at = ready[c]
                while (at, c) in busy or (at, t) in busy:
                    at += 1
                pairs.setdefault(at, []).append((c, t))
                busy |= {(at, c), (at, t)}
                ready[t] = at + 1
            for b in g:
                meas[b] = ro
        n = max(at for at, _ in busy) + 1
        if n > L:
            continue
        lay = [tok(K, "".join(gates.get(i, {}).get(b, "I") for b in range(K)), pairs.get(i, ())) for i in range(n)]
        lay += ["I" * K] * (L - n)
        if cc.logical_products(lay, tuple(prep), tuple(meas)):
            return lay, tuple(prep), tuple(meas)


def struct_names(K):
    ghz = ["ghz_chain", "ghz_tree"] + (["bell2"] if K == 4 else [])
    return [f"{g}_{r}" for g in ghz for r in ("z", "x")] + ["ladder", "rep_h", "rep_s"]


def struct(name, K, L):
    """(layers, prep, meas) of a structured circuit, or None when it has no reliable product."""
    idle = "I" * K
    base, ro = (name[:-2], name[-1].upper()) if name[-2:] in ("_z", "_x") else (name, None)
    if base == "ghz_chain":
        lay = [tok(K, "H" + "I" * (K - 1))] + [tok(K, pairs=[(b, b + 1)]) for b in range(K - 1)]
    elif base == "ghz_tree":
        lay = [tok(K, "H" + "I" * (K - 1)), tok(K, pairs=[(0, 1)]), tok(K, pairs=[(0, 2), (1, 3)][:K - 2])]
    elif base == "bell2":
        lay = [tok(K, "HIHI"), tok(K, pairs=[(0, 1), (2, 3)])]
    elif base == "ladder":
        lay = [tok(K, pairs=[(b, b + 1) if (i // 2) % 2 == 0 else (b + 1, b) for b in range(i % 2, K - 1, 2)])
               for i in range(L)]
    else:
        lay = [{"rep_h": "H", "rep_s": "S"}[base] * K] * L
    lay = (lay + [idle] * L)[:L]
    if ro:
        best = (len(cc.logical_products(lay, ("Z",) * K, (ro,) * K)), ("Z",) * K, (ro,) * K)
    else:
        best = None
        for p_ in itertools.product("ZX", repeat=K):
            for m_ in itertools.product("ZX", repeat=K):
                key = (len(cc.logical_products(lay, p_, m_)), (p_ + m_).count("X"))
                if best is None or key > best[0]:
                    best = (key, p_, m_)
        best = (best[0][0], best[1], best[2])
    return (lay, best[1], best[2]) if best[0] else None


def circuits(family, K, d, L, n, tag):
    """n circuits of a point; numpy default_rng([tag, family, K, d, L]) (tag: 3 development reads, 5 test sets)."""
    rng = np.random.default_rng([tag, FAMILY[family], K, d, L])
    if family == "mix":
        return [mix(rng, K, L, P_SET[i % 4]) for i in range(n)]
    if family == "cnot":
        return [cnot(rng, K, L) for _ in range(n)]
    if family == "f1b":
        return [f1b(rng, L // 2) for _ in range(n)]
    raise ValueError(family)
