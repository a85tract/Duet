"""The open benchmark of transversal-gate circuits: the circuit
families rebuilt from the published descriptions on duet.clifford_circuits (rotated code, one syndrome round per layer,
at most four blocks), their points and seeds, the stored shots and a loader.

External families:
  lom_rep LoMatching's repeated gates (Serra-Peralta, Shaw, Terhal): an idle first layer (their round after
            initialization), then d + 1 gates I, S or H on one block, CNOT or alternating CNOT on two; all blocks
            prepared and read out in Z or all in X
  lom_2q LoMatching's 360 two-qubit Cliffords modulo Paulis and SWAP, each followed by its inverse (their TOTAL
            CIRCUITs, third_party/data-lom-decoders), as layers of H, S and CNOT padded with idle layers to 14, after
            the idle first layer; ZZ or XX
  lom_deep LoMatching's deep four-block circuit (Turner et al.'s, their data repository): 32 source layers of a gate
            from {I, X, Y, Z, H} per block and two CNOTs, each as our gate layer and CNOT layer (64 layers); all |+>,
            readout bases with the most reliable products
  lukin the Lukin group's Fig. 1b family as duet.lukin_bench draws it:
            four blocks in |+>, per depth unit a gate from {I, X, Y, Z, H} per block and then two CNOTs (two layers,
            two rounds), depths 1-10, 20 circuits; readout in the target's bases (Z outside it), products the target
            first, then the rest of a reliable basis
  mccd MCCD's mirror circuits on two blocks: an idle first layer (their round before the first gate), U of D'
            random layers (type 1: a gate from {I, X, Y, Z, H} per block; type 2: such a layer and then one CNOT in a
            uniform direction), then U reversed (each gate is its own inverse); |00>, Z readout
  cain_ghz Cain et al.'s GHZ test: block 0 in |+>, blocks 1 and 2 in |0>, CNOT 0 -> 1, CNOT 0 -> 2; readout XZZ, so the
            one reliable product is Z1 Z2
Physical noisy Paulis (make_circuit pauli="gate") in every external family; H and S as in the earlier study. Models: uniform and lukin on every family but lom_deep (lukin only), SI1000 on lom_* (lom_deep at the source's
p = 0.1 %, a p sweep on lom_rep and lom_2q), Cain et al.'s model on cain_ghz.
Ours: two-block test sets (duet.cliff_sets, by reference) and the K = 3, 4 families of duet.kblock_sets.

disjoint from training): circuit i of a point samples all its shots with stim seed BASE + noise 10^15 + family 10^14 +
d 10^12 + v 10^9 + i, noise code 1 for development shots (the cost probe), 5 uniform, 6 lukin, 7 si1000, 8 cain; v =
100 pcode + the family's variant index. ours_k: KBLOCK_SEED_BASE + (5 uniform, 6 lukin) 10^15 + family 10^14 + K 10^13 +
d 10^12 + pcode 10^11 + L 10^9 + variant 10^6 + i. Random circuits are drawn independently of d and noise (numpy
default_rng([BENCH_TAG, family...])), so every distance and model sees the same circuits.

  python -m duet.bench_sets manifest --out data/open_benchmark/MANIFEST.json (commit it before any shot exists)
  python -m duet.bench_sets generate --manifest MANIFEST.json --dir SETS --workers N [--only NAME...]
Each point is SETS/<name>.npz as duet.cliff_sets writes it (dets and obs bit-packed little-endian per shot, which is
stim's b8 layout, zero-padded to the point's widest circuit; m and k per circuit; the manifest entry); load_point reads
it back and circuit rebuilds each stim circuit."""
import argparse
import collections
import functools
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import struct
import zipfile
import zlib

import numpy as np
import stim

from duet import clifford_circuits as cc
from duet import kblock_sets as ms

BENCH_SEED_BASE, SEED_SPAN = 18_410_000_000_000_000_000, 10 ** 16
BENCH_TAG = 151                                  #: the first key of every circuit generator here
NOISE_CODE = {"dev": 1, "uniform": 5, "lukin": 6, "si1000": 7, "cain": 8}
FAMILY = {"lom_rep": 1, "lom_2q": 2, "lom_deep": 3, "lukin": 4, "mccd": 5, "cain_ghz": 6}
PCODE = {0.001: 1, 0.002: 2, 0.003: 3, 0.004: 4, 0.005: 5}
NETWORK_NOISES = ("uniform", "lukin")            # every decoder; si1000 and cain: the matching decoders only
THIRD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "data-lom-decoders")
LOM_2Q_FILE = "20250130_cliffords_modulo_paulis_and_swap.txt"
LOM_DEEP_FILE = "logical_circuit_deep_logical_32_layers_4_qubits.stim"
LOM_REP_GATES = ("I", "S", "H", "CNOT", "CNOT_alt")
LOM_2Q_DEPTH = 14
LUKIN_CIRCUITS = 20
MCCD_TYPES = {1: "IXYZH", 2: "IXYZH"}
MCCD_DP = tuple(range(1, 10))                    # D' = 1..9: depths 2-18 (type 1) and 4-36 (type 2), their test range


def tok(K, gates=None, pairs=()):
    return ms.tok(K, gates, pairs)


def best_meas(layers, prep):
    """The readout bases with the most reliable products, ties to the most X letters (then itertools order)."""
    best = None
    for m in itertools.product("ZX", repeat=len(prep)):
        key = (len(cc.logical_products(layers, tuple(prep), m)), m.count("X"))
        if best is None or key > best[0]:
            best = (key, "".join(m))
    return best[1]


def complete_products(layers, prep, meas, first):
    """A reliable-product basis that starts with `first` (block lists, reliable and independent) and goes on with the
    products of cc.logical_products that are independent of what came before."""
    masks, out = [], []
    for sub in [sorted(int(b) for b in s) for s in first] + [s for s, _ in cc.logical_products(layers, tuple(prep),
                                                                                                tuple(meas))]:
        m = sum(1 << b for b in sub)
        for v in masks:
            m = min(m, m ^ v)
        if m:
            masks.append(m)
            masks.sort(reverse=True)
            out.append(list(sub))
    return out


# ------------------------------------------------------------------------------------------- external families

def lom_rep(gate, d, basis):
    """LoMatching's repeated-gate circuit of distance d: (layers, prep, meas)."""
    n = d + 1
    if gate in ("I", "S", "H"):
        return ["I"] + [gate] * n, basis, basis
    lay = ["1+"] * n if gate == "CNOT" else ["1+", "+0"] * (n // 2)
    if gate not in ("CNOT", "CNOT_alt") or len(lay) != n:
        raise ValueError(f"lom_rep gate {gate!r} at d = {d}")
    return ["II"] + lay, basis * 2, basis * 2


@functools.lru_cache(maxsize=None)
def lom_2q_cores():
    """The 360 TOTAL CIRCUITs of LoMatching's two-qubit Clifford experiment as our layer tokens, unpadded."""
    text = open(os.path.join(THIRD, LOM_2Q_FILE)).read()
    out = []
    for block in text.split("\n----------\n"):
        if not block.strip():
            continue
        parts = block.split("TOTAL CIRCUIT:\n")[1].split("TICK")
        assert parts[0].strip() == "R 0 1" and parts[-1].strip() == "M 0 1", parts
        lay = []
        for chunk in parts[1:-1]:
            gates, cx = ["I", "I"], None
            for line in chunk.strip().splitlines():
                name, *ts = line.split()
                if name == "CX":
                    cx = (int(ts[0]), int(ts[1]))
                else:
                    for t in ts:
                        gates[int(t)] = name
            if cx is not None:
                assert gates == ["I", "I"] and sorted(cx) == [0, 1], chunk
                lay.append("1+" if cx == (0, 1) else "+0")
            else:
                assert set(gates) <= set("IHS"), chunk
                lay.append("".join(gates))
        out.append(lay)
    assert len(out) == 360
    return tuple(tuple(x) for x in out)


def lom_2q(k, basis):
    """Clifford k (0..359) of LoMatching's list: the idle first layer, its TOTAL CIRCUIT, idle layers to 14."""
    core = list(lom_2q_cores()[k])
    if len(core) > LOM_2Q_DEPTH:
        raise ValueError(f"Clifford {k}: {len(core)} layers")
    return ["II"] + core + ["II"] * (LOM_2Q_DEPTH - len(core)), basis * 2, basis * 2


@functools.lru_cache(maxsize=None)
def lom_deep_layers():
    """The deep logical circuit: per source layer our gate layer (its single-qubit gates, I elsewhere) and our CNOT
    layer (its two CX), 64 layers."""
    c = stim.Circuit(open(os.path.join(THIRD, LOM_DEEP_FILE)).read())
    lay, gates, pairs = [], ["I"] * 4, []
    for inst in c:
        ts = [t.value for t in inst.targets_copy()]
        if inst.name == "RX":
            assert ts == [0, 1, 2, 3] and not lay
        elif inst.name == "TICK":
            lay += ["".join(gates), tok(4, pairs=pairs)]
            gates, pairs = ["I"] * 4, []
        elif inst.name == "CX":
            pairs += list(zip(ts[::2], ts[1::2]))
        else:
            assert inst.name in "XYZH", inst
            for t in ts:
                gates[t] = inst.name
    assert len(lay) == 64 and gates == ["I"] * 4 and not pairs
    return tuple(lay)


def lom_deep():
    lay = list(lom_deep_layers())
    return lay, "XXXX", best_meas(lay, "XXXX")


def lukin(D, i, seed=None):
    """Circuit i of depth D of the Fig. 1b family: (layers, prep, meas, products), products[0] the target; seed:
    random_logical's seed."""
    from duet import lukin_bench as lb
    g = lb.random_logical(D, i) if seed is None else lb.random_logical(D, i, seed=seed)
    lay = []
    for gates, cnots in g["layers"]:
        lay += [gates, tok(4, pairs=[(int(a), int(b)) for a, b in cnots])]
    target = [b for b, ch in enumerate(g["target"]) if ch != "I"]
    return lay, "XXXX", g["bases"], complete_products(lay, "XXXX", g["bases"], [target])


def mccd_circuits(kind, Dp, n):
    """n mirror circuits of MCCD type `kind` (1, 2) with D' random layers of U: [(layers, prep, meas)]."""
    rng = np.random.default_rng([BENCH_TAG, FAMILY["mccd"], kind, Dp])
    out = []
    for _ in range(n):
        U = []
        for _ in range(Dp):
            U.append("".join(MCCD_TYPES[kind][j] for j in rng.integers(5, size=2)))
            if kind == 2:
                U.append("1+" if rng.random() < 0.5 else "+0")
        out.append((["II"] + U + U[::-1], "ZZ", "ZZ"))
    return out


def cain_ghz():
    return ["1+I", "2I+"], "XZZ", "XZZ"


def circuit(entry, c_, p=None, noise=None, meta=False):
    """The stim circuit of circuit c_ of a manifest entry (its noise model and p unless given)."""
    prods = c_.get("products")
    return cc.make_circuit(entry["d"], c_["layers"].split(), entry["p"] if p is None else p, tuple(c_["prep"]),
                           tuple(c_["meas"]), meta=meta, noise=noise or entry["noise"],
                           products=None if prods is None else [list(s) for s in prods], pauli=c_.get("pauli", "frame"))


# ---------------------------------------------------------------------------------------------------- points

def seed(noise, family, d, v, i):
    assert 0 <= v < 1000 and 0 <= i < 10 ** 9 and d < 10
    return BENCH_SEED_BASE + NOISE_CODE[noise] * 10 ** 15 + FAMILY[family] * 10 ** 14 + d * 10 ** 12 + v * 10 ** 9 + i


def dev_seed(noise, family, d, v, i):
    """Development shots (the cost probe, pilots): noise code 1, the model in the 10^13 digit (1-4), family code 7 for
    the K-block families of ours_k."""
    nz = ("uniform", "lukin", "si1000", "cain").index(noise) + 1
    assert 0 <= v < 1000 and 0 <= i < 10 ** 9 and d < 10
    return (BENCH_SEED_BASE + NOISE_CODE["dev"] * 10 ** 15 + dict(FAMILY, ours_k=7)[family] * 10 ** 14 + nz * 10 ** 13
            + d * 10 ** 12 + v * 10 ** 9 + i)


def _c(lay, prep, meas, products=None, **extra):
    out = dict(layers=" ".join(lay), prep="".join(prep), meas="".join(meas), pauli="gate",
               cnot_layers=sum(len(p) > 0 for p, _ in cc.parse_k(lay, len(prep))), **extra)
    if products is not None:
        out["products"] = [list(s) for s in products]
    return out


def family_points(family, d):
    """[(variant index, point label, depth, [circuit dict])] of an external family at distance d (no noise, p, seeds)."""
    if family == "lom_rep":
        return [(v, f"{g}_{b}", d + 1, [_c(*lom_rep(g, d, b), gate=g)])
                for v, (g, b) in enumerate(itertools.product(LOM_REP_GATES, "ZX"))]
    if family == "lom_2q":
        return [(v, b, LOM_2Q_DEPTH, [_c(*lom_2q(k, b), clifford=k) for k in range(360)]) for v, b in enumerate("ZX")]
    if family == "lom_deep":
        return [(0, "deep32", 32, [_c(*lom_deep())])]
    if family == "lukin":
        return [(D, f"D{D}", D, [_c(*lukin(D, i), index=i, target_slot=0) for i in range(LUKIN_CIRCUITS)])
                for D in range(1, 11)]
    if family == "mccd":
        return [(10 * k + Dp, f"t{k}_D{2 * k * Dp}", 2 * k * Dp,
                 [_c(*x, index=i) for i, x in enumerate(mccd_circuits(k, Dp, MCCD_CIRCUITS))])
                for k in (1, 2) for Dp in MCCD_DP]
    if family == "cain_ghz":
        return [(0, "ghz3", 2, [_c(*cain_ghz())])]
    raise ValueError(family)


MCCD_CIRCUITS = 64
EXTERNAL = tuple(FAMILY)
#: SI1000 only on LoMatching's families, Cain et al.'s model only on their GHZ test; lom_deep without its
# saturated uniform points and at the source's p = 0.1 % under SI1000 (an adapted benchmark, not an anchor to their
# Fig. 17); SI1000's p sweep on lom_rep and lom_2q
MODELS = {"lom_rep": ("uniform", "lukin", "si1000"), "lom_2q": ("uniform", "lukin", "si1000"),
          "lom_deep": ("lukin", "si1000"), "lukin": ("uniform", "lukin"), "mccd": ("uniform", "lukin"),
          "cain_ghz": ("uniform", "lukin", "cain")}
SOURCE_P = {("lom_deep", "si1000"): 0.001}
SWEEP_P, SWEEP_D = (0.001, 0.002, 0.003, 0.004, 0.005), {"lom_rep": (3, 5, 7, 9), "lom_2q": (3, 5, 7)}
# shots per circuit of every external point and the prefixes (shots per circuit) that the exact MLE and BP+LoMatching
# decode: {(family, model, p): {d: one number per point of family_points(family, d), in its order}}; ours_k's prefixes
# pilot between the markers.
# --- plan.py begin
SHOTS = {
    ('lom_rep', 'lukin', 0.003): {
        3: (87000, 110000, 70000, 21000, 110000, 65000, 25000, 27000, 22000, 25000),
        5: (1100000, 1200000, 530000, 170000, 880000, 990000, 200000, 200000, 240000, 150000),
        7: (14000000, 14000000, 5600000, 1400000, 13000000, 14000000, 1800000, 1700000, 1700000, 1600000),
        9: (17000000, 17000000, 17000000, 9100000, 17000000, 17000000, 8200000, 8200000, 8200000, 8200000),
    },
    ('lom_rep', 'si1000', 0.001): {
        3: (33000, 33000, 32000, 11000, 34000, 33000, 14000, 11000, 12000, 12000),
        5: (180000, 190000, 130000, 39000, 160000, 180000, 56000, 51000, 45000, 47000),
        7: (1100000, 1100000, 770000, 150000, 1200000, 960000, 250000, 210000, 190000, 200000),
        9: (5200000, 5000000, 3100000, 520000, 5300000, 4700000, 830000, 870000, 760000, 760000),
    },
    ('lom_rep', 'si1000', 0.002): {
        3: (10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000),
        5: (25000, 23000, 18000, 10000, 22000, 21000, 10000, 10000, 10000, 10000),
        7: (64000, 62000, 47000, 13000, 64000, 63000, 14000, 15000, 13000, 12000),
        9: (170000, 150000, 80000, 20000, 160000, 150000, 25000, 25000, 22000, 21000),
    },
    ('lom_rep', 'si1000', 0.003): {
        3: (10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000),
        5: (10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000),
        7: (15000, 13000, 10000, 10000, 14000, 14000, 10000, 10000, 10000, 10000),
        9: (21000, 21000, 12000, 10000, 20000, 20000, 10000, 10000, 10000, 10000),
    },
    ('lom_rep', 'si1000', 0.004): {
        3: (10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000),
        5: (10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000),
        7: (10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000),
        9: (10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000),
    },
    ('lom_rep', 'si1000', 0.005): {
        3: (10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000),
        5: (10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000),
        7: (10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000),
        9: (10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000),
    },
    ('lom_rep', 'uniform', 0.003): {
        3: (10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000),
        5: (16000, 16000, 13000, 10000, 12000, 11000, 10000, 10000, 10000, 10000),
        7: (63000, 59000, 36000, 13000, 43000, 43000, 13000, 12000, 12000, 11000),
        9: (170000, 180000, 84000, 21000, 120000, 130000, 31000, 31000, 24000, 26000),
    },
    ('lom_2q', 'lukin', 0.003): {
        3: (100, 100),
        5: (250, 190),
        7: (2500, 1700),
        9: (17000, 17000),
    },
    ('lom_2q', 'si1000', 0.001): {
        3: (100, 100),
        5: (100, 100),
        7: (620, 360),
    },
    ('lom_2q', 'si1000', 0.002): {
        3: (100, 100),
        5: (100, 100),
        7: (100, 100),
    },
    ('lom_2q', 'si1000', 0.003): {
        3: (100, 100),
        5: (100, 100),
        7: (100, 100),
        9: (100, 100),
    },
    ('lom_2q', 'si1000', 0.004): {
        3: (100, 100),
        5: (100, 100),
        7: (100, 100),
    },
    ('lom_2q', 'si1000', 0.005): {
        3: (100, 100),
        5: (100, 100),
        7: (100, 100),
    },
    ('lom_2q', 'uniform', 0.003): {
        3: (100, 100),
        5: (100, 100),
        7: (100, 100),
        9: (100, 100),
    },
    ('lom_deep', 'lukin', 0.003): {
        3: (10000,),
        5: (13000,),
        7: (180000,),
        9: (730000,),
    },
    ('lom_deep', 'si1000', 0.001): {
        3: (10000,),
        5: (10000,),
        7: (17000,),
        9: (73000,),
    },
    ('lukin', 'lukin', 0.003): {
        3: (8200, 1200, 1300, 1100, 670, 500, 500, 500, 500, 500),
        5: (33000, 13000, 7600, 4700, 4600, 2500, 3000, 2100, 1100, 1600),
        7: (530000, 86000, 55000, 46000, 37000, 21000, 13000, 15000, 7800, 12000),
        9: (930000, 520000, 360000, 230000, 230000, 150000, 74000, 120000, 68000, 100000),
    },
    ('lukin', 'uniform', 0.003): {
        3: (500, 500, 500, 500, 500, 500, 500, 500, 500, 500),
        5: (910, 500, 500, 500, 500, 500, 500, 500, 500, 500),
        7: (7300, 1300, 930, 660, 500, 500, 500, 500, 500, 500),
        9: (33000, 4600, 2800, 1800, 1700, 930, 710, 500, 520, 670),
    },
    ('mccd', 'lukin', 0.003): {
        3: (1300, 690, 500, 350, 270, 180, 190, 160, 200, 390, 220, 160, 160, 160, 160, 160, 160, 160),
        5: (15000, 6900, 2200, 2900, 2600, 2200, 1400, 1800, 1400, 3000, 2500, 1500, 990, 920, 600, 590, 420, 440),
        7: (220000, 88000, 68000, 46000, 33000, 25000, 22000, 23000, 18000, 26000, 15000, 12000, 7300, 4300, 4600,
            3700, 3300, 2500),
        9: (420000, 270000, 200000, 160000, 130000, 110000, 95000, 84000, 75000, 270000, 140000, 110000, 73000, 54000,
            44000, 34000, 24000, 22000),
    },
    ('mccd', 'uniform', 0.003): {
        3: (160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160),
        5: (210, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160),
        7: (1300, 570, 390, 250, 210, 210, 170, 160, 160, 270, 170, 160, 160, 160, 160, 160, 160, 160),
        9: (4700, 1900, 1400, 780, 670, 560, 450, 400, 370, 1000, 440, 290, 200, 160, 160, 160, 160, 160),
    },
    ('cain_ghz', 'cain', 0.003): {
        3: (49000,),
        5: (180000,),
        7: (520000,),
        9: (10000000,),
    },
    ('cain_ghz', 'lukin', 0.003): {
        3: (38000,),
        5: (650000,),
        7: (840000,),
        9: (10000000,),
    },
    ('cain_ghz', 'uniform', 0.003): {
        3: (10000,),
        5: (16000,),
        7: (40000,),
        9: (10000000,),
    },
}
MLE_SHOTS = {
    ('lom_rep', 'lukin', 0.003): {
        3: (87000, 110000, 70000, 21000, 110000, 65000, 25000, 27000, 22000, 25000),
        5: (1100000, 1200000, 530000, 170000, 880000, 990000, 200000, 200000, 240000, 150000),
        7: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        9: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    },
    ('lom_rep', 'si1000', 0.001): {
        3: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        5: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        7: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        9: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    },
    ('lom_rep', 'si1000', 0.002): {
        3: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        5: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        7: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        9: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    },
    ('lom_rep', 'si1000', 0.003): {
        3: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        5: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        7: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        9: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    },
    ('lom_rep', 'si1000', 0.004): {
        3: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        5: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        7: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        9: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    },
    ('lom_rep', 'si1000', 0.005): {
        3: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        5: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        7: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        9: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    },
    ('lom_rep', 'uniform', 0.003): {
        3: (6300, 6500, 5900, 3500, 5800, 5000, 2900, 2800, 3000, 3000),
        5: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        7: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        9: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    },
    ('lom_2q', 'lukin', 0.003): {
        3: (31, 20),
        5: (250, 190),
        7: (0, 0),
        9: (0, 0),
    },
    ('lom_2q', 'si1000', 0.001): {
        3: (0, 0),
        5: (0, 0),
        7: (0, 0),
    },
    ('lom_2q', 'si1000', 0.002): {
        3: (0, 0),
        5: (0, 0),
        7: (0, 0),
    },
    ('lom_2q', 'si1000', 0.003): {
        3: (0, 0),
        5: (0, 0),
        7: (0, 0),
        9: (0, 0),
    },
    ('lom_2q', 'si1000', 0.004): {
        3: (0, 0),
        5: (0, 0),
        7: (0, 0),
    },
    ('lom_2q', 'si1000', 0.005): {
        3: (0, 0),
        5: (0, 0),
        7: (0, 0),
    },
    ('lom_2q', 'uniform', 0.003): {
        3: (3, 3),
        5: (0, 0),
        7: (0, 0),
        9: (0, 0),
    },
    ('lom_deep', 'lukin', 0.003): {
        3: (1200,),
        5: (13000,),
        7: (0,),
        9: (0,),
    },
    ('lom_deep', 'si1000', 0.001): {
        3: (0,),
        5: (0,),
        7: (0,),
        9: (0,),
    },
    ('lukin', 'lukin', 0.003): {
        3: (8200, 1200, 1300, 1100, 670, 470, 220, 380, 210, 250),
        5: (33000, 13000, 7600, 4700, 4600, 2500, 3000, 2100, 1100, 1600),
        7: (48964, 13114, 6794, 3978, 2769, 2417, 1474, 1076, 0, 0),
        9: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    },
    ('lukin', 'uniform', 0.003): {
        3: (210, 97, 74, 47, 66, 51, 30, 37, 22, 30),
        5: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        7: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        9: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    },
    ('mccd', 'lukin', 0.003): {
        3: (1300, 690, 500, 350, 270, 180, 190, 140, 200, 390, 220, 160, 150, 110, 68, 83, 52, 44),
        5: (15000, 6900, 2200, 2900, 2600, 2200, 1400, 1800, 1400, 3000, 2500, 1500, 990, 920, 600, 590, 420, 440),
        7: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        9: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    },
    ('mccd', 'uniform', 0.003): {
        3: (67, 40, 33, 26, 22, 17, 16, 12, 13, 45, 26, 20, 15, 12, 11, 10, 8, 7),
        5: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        7: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        9: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    },
    ('cain_ghz', 'cain', 0.003): {
        3: (0,),
        5: (0,),
        7: (0,),
        9: (0,),
    },
    ('cain_ghz', 'lukin', 0.003): {
        3: (38000,),
        5: (650000,),
        7: (0,),
        9: (0,),
    },
    ('cain_ghz', 'uniform', 0.003): {
        3: (4700,),
        5: (0,),
        7: (0,),
        9: (0,),
    },
}
BP_SHOTS = {
    ('lom_rep', 'lukin', 0.003): {
        3: (4700, 4900, 3900, 1200, 5800, 4800, 1400, 1500, 1300, 1400),
        5: (34000, 39000, 18000, 5600, 30000, 34000, 6800, 6400, 5700, 5600),
        7: (110000, 99000, 41000, 9700, 91000, 96000, 13000, 12000, 12000, 12000),
        9: (460000, 390000, 130000, 20000, 380000, 330000, 39000, 34000, 38000, 36000),
    },
    ('lom_rep', 'si1000', 0.001): {
        3: (2300, 3300, 3200, 2200, 3200, 3300, 980, 1100, 1200, 1200),
        5: (18000, 19000, 13000, 3900, 16000, 18000, 5500, 5100, 4500, 5300),
        7: (110000, 110000, 77000, 15000, 120000, 96000, 25000, 21000, 19000, 20000),
        9: (520000, 500000, 310000, 52000, 530000, 470000, 83000, 87000, 76000, 76000),
    },
    ('lom_rep', 'si1000', 0.002): {
        3: (990, 870, 860, 620, 730, 840, 320, 330, 300, 290),
        5: (2500, 2100, 1800, 780, 2200, 2100, 640, 550, 510, 640),
        7: (6400, 5900, 6000, 1500, 6400, 4300, 1500, 1400, 1600, 1500),
        9: (17000, 15000, 8000, 2000, 16000, 15000, 2500, 2500, 2200, 2100),
    },
    ('lom_rep', 'si1000', 0.003): {
        3: (550, 460, 460, 150, 270, 440, 210, 160, 180, 170),
        5: (500, 820, 480, 300, 760, 680, 170, 200, 240, 220),
        7: (1600, 1300, 1500, 470, 740, 1400, 350, 320, 410, 480),
        9: (2100, 2400, 1200, 450, 1700, 2500, 580, 490, 460, 430),
    },
    ('lom_rep', 'si1000', 0.004): {
        3: (300, 270, 250, 140, 260, 250, 120, 100, 99, 110),
        5: (340, 170, 320, 190, 350, 370, 120, 130, 110, 120),
        7: (570, 380, 360, 170, 570, 650, 130, 160, 160, 110),
        9: (580, 730, 440, 120, 490, 480, 130, 120, 140, 110),
    },
    ('lom_rep', 'si1000', 0.005): {
        3: (180, 150, 200, 97, 190, 190, 87, 80, 79, 82),
        5: (150, 200, 180, 90, 210, 220, 92, 66, 78, 76),
        7: (160, 290, 160, 73, 170, 230, 64, 91, 54, 76),
        9: (220, 230, 180, 61, 250, 280, 76, 64, 49, 52),
    },
    ('lom_rep', 'uniform', 0.003): {
        3: (430, 480, 610, 230, 430, 470, 210, 190, 170, 180),
        5: (1500, 1200, 1200, 400, 1100, 950, 320, 370, 280, 420),
        7: (2000, 3000, 2100, 770, 2100, 2200, 680, 800, 460, 540),
        9: (6100, 4800, 3100, 910, 4100, 4200, 1200, 940, 860, 940),
    },
    ('lom_2q', 'lukin', 0.003): {
        3: (4, 3),
        5: (16, 12),
        7: (54, 36),
        9: (280, 200),
    },
    ('lom_2q', 'si1000', 0.001): {
        3: (3, 2),
        5: (11, 11),
        7: (62, 36),
    },
    ('lom_2q', 'si1000', 0.002): {
        3: (1, 1),
        5: (2, 2),
        7: (9, 4),
    },
    ('lom_2q', 'si1000', 0.003): {
        3: (1, 1),
        5: (1, 1),
        7: (1, 1),
        9: (3, 2),
    },
    ('lom_2q', 'si1000', 0.004): {
        3: (1, 1),
        5: (1, 1),
        7: (1, 1),
    },
    ('lom_2q', 'si1000', 0.005): {
        3: (1, 1),
        5: (1, 1),
        7: (1, 1),
    },
    ('lom_2q', 'uniform', 0.003): {
        3: (1, 1),
        5: (1, 1),
        7: (2, 2),
        9: (6, 4),
    },
    ('lom_deep', 'lukin', 0.003): {
        3: (70,),
        5: (280,),
        7: (410,),
        9: (2800,),
    },
    ('lom_deep', 'si1000', 0.001): {
        3: (92,),
        5: (420,),
        7: (1700,),
        9: (7300,),
    },
    ('lukin', 'lukin', 0.003): {
        3: (620, 89, 91, 78, 55, 28, 24, 25, 16, 20),
        5: (2400, 430, 230, 130, 190, 93, 97, 92, 52, 93),
        7: (14000, 2200, 1400, 1200, 940, 520, 320, 370, 200, 310),
        9: (2676, 9100, 4500, 2600, 3200, 1700, 850, 1300, 780, 1200),
    },
    ('lukin', 'uniform', 0.003): {
        3: (39, 8, 6, 5, 8, 3, 3, 3, 2, 3),
        5: (94, 24, 11, 9, 9, 6, 5, 6, 4, 5),
        7: (120, 36, 27, 16, 31, 14, 9, 6, 6, 8),
        9: (820, 120, 69, 45, 43, 24, 18, 12, 14, 17),
    },
    ('mccd', 'lukin', 0.003): {
        3: (100, 54, 39, 28, 22, 18, 17, 14, 13, 31, 19, 12, 10, 10, 7, 4, 5, 4),
        5: (600, 280, 200, 150, 110, 89, 73, 64, 39, 120, 97, 58, 50, 26, 26, 18, 14, 14),
        7: (5800, 2400, 1800, 1200, 870, 660, 570, 600, 470, 670, 390, 310, 200, 120, 120, 96, 87, 66),
        9: (32000, 12000, 6800, 4800, 3300, 2800, 2400, 2100, 1700, 2800, 1300, 1100, 650, 480, 400, 310, 220, 200),
    },
    ('mccd', 'uniform', 0.003): {
        3: (7, 3, 3, 3, 2, 2, 2, 2, 2, 4, 2, 2, 1, 1, 1, 1, 1, 1),
        5: (18, 9, 9, 5, 5, 3, 4, 3, 3, 7, 5, 3, 3, 2, 2, 2, 1, 2),
        7: (51, 18, 16, 15, 12, 10, 8, 8, 6, 27, 8, 6, 5, 4, 3, 4, 3, 2),
        9: (190, 75, 31, 32, 27, 22, 18, 16, 15, 40, 15, 10, 8, 7, 6, 5, 4, 4),
    },
    ('cain_ghz', 'cain', 0.003): {
        3: (4900,),
        5: (18000,),
        7: (52000,),
        9: (220000,),
    },
    ('cain_ghz', 'lukin', 0.003): {
        3: (3800,),
        5: (17000,),
        7: (84000,),
        9: (430000,),
    },
    ('cain_ghz', 'uniform', 0.003): {
        3: (420,),
        5: (2000,),
        7: (3300,),
        9: (5200,),
    },
}
OURS_K_MLE = {
    ('lukin', 3, 'cnot', 3): (4000, 3500, 2100, 1500, 1300, 860, 770, 670, 580, 530),
    ('lukin', 3, 'cnot', 4): (4000, 2400, 1300, 920, 760, 580, 440, 350, 340, 320),
    ('lukin', 3, 'f1b', 4): (3900, 4000, 850, 1700, 360, 230, 410, 330, 210, 140),
    ('lukin', 3, 'mix', 3): (4000, 1900, 1400, 1500, 500, 1100, 230, 200, 270, 280, 340, 160, 190, 100, 210, 120),
    ('lukin', 3, 'mix', 4): (4000, 3300, 2000, 460, 290, 240, 860, 320, 120, 210, 140, 150, 300, 130, 43, 53),
    ('lukin', 3, 'struct', 3): (64000, 64000, 64000, 64000, 52000, 64000, 16000, 35000, 40000, 34000, 40000, 22000,
        36000, 7500, 18000, 19000, 18000, 20000, 11000, 18000, 3500, 8400, 9700, 8600, 9500, 4700, 8200, 1800),
    ('lukin', 3, 'struct', 4): (64000, 64000, 64000, 64000, 64000, 64000, 47000, 64000, 12000, 23000, 37000, 23000,
        31000, 31000, 29000, 17000, 30000, 5700, 13000, 16000, 13000, 15000, 14000, 15000, 7600, 13000, 2700, 5900,
        3400, 6400, 7900, 7000, 7400, 3500, 6300, 1400),
    ('lukin', 5, 'cnot', 3): (10000, 10000, 10000, 0, 0, 0, 0, 0, 0, 0),
    ('lukin', 5, 'cnot', 4): (10000, 10000, 10000, 0, 0, 0, 0, 0, 0, 0),
    ('lukin', 5, 'f1b', 4): (10000, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('lukin', 5, 'mix', 3): (10000, 10000, 0, 0, 2500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('lukin', 5, 'mix', 4): (10000, 6600, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('lukin', 5, 'struct', 3): (160000, 160000, 160000, 160000, 160000, 160000, 160000, 160000, 160000, 160000,
        160000, 160000, 160000, 68000, 0, 0, 0, 0, 0, 0, 35000, 0, 0, 0, 0, 0, 0, 0),
    ('lukin', 5, 'struct', 4): (160000, 160000, 160000, 160000, 160000, 160000, 160000, 160000, 160000, 160000,
        160000, 160000, 0, 0, 160000, 0, 160000, 63000, 0, 0, 0, 0, 0, 0, 0, 0, 28000, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('uniform', 3, 'cnot', 3): (460, 190, 120, 87, 70, 59, 46, 43, 40, 38),
    ('uniform', 3, 'cnot', 4): (330, 140, 84, 53, 49, 40, 34, 32, 30, 25),
    ('uniform', 3, 'f1b', 4): (200, 99, 47, 56, 37, 47, 24, 30, 40, 18),
    ('uniform', 3, 'mix', 3): (300, 70, 69, 66, 39, 41, 14, 16, 11, 17, 9, 15, 16, 14, 14, 8),
    ('uniform', 3, 'mix', 4): (150, 150, 40, 22, 20, 12, 41, 15, 15, 18, 14, 15, 23, 8, 7, 7),
    ('uniform', 3, 'struct', 3): (4100, 6600, 3800, 6500, 2700, 3200, 1700, 1800, 2500, 2000, 2100, 1200, 1900, 770,
        820, 1100, 920, 1300, 700, 890, 510, 560, 620, 550, 660, 410, 490, 310),
    ('uniform', 3, 'struct', 4): (3000, 6400, 3000, 6200, 3000, 3100, 2100, 2600, 1200, 1200, 2100, 1400, 1800, 1600,
        1600, 960, 1300, 630, 680, 910, 710, 900, 840, 900, 490, 660, 370, 430, 590, 450, 580, 470, 490, 320, 380,
        260),
}
OURS_K_BP = {
    ('lukin', 3, 'cnot', 3): (1100, 250, 150, 110, 91, 68, 75, 60, 36, 38),
    ('lukin', 3, 'cnot', 4): (630, 170, 89, 64, 53, 41, 31, 25, 24, 26),
    ('lukin', 3, 'f1b', 4): (270, 310, 60, 120, 25, 17, 21, 23, 22, 14),
    ('lukin', 3, 'mix', 3): (720, 130, 69, 33, 35, 13, 32, 23, 17, 15, 25, 9, 11, 10, 8, 9),
    ('lukin', 3, 'mix', 4): (300, 230, 140, 32, 20, 17, 61, 23, 9, 15, 10, 10, 21, 9, 3, 4),
    ('lukin', 3, 'struct', 3): (4400, 8600, 4400, 8400, 3700, 6000, 1100, 2400, 2800, 2400, 2800, 1600, 2100, 830,
        1200, 1300, 1200, 1400, 710, 1300, 250, 590, 680, 520, 670, 340, 580, 260),
    ('lukin', 3, 'struct', 4): (5600, 9400, 5000, 9700, 4600, 4500, 3300, 5300, 820, 1600, 2600, 1600, 2200, 2200,
        2100, 1200, 2100, 400, 890, 1100, 900, 1100, 950, 990, 530, 890, 190, 410, 450, 550, 540, 490, 520, 250, 420,
        98),
    ('lukin', 5, 'cnot', 3): (9700, 1400, 660, 410, 370, 230, 200, 190, 120, 140),
    ('lukin', 5, 'cnot', 4): (6900, 840, 410, 320, 200, 140, 120, 110, 83, 69),
    ('lukin', 5, 'f1b', 4): (540, 330, 970, 550, 420, 76, 170, 37, 65, 77),
    ('lukin', 5, 'mix', 3): (5200, 1000, 570, 210, 75, 94, 150, 150, 47, 110, 130, 31, 29, 51, 25, 23),
    ('lukin', 5, 'mix', 4): (4800, 210, 160, 320, 220, 70, 75, 140, 120, 27, 32, 38, 29, 17, 14, 19),
    ('lukin', 5, 'struct', 3): (51000, 60000, 48000, 62000, 25000, 54000, 7000, 9500, 19000, 15000, 17000, 6200,
        16000, 2100, 5600, 7400, 6400, 7700, 2800, 6200, 1300, 3000, 3500, 3600, 3400, 1100, 3300, 500),
    ('lukin', 5, 'struct', 4): (39000, 66000, 33000, 65000, 36000, 30000, 18000, 41000, 4700, 6400, 15000, 6600,
        13000, 12000, 13000, 3700, 13000, 2000, 3900, 5500, 3900, 5300, 5300, 5500, 1800, 4800, 860, 1900, 2600, 2600,
        2300, 2500, 2400, 880, 2400, 400),
    ('lukin', 7, 'cnot', 3): (20000, 13000, 4400, 2100, 1900, 1400, 1100, 900, 810, 790),
    ('lukin', 7, 'cnot', 4): (20000, 5800, 1700, 1700, 880, 520, 440, 460, 390, 300),
    ('lukin', 7, 'f1b', 4): (20000, 8800, 2300, 670, 1100, 450, 290, 390, 240, 89),
    ('lukin', 7, 'mix', 3): (20000, 7600, 1300, 1100, 800, 730, 730, 280, 190, 150, 150, 370, 470, 120, 260, 140),
    ('lukin', 7, 'mix', 4): (20000, 3600, 2000, 1900, 750, 270, 420, 240, 160, 74, 120, 270, 140, 300, 140, 120),
    ('lukin', 7, 'struct', 3): (320000, 320000, 320000, 320000, 200000, 320000, 49000, 73000, 120000, 97000, 110000,
        27000, 130000, 12000, 38000, 48000, 42000, 46000, 12000, 45000, 4600, 22000, 21000, 23000, 19000, 6000, 21000,
        1900),
    ('lukin', 7, 'struct', 4): (320000, 320000, 320000, 320000, 320000, 310000, 150000, 320000, 26000, 34000, 110000,
        49000, 93000, 92000, 86000, 21000, 110000, 8900, 21000, 36000, 29000, 35000, 43000, 36000, 7900, 38000, 3600,
        13000, 17000, 16000, 16000, 20000, 17000, 3900, 16000, 1500),
    ('lukin', 9, 'cnot', 3): (10000, 3675, 26000, 14000, 7800, 6200, 6200, 4700, 3600, 4000),
    ('lukin', 9, 'cnot', 4): (10000, 36000, 12000, 5300, 4300, 2900, 2100, 1700, 1400, 1400),
    ('lukin', 9, 'f1b', 4): (40000, 8100, 4000, 3800, 1300, 470, 520, 250, 1200, 110),
    ('lukin', 9, 'mix', 3): (7500, 40000, 874, 9100, 140, 4200, 2500, 1800, 1500, 1200, 1200, 780, 790, 380, 120, 840),
    ('lukin', 9, 'mix', 4): (3750, 832, 7200, 5800, 3300, 1100, 1900, 800, 1100, 910, 1100, 27, 780, 31, 260, 440),
    ('lukin', 9, 'struct', 3): (68239, 89138, 70962, 130909, 59511, 80519, 250000, 380000, 30708, 26443, 32312,
        150000, 33711, 52000, 220000, 13514, 9991, 13690, 50000, 13540, 16000, 3229, 3781, 3208, 3417, 26000, 3233,
        7300),
    ('lukin', 9, 'struct', 4): (50162, 71173, 50966, 70673, 47152, 47697, 41582, 53314, 200000, 190000, 24558, 330000,
        22927, 22764, 21524, 110000, 23936, 39000, 120000, 6935, 160000, 9951, 6731, 6893, 36000, 8274, 13000, 2012,
        2299, 1568, 2531, 2114, 2199, 16000, 1882, 5100),
    ('uniform', 3, 'cnot', 3): (59, 24, 16, 12, 9, 8, 6, 6, 6, 5),
    ('uniform', 3, 'cnot', 4): (43, 18, 11, 7, 7, 6, 5, 6, 4, 4),
    ('uniform', 3, 'f1b', 4): (25, 13, 7, 8, 5, 6, 4, 4, 4, 4),
    ('uniform', 3, 'mix', 3): (39, 9, 9, 9, 5, 6, 2, 2, 2, 3, 2, 2, 2, 2, 2, 1),
    ('uniform', 3, 'mix', 4): (20, 20, 6, 3, 3, 2, 6, 2, 2, 3, 2, 2, 3, 1, 2, 1),
    ('uniform', 3, 'struct', 3): (530, 860, 490, 840, 350, 420, 220, 230, 320, 260, 270, 150, 250, 100, 110, 140, 120,
        160, 90, 120, 65, 73, 100, 60, 86, 54, 64, 40),
    ('uniform', 3, 'struct', 4): (390, 830, 380, 800, 390, 400, 270, 340, 150, 150, 270, 180, 240, 200, 200, 130, 170,
        81, 88, 120, 92, 120, 110, 120, 78, 85, 50, 56, 76, 58, 78, 60, 63, 41, 49, 33),
    ('uniform', 5, 'cnot', 3): (260, 64, 34, 24, 18, 23, 16, 16, 12, 12),
    ('uniform', 5, 'cnot', 4): (170, 41, 29, 20, 16, 15, 11, 8, 9, 7),
    ('uniform', 5, 'f1b', 4): (36, 31, 18, 26, 29, 7, 11, 5, 6, 8),
    ('uniform', 5, 'mix', 3): (150, 63, 17, 13, 6, 10, 11, 8, 3, 5, 3, 3, 2, 4, 3, 2),
    ('uniform', 5, 'mix', 4): (76, 23, 7, 12, 9, 5, 4, 5, 5, 3, 5, 4, 3, 2, 2, 2),
    ('uniform', 5, 'struct', 3): (1800, 2800, 1800, 2700, 1200, 1600, 520, 560, 770, 580, 820, 480, 560, 260, 290,
        370, 320, 250, 210, 260, 97, 170, 240, 130, 220, 89, 130, 61),
    ('uniform', 5, 'struct', 4): (1300, 2700, 1200, 2800, 1500, 1400, 840, 1100, 380, 370, 690, 390, 670, 380, 520,
        360, 440, 200, 250, 300, 210, 260, 240, 240, 130, 190, 90, 110, 150, 140, 160, 81, 130, 68, 140, 57),
}
# --- plan.py end


def external_points(families=EXTERNAL, noises=None, ds=(3, 5, 7, 9), shots=None):
    """Every external point: per family its models (MODELS, or those of `noises` among them), p = 0.3 % (SOURCE_P where
    the source's p is used, SWEEP_P for SI1000 on SWEEP_D), shots per circuit from `shots` (default SHOTS; None where
    not set)."""
    shots = SHOTS if shots is None else shots
    pts = []
    for fam in families:
        for d in ds:
            for j, (v, label, depth, circs) in enumerate(family_points(fam, d)):
                for noise in MODELS[fam]:
                    if noises is not None and noise not in noises:
                        continue
                    ps = SWEEP_P if noise == "si1000" and d in SWEEP_D.get(fam, ()) else (SOURCE_P.get((fam, noise),
                                                                                                         0.003),)
                    for p in ps:
                        vv = 100 * PCODE[p] + v
                        cs = [dict(c_, seed=str(seed(noise, fam, d, vv, i))) for i, c_ in enumerate(circs)]
                        n = shots.get((fam, noise, p), {}).get(d)
                        cov = {f"{x}_shots": int(t.get((fam, noise, p), {}).get(d, (0,) * (j + 1))[j])
                               for x, t in (("mle", MLE_SHOTS), ("bplomr", BP_SHOTS))}
                        pts.append(dict(name=f"{fam}_{label}_d{d}_{noise}_p{PCODE[p]}", family=fam, d=d, noise=noise,
                                        p=p, depth=depth, L=len(circs[0]["layers"].split()), K=len(circs[0]["prep"]),
                                        shots=None if n is None else int(n[j]), **cov,
                                        seeds=[cs[0]["seed"], str(int(cs[-1]["seed"]) + 1)], circuits=cs))
    return pts


# ours_k: the K-block families with test-set seeds
OURS_K_SHOTS = {"lukin": {3: 4000, 5: 10000, 7: 20000, 9: 40000}, "uniform": {3: 2000, 5: 5000}}
OURS_K_N = {"mix": 64, "cnot": 16, "f1b": 16}


def ours_k_points(Ks=(3, 4), p=0.003):
    """The K-block families: mix (64 circuits, L = 1-16), cnot (16, L = 1-10), f1b (K = 4, 16, depth
    units 1-10, L = 2 units) and struct (one circuit per name, L = 2, 4, 8, 16, a quarter of a mix point's shots);
    circuits from duet.kblock_sets.circuits(..., tag 5), the same for both models."""
    pts, pc, pos = [], PCODE[p], collections.Counter()
    Ls = {"mix": range(1, 17), "cnot": range(1, 11), "f1b": range(2, 21, 2), "struct": (2, 4, 8, 16)}
    for noise, by_d in OURS_K_SHOTS.items():
        for K, d, fam in itertools.product(Ks, sorted(by_d), ("mix", "cnot", "f1b", "struct")):
            if fam == "f1b" and K != 4:
                continue
            for L in Ls[fam]:
                base = (cc.KBLOCK_SEED_BASE + NOISE_CODE[noise] * 10 ** 15 + ms.FAMILY[fam] * 10 ** 14 + K * 10 ** 13
                        + d * 10 ** 12 + pc * 10 ** 11 + L * 10 ** 9)
                if fam == "struct":
                    groups = [(f"{nm}_", [(base + v * 10 ** 6, x)]) for v, nm in enumerate(ms.struct_names(K))
                              if (x := ms.struct(nm, K, L))]
                else:
                    groups = [("", [(base + i, x) for i, x in enumerate(ms.circuits(fam, K, d, L, OURS_K_N[fam], 5))])]
                for label, circs in groups:
                    cs = [dict(layers=" ".join(lay), prep="".join(prep), meas="".join(meas), pauli="frame",
                               cnot_layers=sum(len(q) > 0 for q, _ in cc.parse_k(lay, K)), seed=str(s))
                          for s, (lay, prep, meas) in circs]
                    j, key = pos[(noise, d, fam, K)], (noise, d, fam, K)
                    pos[key] += 1
                    pts.append(dict(name=f"ours_k_{fam}_{label}K{K}_d{d}_L{L}_{noise}_p{pc}", family="ours_k", sub=fam,
                                    d=d, noise=noise, p=p, depth=L // 2 if fam == "f1b" else L, L=L, K=K,
                                    shots=by_d[d] * (OURS_K_N["mix"] // 4 if fam == "struct" else 1),
                                    mle_shots=int(OURS_K_MLE[key][j]) if key in OURS_K_MLE else 0,
                                    bplomr_shots=int(OURS_K_BP[key][j]) if key in OURS_K_BP else 0,
                                    seeds=[cs[0]["seed"], str(int(cs[-1]["seed"]) + 1)], circuits=cs))
    return pts


def prefix(pt, decoder):
    """Shots per circuit of point pt that `decoder` decodes: every shot for LoMatching, Cain et al. and the networks,
    the manifest's prefix for BP+LoMatching and the exact MLE (0: none)."""
    return pt["shots"] if decoder in ("lomr", "cain", "net") else pt.get(f"{decoder}_shots", 0)


def manifest(external=True, ours=True):
    pts = (external_points() if external else []) + (ours_k_points() if ours else [])
    missing = [pt["name"] for pt in pts if not pt["shots"]]
    if missing:
        raise ValueError(f"{len(missing)} points without shots (bench_sets.SHOTS), e.g. {missing[0]}")
    return dict(version=1, seed_base=str(BENCH_SEED_BASE), seed_range=[str(BENCH_SEED_BASE),
                str(BENCH_SEED_BASE + SEED_SPAN)], points=pts)


# ---------------------------------------------------------------------------------------- stored shots, loader

def _point(job):
    """Sample a point's circuits bit-packed into one zero-padded array per kind and
    write SETS/<name>.npz; returns (name, SHA-256 of the file)."""
    pt, out = job
    cs = [circuit(pt, c_) for c_ in pt["circuits"]]
    m = np.array([c.num_detectors for c in cs])
    k = np.array([c.num_observables for c in cs])
    D = np.zeros((len(cs), pt["shots"], (m.max() + 7) // 8), np.uint8) if len(cs) > 1 else None
    O = np.zeros((len(cs), pt["shots"], (k.max() + 7) // 8), np.uint8) if len(cs) > 1 else None
    for i, (c, c_) in enumerate(zip(cs, pt["circuits"])):
        dets, obs = c.compile_detector_sampler(seed=int(c_["seed"])).sample(pt["shots"], separate_observables=True,
                                                                            bit_packed=True)
        if len(cs) == 1:                         # no copy for one circuit (single-circuit points reach about 2 GB)
            D, O = dets[None], obs[None]
            break
        D[i, :, :dets.shape[1]] = dets
        O[i, :, :obs.shape[1]] = obs
    path = os.path.join(out, pt["name"] + ".npz")
    np.savez(path + ".tmp.npz", dets=D, obs=O, m=m, k=k, entry=json.dumps(pt))
    os.replace(path + ".tmp.npz", path)
    return pt["name"], hashlib.sha256(open(path, "rb").read()).hexdigest()


def packed(path, name):
    """Array `name` ("dets" or "obs": (circuits, shots, bytes) uint8) of a stored point as a read-only memmap: np.savez
    stores each array uncompressed, so its data sit at a fixed offset of the file. A job then reads only its slice."""
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(name + ".npy")
    assert info.compress_type == zipfile.ZIP_STORED
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        n, e = struct.unpack("<HH", f.read(30)[26:30])
        f.seek(info.header_offset + 30 + n + e)
        version = np.lib.format.read_magic(f)
        shape, fortran, dtype = (np.lib.format.read_array_header_1_0 if version == (1, 0) else
                                 np.lib.format.read_array_header_2_0)(f)
        return np.memmap(path, dtype=dtype, mode="r", shape=shape, offset=f.tell(), order="F" if fortran else "C")


def unpack(x, n):
    """bool (shots, n) array of packed rows x (shots, bytes), little-endian bits (stim's b8)."""
    return np.unpackbits(np.asarray(x), axis=1, count=int(n), bitorder="little").astype(bool)


def iter_point(path):
    """A stored point circuit by circuit: (entry, [(i, dets, obs)]) with dets, obs the bool (shots, m[i]) and
    (shots, k[i]) arrays of circuit i, unpacked one circuit at a time from memmaps (each array read once, finding 7)."""
    z = np.load(path)
    m, k, entry = z["m"], z["k"], json.loads(str(z["entry"]))
    D, O = packed(path, "dets"), packed(path, "obs")
    return entry, ((i, unpack(D[i], m[i]), unpack(O[i], k[i])) for i in range(len(m)))


def load_point(path):
    """A stored point: dict(entry, m, k, dets, obs) with dets[i], obs[i] the bool (shots, m[i]) and (shots, k[i])
    arrays of circuit i (stim's detection events and observable flips)."""
    z = np.load(path)
    entry, it = iter_point(path)
    dets, obs = [], []
    for _, x, y in it:
        dets.append(x)
        obs.append(y)
    return dict(entry=entry, m=z["m"], k=z["k"], dets=dets, obs=obs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["manifest", "generate"])
    ap.add_argument("--out")
    ap.add_argument("--manifest")
    ap.add_argument("--dir")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default="")
    ap.add_argument("--shard", default="", help="J/N: this job's share of the points (by name)")
    a = ap.parse_args()
    if a.what == "manifest":
        m = manifest()
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(m, open(a.out, "w"), indent=0)
        print(len(m["points"]), "points")
        return
    m = json.load(open(a.manifest))
    os.makedirs(a.dir, exist_ok=True)
    only = set(a.only.split(",")) if a.only else None
    todo = [(pt, a.dir) for pt in m["points"] if (only is None or pt["name"] in only)
            and not os.path.exists(os.path.join(a.dir, pt["name"] + ".npz"))]
    if a.shard:
        j, n = map(int, a.shard.split("/"))
        todo = [t for t in todo if zlib.crc32(t[0]["name"].encode()) % n == j]
    todo.sort(key=lambda j: -j[0]["d"] ** 2 * j[0]["L"] * j[0]["K"] * len(j[0]["circuits"]))
    with mp.get_context("fork").Pool(a.workers) as pool:
        for name, sha in pool.imap_unordered(_point, todo):
            with open(os.path.join(a.dir, f"SHA256SUMS{'.' + a.shard.replace('/', 'of') if a.shard else ''}"),
                      "a") as f:
                f.write(f"{sha}  {name}.npz\n")
            print(name, flush=True)


if __name__ == "__main__":
    main()
