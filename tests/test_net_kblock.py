"""Tests of the K-block network tables and model: the logical line
of the observables, the labels and the inference-time correction agree for every gate in every frame (review finding
1); K-block tables are consistent; two-block tables, outputs, loss and gradients equal the earlier two-block code's bit for bit
(tests/kblock_net_k2_ref.py); relabelling the blocks, with the product basis carried along, permutes the tables, the
index gather equals its one-hot product; the trainer runs on mixed block counts."""
import itertools
import os

import numpy as np
import pytest
import stim
import torch

from duet import clifford_circuits as cc
from duet.circuits import block_offset
from duet.net import labels_ro as LR
from duet.net import model_cl as M
from duet.net import spec_cl
from duet.net import train_cl

P = 0.003
DATA = os.path.join(os.path.dirname(__file__), "data")


def frame_cases(K):
    """Two-layer circuits that put every gate in every frame: H on any subset of the blocks first (the frames), then
    I, H or S on each block, or a CNOT pair in either direction with I, H or S on the other blocks."""
    seconds = ["".join(v) for v in itertools.product("IHS", repeat=K)]
    for c, t in itertools.permutations(range(K), 2):
        rest = [b for b in range(K) if b not in (c, t)]
        for gs in itertools.product("IHS", repeat=len(rest)):
            tok = [""] * K
            tok[c], tok[t] = str(t), "+"
            for b, g in zip(rest, gs):
                tok[b] = g
            seconds.append("".join(tok))
    return [[a, b] for a in ("".join(v) for v in itertools.product("IH", repeat=K)) for b in seconds]


def bases_with_product(rng, lay, K, tries=32):
    for _ in range(tries):
        prep, meas = tuple(rng.choice(["Z", "X"], K)), tuple(rng.choice(["Z", "X"], K))
        if cc.logical_products(lay, prep, meas):
            return prep, meas
    return None


def observable_lines(d, K, labelled):
    """{observable: data coordinates it reads in the final measurement} of a labelled circuit."""
    lab = stim.Circuit(labelled)
    mq = [t.value for inst in lab.flattened() if stim.gate_data(inst.name).produces_measurements
          for t in inst.targets_copy()]
    n_rec, xy = len(mq), lab.get_final_qubit_coordinates()
    final = set(range(n_rec - K * d * d, n_rec))
    recs = {}
    for inst in lab.flattened():
        if inst.name == "OBSERVABLE_INCLUDE":
            recs.setdefault(int(inst.gate_args_copy()[0]), set()).symmetric_difference_update(
                n_rec + t.value for t in inst.targets_copy())
    return {k: {tuple(xy[mq[r]]) for r in v & final} for k, v in recs.items()}


def line_coords(d, blocks, meas, f_end, c):
    out = set()
    for b in blocks:
        row = (meas[b] == "Z") == (f_end[b] == 0)
        for a in range(1, 2 * d, 2):
            x, y = (a, c) if row else (c, a)
            out.add((float(x + block_offset(d, b)), float(y)))
    return out


@pytest.mark.parametrize("K", [2, 3])
def test_lines_every_gate_every_frame(K):
    """The observables (labels y) sit on line0's line, and representative j of the labelled circuit (labels y_kj, the
    per-layer labels) and A (the correction g at inference) on line j, for every gate in every frame; H2+ is the
    review's case (line0 gave 1 for parse_k's layers)."""
    d = 3
    rng = np.random.default_rng(3 + K)
    n = 0
    for lay in frame_cases(K) + ([["H2+"], ["1+S", "H2+"]] if K == 3 else []):
        bases = bases_with_product(rng, lay, K)
        if bases is None:
            continue
        prep, meas = bases
        ops = cc.parse_k(lay, K)
        c0 = cc.line0(d, ops)
        assert c0 == (d if any(set(g) & set("HS") for _, g in ops) else 1)
        s = spec_cl.build(d, lay, prep, meas, P)
        f_end = cc.make_circuit(d, lay, P, prep, meas, meta=True)[1]["final_frames"]
        got = observable_lines(d, K, s["labelled"])
        for k, target in enumerate(s["targets"]):
            blocks = [b for b, _, _ in target]
            assert got[k] == line_coords(d, blocks, meas, f_end, c0), (lay, prep, meas, k)
            for j in range(d):
                assert got[s["n_obs"] + k * d + j] == line_coords(d, blocks, meas, f_end, 2 * j + 1), (lay, k, j)
        n += 1
    assert n >= (40 if K == 2 else 300)
    assert cc.line0(d, cc.parse_k(["H2+"])) == d and cc.line0(d, cc.parse(["C0", "HS"])) == d


@pytest.mark.parametrize("K", [3, 4])
def test_k_tables_and_labels(K):
    d = 3
    rng = np.random.default_rng(17 + K)
    for _ in range(6):
        lay, prep, meas = cc.sample_circuit_k(rng, K, int(rng.integers(2, 7)), 0.5)
        s = spec_cl.build(d, lay, prep, meas, P)
        T, S, n_obs = s["T"], s["slots"], s["n_obs"]
        assert s["K"] == K and S == max(2, K) and s["mx"].shape == (K, T, d + 1, d + 1)
        assert s["gate"].shape == (K, T, 2, 4)
        assert s["parts"].shape == (S, T, K, 2) and s["A"].shape == (s["m"], S * d) and s["bound"].shape == (K, T, 4)
        assert s["bound"][:, 0, 0].all() and s["bound"][:, -1, 1].all() and s["bound"].sum() == 2 * K
        assert all(len(e) == 3 and e[1] == T - 1 and e[2] == 0 for tg in s["targets"] for e in tg)
        assert s["mx"].sum() + s["mz"].sum() == s["m"]
        for b, t, side in zip(*np.nonzero(np.ones((K, T, 2)))):
            pc, pb = s["pcode"][b, t, side], s["pblk"][b, t, side]
            if pc in (1, 2, 3):
                assert pb != b and s["pblk"][pb, t, side] == b and s["pcode"][pb, t, side] in (1, 2, 3)
            else:
                assert pb == b
        for k in range(n_obs):
            for t in range(T):
                for b in range(K):
                    for part in range(2):
                        par = [bin(int(v)).count("1") % 2 for v in s["lines"][k, t, b, part]]
                        assert set(par) == {int(s["parts"][k, t, b, part])}, (lay, k, t, b, part)
        ext, _ = LR.extended_dem(s)
        dets, obs, _ = ext.compile_sampler(seed=3).sample(1000)
        y, yl, u = LR.split_labels(obs.astype(np.uint8), s)
        g = (dets.astype(np.int64) @ s["A"].astype(np.int64) % 2).reshape(-1, S, d)[:, :n_obs]
        assert np.array_equal(yl, y[:, :, None] ^ g) and np.array_equal(np.bitwise_xor.reduce(u, axis=3), yl)
        tab, _ = train_cl.tables(d, lay, prep, meas, P)
        assert tab["offb"].shape == (S, T, K, (d + 1) ** 2, d) and tab["cnt"].shape == (S, T, 3)


@pytest.mark.parametrize("name", ["kblock_net_k2.pt", "kblock_net_k2_v2.pt"])
def test_two_block_bit_for_bit(name):
    """Two-block circuits: the K-block code gives the earlier two-block code's tables, outputs (c, z, l in training and evaluation
    form), losses and every parameter's gradient bit for bit with the same weights on the same shots: random
    weights, and the Lukin d = 3 v2 network of. 
    gave NetCL NetRO's representatives, so A, the labelled circuit, the extended model and the evaluation-form c (which
    training-form outputs, losses and gradients comparable."""
    import kblock_net_k2_ref as R
    ref = torch.load(os.path.join(DATA, name), weights_only=False)
    if R.cpu_model() != ref["cpu"]:
        pytest.skip(f"reference made on {ref['cpu']}; this CPU's kernels may round differently")
    if "ckpt" in ref and not os.path.exists(R.ckpt_path(ref)):
        pytest.skip(f"{R.ckpt_path(ref)} is not on this machine")
    label = ("spec.A", "spec.labelled", "tab.A", "ext")
    keep = lambda rows: [{k: v for k, v in r.items() if k not in label} for r in rows]   # noqa: E731
    assert keep(R.table_digests(ref["cases"])) == keep(ref["digests"])
    bad = [k for k in R.unequal(ref, R.outputs(ref)) if not k.endswith(".c_eval")]
    assert not bad, bad[:5]


def relabel(lay, prep, meas, perm):
    """The circuit with block b renamed perm[b]."""
    K = len(prep)
    out = []
    for tok in lay:
        new = [""] * K
        for b, ch in enumerate(tok):
            new[perm[b]] = str(perm[int(ch)]) if ch.isdigit() else ch
        out.append("".join(new))
    inv = [perm.index(b) for b in range(K)]
    return out, tuple(prep[i] for i in inv), tuple(meas[i] for i in inv)


def canonical(dem, dmap=None):
    out = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        ts = inst.targets_copy()
        key = (frozenset(dmap[t.val] if dmap is not None else t.val for t in ts if t.is_relative_detector_id()),
               frozenset(t.val for t in ts if t.is_logical_observable_id()))
        p = inst.args_copy()[0]
        out[key] = out.get(key, 0.0) * (1 - p) + p * (1 - out.get(key, 0.0))
    return out


def small_net(d, seed=0):
    torch.manual_seed(seed)
    net = M.NetCL(d, width=32, bott=8, blocks=2, heads=4).eval()
    with torch.no_grad():
        for q in net.parameters():
            q.add_(0.1 * torch.randn_like(q))
    return net


def batch(tab, dets, obs, d):
    pack = lambda a: np.packbits(a.astype(np.uint8), axis=1, bitorder="little")   # noqa: E731
    return train_cl.collate([(tab, pack(dets), pack(obs))], d, torch.device("cpu"))


@pytest.mark.parametrize("K,n", [(3, 5), (4, 4)])
def test_block_permutation_equivariance(K, n):
    """Lukin model (every detector one check; under the uniform model the round-0 combinations sit on the higher block
    of their pair, so a relabelling can move one site: the stated limit). With the basis of the first circuit carried
    to the relabelled one, the tables permute exactly, the extended error models (labels included) are equal under the
    detector map, the network's outputs agree to float rounding, LoMatching's regions and Cain et al.'s subgraphs map
    exactly, and their decisions on the same shots agree."""
    from duet.decoders.cain2025 import CainDecoder
    from duet.decoders.lomatching_dec import LoMatchingRegions
    d = 3
    rng = np.random.default_rng(91 + K)
    net = small_net(d)
    for _ in range(n):
        lay, prep, meas = cc.sample_circuit_k(rng, K, int(rng.integers(2, 6)), 0.5)
        perm = [int(v) for v in rng.permutation(K)]
        lay2, prep2, meas2 = relabel(lay, prep, meas, perm)
        s1 = spec_cl.build(d, lay, prep, meas, P, "lukin")
        basis = [sorted(perm[b] for b, _, _ in tg) for tg in s1["targets"]]
        s2 = spec_cl.build(d, lay2, prep2, meas2, P, "lukin", products=basis)
        pa = np.array(perm)
        idx2 = {(int(b), *map(int, tij)): k for k, (b, tij) in enumerate(zip(s2["block"], s2["tij"]))}
        dmap = np.array([idx2[(perm[int(b)], *map(int, tij))] for b, tij in zip(s1["block"], s1["tij"])])
        for key in ("mx", "mz", "gate", "pcode", "bound"):
            assert np.array_equal(s2[key][pa], s1[key]), key
        assert np.array_equal(s2["pblk"][pa], pa[s1["pblk"]])
        for key in ("parts", "lines", "orient"):
            assert np.array_equal(s2[key][:, :, pa], s1[key]), key
        assert np.array_equal(s2["A"][dmap], s1["A"])
        tab1, ext1 = train_cl.tables(d, lay, prep, meas, P, "lukin")
        tab2, ext2 = train_cl.tables(d, lay2, prep2, meas2, P, "lukin", products=basis)
        a, b = canonical(ext1, dmap), canonical(ext2)
        assert a.keys() == b.keys() and all(abs(a[k] - b[k]) < 1e-12 for k in a)
        # the network: shots of the first circuit, their detectors moved by the map
        dets, obs, _ = ext1.compile_sampler(seed=5).sample(64)
        dets2 = np.zeros_like(dets)
        dets2[:, dmap] = dets
        with torch.no_grad():
            for train in (False, True):
                o1 = train_cl.forward(net, batch(tab1, dets, obs, d), train)
                o2 = train_cl.forward(net, batch(tab2, dets2, obs, d), train)
                for x, y in zip(o1, o2):
                    assert torch.allclose(x, y, atol=1e-5), (lay, perm, (x - y).abs().max())
        # the opponents
        c1, m1 = cc.make_circuit(d, lay, P, prep, meas, meta=True, noise="lukin")
        c2, m2 = cc.make_circuit(d, lay2, P, prep2, meas2, meta=True, noise="lukin", products=basis)
        r1, r2 = cc.regions(m1, lay, meas), cc.regions(m2, lay2, meas2)
        assert all(set(dmap[x].tolist()) == set(y.tolist()) for x, y in zip(r1, r2))
        cd1, cd2 = CainDecoder(c1), CainDecoder(c2)
        for (v1, k1), (v2, k2) in zip(cd1.subgraphs, cd2.subgraphs):
            assert set(dmap[list(v1)].tolist()) == set(v2) and \
                {frozenset(dmap[list(x)].tolist()) for x in k1} == {frozenset(x) for x in k2}
        sd = c1.compile_detector_sampler(seed=6).sample(2000)
        sd2 = np.zeros_like(sd)
        sd2[:, dmap] = sd
        for dec1, dec2 in ((LoMatchingRegions(c1, r1), LoMatchingRegions(c2, r2)), (cd1, cd2)):
            p1, p2 = np.asarray(dec1.decode_batch(sd), bool), np.asarray(dec2.decode_batch(sd2), bool)
            assert (p1 == p2).all(1).mean() >= 0.995, (lay, perm, (p1 != p2).any(1).sum())


def test_gather_equals_onehot():
    """The index gather gives the one-hot product's bias values: equal outputs; gradients equal to rounding."""
    d, K = 3, 4
    rng = np.random.default_rng(23)
    lay, prep, meas = cc.sample_circuit_k(rng, K, 4, 0.5)
    tab, ext = train_cl.tables(d, lay, prep, meas, P)
    dets, obs, _ = ext.compile_sampler(seed=2).sample(32)
    b = batch(tab, dets, obs, d)
    outs, grads = [], []
    for lookup in ("onehot", "gather"):
        net = small_net(d, 1).train()
        net.head.lookup = lookup
        c, z, l = train_cl.forward(net, b)
        loss = M.loss_cl(c, z, l, b["y"], b["yl"], b["u"], b["slot"], 1.0)[0]
        loss.backward()
        outs.append((c, z, l))
        grads.append({k: q.grad.clone() for k, q in net.named_parameters()})
    assert all(torch.equal(x, y) for x, y in zip(*outs))
    assert all(torch.allclose(grads[0][k], grads[1][k], rtol=1e-5, atol=1e-7) for k in grads[0])


def test_given_products():
    lay, prep, meas = ["1+I", "I2+"], ("X", "Z", "Z"), ("Z", "Z", "Z")
    canon = cc.logical_products(lay, prep, meas)
    assert [s for s, _ in canon] == [[0, 1], [0, 2]]
    alt = cc.logical_products(lay, prep, meas, basis=[[1, 2], [0, 1]])
    assert [s for s, _ in alt] == [[1, 2], [0, 1]]
    for bad in ([[0]], [[0, 1], [1, 0]], [[0, 1], [0, 2], [1, 2]]):
        with pytest.raises(ValueError):
            cc.logical_products(lay, prep, meas, basis=bad)
    c = cc.make_circuit(3, lay, P, prep, meas, products=[[1, 2], [0, 1]])
    assert c.num_observables == 2


def test_train_mixed_block_counts(tmp_path):
    """The trainer runs batches of 2, 3 and 4 blocks (K > 2 with the ent share) and monitors each block count."""
    out = str(tmp_path / "run")
    rc = train_cl.main(["--out", out, "--d", "3", "--Ls", "2", "--samples", "96", "--batch", "16", "--chunk", "8",
                        "--width", "16", "--bott", "4", "--blocks", "1", "--workers", "1", "--prefetch", "4",
                        "--eval-circuits", "1", "--eval-cnot", "1", "--eval-per", "8", "--eval-every", "96",
                        "--block-counts", "2,3,4", "--p-cnot-set", "0,0.5,1", "--cnot-share", "0.25",
                        "--ent-share", "0.5",
                        "--lookup", "gather", "--noise", "lukin", "--device", "cpu"])
    assert rc == 0
    rows = [__import__("json").loads(x) for x in open(os.path.join(out, "log.jsonl"))]
    assert set(rows[-1]["ema_failures"]) == {"mix", "cnot", "mix3", "cnot3", "mix4", "cnot4"}
