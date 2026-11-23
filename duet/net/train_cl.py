"""Train NetCL (duet/net/model_cl.py) on random circuits of the Clifford family:

    python -m duet.net.train_cl --d 3 --p 0.003 --Ls 4,6,8 --samples 1e8 --out runs/cl_d3 [options]

train_ro's recipe (AdamW 1e-3, betas 0.9 / 0.99, weight decay 0.01, warm-up, cosine to 2 % (--lr-floor), clip 1, EMA 0.999, bf16
autocast, final + per-layer loss with lambda 1) with a fresh circuit per chunk: chunk k draws depth Ls[(k // per batch)
mod |Ls|] and a circuit of that depth (generator numpy default_rng([seed, k])): with probability --cnot-share one of
today's CNOT-only circuits (random directions, |00>, Z readout), otherwise duet.clifford_circuits.sample_circuit with a
CNOT probability drawn per circuit from --p-cnot-set (or the fixed --p-cnot). It builds the circuit's tables
(duet.net.spec_cl.build) and extended detector error model (duet.net.labels_ro.extended_dem) and samples --chunk shots
with stim seed (seed * 1,000,003 + k) mod 2^63 (training seeds stay below 2^63). A batch holds --batch / --chunk chunks
of one depth, in --accum micro-batches. Monitor: --eval-circuits mixed circuits (the mix without its CNOT-only share)
and --eval-cnot CNOT-only circuits per depth (seeds duet.clifford_circuits.SEED_BASE + 9 * 10^15 +...), --eval-per shots
each, failures of the EMA and live networks per family and depth every --eval-every samples. --noise picks the circuit
noise model of training and monitor alike (duet.clifford_circuits: uniform, the default, or lukin).
K blocks: --block-counts lists block counts K, one per share of the batches;
batch i holds one (K, depth) pair, the i-th of the list (K, L) over the block counts and depths in turn, so the default
2 keeps two-block stream; K > 2 draws the K-block family (train_cl.draw) and its monitor circuits (families
mix<K> and cnot<K>, seeds duet.clifford_circuits.KBLOCK_SEED_BASE + 9 * 10^15 +...); --ent-share of the K > 2 chunks come
from the entangled-then-idle family (duet.kblock_sets.ent), and --ent-k2 of the K = 2 chunks. --lookup gather swaps the read-out's one-hot table product for
an index gather (equal forward values, less memory).
--init (fine-tuning) logs the loaded weights' monitor row at step 0. --init may hold fewer trunk blocks than --blocks
(model_cl.load_grown: the added blocks start at zero output). --Ls may repeat a depth to give it more batches (each
entry one share; the monitor reads each depth once).
--joint 1 adds model_cl's joint read-out of single-layer circuits (zero output at start); --freeze-trunk trains the
read-out alone.
Single-layer focus: the first --l1-first samples, and afterwards a share --l1-share of
the batches spread evenly, are two-block single-layer batches whose chunks draw one of the six single-layer families with equal share, or with the shares of --l1-fam F:W..., and a circuit uniformly within it; the other batches take
the (K, L) list over --block-counts and --Ls in turn as before, in a fixed shuffled order (numpy default_rng(0)) when
such batches exist. An --Ls entry L:n stands for n entries. --pin 1 adds the input channel log(p / 0.003) (model_cl,
zero stem weights when --init has none); --p-l1 P:W... draws each single-layer chunk's p from that set with those
weights (after its circuit, so the circuit stream is unchanged), over the first --p-ramp samples moved linearly from --p
to the drawn p, except in the last --p-last samples,
which keep --p;
the chunk's tables, extended model and labels use its p. --save-at N... saves the EMA weights as ema_<N/1e6>M.pt when
training passes N samples. --init may also hold a narrower trunk bottleneck or no p channel (model_cl.load_grown).
Single-layer residual: --res N adds model_cl's residual path of whole single-layer
circuits (N blocks; --route 1 its error-model routing, spec_cl.routes, --prod 1 its product head; zero output at the
start when --init has none) and trains it alone, every other weight frozen; tables adds the routing of every
single-layer circuit (nbr, rel).
Single-layer path: --l1path 1 adds model_cl's single-layer path, a copy of the
network with the residual of --res / --route / --prod that decodes whole single-layer circuits; only the path trains, the network stays as loaded (--init: a
checkpoint without the path, or with a path without the p channel or with fewer blocks, model_cl.load_grown).
Lukin family: --lukin-share S makes the share S of the batches, spread evenly, batches of
the open benchmark's lukin family (duet.bench_sets.lukin: four blocks in |+>, per depth unit a gate from {I, X, Y, Z, H}
per block and then two CNOTs, L = 2D, the target first among the products), chunk k's logical circuit random_logical(D,
k, seed=--seed) and its Paulis physical (tables(..., pauli="gate")); the batches' D cycles through --lukin-depths. --p-tri
F: over the last fraction F of the samples each chunk's p rises linearly from p to 2p and falls back (arm MH).
Exact resume: ckpt.pt (weights, EMA, optimizer, scheduler, step, loss sums) every --ckpt-minutes and on SIGTERM (then
exit 99, PREEMPTED); a restart with the same --out resumes from it, the chunk stream continuing at chunk step * per
batch. A finished run (run.json) exits at once. Output: log.jsonl, ckpt.pt, last_ema.pt, final_ema.pt (FORMAT),
run.json."""
import argparse
import collections
import copy
import itertools
import json
import math
import multiprocessing as mp
import os
import signal
import sys
import time

import numpy as np
import torch

from duet import clifford_circuits as cc
from duet import kblock_sets
from duet.net import labels_ro as LR
from duet.net import spec_cl

FORMAT = "duet_net/3"


def format_ok(f):
    """True for this version's checkpoint format; also for the same version written under an earlier name."""
    return isinstance(f, str) and f.rsplit("/", 1)[-1] == FORMAT.rsplit("/", 1)[-1]            # 3: NetRO's line representatives; 2: per-block sum of
# the part vectors (model_cl); a run resumes only under the format it started with
MON_BASE = cc.SEED_BASE + 9 * 10 ** 15
KBLOCK_MON_BASE = cc.KBLOCK_SEED_BASE + 9 * 10 ** 15
PREEMPTED = 99
# settings a resumed run may change (none changes what is trained)
FREE = {"workers", "prefetch", "compile", "device", "max_minutes", "max_steps", "ckpt_minutes", "save_at",
        "out", "init"}     #: a run moves between hosts (the paths differ; init is read at step 0 only)


def parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--noise", default="uniform", choices=["uniform", "lukin"], help="circuit noise model "
                    "(duet.clifford_circuits: uniform = every operation at p; lukin = Bonilla Ataides et al.'s)")
    ap.add_argument("--p", type=float, default=0.003)
    ap.add_argument("--Ls", default="4,6,8")
    ap.add_argument("--p-cnot", type=float, default=cc.P_CNOT)
    ap.add_argument("--gates", default="IHS")
    ap.add_argument("--samples", type=float, default=5e7)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--bott", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--lr-floor", type=float, default=0.02, help="the cosine's end as a share of --lr ()")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--eval-circuits", type=int, default=100)
    ap.add_argument("--eval-per", type=int, default=200)
    ap.add_argument("--eval-every", type=float, default=5e6)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--prefetch", type=int, default=32)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--cnot-share", type=float, default=0.0, help="fraction of chunks from today's CNOT-only family")
    ap.add_argument("--ent-share", type=float, default=0.0, help="K > 2: fraction of chunks from the "
                    "entangled-then-idle family (duet.kblock_sets.ent)")
    ap.add_argument("--ent-k2", type=float, default=0.0, help="K = 2: fraction of chunks from the entangled-then-idle "
                    "family (; 0 keeps the two-block stream)")
    ap.add_argument("--p-cnot-set", default="", help="comma list: a CNOT probability drawn per circuit")
    ap.add_argument("--eval-cnot", type=int, default=0, help="monitor circuits per depth from the CNOT-only family")
    ap.add_argument("--accum", type=int, default=1, help="micro-batches per batch (memory)")
    ap.add_argument("--ckpt-minutes", type=float, default=5.0)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many steps in this process (exit 3)")
    ap.add_argument("--init", help="start from this checkpoint's weights (fine-tuning)")
    ap.add_argument("--joint", type=int, default=0, help="1: the joint read-out of single-layer circuits")
    ap.add_argument("--freeze-trunk", action="store_true", help="train the read-out alone")
    ap.add_argument("--pin", type=int, default=0, help="1: the p input channel")
    ap.add_argument("--res", type=int, default=0, help="blocks of the single-layer residual, trained alone ()")
    ap.add_argument("--route", type=int, default=0, help="1: the residual's error-model routing")
    ap.add_argument("--prod", type=int, default=0, help="1: the residual's product head")
    ap.add_argument("--l1path", type=int, default=0, help="1: the single-layer path (), trained alone")
    ap.add_argument("--troute", type=int, default=0, help="1: the single-layer path's routing before every trunk block")
    ap.add_argument("--flags", type=int, default=0, help="1: the single-layer path's two idle-tick input channels")
    ap.add_argument("--l1-blocks", type=int, default=0, help="the single-layer path's own trunk blocks (0: --blocks; "
                    ")")
    ap.add_argument("--l1-first", type=float, default=0.0, help="samples of two-block single-layer batches first")
    ap.add_argument("--l1-share", type=float, default=0.0, help="then this share of two-block single-layer batches")
    ap.add_argument("--l1-fam", default="", help="F:W,... single-layer chunks draw the family with these weights "
                    "(all six of single_layer_families; default equal shares)")
    ap.add_argument("--p-l1", default="", help="P:W,...: single-layer chunks draw p from this set with these weights")
    ap.add_argument("--p-last", type=float, default=0.0, help="the last samples at --p whatever --p-l1")
    ap.add_argument("--p-ramp", type=float, default=0.0, help="--p-l1's p reached linearly from --p over these samples")
    ap.add_argument("--lukin-share", type=float, default=0.0, help="share of the batches from the lukin family (run "
                    ": duet.bench_sets.lukin with a fresh logical circuit per chunk, physical Paulis, K = 4)")
    ap.add_argument("--lukin-depths", default="1,2,3,4,5,6,7,8,9,10", help="the lukin batches' depth units D (L = 2D) "
                    "in turn; an entry D:n stands for n entries")
    ap.add_argument("--p-tri", type=float, default=0.0, help="over this last fraction of the samples every chunk's p "
                    "rises linearly to 2p and falls back to p (arm MH)")
    ap.add_argument("--save-at", default="", help="comma list of sample counts: EMA snapshots ema_<N/1e6>M.pt")
    ap.add_argument("--block-counts", default="2", help="comma list of block counts K, one entry per share of the "
                    "batches (2 = two-block stream; e.g. 2,3,4: equal shares); a batch holds one K and depth")
    ap.add_argument("--lookup", default="onehot", choices=["onehot", "gather"], help="the read-out's line-bias table "
                    "(model_cl.LayerLinesCL.lookup; gather saves memory at large K and d)")
    ap.add_argument("--teacher-lom", type=float, default=0.0, help="B: add B times the BCE of the final logits "
                    "against LoMatching's per-observable decisions on the single-layer chunks of the one-block "
                    "families (TEACH_FAMS;); 0: today's loss")
    ap.add_argument("--replay", default="", help="comma list of pool files (pool-mining script not included): each micro-batch "
                    "replaces --replay-share of its shots with shots drawn from the pools (, arm R)")
    ap.add_argument("--replay-share", type=float, default=0.125)
    ap.add_argument("--replay-views", type=float, default=0, help="if > 0, the share becomes min(--replay-share, "
                    "this x pool shots / --samples), set at the start from the pool files (); "
                    "run.json and the checkpoints keep it as replay_share")
    ap.add_argument("--couple", default="cross", choices=["cross", "late", "none"], help="ablations: "
                    "cross-block exchange in the trunk (cross), only at the read-out (late) or nowhere (none: the "
                    "read-out per block; model_cl.NetCL)")
    ap.add_argument("--head", default="lines", choices=["lines", "pool"], help="the per-layer read-out (lines) or run "
                    "the pooled read-out (pool: model_cl.PoolCL, trained on the final BCE alone)")
    ap.add_argument("--device", default="")
    ap.add_argument("--max-minutes", type=float, default=0)
    return ap


def tables(d, layers, prep, meas, p, noise="uniform", products=None, pauli="frame"):
    """Everything one circuit contributes to a batch (numpy), plus its extended model; products: the reliable-product
    basis to use (spec_cl.build); pauli="gate": the circuit's Paulis as noisy physical gates."""
    from duet.net import model_cl as M
    s = spec_cl.build(d, layers, prep, meas, p, noise, products, pauli=pauli)
    ext, _ = LR.extended_dem(s)
    tab = dict(p=p, m=s["m"], T=s["T"], K=s["K"], S=s["slots"], n_obs=s["n_obs"], flat=s["flat"].astype(np.int64),
               mx=s["mx"], mz=s["mz"], gate=s["gate"], pcode=s["pcode"], pblk=s["pblk"], bound=s["bound"],
               offb=M.offset_bins(d, s["lines"], s["orient"], s["mx"], s["mz"]),
               cnt=M.part_counts(s["parts"]), A=s["A"], layers=" ".join(layers), prep=s["prep"], meas=s["meas"],
               n_total=LR.n_total(s))
    if s["T"] == 2:               # the single-layer residual's routing
        tab["nbr"], tab["rel"], _ = spec_cl.routes(s, ext)
    return tab, ext


def load_pool(paths, d):
    """Replay pools -> dict(ci, dets, obs, cfgs, tabs): every file's
    shots with ci the index of their configuration in cfgs [(layers, prep, meas, p, noise)], dets their packed
    detector samples and obs the extended model's bits as _chunk samples them (each padded to the widest file), tabs
    each configuration's tables."""
    cfgs, ci, dets, obs = {}, [], [], []
    for f in paths:
        z = np.load(f)
        loc = []
        for d_, lay, prep, meas, p, noise in json.loads(str(z["cfg"])):
            assert d_ == d, (f, d_, d)
            loc.append(cfgs.setdefault((tuple(lay), prep, meas, float(p), noise), len(cfgs)))
        ci.append(np.asarray(loc, np.int64)[z["ci"]])
        dets.append(z["dets"])
        obs.append(z["obs"])
    pad = lambda xs: np.concatenate([np.pad(x, ((0, 0), (0, max(y.shape[1] for y in xs) - x.shape[1]))) for x in xs])  # noqa: E731
    cl = sorted(cfgs, key=cfgs.get)
    return dict(ci=np.concatenate(ci), dets=pad(dets), obs=pad(obs), cfgs=cl,
                tabs=[tables(d, list(lay), tuple(prep), tuple(meas), p, noise)[0] for lay, prep, meas, p, noise in cl])


def draw(rng, L, p_cnot, gates, cnot_share=0.0, p_set=(), K=2, ent_share=0.0, ent_k2=0.0):
    """One training circuit (layers, prep, meas); the generator calls are those of the prototype when cnot_share = 0
    and p_set is empty. K > 2: the K-block family: with probability cnot_share the
    cnot family (K // 2 CNOT pairs per layer, all Z), with probability ent_share the entangled-then-idle family
    (duet.kblock_sets.ent; one uniform draw decides both, so ent_share = 0 keeps the earlier stream), otherwise
    duet.clifford_circuits.sample_circuit_k. ent_k2 > 0: two-block chunks the same way, one uniform draw
    picking the cnot family below cnot_share, the entangled-then-idle family (Bell pairs) below cnot_share + ent_k2, the
    mix otherwise; ent_k2 = 0 keeps the two-block stream."""
    if K == 2 and ent_k2:
        u = rng.random()
        if u < cnot_share:
            return cc.cnot_layers(rng.integers(0, 2, size=L).tolist()), ("Z", "Z"), ("Z", "Z")
        if u < cnot_share + ent_k2:
            return kblock_sets.ent(rng, 2, L)
        q = float(p_set[rng.integers(len(p_set))]) if len(p_set) else p_cnot
        return cc.sample_circuit(rng, L, q, gates)
    if K != 2:
        u = rng.random() if cnot_share or ent_share else 1.0
        if u < cnot_share:
            return cc.sample_circuit_k(rng, K, L, 1.0, "I", prep=("Z",) * K, meas=("Z",) * K)
        if u < cnot_share + ent_share:
            return kblock_sets.ent(rng, K, L)
        q = float(p_set[rng.integers(len(p_set))]) if len(p_set) else p_cnot
        return cc.sample_circuit_k(rng, K, L, q, gates)
    if cnot_share and rng.random() < cnot_share:
        # today's family: CNOT layers of random direction, |00>, Z readout
        return cc.cnot_layers(rng.integers(0, 2, size=L).tolist()), ("Z", "Z"), ("Z", "Z")
    q = float(p_set[rng.integers(len(p_set))]) if len(p_set) else p_cnot
    return cc.sample_circuit(rng, L, q, gates)


_FAMS = {}


def _family(t, prep):
    """Stratum of a single-layer circuit."""
    if t in ("C0", "C1"):
        ctrl, targ = (0, 1) if t == "C0" else (1, 0)
        return "cnot_bell" if prep[ctrl] == "X" and prep[targ] == "Z" else "cnot_other"
    if "S" in t:
        return "gate_HS" if "H" in t else "gate_S"
    return "gate_H" if "H" in t else "idle"


def single_layer_families():
    """Every two-block single-layer circuit of the generator's gates (a CNOT either way or a gate in I, H, S per block;
    bases Z, X per block) with a reliable product, by family: {family: [(layers, prep, meas)]}."""
    if not _FAMS:
        for t in ["C0", "C1"] + [x + y for x in "IHS" for y in "IHS"]:
            for pr in itertools.product("ZX", repeat=2):
                for me in itertools.product("ZX", repeat=2):
                    if cc.logical_products([t], pr, me):
                        _FAMS.setdefault(_family(t, pr), []).append(([t], pr, me))
    return _FAMS


def _draw_chunk(job):
    """Chunk k's circuit and noise strength (layers, prep, meas, p): l1, one of the six single-layer families with
    equal share (or the weights fam_w over the sorted families) and a circuit uniformly within it; p_l1 = (ps, weights,
    ramp), p drawn after the circuit and moved from the job's p by the fraction ramp."""
    d, p, L, seed, k, n, p_cnot, gates, cnot_share, p_set, noise, K, ent_share, ent_k2, l1, p_l1, fam_w, *_ = job
    if l1 == "lukin":            #: the benchmark's builder on random_logical's circuit k of the run's seed
        from duet import bench_sets, lukin_bench
        assert seed != lukin_bench.SEED, "the benchmark's own circuits"
        lay, prep, meas, prods = bench_sets.lukin(L // 2, k, seed=seed)
        return lay, tuple(prep), tuple(meas), p, [list(x) for x in prods], "gate"
    rng = np.random.default_rng([seed, k])
    if l1:
        fams = single_layer_families()
        f = sorted(fams)[rng.choice(len(fams), p=fam_w) if fam_w else rng.integers(len(fams))]
        lay, prep, meas = fams[f][rng.integers(len(fams[f]))]
    else:
        lay, prep, meas = draw(rng, L, p_cnot, gates, cnot_share, p_set, K, ent_share, ent_k2)
    if p_l1:
        p = p + (p_l1[0][rng.choice(len(p_l1[0]), p=p_l1[1])] - p) * p_l1[2]
    return lay, prep, meas, p


TEACH_FAMS = ("gate_H", "gate_HS", "gate_S", "idle")     # the one-block families the teacher covers (192 (2))
_LOM = {}


def lom_decoder(d, layers, prep, meas, p, noise="uniform"):
    """(stim circuit, LoMatching) of one circuit, the lomr decoder of duet.bench_decode, built once per
    configuration in each process."""
    key = (d, tuple(layers), tuple(prep), tuple(meas), float(p), noise)
    if key not in _LOM:
        from duet import bench_decode as bd
        _LOM[key] = bd.build("lomr", dict(d=d, p=p, noise=noise),
                             dict(layers=" ".join(layers), prep="".join(prep), meas="".join(meas)))
    return _LOM[key]


def lom_decide(d, layers, prep, meas, p, noise, dets, m):
    """LoMatching's per-observable decisions (n, n_obs) uint8 on packed detector samples (n, bytes)."""
    c, dec = lom_decoder(d, layers, prep, meas, p, noise)
    assert c.num_detectors == m, (layers, prep, meas)
    return np.asarray(dec.decode_batch(unpack(dets, m).astype(bool)), np.uint8).reshape(len(dets), -1)


def _chunk(job):
    d, seed, k, n, noise = job[0], job[3], job[4], job[5], job[10]
    lay, prep, meas, p, *extra = _draw_chunk(job)            # extra: products and pauli of a lukin chunk
    tab, ext = tables(d, lay, prep, meas, p, noise, *extra)
    dets, obs, _ = ext.compile_sampler(seed=(seed * 1_000_003 + k) % 2 ** 63).sample(n, bit_packed=True)
    if len(job) > 17 and job[17] and job[14] is True and _family(lay[0], prep) in TEACH_FAMS:
        tab["lom"] = lom_decide(d, lay, prep, meas, p, noise, dets, tab["m"])
    return k, tab, dets, obs


def depths(spec):
    """--Ls: comma list of depths; an entry L:n stands for n entries of L."""
    out = []
    for v in spec.split(","):
        L, _, n = v.partition(":")
        out += [int(L)] * int(n or 1)
    return out


def p_weights(spec):
    """--p-l1 P:W... -> ((p...), (weight...)) with the weights normalised;  when empty."""
    if not spec:
        return ()
    pw = [v.split(":") for v in spec.split(",")]
    w = np.array([float(y) for _, y in pw])
    return tuple(float(x) for x, _ in pw), tuple(w / w.sum())


def fam_weights(spec):
    """--l1-fam F:W... -> the weights of the sorted single-layer families, normalised;  when empty."""
    if not spec:
        return ()
    fw = {f: float(w) for f, w in (v.split(":") for v in spec.split(","))}
    fams = sorted(single_layer_families())
    assert sorted(fw) == fams and min(fw.values()) >= 0, f"--l1-fam names each of {fams}"
    return tuple(fw[f] / sum(fw.values()) for f in fams)


def chunk_job(a, k, kind, p_set, p_l1):
    """Chunk k's job for _chunk; kind = batch_plan's (K, L, l1, last) of its batch; p_l1 = p_weights(--p-l1), to which
    the ramp fraction of the chunk's first sample is added."""
    K, L, one, last = kind
    ramp = min(1.0, k * a.chunk / a.p_ramp) if a.p_ramp else 1.0
    p, tri = a.p, getattr(a, "p_tri", 0.0)
    if tri and k * a.chunk >= (1 - tri) * a.samples:     # MH: p -> 2p -> p over the last fraction tri
        u = (k * a.chunk - (1 - tri) * a.samples) / (tri * a.samples)
        p = a.p * (2 - abs(2 * u - 1))
    return (a.d, p, L, a.seed, k, a.chunk, a.p_cnot, a.gates, a.cnot_share, p_set, a.noise, K, a.ent_share, a.ent_k2,
            one, p_l1 + (ramp,) if p_l1 and one is True and not last else (), fam_weights(getattr(a, "l1_fam", "")),
            getattr(a, "teacher_lom", 0.0) > 0)


def batch_plan(a, steps):
    """Batch i -> (K, L, l1, last): l1 a two-block single-layer batch of the balanced families (the first --l1-first
    samples, then the share --l1-share spread evenly), or "lukin" a batch of the lukin family (the share --lukin-share spread
    evenly, K = 4, L = 2D with D the next entry of --lukin-depths), else the next entry of the (K, L) list; last: batch i
    lies in the last --p-last samples."""
    Ks = [int(v) for v in a.block_counts.split(",")]
    kl = [(K, L) for K in Ks for L in depths(a.Ls)]
    n1, s = int(math.ceil(a.l1_first / a.batch)), a.l1_share
    if n1 or s:
        kl = [kl[j] for j in np.random.default_rng(0).permutation(len(kl))]
    n_last = steps - int(math.ceil(a.p_last / a.batch)) if a.p_last else steps
    s_lukin = getattr(a, "lukin_share", 0.0)
    lukin_D = depths(getattr(a, "lukin_depths", "1")) if s_lukin else []
    assert not (s_lukin and (n1 or s)), "--lukin-share with single-layer batches"

    def plan(i):
        last = i >= n_last
        if s_lukin:              #: the share s_lukin of the batches, spread evenly, from the lukin family
            if math.floor((i + 1) * s_lukin) > math.floor(i * s_lukin):
                return 4, 2 * lukin_D[math.floor(i * s_lukin) % len(lukin_D)], "lukin", last
            K, L = kl[(i - math.floor(i * s_lukin)) % len(kl)]
            return K, L, False, last
        if i < n1:
            return 2, 1, True, last
        j = i - n1
        if math.floor((j + 1) * s) > math.floor(j * s):
            return 2, 1, True, last
        K, L = kl[(j - math.floor(j * s)) % len(kl)]
        return K, L, False, last
    return plan


MON_FAMS = ("mix", "cnot")


def _mon(job):
    d, p, fam, L, i, n, p_cnot, gates, p_set, noise, K = job
    f = MON_FAMS.index(fam)
    if K == 2:
        rng = np.random.default_rng([MON_BASE % 2 ** 32, f, L, i])
        seed = MON_BASE + f * 10 ** 12 + L * 10 ** 9 + i
    else:                                 # K blocks: range
        rng = np.random.default_rng([KBLOCK_MON_BASE % 2 ** 32, K, f, L, i])
        seed = KBLOCK_MON_BASE + K * 10 ** 13 + f * 10 ** 12 + L * 10 ** 9 + i
    lay, prep, meas = draw(rng, L, p_cnot, gates, 1.0 if fam == "cnot" else 0.0, p_set, K)
    tab, ext = tables(d, lay, prep, meas, p, noise)
    dets, obs, _ = ext.compile_sampler(seed=seed).sample(n, bit_packed=True)
    return (fam if K == 2 else f"{fam}{K}"), L, tab, dets, obs


def unpack(a, n):
    return np.unpackbits(a, axis=1, bitorder="little")[:, :n]


def collate(items, d, dev, train=True):
    """items: [(tab, dets packed, obs packed)] of one depth and block count -> tensors on dev; product slots padded to
    the most of the items (empty slots: outside bins, no labels, slot 0)."""
    T = items[0][0]["T"]
    S = max(tab.get("S", 2) for tab, _, _ in items)
    from duet.net.model_cl import P_REF
    cols = {k: [] for k in ("bits", "gidx", "mx", "mz", "gate", "pcode", "offb", "cnt", "g", "y", "yl", "u", "slot",
                            "pblk", "bound", "lp")}
    if all("nbr" in tab for tab, _, _ in items):
        cols.update(nbr=[], rel=[])
    if any("lom" in tab for tab, _, _ in items):         # the teacher's targets and mask
        cols.update(t=[], tm=[])
    for tab, dp, op in items:
        n, m, n_obs, S_t = len(dp), tab["m"], tab["n_obs"], tab.get("S", 2)
        bits = unpack(dp, m)
        o = unpack(op, tab["n_total"])
        y = np.zeros((n, S), np.uint8)
        yl = np.zeros((n, S, d), np.uint8)
        u = np.zeros((n, S, d, T), np.uint8)
        y[:, :n_obs] = o[:, :n_obs]
        yl[:, :n_obs] = o[:, n_obs:n_obs * (d + 1)].reshape(n, n_obs, d)
        u[:, :n_obs] = o[:, n_obs * (d + 1):].reshape(n, T, n_obs, d).transpose(0, 2, 3, 1)
        g = np.zeros((n, S, d), np.uint8)
        g[:, :S_t] = ((bits.astype(np.int64) @ tab["A"].astype(np.int64)) % 2).reshape(n, S_t, d)
        slot = np.zeros((n, S), np.uint8)
        slot[:, :n_obs] = 1
        offb, cnt = tab["offb"], tab["cnt"]
        if S_t < S:
            offb = np.concatenate([offb, np.full((S - S_t,) + offb.shape[1:], 4 * d - 1, offb.dtype)])
            cnt = np.concatenate([cnt, np.zeros((S - S_t,) + cnt.shape[1:], cnt.dtype)])
        rep = lambda a: np.broadcast_to(a, (n,) + a.shape)   # noqa: E731
        for k, v in (("bits", bits), ("gidx", rep(tab["flat"])), ("mx", rep(tab["mx"])), ("mz", rep(tab["mz"])),
                     ("gate", rep(tab["gate"])), ("pcode", rep(tab["pcode"])), ("offb", rep(offb)),
                     ("cnt", rep(cnt)), ("g", g), ("y", y), ("yl", yl), ("u", u), ("slot", slot),
                     ("pblk", rep(tab["pblk"])), ("bound", rep(tab["bound"])),
                     ("lp", np.full(n, math.log(tab.get("p", P_REF) / P_REF), np.float32))):
            cols[k].append(v)
        if "nbr" in cols:
            cols["nbr"].append(rep(tab["nbr"]))
            cols["rel"].append(rep(tab["rel"]))
        if "t" in cols:
            t, tm = np.zeros((n, S), np.uint8), np.zeros((n, S), np.uint8)
            if "lom" in tab:
                t[:, :n_obs], tm[:, :n_obs] = tab["lom"][:n], 1
            cols["t"].append(t)
            cols["tm"].append(tm)
    out = {k: torch.from_numpy(np.ascontiguousarray(np.concatenate(v))) for k, v in cols.items()}
    out = {k: (v.pin_memory() if dev.type == "cuda" else v).to(dev, non_blocking=True) for k, v in out.items()}
    out["gidx"] = out["gidx"].long()
    if "nbr" in out:
        out["nbr"] = out["nbr"].long()
    return out


def forward(net, b, train=True):
    g = (b["yl"] ^ b["y"][:, :, None]) if train else b["g"]
    return net(b["bits"], b["gidx"], b["mx"], b["mz"], b["gate"], b["pcode"], b["offb"], b["cnt"], g, b.get("pblk"),
               b.get("bound"), lp=b.get("lp"), nbr=b.get("nbr"), rel=b.get("rel"))


def line_map(d, d_s):
    """The source line of each line of a distance-d network taken from a distance-d_s one: the line at the same distance
    from the nearer edge; the centre line and lines farther from both edges than the source has lines: its centre line."""
    out = []
    for j in range(d):
        e = min(j, d - 1 - j)
        out.append(d_s // 2 if 2 * j == d - 1 or e >= d_s // 2 else e if 2 * j < d - 1 else d_s - 1 - e)
    return out




def main(argv=None):
    from duet.net import model_cl as M
    a = parser().parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if os.path.exists(os.path.join(a.out, "run.json")):
        print(json.dumps(dict(done=a.out)), flush=True)
        return 0
    term = []
    signal.signal(signal.SIGTERM, lambda *_: term.append(time.time()))   # preemption: checkpoint, exit PREEMPTED
    Ls = depths(a.Ls)
    p_set = tuple(float(v) for v in a.p_cnot_set.split(",")) if a.p_cnot_set else ()
    p_l1 = p_weights(a.p_l1)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = dev.type == "cuda"
    torch.manual_seed(a.seed)
    if a.l1path:                  #: the residual and its options belong to the single-layer path
        net = M.NetCL(a.d, a.width, a.bott, a.blocks, a.heads, l1path={k: v for k, v in dict(
            res=a.res, route=a.route, prod=a.prod, troute=a.troute, flags=a.flags, pin=a.pin,
            blocks=a.l1_blocks).items() if v}).to(dev).train()
    else:
        assert not (a.troute or a.flags), "--troute and --flags are options of the single-layer path (--l1path 1)"
        net = M.NetCL(a.d, a.width, a.bott, a.blocks, a.heads, joint=a.joint, pin=a.pin, res=a.res, route=a.route,
                      prod=a.prod, couple=a.couple, head=a.head).to(dev).train()
    if a.freeze_trunk:
        for k, q in net.named_parameters():
            q.requires_grad_(k.split(".")[0] in ("head", "w", "b", "joint"))
    if a.res or a.l1path:         # the single-layer residual or path (179 (3)) alone
        for k, q in net.named_parameters():
            q.requires_grad_(k.split(".")[0] == ("l1path" if a.l1path else "res"))
    net.head.lookup = a.lookup
    if a.l1path:
        net.l1path.head.lookup = a.lookup
    Ks = [int(v) for v in a.block_counts.split(",")]
    kl_order = [(K, L) for K in Ks for L in Ls]        # batch i: block count and depth kl_order[i mod |kl_order|]
    ema = copy.deepcopy(net).eval()
    for q in ema.parameters():
        q.requires_grad_(False)
    steps = int(a.samples // a.batch)
    plan = batch_plan(a, steps)                         # with no single-layer batches: kl_order in turn
    l1 = bool(a.l1_first or a.l1_share)
    save_at = {int(math.ceil(float(v) / a.batch)): f"ema_{float(v) / 1e6:g}M.pt" for v in a.save_at.split(",") if v}
    per_batch = a.batch // a.chunk
    # --accum divides the chunks per batch, or is a multiple of them: each chunk then splits into
    # accum / per_batch equal shot slices, one micro-batch each (the same samples and gradient, less memory)
    assert per_batch % a.accum == 0 or (a.accum % per_batch == 0 and a.chunk % (a.accum // per_batch) == 0 and
                                        not a.replay and not a.teacher_lom), "--accum and the chunks per batch"
    rpool, n_rep = None, 0
    if a.replay:                  #, arm R: a share of every micro-batch from the pools
        assert a.l1_first >= a.samples, "--replay: single-layer batches throughout"
        rpool = load_pool(a.replay.split(","), a.d)
        if a.replay_views:        #: each pool shot about replay_views times
            a.replay_share = min(a.replay_share, a.replay_views * len(rpool["ci"]) / a.samples)
        n_rep = int(round(per_batch // a.accum * a.chunk * a.replay_share))
        assert 0 < n_rep < a.chunk and len(rpool["ci"]) >= n_rep, (n_rep, len(rpool["ci"]))
        print(json.dumps(dict(replay=a.replay, pool=len(rpool["ci"]), configs=len(rpool["cfgs"]), share=a.replay_share,
                         per_microbatch=n_rep)),
              flush=True)

    def replay(mb, step, j):
        """Micro-batch j of step: its last chunk's last n_rep shots give way to n_rep shots drawn without replacement
        from the pools; the draw depends on the seed, step and j only, so a resumed run draws the same."""
        tab, dp, op = mb[-1]
        n = len(dp) - n_rep
        out = mb[:-1] + [(dict(tab, lom=tab["lom"][:n]) if "lom" in tab else tab, dp[:n], op[:n])]
        idx = np.sort(np.random.default_rng([a.seed, 192, step, j]).choice(len(rpool["ci"]), n_rep, replace=False))
        for c in np.unique(rpool["ci"][idx]):
            w = idx[rpool["ci"][idx] == c]
            out.append((rpool["tabs"][c], rpool["dets"][w], rpool["obs"][w]))
        assert sum(len(x[1]) for x in out) == sum(len(x[1]) for x in mb)
        return out
    opt = torch.optim.AdamW([q for q in net.parameters() if q.requires_grad], lr=a.lr, betas=(0.9, 0.99),
                            weight_decay=a.wd)
    warm = min(a.warmup, max(1, steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              a.lr_floor + (1 - a.lr_floor) * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    n_loss = 4 if a.teacher_lom else 3            # train_loss: total, final + representative, per-layer[, teacher]
    step, acc, n_acc = 0, torch.zeros(n_loss, device=dev), 0
    ck_path = os.path.join(a.out, "ckpt.pt")
    if os.path.exists(ck_path):                                   # exact resume
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        assert format_ok(ck["format"]), f"a {ck['format']} run: resume it with the code it started with"
        ck["args"].setdefault("noise", "uniform")                 # checkpoints from before the switch
        ck["args"].setdefault("block_counts", "2")                # and from before K blocks
        ck["args"].setdefault("lookup", "onehot")
        ck["args"].setdefault("ent_share", 0.0)
        ck["args"].setdefault("ent_k2", 0.0)
        ck["args"].setdefault("joint", 0)
        ck["args"].setdefault("freeze_trunk", False)
        for k_, v_ in (("pin", 0), ("l1_first", 0.0), ("l1_share", 0.0), ("l1_fam", ""), ("p_l1", ""), ("p_last", 0.0), ("p_ramp", 0.0),
                       ("res", 0), ("route", 0), ("prod", 0), ("l1path", 0), ("troute", 0), ("flags", 0),
                       ("lukin_share", 0.0), ("lukin_depths", "1,2,3,4,5,6,7,8,9,10"), ("p_tri", 0.0), ("lr_floor", 0.02),
                       ("l1_blocks", 0), ("couple", "cross"), ("head", "lines")):
            ck["args"].setdefault(k_, v_)
        diff = {k for k in set(ck["args"]) | set(vars(a)) if k not in FREE and ck["args"].get(k) != vars(a).get(k)}
        assert not diff, f"resume with other settings: {sorted(diff)}"
        net.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step, n_acc = ck["step"], ck["n_acc"]
        acc = ck["acc"].to(dev)
        # a kill between a log row and its checkpoint leaves rows past the checkpoint (recomputed now) or a cut line
        lp = os.path.join(a.out, "log.jsonl")
        if os.path.exists(lp):
            keep = []
            for line in open(lp):
                try:
                    if json.loads(line)["step"] <= step:
                        keep.append(line.rstrip("\n") + "\n")
                except (json.JSONDecodeError, KeyError):
                    pass
            open(lp + ".tmp", "w").writelines(keep)
            os.replace(lp + ".tmp", lp)
        print(json.dumps(dict(resumed=step)), flush=True)
    elif a.init:
        ck = torch.load(a.init, map_location="cpu", weights_only=False)
        assert format_ok(ck["format"]), "checkpoint of another network"
        if ck["cfg"] == net.cfg:
            net.load_state_dict(ck["model"])
            ema.load_state_dict(ck["model"])
        else:                     # more trunk blocks, a wider bottleneck or the p channel: the network
            # starts with the checkpoint's function (model_cl.load_grown)
            print(json.dumps(dict(init=a.init, added_blocks=M.load_grown(net, ck), cfg=net.cfg)), flush=True)
            ema.load_state_dict(net.state_dict())
    if a.compile:                 # one graph per (block count, depth): past dynamo's default limit of 8 the rest ran eager
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit,
                                                   2 * (len(kl_order) + (len(set(depths(a.lukin_depths))) if a.lukin_share else 0)))
    fwd = torch.compile(net, dynamic=False) if a.compile else net
    ctx = mp.get_context("fork")
    pool = ctx.Pool(a.workers, initializer=signal.signal, initargs=(signal.SIGTERM, signal.SIG_DFL))
    t_start = time.time()
    mon_Ls = dict.fromkeys(([1] if l1 else []) + Ls)
    mon_jobs = [(a.d, a.p, fam, L, i, a.eval_per, a.p_cnot, a.gates, p_set, a.noise, K) for K in dict.fromkeys(Ks)
                for fam, n in (("mix", a.eval_circuits), ("cnot", a.eval_cnot)) for L in mon_Ls for i in range(n)]
    mon = {}
    for fam, L, tab, dp, op in pool.imap(_mon, mon_jobs, chunksize=4):
        mon.setdefault((fam, L), []).append((tab, dp, op))
    # the monitor's batches stay in (pinned) host memory and go to the GPU one at a time: at d = 7 the whole set, with
    # every shot's copy of its circuit's tables, crowds training out of a 40 GB card (3.5 times slower steps)
    mon_b = {key: [collate(v[i:i + 16], a.d, torch.device("cpu"), train=False) for i in range(0, len(v), 16)]
             for key, v in mon.items()}
    if dev.type == "cuda":
        mon_b = {key: [{k: t.pin_memory() for k, t in b.items()} for b in bs] for key, bs in mon_b.items()}
    eval_shots = {f"{fam}_L{L}": sum(len(dp) for _, dp, _ in v) for (fam, L), v in mon.items()}
    print(json.dumps(dict(monitor_shots=eval_shots, minutes=(time.time() - t_start) / 60)), flush=True)

    def jobs(k):
        while True:
            # chunk k feeds batch k // per_batch, whose block count and depth are kl_order[(k // per_batch) mod its
            # length] (two blocks only: the depth Ls[(k // per_batch) mod |Ls|], as before K blocks), or plan's
            yield chunk_job(a, k, plan(k // per_batch), p_set, p_l1)
            k += 1

    # bounded prefetch: pool.imap would run the workers flat out and pile results up in this process
    job_it, pending = jobs(step * per_batch), collections.deque()
    for _ in range(a.prefetch):
        pending.append(pool.apply_async(_chunk, (next(job_it),)))

    def next_chunk():
        while True:
            try:
                r = pending[0].get(timeout=5)
                break
            except mp.TimeoutError:
                if term:
                    return None
        pending.popleft()
        pending.append(pool.apply_async(_chunk, (next(job_it),)))
        return r

    @torch.no_grad()
    def failures(model):
        model.eval()
        per = {}
        for (fam, L), bs in mon_b.items():
            f = 0
            for b in bs:
                b = {k: t.to(dev, non_blocking=True) for k, t in b.items()}
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                    c, _, _ = forward(model, b, train=False)
                f += int((((c > 0).to(b["y"].dtype) != b["y"]) & (b["slot"] > 0)).any(1).sum())
            per.setdefault(fam, {})[str(L)] = f
        return {fam: sum(v.values()) for fam, v in per.items()}, per

    logf = open(os.path.join(a.out, "log.jsonl"), "a")
    eval_steps = max(1, int(a.eval_every // a.batch))
    t0, wait, t_ck, s0, s_start = time.time(), 0.0, time.time(), step, step
    pe, pw = list(ema.parameters()), list(net.parameters())

    def record(final=False):
        dt, te = time.time() - t0, time.time()             # the rate covers training only; eval_s is the monitor's time
        fe, fe_L = failures(ema)
        fc, fc_L = failures(net)
        net.train()
        row = dict(step=step, samples=step * a.batch, lr=sched.get_last_lr()[0],
                   train_loss=(acc / max(1, n_acc)).tolist(), ema_failures=fe, failures=fc, ema_per_depth=fe_L,
                   per_depth=fc_L, eval_shots=eval_shots, minutes=(time.time() - t_start) / 60,
                   samples_per_s=(step - s0) * a.batch / max(dt, 1e-9) if not final else None,
                   sampler_wait=wait / max(dt, 1e-9), eval_s=round(time.time() - te, 1), final=final,
                   gpu_peak_gb=round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 2) if dev.type == "cuda" else None)
        logf.write(json.dumps(row) + "\n")
        logf.flush()
        print(json.dumps(row), flush=True)

    def save(path, model, extra=None):
        torch.save(dict(format=FORMAT, model=model.state_dict(), cfg=model.cfg, args=vars(a), step=step, **(extra or {})),
                   path + ".tmp")
        os.replace(path + ".tmp", path)

    def checkpoint():
        torch.save(dict(format=FORMAT, args=vars(a), step=step, model=net.state_dict(), ema=ema.state_dict(),
                        opt=opt.state_dict(), sched=sched.state_dict(), acc=acc.cpu(), n_acc=n_acc), ck_path + ".tmp")
        os.replace(ck_path + ".tmp", ck_path)

    def preempted():
        checkpoint()
        print(json.dumps(dict(preempted=step, minutes=(time.time() - t_start) / 60)), flush=True)
        pool.terminate()
        return PREEMPTED

    if a.init and step == 0 and not os.path.exists(ck_path):
        record()                                       # the loaded weights' row (a restart resumes instead)
        checkpoint()
    stopped = False
    while step < steps:
        if term:
            return preempted()
        tw = time.time()
        items = []
        while len(items) < per_batch:
            r = next_chunk()
            if r is None:
                return preempted()
            items.append(r[1:])
        wait += time.time() - tw
        assert len({(tab["T"], tab["K"]) for tab, _, _ in items}) == 1
        opt.zero_grad(set_to_none=True)
        if a.accum <= per_batch:
            part = per_batch // a.accum
            mbs = [items[j * part:(j + 1) * part] for j in range(a.accum)]
        else:                                   #: micro-batches of one chunk's shot slice
            h = a.chunk // (a.accum // per_batch)
            mbs = [[(tab, dp[i:i + h], op[i:i + h])] for tab, dp, op in items for i in range(0, a.chunk, h)]
        tot = torch.zeros(n_loss, device=dev)
        for j, mb in enumerate(mbs):
            b = collate(replay(mb, step, j) if rpool else mb, a.d, dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                c, z, l = forward(fwd, b)
                loss, lf, ll = M.loss_cl(c, z, l, b["y"], b["yl"], b["u"], b["slot"], a.lam)
                if a.teacher_lom:
                    lt = M.teacher_bce(c, b["t"], b["tm"]) if "t" in b else torch.zeros((), device=dev)
                    loss = loss + a.teacher_lom * lt
            (loss / a.accum).backward()
            tot += torch.stack([loss.detach(), lf.detach(), ll.detach()] +
                               ([lt.detach()] if a.teacher_lom else [])).float() / a.accum
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            torch._foreach_lerp_(pe, pw, 1.0 - a.ema)
        step += 1
        acc += tot
        n_acc += 1
        if step in save_at:
            save(os.path.join(a.out, save_at[step]), ema)
        if step % eval_steps == 0 or step == steps:
            record(final=step == steps)
            acc, n_acc, t0, wait, s0 = torch.zeros(n_loss, device=dev), 0, time.time(), 0.0, step
            save(os.path.join(a.out, "last_ema.pt"), ema)
            checkpoint()
            t_ck = time.time()
        elif time.time() - t_ck >= 60 * a.ckpt_minutes:
            checkpoint()
            t_ck = time.time()
            #: a progress line per checkpoint (the rate since the last monitor row, training only)
            print(json.dumps(dict(progress=step, samples=step * a.batch, minutes=round((t_ck - t_start) / 60, 2),
                                  samples_per_s=round((step - s0) * a.batch / max(t_ck - t0, 1e-9), 1),
                                  sampler_wait=round(wait / max(t_ck - t0, 1e-9), 4),
                                  gpu_peak_gb=round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 2)
                                  if dev.type == "cuda" else None)), flush=True)
        if a.max_minutes and (time.time() - t_start) / 60 >= a.max_minutes or a.max_steps and step - s_start >= a.max_steps:
            stopped = True
            break
    if stopped:                                     # resumable later: keep ckpt.pt, write no run.json
        checkpoint()
        pool.terminate()
        return 3
    save(os.path.join(a.out, "final_ema.pt"), ema, dict(stopped=False))
    json.dump(dict(args=vars(a), steps=steps, step=step, samples=step * a.batch, stopped=False,
                   minutes=(time.time() - t_start) / 60, format=FORMAT), open(os.path.join(a.out, "run.json"), "w"),
              indent=1)
    pool.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
