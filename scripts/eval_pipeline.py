"""Benchmark I's circuit family, its development banks and test sets, the opponents (LoMatching, the Cain et al.
reimplementation, BP+LoMatching), the network reads, the routing groundwork and the Tesseract trials, mostly on CPU.

The family. One generator for K = 2, 3, 4:
duet.clifford_circuits.sample_circuit_k(rng, K, L, q, "IHS"), the K-block mix law of that train_cl.draw uses for the mix share of every K > 2 chunk of the start networks ft4_lukin_d* and of the
main networks trained from them; for K = 2 its law is sample_circuit's (a CNOT layer with probability q in a
uniform direction, otherwise one gate per block), with other generator calls. Per layer the blocks are shuffled and
taken in consecutive pairs; each pair is a CNOT with probability q (uniform direction); every block outside a CNOT gets
I, H or S (uniform); so K = 3 always leaves one block out of the pairs, and K = 4 may hold one, two or no pair. q is the
circuit's CNOT share, q = P_SET[i mod 4] = 0, 1/3, 2/3, 1 for circuit i (16 circuits of each per point; training draws q
uniformly from the same set). Preparation and readout uniform in Z, X per block; circuits without a reliable product
are drawn again (as training). Paulis are software, and S_DAG = S Z is a physical S with a software Z, so the
decoding problem of {I, H, S, S_DAG} with software Paulis is that of the trained gates {I, H, S}; no Pauli letter is
drawn. One syndrome round per layer, the Lukin noise model, p = 0.3 % (L = 8 also at 0.1 % and 0.2 %). A decoder fails a
shot when any reliable product is decoded wrong.

Points: K in (2, 3, 4) x d in (3, 5, 7, 9) x (L, p) in POINTS, 64 circuits each, SHOTS[(L, p)] shots per circuit. The 64 logical circuits of a (purpose, K, L) are the same at every d and
p (numpy default_rng([FIRST_TAG, purpose, K, L])); development and test circuits differ (purpose 1 and 2).

Seeds: range [FIRST_SEED_BASE, FIRST_SEED_BASE + 10^14) = [1.8446e19, 1.84461e19), above [1.8445e19,
1.8446e19) and every earlier range (scripts/dev_bank.py EARLIER), below 2^64 and above 2^63 (disjoint from training); a git
samples all its shots with stim seed FIRST_SEED_BASE + g 10^13 + d 10^11 + K 10^10 + L 10^8 + pcode 10^6 + i, g = 1
development, 2 test, 3 the timing probe; pcode 1, 2, 3 for p = 0.1, 0.2, 0.3 %.

Main result 2's fresh test set:
new noise samples on the open benchmark's fixed lukin-family circuits, d = 3, 5, 7, D = 1-10, the manifest's shots per circuit and
exact-MLE prefix, point names unchanged; circuit i of depth D at distance d
samples with stim seed FIRST_SEED_BASE + 4 10^13 + d 10^11 + D 10^8 + i (g = 4; disjoint from g = 1-3). scripts/dev_bank.py
test.py route / rep, dchk.py cover) run on the set unchanged.

  python scripts/eval_pipeline.py manifest --purpose dev|test|bench2 --out F the points (duet.bench_sets' manifest format)
  python scripts/eval_pipeline.py gen --manifest F --dir SETS [--only RE] sample and store every point (bench_sets._point;
                                                                     SETS/<point>.npz, SETS/sha256.json)
  python scripts/eval_pipeline.py dec --manifest F --dir SETS --out DEC --decoder lomr|cain|bplomr [--only RE] [--part j/P]
        decode stored shots (duet.bench_decode's builders, ignore_decomposition_failures as the open benchmark):
        DEC/<decoder>/<point>/c<i>_<first>.npz (shot indices, predicted flips as bits of a byte, CPU seconds);
        lomr and cain every shot, bplomr the first bplomr_shots (1,000) of each circuit
  python scripts/eval_pipeline.py net --manifest F --dir SETS --out NET --name N --ckpt CK [--frac F] [--device cpu|cuda]
        a network's per-shot logits (frozen forward: train_cl.tables, collate, forward; fp32, TF32 off) of the first
        ceil(frac x shots) shots of each circuit: NET/<N>/<point>.npz (circuits, shots, 4) float32, NaN past the products
  python scripts/eval_pipeline.py sel --manifest F --net NET --name N --frac 0.25 --out LISTS
        the least confident shots (score: the smallest |logit| over the circuit's products) per point (d, K, L, p) and
        per depth (d, L, p; K pooled), the union of both, as {point: {circuit: [shot...]}}
  python scripts/eval_pipeline.py tess --manifest F --dir SETS --lists LISTS --out TESS [--deadline none|L<c>|S] [--part j/P]
        the long beam's 21 trials of every listed shot, each timed alone
        (perf_counter_ns around decode_to_errors, every core busy): seconds, flag, dropped (deadline), cost, flips;
  python scripts/eval_pipeline.py probe --d D --out F [--workers W] [--ckpt CK] decoder and network costs per (K, L): timing
  python scripts/eval_pipeline.py labcheck --ds 3,5 --workers W labels, detectors, the law and hand-checked
                                                                         circuits (item 1's tests)
  python scripts/eval_pipeline.py score --manifest F --dir SETS --dec DEC --out J the opponents' failures per point
"""
import argparse
import glob
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import re
import sys
import time
import zlib

import numpy as np

from duet import bench_sets as bs
from duet import clifford_circuits as cc
from duet import kblock_sets as ms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIRST_SEED_BASE, FIRST_SEED_SPAN = 18_446_000_000_000_000_000, 10 ** 14
FIRST_TAG = 20261110
PURPOSE = {"dev": 1, "test": 2, "probe": 3, "bench2": 4}
PCODE = {0.001: 1, 0.002: 2, 0.003: 3}
GATES = "IHS"
P_SET = ms.P_SET                                   # (0, 1/3, 2/3, 1): circuit i has q = P_SET[i % 4]
NCIRC = 64
DS, KS = (3, 5, 7, 9), (2, 3, 4)
POINTS = ((1, 0.003), (2, 0.003), (4, 0.003), (8, 0.003), (12, 0.003), (16, 0.003), (8, 0.001), (8, 0.002))
BPLOMR = 1000                                      # BP+LoMatching: the first 1,000 shots of each circuit
K_TRIALS = 21
CHUNK = {"lomr": 20000, "cain": 20000, "bplomr": 250}
def tess_chunk(d, L):
    """Shots per trial part: about half a minute of trials, so the decoder's build (up to 3 s) stays small."""
    return int(min(4000, max(20, 1000 / L ** 1.25) * {3: 20, 5: 5, 7: 2, 9: 1}.get(d, 1)))
NET_BATCH = {3: 32, 5: 8, 7: 8, 9: 8}               # the CPU read's batch: small batches stay in cache (8 is 2.2 x faster per shot than 128 at d = 5)


def seed(purpose, d, K, L, p, i):
    g = PURPOSE[purpose]
    assert 0 < d < 100 and 0 < K < 10 and 0 < L < 100 and 0 <= i < 10 ** 6, (d, K, L, i)
    s = FIRST_SEED_BASE + g * 10 ** 13 + d * 10 ** 11 + K * 10 ** 10 + L * 10 ** 8 + PCODE[p] * 10 ** 6 + i
    assert FIRST_SEED_BASE <= s < FIRST_SEED_BASE + FIRST_SEED_SPAN < 2 ** 64
    return s


def seed_bench2(d, D, i):
    """Main result 2's fresh test set: circuit i (0-19) of depth unit count D at distance d."""
    assert d in (3, 5, 7) and 0 < D <= 10 and 0 <= i < 20, (d, D, i)
    s = FIRST_SEED_BASE + PURPOSE["bench2"] * 10 ** 13 + d * 10 ** 11 + D * 10 ** 8 + i
    assert FIRST_SEED_BASE + 4 * 10 ** 13 <= s < FIRST_SEED_BASE + 5 * 10 ** 13 <= FIRST_SEED_BASE + FIRST_SEED_SPAN < 2 ** 64
    return s


def bench2_points(ds=(3, 5, 7)):
    """The open benchmark's lukin-family test points under the Lukin model (scripts/dev_bank.py test_points: the manifest's
    circuits i = 0-19, shots per circuit and exact-MLE prefix) with main result 2's new seeds."""
    with open(os.path.join(ROOT, "data", "open_benchmark", "MANIFEST.json")) as f:
        pts = [pt for pt in json.load(f)["points"] if pt["family"] == "lukin" and pt["noise"] == "lukin" and pt["d"] in ds]
    out = []
    for pt in sorted(pts, key=lambda pt: (ds.index(pt["d"]), pt["depth"])):
        cs = [dict(c, seed=str(seed_bench2(pt["d"], pt["depth"], int(c["index"])))) for c in pt["circuits"]]
        assert [int(c["index"]) for c in cs] == list(range(20)) and pt["K"] == 4 and pt["L"] == 2 * pt["depth"], pt["name"]
        out.append(dict(pt, circuits=cs, seeds=[cs[0]["seed"], str(int(cs[-1]["seed"]) + 1)], purpose="bench2"))
    return out


def circuits(purpose, K, L, n=NCIRC):
    """The n logical circuits of (purpose, K, L): (layers, prep, meas, q); the same at every d and p."""
    rng = np.random.default_rng([FIRST_TAG, PURPOSE[purpose], K, L])
    out = []
    for i in range(n):
        q = P_SET[i % 4]
        lay, prep, meas = cc.sample_circuit_k(rng, K, L, q, GATES)
        out.append((lay, prep, meas, q))
    return out


def point_name(purpose, d, K, L, p):
    return f"b1{purpose[0]}_K{K}_d{d}_L{L}_p{PCODE[p]}"


def points(purpose, ds=DS, Ks=KS, pts=POINTS, shots=None):
    """Manifest entries (duet.bench_sets' format) of a purpose's points."""
    shots = shots or SIZE[purpose]
    out = []
    for K in Ks:
        for L, p in pts:
            circs = circuits(purpose, K, L)
            for d in ds:
                cs = []
                for i, (lay, prep, meas, q) in enumerate(circs):
                    prods = cc.logical_products(lay, prep, meas)
                    cs.append(dict(layers=" ".join(lay), prep="".join(prep), meas="".join(meas), pauli="frame",
                                   cnot_layers=sum(len(pr) > 0 for pr, _ in cc.parse_k(lay, K)), q=round(q, 7),
                                   products=[list(s) for s, _ in prods], values=[v for _, v in prods],
                                   seed=str(seed(purpose, d, K, L, p, i))))
                n = int(shots[(L, p)])
                out.append(dict(name=point_name(purpose, d, K, L, p), family="general_mix", sub="mix", purpose=purpose,
                                d=d, noise="lukin", p=p, depth=L, L=L, K=K, shots=n, bplomr_shots=min(BPLOMR, n),
                                seeds=[cs[0]["seed"], str(int(cs[-1]["seed"]) + 1)], circuits=cs))
    return out


def load_manifest(path):
    return json.load(open(path))


def select(m, only=""):
    rx = re.compile(only) if only else None
    return [pt for pt in m["points"] if rx is None or rx.search(pt["name"])]


def set_path(sets, pt):
    return os.path.join(sets, pt["name"] + ".npz")


def git_commit():
    f = os.path.join(ROOT, "COMMIT")
    return open(f).read().strip() if os.path.exists(f) else None


def cpu():
    for line in open("/proc/cpuinfo"):
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def part_of(jobs, spec, key=lambda j: j, weight=None):
    """Part j of P of a job list: "j/P" by a hash of key; "bj/P" balanced by weight (greedy, largest first, over the
    whole list in a fixed order, so every part sees the same assignment)."""
    if not spec:
        return jobs
    if spec.startswith("b"):
        j0, n = (int(x) for x in spec[1:].split("/"))
        load, out = [0.0] * n, []
        for jb in sorted(jobs, key=lambda x: (-weight(x), str(key(x)))):
            k = min(range(n), key=lambda i: (load[i], i))
            load[k] += weight(jb)
            if k == j0:
                out.append(jb)
        return out
    j0, n = (int(x) for x in spec.split("/"))
    return [j for j in jobs if zlib.crc32(str(key(j)).encode()) % n == j0]


def net_weight(pt):
    """The start network's CPU cost of a point, roughly (timing sample): shots x K x (L + 1) x (d + 1)^2."""
    return pt["shots"] * len(pt["circuits"]) * pt["K"] * (pt["L"] + 1) * (pt["d"] + 1) ** 2


# ------------------------------------------------------------------------------------------------- manifest, gen

def cmd_manifest(a):
    if a.purpose == "bench2":
        src = os.path.join(ROOT, "data", "open_benchmark", "MANIFEST.json")
        pts = bench2_points()
        m = dict(version=1, run="", purpose="bench2", family="lukin",
                 seed_base=str(FIRST_SEED_BASE), seed_range=[str(FIRST_SEED_BASE + 4 * 10 ** 13), str(FIRST_SEED_BASE + 5 * 10 ** 13)],
                 seed_rule="FIRST_SEED_BASE + 4 10^13 + d 10^11 + D 10^8 + i (circuit i = 0-19 of depth D at distance d)",
                 circuit_rule="the open benchmark's fixed lukin-family circuits (data/open_benchmark/MANIFEST.json, family and "
                              "noise lukin, circuits i = 0-19; duet.bench_sets.lukin), unchanged",
                 source_manifest_sha256=hashlib.sha256(open(src, "rb").read()).hexdigest(),
                 sizes="the manifest's shots per circuit and mle_shots prefix ( at d = 3, 5; at d = 7)",
                 noise="lukin", points=pts)
        json.dump(m, open(a.out, "w"), indent=0)
        print(json.dumps(dict(points=len(pts), shots=int(sum(pt["shots"] * len(pt["circuits"]) for pt in pts)),
                              mle=int(sum(bs.prefix(pt, "mle") * len(pt["circuits"]) for pt in pts)))))
        return
    shots = {(L, p): v for (L, p), v in SIZE[a.purpose].items()}
    pts = points(a.purpose, shots=shots)
    m = dict(version=1, run="", purpose=a.purpose, family="general_mix",
             seed_base=str(FIRST_SEED_BASE), seed_range=[str(FIRST_SEED_BASE), str(FIRST_SEED_BASE + FIRST_SEED_SPAN)],
             seed_rule="FIRST_SEED_BASE + g 10^13 + d 10^11 + K 10^10 + L 10^8 + pcode 10^6 + i (g: 1 dev, 2 test, 3 probe)",
             circuit_rule=f"numpy default_rng([{FIRST_TAG}, g, K, L]); circuit i: sample_circuit_k(rng, K, L, P_SET[i % 4], "
                          f"'{GATES}'), the same circuits at every d and p",
             generator="duet.clifford_circuits.sample_circuit_k (K-block mix law; train_cl.draw's mix)",
             noise="lukin", points=pts)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(m, open(a.out, "w"), indent=0)
    print(json.dumps(dict(points=len(pts), shots=int(sum(pt["shots"] * len(pt["circuits"]) for pt in pts)))))


def cmd_gen(a):
    m = load_manifest(a.manifest)
    os.makedirs(a.dir, exist_ok=True)
    todo = [(pt, a.dir) for pt in select(m, a.only) if not os.path.exists(set_path(a.dir, pt))]
    todo = part_of(todo, a.part, key=lambda j: j[0]["name"])
    todo.sort(key=lambda j: -j[0]["shots"] * j[0]["L"] * j[0]["K"] * j[0]["d"] ** 2)
    t0, rec = time.time(), {}
    with mp.get_context("fork").Pool(a.workers) as pool:
        for name, sha in pool.imap_unordered(bs._point, todo):
            rec[name] = sha
            with open(os.path.join(a.dir, f"sha256{'.' + a.part.replace('/', 'of') if a.part else ''}.txt"), "a") as f:
                f.write(f"{sha}  {name}.npz\n")
            print(json.dumps(dict(point=name, sha256=sha, minutes=round((time.time() - t0) / 60, 2))), flush=True)


# ------------------------------------------------------------------------------------------------------------ dec

def _dec_job(job):
    """Decode shots idx of circuit i of a stored point with an opponent; write the part file."""
    name, path, i, idx, out = job
    if os.path.exists(out):
        return out, 0, 0.0
    from duet import bench_decode as bd
    z = np.load(path)
    entry, m = json.loads(str(z["entry"])), int(z["m"][i])
    c_ = entry["circuits"][i]
    rows = bs.packed(path, "dets")[i]
    dets = bs.unpack(rows[idx[0]:idx[-1] + 1] if np.array_equal(idx, np.arange(idx[0], idx[-1] + 1)) else rows[idx], m)
    t0 = time.time()
    c, dec = bd.build(name, entry, c_)
    build = time.time() - t0
    pred, ok, s = bd.decode(name, dec, dets)
    sec = np.full(len(idx), s / len(idx))
    settings = getattr(dec, "settings", {})
    assert pred.shape[1] <= 8
    pk = np.packbits(pred, axis=1, bitorder="little")[:, 0] if pred.shape[1] else np.zeros(len(idx), np.uint8)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out + ".tmp.npz", idx=np.asarray(idx, np.int64), pred=pk, ok=ok, sec=np.asarray(sec, np.float32),
             build=build, cpu=cpu(), settings=json.dumps(settings, default=str))
    os.replace(out + ".tmp.npz", out)
    return out, len(idx), float(np.sum(sec)) + build


def cmd_dec(a):
    m = load_manifest(a.manifest)
    jobs = []
    for pt in select(m, a.only):
        base = os.path.join(a.out, a.decoder, pt["name"])
        n = pt["shots"] if a.decoder != "bplomr" else pt["bplomr_shots"]
        n = min(n, a.max_shots or n)
        ch = CHUNK[a.decoder]
        for i in range(len(pt["circuits"])):
            for s in range(0, n, ch):
                idx = np.arange(s, min(n, s + ch))
                out = os.path.join(base, f"c{i}_{s}.npz")
                jobs.append((a.decoder, set_path(a.dir, pt), i, idx, out,
                             pt["L"] * len(idx) * pt["d"] ** 3 * pt["K"]))
    jobs = part_of(jobs, a.part, key=lambda j: j[4], weight=lambda j: j[-1])     # the same split whatever is done
    jobs = [j for j in jobs if not os.path.exists(j[4])]
    jobs.sort(key=lambda j: -j[-1])
    t0, n, cpu_s = time.time(), 0, 0.0
    print(json.dumps(dict(decoder=a.decoder, parts=len(jobs), shots=int(sum(len(j[3]) for j in jobs)), cpu=cpu())),
          flush=True)
    with mp.get_context("fork").Pool(a.workers) as pool:
        for k, (_, ns, s) in enumerate(pool.imap_unordered(_dec_job, [j[:5] for j in jobs])):
            n, cpu_s = n + ns, cpu_s + s
            if (k + 1) % max(1, len(jobs) // 20) == 0 or k + 1 == len(jobs):
                print(json.dumps(dict(done=k + 1, shots=n, core_h=round(cpu_s / 3600, 3),
                                      minutes=round((time.time() - t0) / 60, 1))), flush=True)


def load_dec(dec_dir, decoder, pt):
    """{circuit: (shot indices, predicted flips packed in a byte, certified, seconds)} of a decoder's parts."""
    out = {}
    for f in glob.glob(os.path.join(dec_dir, decoder, pt["name"], "c*_*.npz")):
        if f.endswith(".tmp.npz"):
            continue
        i = int(os.path.basename(f)[1:].split("_")[0])
        z = np.load(f)
        out.setdefault(i, []).append((z["idx"], z["pred"], z["ok"], z["sec"]))
    return {i: tuple(np.concatenate([x[j] for x in v]) for j in range(4)) for i, v in out.items()}


# ------------------------------------------------------------------------------------------------------------ net

def bench_net():
    spec = importlib.util.spec_from_file_location("bench_net", os.path.join(ROOT, "scripts", "bench_net.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def frozen_logits(net, tab, packed, d, batch):
    """(n, n_obs) fp32 logits of packed shots by the frozen forward (train_cl.collate, forward; eval form)."""
    import torch
    from duet.net import train_cl
    dev = next(net.parameters()).device
    out = []
    for s in range(0, len(packed), batch):
        b = packed[s:s + batch]
        zero = np.zeros((len(b), (tab["n_total"] + 7) // 8), np.uint8)
        with torch.no_grad():
            c, _, _ = train_cl.forward(net, train_cl.collate([(tab, np.ascontiguousarray(b), zero)], d, dev,
                                                             train=False), train=False)
        out.append(c[:, :tab["n_obs"]].float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, tab["n_obs"]), np.float32)


def _net_init(ckpt, d, device, threads):
    import torch
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    global _NET, _NETREC
    _NET, _NETREC = bench_net().load_net("lukin", d, device, ckpt)


def _net_job(job):
    """Logits of the first n shots of circuit i of a point (one CPU worker, its own threads)."""
    from duet.net import train_cl
    path, i, n, batch = job
    z = np.load(path)
    entry = json.loads(str(z["entry"]))
    c_ = entry["circuits"][i]
    d = entry["d"]
    t0 = time.time()
    tab = train_cl.tables(d, c_["layers"].split(), tuple(c_["prep"]), tuple(c_["meas"]), entry["p"], entry["noise"],
                          [list(s) for s in c_["products"]])[0]
    t1 = time.time()
    D = bs.packed(path, "dets")
    c = frozen_logits(_NET, tab, D[i, :n, :(tab["m"] + 7) // 8], d, batch)
    return entry["name"], i, c, t1 - t0, time.time() - t1


def cmd_net(a):
    import torch
    m = load_manifest(a.manifest)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    d = ck["cfg"]["d"]
    sha = hashlib.sha256(open(a.ckpt, "rb").read()).hexdigest()
    pts = [pt for pt in select(m, a.only) if pt["d"] == d]
    os.makedirs(os.path.join(a.out, a.name), exist_ok=True)
    todo = part_of(pts, a.part, key=lambda pt: pt["name"], weight=net_weight)      # the same split whatever is done
    todo = [pt for pt in todo if not os.path.exists(os.path.join(a.out, a.name, pt["name"] + ".npz"))]
    rec = dict(ckpt=a.ckpt, sha256=sha, step=ck.get("step"), cfg=ck["cfg"], device=a.device, frac=a.frac,
               path="frozen forward (train_cl.tables, collate, forward), fp32, TF32 off", code=os.environ.get("CODE"),
               cpu=cpu())
    del ck
    jobs = []
    for pt in todo:
        n = int(math.ceil(a.frac * pt["shots"]))
        for i in range(len(pt["circuits"])):
            jobs.append((set_path(a.dir, pt), i, n, a.batch or NET_BATCH[d]))
    print(json.dumps(dict(points=len(todo), circuits=len(jobs), shots=int(sum(j[2] for j in jobs)), **rec)),
          flush=True)
    acc, t0 = {}, time.time()
    pool = mp.get_context("fork").Pool(a.workers, initializer=_net_init, initargs=(a.ckpt, d, a.device, a.threads))
    left = {pt["name"]: len(pt["circuits"]) for pt in todo}
    byname = {pt["name"]: pt for pt in todo}
    for name, i, c, tt, tn in pool.imap_unordered(_net_job, jobs, chunksize=1):
        pt = byname[name]
        n = int(math.ceil(a.frac * pt["shots"]))
        x = acc.setdefault(name, dict(lg=np.full((len(pt["circuits"]), n, 4), np.nan, np.float32),
                                      tab=np.zeros(len(pt["circuits"])), sec=np.zeros(len(pt["circuits"]))))
        x["lg"][i, :, :c.shape[1]] = c
        x["tab"][i], x["sec"][i] = tt, tn
        left[name] -= 1
        if left[name] == 0:
            out = os.path.join(a.out, a.name, name + ".npz")
            np.savez(out + ".tmp.npz", logits=x["lg"], tab_sec=x["tab"], secs=x["sec"], batch=a.batch or NET_BATCH[d],
                     info=json.dumps(rec))
            os.replace(out + ".tmp.npz", out)
            print(json.dumps(dict(point=name, shots=n * len(pt["circuits"]), core_s_per_shot=round(
                float(x["sec"].sum()) * a.threads / (n * len(pt["circuits"])), 5),
                minutes=round((time.time() - t0) / 60, 1))), flush=True)
            del acc[name]
    pool.close()


# ------------------------------------------------------------------------------------------------------------ sel

def scores(net_dir, name, pt):
    """(circuits, n) the smallest |logit| over each circuit's products."""
    z = np.load(os.path.join(net_dir, name, pt["name"] + ".npz"))
    lg = z["logits"]
    return np.nanmin(np.abs(lg), axis=2)


def least(sc, frac):
    """Flat indices (circuit * n + shot) of the ceil(frac x size) smallest scores (ties: the lower index)."""
    flat = sc.reshape(-1)
    k = int(math.ceil(frac * flat.size))
    return np.sort(np.argsort(flat, kind="stable")[:k])


def cmd_sel(a):
    m = load_manifest(a.manifest)
    pts = [pt for pt in select(m, a.only) if os.path.exists(os.path.join(a.net, a.name, pt["name"] + ".npz"))]
    out, tot, by_depth = {}, 0, {}
    for pt in pts:
        sc = scores(a.net, a.name, pt)
        n = sc.shape[1]
        keep = set(least(sc, a.frac).tolist())
        by_depth.setdefault((pt["d"], pt["L"], pt["p"]), []).append((pt, sc))
        out[pt["name"]] = keep
    for key, v in by_depth.items():                  # per depth, K pooled: the same share of the pooled scores
        allsc = np.concatenate([sc.reshape(-1) for _, sc in v])
        k = int(math.ceil(a.frac * allsc.size))
        thr = np.sort(allsc, kind="stable")[k - 1] if k else -np.inf
        for pt, sc in v:
            out[pt["name"]] |= set(np.flatnonzero(sc.reshape(-1) <= thr).tolist())
    lists = {}
    for pt in pts:
        n = int(np.load(os.path.join(a.net, a.name, pt["name"] + ".npz"))["logits"].shape[1])
        x = np.array(sorted(out[pt["name"]]), np.int64)
        tot += len(x)
        lists[pt["name"]] = {str(int(c)): (x[x // n == c] % n).tolist() for c in np.unique(x // n)}
    json.dump(dict(net=a.name, frac=a.frac, rule="per point and per depth (K pooled), union", shots=tot, lists=lists),
              open(a.out, "w"))
    print(json.dumps(dict(points=len(pts), shots=tot)))


# ----------------------------------------------------------------------------------------------------------- tess

def tess_mod():
    import tesseract_decoder
    from tesseract_decoder import tesseract as tt, utils as tu
    return tesseract_decoder, tt, tu


def tess_config(circ):
    """The long beam's TesseractConfig (duet.decoders.tesseract_dec.SETTINGS), as pdec.config."""
    from duet.decoders.tesseract_dec import SETTINGS as s
    _, tt, tu = tess_mod()
    dem = circ.detector_error_model(decompose_errors=False)
    orders = tu.build_det_orders(dem, s["num_det_orders"], getattr(tu.DetOrder, s["det_order"]), s["det_order_seed"])
    return tt.TesseractConfig(dem=dem, det_beam=s["det_beam"], beam_climbing=s["beam_climbing"],
                              no_revisit_dets=s["no_revisit_dets"], pqlimit=s["pqlimit"], det_orders=orders)


def capsec(spec, L, d=None, fac=None):
    """A trial's local deadline in seconds: none; L<c>: c x L; B<c>: 2 c L / F with
    F this node's CPU model's factor to the choice's basis at d (route.py parse_fac: --cal, --basis; no default)."""
    if spec in (None, "none", ""):
        return None
    if spec.startswith("L"):
        return float(spec[1:]) * L
    if spec.startswith("B"):
        F = fac.get(d, {}).get(cpu(), fac.get(None, {}).get(cpu()))
        assert F, (d, cpu(), "no factor from this CPU model to the basis: calibrate it first (README, Timing)")
        return 2 * float(spec[1:]) * L / F
    return float(spec)


def pk(obs):
    return sum(1 << q for q, v in enumerate(obs) if v)


def trials(dec, dets, dl):
    """Every trial of each shot alone: (seconds, flag, dropped, cost, flips byte) arrays (n, 21)."""
    n, clock = len(dets), time.perf_counter_ns
    sec, cost = np.zeros((n, K_TRIALS), np.float32), np.full((n, K_TRIALS), np.nan)
    flag, drop, pred = np.zeros((n, K_TRIALS), bool), np.zeros((n, K_TRIALS), bool), np.zeros((n, K_TRIALS), np.uint8)
    for k, row in enumerate(dets):
        for j in range(K_TRIALS):
            t1 = clock()
            e = dec.decode_to_errors(row, j, j) if dl is None else dec.decode_to_errors(row, j, j, dl)
            sec[k, j] = (clock() - t1) / 1e9
            flag[k, j] = bool(dec.low_confidence_flag)
            drop[k, j] = bool(getattr(dec, "deadline_exceeded", False))
            if not flag[k, j]:
                cost[k, j] = dec.cost_from_errors(e)
                pred[k, j] = pk(dec.get_observables_from_errors(e))
    return sec, flag, drop, cost, pred


def fac_of(fac, d, c):
    """route.parse_fac's factor of (d, CPU model c), else of c for every d; None without one."""
    return fac.get(d, {}).get(c, fac.get(None, {}).get(c))


def done_shots(out, i, dl, fac=None, d=None):
    """The shots of circuit i already decoded beside out in a part whose deadline covers dl (seconds on this node's CPU;
    None: no deadline, then only parts without one count), so overlapping lists never decode a shot twice and a
    shorter deadline never stands for a larger one. Finite deadlines compare in basis seconds: each part's local deadline x its CPU model's factor at d against dl x this CPU's (fac: route.parse_fac's
    Grace factors; one basis is enough, the Spark ones being the Grace ones x 1.124); a part of a CPU model without a
    factor never counts, and without fac only parts of this CPU model count (the same clock). route.load_trials keeps
    each shot's largest basis-deadline part."""
    done, here, want = set(), cpu(), None
    if dl is not None and fac is not None:
        fh = fac_of(fac, d, here)
        assert fh, (d, here, "no factor of this CPU model: calibrate it first (README, Timing)")
        want = dl * fh
    for f in glob.glob(os.path.join(os.path.dirname(out), f"c{i}_*.npz")):
        if f.endswith(".tmp.npz"):
            continue
        with np.load(f) as z:
            d0 = float(z["deadline"])
            d0 = math.inf if math.isnan(d0) else d0
            if d0 == math.inf:
                ok = True
            elif dl is None:
                ok = False
            elif fac is None:
                ok = str(z["cpu"]) == here and d0 >= dl * (1 - 1e-9)
            else:
                fp = fac_of(fac, d, str(z["cpu"]))
                ok = bool(fp) and d0 * fp >= want * (1 - 1e-9)
            if ok:
                done.update(z["idx"].tolist())
    return done


def req_deadline(req, L, d, facs):
    """: the local deadline of a shot whose choices are req ("none" or "basis:c;basis:c"): max over
    them of 2 c L / F_basis (this node's CPU model's factor at d; no default), none when one is uncapped."""
    if req == "none":
        return None
    out = 0.0
    for x in req.split(";"):
        b, c = x.split(":")
        F = fac_of(facs[b], d, cpu())
        assert F, (d, b, cpu(), "no factor from this CPU model to the basis: calibrate it first (README, Timing)")
        out = max(out, 2 * float(c) * L / F)
    return out


def _tess_job(job):
    """One chunk's trials: the shots not yet decoded beside it (done_shots), in a part named by its first remaining
    shot (c<i>_<x>, or c<i>_<x>_<k> beside a shorter-deadline part of the same first shot)."""
    path, i, idx, out, dl, cfac, d = job
    done = done_shots(out, i, dl, cfac, d)
    idx = np.array([x for x in idx if int(x) not in done], np.int64)
    if not len(idx):
        return 0, 0.0, 0
    out, k = os.path.join(os.path.dirname(out), f"c{i}_{int(idx[0])}.npz"), 1
    while os.path.exists(out):            # a shorter-deadline part of the same first shot: a new name beside it
        out, k = os.path.join(os.path.dirname(out), f"c{i}_{int(idx[0])}_{k}.npz"), k + 1
    tmod, _, _ = tess_mod()
    z = np.load(path)
    entry, m = json.loads(str(z["entry"])), int(z["m"][i])
    dets = bs.unpack(bs.packed(path, "dets")[i][idx], m)
    t0 = time.time()
    dec = tess_config(bs.circuit(entry, entry["circuits"][i])).compile_decoder()
    build = time.time() - t0
    if dl is not None and "deadline_exceeded" not in dir(dec):
        raise RuntimeError("a deadline needs stoppable build on PYTHONPATH")
    sec, flag, drop, cost, pred = trials(dec, dets, dl)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out + ".tmp.npz", idx=np.asarray(idx, np.int64), sec=sec, cost=cost, flag=flag, drop=drop, pred=pred,
             build=build, cpu=cpu(), module=tmod.__file__, deadline=np.nan if dl is None else dl)
    os.replace(out + ".tmp.npz", out)
    return len(idx), float(sec.sum()) + build, int(drop.sum())


def cmd_tess(a):
    m = load_manifest(a.manifest)
    pts = {pt["name"]: pt for pt in m["points"]}
    Ls = json.load(open(a.lists))["lists"]
    rx = re.compile(a.only) if a.only else None
    fac, facs, R = None, None, None
    if a.deadline.startswith("B") or a.deadline == "req":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import operating_points as route  # noqa: E402
        ds = sorted({pts[nm]["d"] for nm in Ls})
        assert a.cal, "--deadline B<c> / req needs --cal"
        facs = {b: route.parse_fac(a.fac, a.cal, route.SPARK_CPU if b == "spark" else 1.0, cal_ds=ds)
                for b in ("grace", "spark")}
        if a.deadline.startswith("B"):
            assert a.basis in ("grace", "spark"), "--deadline B<c> needs --basis grace|spark"
            fac = facs[a.basis]
        else:
            R = json.load(open(a.lists))["req"]
    cfac = facs["grace"] if facs else None        # done_shots compares deadlines in (Grace) basis seconds
    jobs = []
    for nm, cs in Ls.items():
        if rx is not None and not rx.search(nm):
            continue
        pt = pts[nm]
        ch = tess_chunk(pt["d"], pt["L"])
        for c, ss in cs.items():
            if R is None:
                groups = {None: sorted(ss)}
            else:                                      # one group per requirement: one deadline per part
                groups = {}
                for s_, rq in zip(ss, R[nm][c]):
                    groups.setdefault(rq, []).append(s_)
                groups = {rq: sorted(v) for rq, v in groups.items()}
            for rq, ss in groups.items():
                dl = capsec(a.deadline, pt["L"], pt["d"], fac) if rq is None else \
                    req_deadline(rq, pt["L"], pt["d"], facs)
                for k in range(0, len(ss), ch):
                    idx = np.array(ss[k:k + ch], np.int64)
                    out = os.path.join(a.out, nm, f"c{c}_{int(idx[0])}.npz")
                    jobs.append((pt["d"] ** 4 * pt["K"] * pt["L"] ** 1.5 * len(idx),
                                 (set_path(a.dir, pt), int(c), idx, out, dl, cfac, pt["d"])))
    jobs.sort(key=lambda j: -j[0])
    jobs = part_of(jobs, a.part, key=lambda j: j[1][3], weight=lambda j: j[0])    # the same split whatever is done
    # (no skip by name here: _tess_job drops the shots already decoded, whatever list decoded them)
    print(json.dumps(dict(parts=len(jobs), shots=int(sum(len(j[1][2]) for j in jobs)), deadline=a.deadline,
                          module=tess_mod()[0].__file__, cpu=cpu())), flush=True)
    t0, n, s, dr = time.time(), 0, 0.0, 0
    with mp.get_context("fork").Pool(a.workers) as pool:
        for k, (ns, ss, dd) in enumerate(pool.imap_unordered(_tess_job, [j[1] for j in jobs])):
            n, s, dr = n + ns, s + ss, dr + dd
            if (k + 1) % max(1, len(jobs) // 40) == 0 or k + 1 == len(jobs):
                print(json.dumps(dict(done=k + 1, shots=n, dropped=dr, core_h=round(s / 3600, 3),
                                      minutes=round((time.time() - t0) / 60, 1))), flush=True)


# ---------------------------------------------------------------------------------------------------------- probe

def _probe_job(job):
    """Costs on one probe circuit (parts: dec = builds and per-shot CPU seconds of LoMatching, Cain et al. and
    BP+LoMatching; tess = the long beam's 21 trials on shots of the top quartile by detection events and on random
    ones; net = the start network's frozen forward on CPU, one thread)."""
    d, K, L, i, n, ntess, nbp, ckpt, parts = job
    lay, prep, meas, q = circuits("probe", K, L)[i]
    prods = cc.logical_products(lay, prep, meas)
    entry = dict(name=f"probe_K{K}_d{d}_L{L}", d=d, noise="lukin", p=0.003, K=K, L=L)
    c_ = dict(layers=" ".join(lay), prep="".join(prep), meas="".join(meas), pauli="frame",
              products=[list(s) for s, _ in prods])
    circ = bs.circuit(entry, c_)
    dets, obs = circ.compile_detector_sampler(seed=seed("probe", d, K, L, 0.003, i)).sample(n, separate_observables=True)
    r = dict(d=d, K=K, L=L, i=i, q=q, m=circ.num_detectors, k=circ.num_observables, n=n,
             events_mean=float(dets.sum(1).mean()), cpu=cpu())
    if "dec" in parts:
        from duet import bench_decode as bd
        for name in ("lomr", "cain"):
            t0 = time.process_time()
            _, dec = bd.build(name, entry, c_)
            r[name + "_build"] = time.process_time() - t0
            pred, _, s = bd.decode(name, dec, dets)
            r[name + "_s"] = s / n
            r[name + "_fail"] = int((pred != obs).any(1).sum())
            r[name + "_dropped"] = int(np.sum(getattr(dec, "dropped", [0])))
        t0 = time.process_time()
        _, dec = bd.build("bplomr", entry, c_)
        r["bplomr_build"] = time.process_time() - t0
        pred, _, s = bd.decode("bplomr", dec, dets[:nbp])
        r["bplomr_s"] = s / nbp
        r["bplomr_fail"] = int((pred != obs[:nbp]).any(1).sum())
    if "tess" in parts:
        ev = dets.sum(1)
        g = np.random.default_rng([FIRST_TAG, 3, d, K, L, i])
        top = np.flatnonzero(ev >= np.quantile(ev, 0.75))
        pick_top = np.sort(g.choice(top, min(ntess, len(top)), replace=False))
        pick_any = np.sort(g.choice(n, 2, replace=False))
        t0 = time.process_time()
        tdec = tess_config(circ).compile_decoder()
        r["tess_build"] = time.process_time() - t0
        for lab, pick in (("tess_top", pick_top), ("tess_any", pick_any)):
            sec, flag, drop, cost, tp = trials(tdec, dets[pick], None)
            r[lab + "_shot_s"] = sec.sum(1).tolist()
            r[lab + "_max_trial_s"] = sec.max(1).tolist()
            r[lab + "_events"] = ev[pick].tolist()
            r[lab + "_all_flagged"] = int(flag.all(1).sum())
    if "net" in parts:
        import torch
        from duet.net import train_cl
        torch.set_num_threads(1)
        net = bench_net().load_net("lukin", d, "cpu", ckpt)[0]
        t0 = time.time()
        tab = train_cl.tables(d, lay, prep, meas, 0.003, "lukin", [list(s) for s, _ in prods])[0]
        r["net_tab_s"] = time.time() - t0
        nn = min(n, 64)
        pkd = np.packbits(dets[:nn], axis=1, bitorder="little")
        t0 = time.time()
        lg = frozen_logits(net, tab, pkd, d, max(8, NET_BATCH[d]))
        r["net_s"] = (time.time() - t0) / nn
        r["net_fail"] = int(((lg > 0) != obs[:nn]).any(1).sum())
        r["net_n"] = nn
    return r


def cmd_probe(a):
    Ls = [int(x) for x in a.Ls.split(",")]
    jobs = [(a.d, K, L, i, a.n, a.ntess, a.nbp, a.ckpt, a.parts.split(",")) for K in KS for L in Ls
            for i in range(a.circuits)]
    jobs.sort(key=lambda j: -j[1] * j[2])
    rows = []
    with mp.get_context("fork").Pool(a.workers) as pool:
        for r in pool.imap_unordered(_probe_job, jobs, chunksize=1):
            rows.append(r)
            print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()
                              if not isinstance(v, list)}), flush=True)
    json.dump(rows, open(a.out, "w"))


# ------------------------------------------------------------------------------------------------------- labcheck

def _lab_job(job):
    """Item 1's checks on one circuit: (a) representatives on line j's final data, (b) per-layer
    labels encode every error's representative flips (labels_ro.check_encoding), (c) y_{k,j} = y_k + (D A) for every
    fault (geometry.check_faults); detectors: count K (2L + 1)(d^2 - 1) / 2, noiseless samples zero, noiseless
    observables equal the tableau's values, the extended model's detectors equal the circuit's."""
    sys.path.insert(0, os.path.join(ROOT, "ops", "multi"))
    import labcheck as LC
    from duet.net import geometry, spec_cl
    from duet.net import labels_ro as LR
    purpose, d, K, L, i = job
    lay, prep, meas, q = circuits(purpose, K, L)[i]
    prods = [list(s) for s, _ in cc.logical_products(lay, prep, meas)]
    s = spec_cl.build(d, lay, prep, meas, 0.003, "lukin", prods)
    ext, _ = LR.extended_dem(s)
    c, meta = cc.make_circuit(d, lay, 0.003, prep, meas, meta=True, noise="lukin", products=prods)
    c0 = cc.make_circuit(d, lay, 0.0, prep, meas, noise="lukin", products=prods)
    dz, oz = c0.compile_detector_sampler(seed=1).sample(64, separate_observables=True)
    vals = [int(v) for _, v in meta["products"]]
    r = dict(purpose=purpose, d=d, K=K, L=L, i=i, layers=" ".join(lay), prep="".join(prep), meas="".join(meas),
             products=prods, rep_errors=LC.rep_errors(s, lay, prep, meas, "lukin"), encoding=LR.check_encoding(s, ext),
             faults=geometry.check_faults(s),
             det_count_ok=c.num_detectors == K * (2 * L + 1) * (d * d - 1) // 2 == ext.num_detectors,
             noiseless_dets_zero=not dz.any(), noiseless_obs_zero=not oz.any(),
             products_match=[sorted(x) for x, _ in meta["products"]] == [sorted(x) for x in prods],
             n_obs=len(prods), values=vals, values_match=ref_values(c0) == vals)
    r["bad"] = int(r["rep_errors"] + r["encoding"] + r["faults"] + (not r["det_count_ok"]) + (not r["noiseless_dets_zero"])
                   + (not r["noiseless_obs_zero"]) + (not r["products_match"]) + (not r["values_match"]))
    return r


HAND = [   # (layers, prep, meas, the reliable products and values worked out by hand)
    # K = 2: CNOT 0 -> 1 on |+>|0> is the Bell state: only Z0 Z1 (= +1, value 0) is reliable in ZZ
    (["1+"], "XZ", "ZZ", [([0, 1], 0)]),
    # K = 2: the same read in XX: X0 X1 = +1
    (["1+"], "XZ", "XX", [([0, 1], 0)]),
    # K = 2: H on block 0 then CNOT 0 -> 1 from |00>: the same Bell state, Z0 Z1 and (XX) X0 X1
    (["HI", "1+"], "ZZ", "ZZ", [([0, 1], 0)]),
    # K = 2: S on |+> gives |+i>: X0 is random; block 1 idles in |0>: Z1 = +1
    (["SI"], "XZ", "XZ", [([1], 0)]),
    # K = 2: S S = Z on |+>: X0 = -1 (value 1); H on block 1 from |0> gives |+>: X1 = +1
    (["SH", "SI"], "XZ", "XX", [([0], 1), ([1], 0)]),
    # K = 3: H on 0, CNOT 1 -> 2 from |0>|+>|0>: X0 = +1, Z1 Z2 = +1
    (["H2+"], "ZXZ", "XZZ", [([0], 0), ([1, 2], 0)]),
    # K = 3: CNOT 0 -> 1 and S on 2, then CNOT 1 -> 2, from |+00>: GHZ on 0, 1, 2 (S on |0> does nothing); the basis
    # by size, then lexicographic: Z0 Z1, then Z0 Z2 (Z1 Z2 is their product)
    (["1+S", "I2+"], "XZZ", "ZZZ", [([0, 1], 0), ([0, 2], 0)]),
    # K = 4: two Bell pairs (0, 1) and (2, 3) from |+0+0>, read in X: X0 X1 = +1, X2 X3 = +1
    (["1+3+"], "XZXZ", "XXXX", [([0, 1], 0), ([2, 3], 0)]),
    # K = 4: CNOT 3 -> 0, H on 1, S on 2 from |0000>: Z0, Z3 (= +1), X1 = +1, Z2 = +1
    (["+HS0"], "ZZZZ", "ZXZZ", [([0], 0), ([1], 0), ([2], 0), ([3], 0)]),
]


def ref_values(c0):
    """Each observable's parity of the noiseless reference sample's records (stim's samplers report flips relative to
    it, so this is the noiseless outcome itself); OBSERVABLE_INCLUDE comes after the last measurement."""
    sys.path.insert(0, os.path.join(ROOT, "ops", "multi"))
    import labcheck as LC
    recs, _, _ = LC.rep_records(str(c0))
    ref = c0.reference_sample()
    return [int(np.bitwise_xor.reduce(ref[sorted(recs[k])].astype(np.uint8))) for k in range(c0.num_observables)]


def _hand_job(job):
    """A hand-checked circuit at d: products and values as worked out; noiseless circuit gives those values in the
    final data's parities (no flip); detector count."""
    d, (lay, prep, meas, want) = job
    K = len(prep)
    got = [(sorted(s), v) for s, v in cc.logical_products(lay, tuple(prep), tuple(meas))]
    c0 = cc.make_circuit(d, lay, 0.0, tuple(prep), tuple(meas), noise="lukin")
    dz, oz = c0.compile_detector_sampler(seed=2).sample(32, separate_observables=True)
    c = cc.make_circuit(d, lay, 0.003, tuple(prep), tuple(meas), noise="lukin")
    vals = ref_values(c0)
    return dict(d=d, layers=" ".join(lay), prep=prep, meas=meas, want=want, got=got, products_ok=got == want,
                noiseless_ok=not dz.any() and not oz.any(), ref_values=vals,
                values_ok=vals == [v for _, v in want],
                det_count_ok=c.num_detectors == K * (2 * len(lay) + 1) * (d * d - 1) // 2)


def law_census(draw, n, K, L, seed_):
    """Frequencies of the generator's law over n circuits: CNOT pairs per layer, gate letters outside pairs, bases,
    products per circuit."""
    rng = np.random.default_rng(seed_)
    pairs, gates, bases, prods = np.zeros(K // 2 + 1), {g: 0 for g in GATES}, {"Z": 0, "X": 0}, np.zeros(K + 1)
    for _ in range(n):
        lay, prep, meas = draw(rng)
        for pr, g in cc.parse_k(lay, K):
            pairs[len(pr)] += 1
            for b, x in enumerate(g):
                if all(b not in p_ for p_ in pr):
                    gates[x] += 1
        for x in prep + meas:
            bases[x] += 1
        prods[len(cc.logical_products(lay, prep, meas))] += 1
    tot_l = pairs.sum()
    return dict(pairs=(pairs / tot_l).round(4).tolist(), gates={g: round(v / sum(gates.values()), 4)
                                                               for g, v in gates.items()},
                bases={x: round(v / sum(bases.values()), 4) for x, v in bases.items()},
                products=(prods / n).round(4).tolist())


def cmd_labcheck(a):
    from duet.net import train_cl
    ds = [int(x) for x in a.ds.split(",")]
    jobs = [(pu, d, K, L, i) for pu in ("dev", "test") for d in ds for K in KS for L in (1, 2, 4, 8, 12, 16)
            for i in range(a.per)]
    bad, n = 0, 0
    with mp.get_context("fork").Pool(a.workers) as pool:
        for r in pool.imap_unordered(_lab_job, jobs, chunksize=1):
            bad += r["bad"]
            n += 1
            if r["bad"] or n % 50 == 0:
                print(json.dumps(r), flush=True)
        print(json.dumps(dict(label_check_circuits=n, violations=bad)), flush=True)
        hb = 0
        for r in pool.imap(_hand_job, [(d, h) for d in ds for h in HAND]):
            hb += (not r["products_ok"]) + (not r["noiseless_ok"]) + (not r["det_count_ok"]) + (not r["values_ok"])
            print(json.dumps(r), flush=True)
        print(json.dumps(dict(hand_circuits=len(ds) * len(HAND), hand_violations=hb)), flush=True)
    # the law against the trainer's own draw (train_cl.draw: no cnot or ent share, q from P_SET) at the training depths
    for K in KS:
        for L in (4, 8):
            ours = law_census(lambda g: cc.sample_circuit_k(g, K, L, P_SET[int(g.integers(4))], GATES), a.n, K, L,
                              [FIRST_TAG, 9, K, L])
            theirs = law_census(lambda g: train_cl.draw(g, L, cc.P_CNOT, GATES, 0.0, P_SET, K, 0.0), a.n, K, L,
                                [FIRST_TAG, 8, K, L])
            print(json.dumps(dict(law=True, K=K, L=L, n=a.n, ours=ours, train_cl_draw=theirs)), flush=True)


# ---------------------------------------------------------------------------------------------------------- score

def cmd_score(a):
    m = load_manifest(a.manifest)
    rows = []
    for pt in select(m, a.only):
        path = set_path(a.dir, pt)
        assert os.path.exists(path), (path, "a point's set is missing: no point is skipped")
        z = np.load(path)
        O, k = bs.packed(path, "obs"), z["k"]
        r = dict(point=pt["name"], d=pt["d"], K=pt["K"], L=pt["L"], p=pt["p"], circuits=len(pt["circuits"]),
                 shots=pt["shots"])
        for dec_ in ("lomr", "cain", "bplomr"):
            got = load_dec(a.dec, dec_, pt)
            want = np.arange(pt["bplomr_shots"] if dec_ == "bplomr" else pt["shots"])
            nf, ns, sec, bad = 0, 0, 0.0, []
            for i in range(len(pt["circuits"])):          # the prescribed coverage
                if i not in got:
                    bad.append(f"c{i}: no part")
                    continue
                idx, pred, ok, s = got[i]
                o = np.argsort(idx, kind="stable")
                if not np.array_equal(idx[o], want) or not ok.all():
                    bad.append(f"c{i}: {len(idx)} shots, {len(np.unique(idx))} distinct, {int((~ok).sum())} not ok")
                    continue
                y = np.packbits(bs.unpack(O[i][idx], k[i]), axis=1, bitorder="little")[:, 0]
                nf += int((pred != y).sum())
                ns += len(idx)
                sec += float(s.sum())
            r[dec_] = dict(shots=ns, fails=nf, core_s=round(sec, 1), complete=not bad, problems=bad[:3],
                           incomplete_circuits=len(bad))
        rows.append(r)
        print(json.dumps(r), flush=True)
    inc = [(r["point"], d_) for r in rows for d_ in ("lomr", "cain", "bplomr") if not r[d_]["complete"]]
    json.dump(dict(complete=not inc, incomplete=inc, rows=rows), open(a.out, "w"), indent=0)
    assert not inc, (f"{len(inc)} (point, decoder) pairs lack the prescribed coverage", inc[:6])


# ----------------------------------------------------------------------------------------------------------- main

# shots per circuit by (L, p), the same at every d and K
SIZE = {"test": {(1, 0.003): 100000, (2, 0.003): 60000, (4, 0.003): 21000, (8, 0.003): 8000, (12, 0.003): 4500,
                 (16, 0.003): 2800, (8, 0.001): 16000, (8, 0.002): 16000}}
SIZE["dev"] = {k: v // 4 for k, v in SIZE["test"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["manifest", "gen", "dec", "net", "sel", "tess", "probe", "labcheck", "score"])
    ap.add_argument("--purpose", default="dev")
    ap.add_argument("--out", default="")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--dir", default="")
    ap.add_argument("--only", default="")
    ap.add_argument("--part", default="")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("NCPU", 4)))
    ap.add_argument("--decoder", default="lomr", choices=["lomr", "cain", "bplomr"])
    ap.add_argument("--max-shots", type=int, default=0)
    ap.add_argument("--name", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--net", default="")
    ap.add_argument("--lists", default="")
    ap.add_argument("--deadline", default="none", help="tess: none, L<c> (c x L s), B<c> (2 c L / F: --basis, --cal), "
                    "req (the lists' per-shot choices: score.py tlist) or seconds")
    ap.add_argument("--basis", default="", help="tess --deadline B<c>: grace or spark")
    ap.add_argument("--cal", default="", help="tess --deadline B<c>: the calibration's directory")
    ap.add_argument("--fac", default="", help="tess --deadline B<c>: route.py's extra factors")
    ap.add_argument("--dec", default="")
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--Ls", default="1,2,4,8,12,16")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--circuits", type=int, default=4)
    ap.add_argument("--ntess", type=int, default=6)
    ap.add_argument("--nbp", type=int, default=40)
    ap.add_argument("--parts", default="dec,tess")
    ap.add_argument("--ds", default="3,5")
    ap.add_argument("--per", type=int, default=4)
    a = ap.parse_args()
    dict(manifest=cmd_manifest, gen=cmd_gen, dec=cmd_dec, net=cmd_net, sel=cmd_sel, tess=cmd_tess, probe=cmd_probe,
         labcheck=cmd_labcheck, score=cmd_score)[a.mode](a)


if __name__ == "__main__":
    main()
