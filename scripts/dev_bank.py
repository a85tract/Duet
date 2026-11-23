"""The development bank of Benchmark II: the open benchmark's lukin family under the Lukin model at
d = 3, 5, 7 and D = 1-10, on other logical circuits than the test set's: random_logical(D, i) for i = 20-39 (the test
set has i = 0-19), every choice of the run made on it.

  python scripts/dev_bank.py gen --bank B [--ds 3,5,7] [--workers W] sample and store every point (B/sets/<point>.npz,
                                                                         duet.bench_sets' format)
  python scripts/dev_bank.py dec --bank B --decoder lomr|tess|mle [--ds..] [--only RE] [--select FILE] [--max-shots N]
                               [--tag T] [--workers W] [--reverse] decode stored shots: B/dec/<decoder>[_T]/<point>/
                                                                         c<i>_<first>.npz (shot indices, predicted flips
                                                                         as bits of one byte, certified, seconds per shot)
  python scripts/dev_bank.py net --bank B --net NAME=CKPT [--frac F] a network's per-shot logits in fp32 (TF32 off;
                                                                         the frozen forward of scripts/bench_net.py):
                                                                         B/net/NAME/<point>.npz, (circuits, shots, 4)
                                                                         float32, NaN beyond the circuit's products
Decoders: lomr (LoMatching on Definition-1 regions, duet.bench_decode), tess, mle. --select FILE decodes only the listed shots ({point: {circuit: [shot...]}}), --max-shots N the first N
shots of each circuit (timing samples). Parts that exist are skipped, so a job resumes.

Shots per circuit (SHOTS): about F target-parity failures of the network per depth at the benchmark's source networks'
rates: F = 700 at
d = 3 and 5 (a 10 % difference per depth resolved unpaired to about 11 %, paired better), and at d = 7 F = 60 at
D = 1-4 and 180 at D = 5-10 (the selection score, target failures with each depth weighted equally, resolves 10 %
unpaired). Seeds: range [BANK_SEED_BASE, BANK_SEED_BASE + 10^15) = [1.8445e19, 1.8446e19), above 
[1.8443e19, 1.8445e19) and every earlier range (EARLIER; tests/test_dev_bank.py); circuit i of depth D at distance d samples
bank of our own families, D its variant; g = 3: the fresh d = 7 test set, BANK_SET=fresh7, fresh_points)."""
import argparse
import glob
import importlib.util
import json
import math
import multiprocessing as mp
import os
import re
import time

import numpy as np

from duet import bench_sets as bs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANK_SEED_BASE, BANK_SEED_SPAN = 18_445_000_000_000_000_000, 10 ** 15
EARLIER = {   # seed ranges of training and of earlier sets: new sets must not overlap them
    "training and monitors (seeds mod 2^63)": (0, 2 ** 63),
    "earlier range 1": (10 ** 19, 10 ** 19 + 10 ** 16),
    "earlier range 2": (12_340_000_000_000_000_000, 12_350_000_000_000_000_000),
    "earlier range 3": (13 * 10 ** 18, 13 * 10 ** 18 + 10 ** 17),
    "earlier range 4": (13_100_000_000_000_000_000, 13_110_000_000_000_000_000),
    "earlier range 5": (14 * 10 ** 18, 14 * 10 ** 18 + 10 ** 16),
    "earlier range 6": (14_200_000_000_000_000_000, 14_210_000_000_000_000_000),
    "earlier range 7": (15 * 10 ** 18, 15 * 10 ** 18 + 10 ** 17),
    "earlier range 8": (16_400_000_000_000_000_000, 16_410_000_000_000_000_000),
    "earlier range 9": (17_500_000_000_000_000_000, 17_510_000_000_000_000_000),
    "earlier range 10": (18 * 10 ** 18, 18 * 10 ** 18 + 10 ** 16),
    "earlier range 11": (18_200_000_000_000_000_000, 18_210_000_000_000_000_000),
    "earlier range 12": (18_400_000_000_000_000_000, 18_400_010_000_000_000_000),
    "open benchmark (duet.bench_sets)": (18_410_000_000_000_000_000, 18_420_000_000_000_000_000),
    "earlier range 13": (18_420_000_000_000_000_000, 18_430_000_000_000_000_000),
    "earlier range 14": (18_430_000_000_000_000_000, 18_440_000_000_000_000_000),
    "earlier range 15": (18_442_000_000_000_000_000, 18_443_000_000_000_000_000),
    "earlier range 16": (18_443_000_000_000_000_000, 18_445_000_000_000_000_000),
}
INDEX = tuple(range(20, 40))      # random_logical's circuit indices (the test set: 0-19)
DEPTHS = tuple(range(1, 11))
SHOTS = {   # shots per circuit, D = 1-10 (module docstring)
    3: (15000, 5600, 3900, 2600, 1900, 1500, 1200, 1100, 800, 800),
    5: (160700, 42500, 25300, 15200, 11500, 8100, 6900, 5900, 5100, 4400),
    7: (95500, 24600, 11700, 6800, 13700, 10500, 7000, 6900, 4500, 4800),
}
CHUNK = {"lomr": {3: 20000, 5: 20000, 7: 20000}, "tess": {3: 4000, 5: 1000, 7: 300}, "mle": {3: 200, 5: 100, 7: 50}}


def seed(g, d, D, i):
    assert g in (1, 2, 3, 9) and 0 < d < 100 and 0 < D < 1000 and 0 <= i < 10 ** 9, (g, d, D, i)
    s = BANK_SEED_BASE + g * 10 ** 14 + d * 10 ** 12 + D * 10 ** 9 + i
    assert BANK_SEED_BASE <= s < BANK_SEED_BASE + BANK_SEED_SPAN < 2 ** 64
    return s


TEST = os.environ.get("BANK_SET", "bank") == "test"   # the benchmark's test points instead
FRESH = os.environ.get("BANK_SET", "bank") in ("fresh7", "first_bench2", "fresh_bench2")   # a fresh test set
FIRST_BENCH2 = os.environ.get("BANK_SET", "bank") in ("first_bench2", "fresh_bench2")   # fresh main-result-2 test set at d = 3, 5, 7
# (first_bench2_points); fresh_bench2: re-test set (data/fresh_test/MANIFEST_bench2.json: the same points, new noise seeds)


def test_points(ds=(3, 5, 7)):
    """The open benchmark's lukin-family test points under the Lukin model, one per (d, D), in the bank's order."""
    with open(os.path.join(ROOT, "data", "open_benchmark", "MANIFEST.json")) as f:
        pts = [pt for pt in json.load(f)["points"] if pt["family"] == "lukin" and pt["noise"] == "lukin" and pt["d"] in ds]
    return sorted(pts, key=lambda pt: (ds.index(pt["d"]), pt["depth"]))


def fresh_points(ds=(7,)):
    """A fresh test set: test_points' entries (the benchmark's circuits
    i = 0-19, its shots per circuit and MLE prefix) with new stim seeds seed(3, d, D, i), sampled by the first test
    set's generator (duet.bench_sets._point, which made the release's sets); only d = 7."""
    assert tuple(ds) == (7,), ds
    out = []
    for pt in test_points(tuple(ds)):
        cs = [dict(c, seed=str(seed(3, pt["d"], pt["depth"], int(c["index"])))) for c in pt["circuits"]]
        out.append(dict(pt, circuits=cs, seeds=[cs[0]["seed"], str(int(cs[-1]["seed"]) + 1)]))   # [first, last + 1]
    return out


def decode_timed(dec, dets):
    """Decode shot by shot; return predictions (shots x obs) and per-shot wall times in microseconds."""
    preds, times = [], np.empty(len(dets), np.float64)
    clock = time.perf_counter_ns
    for i, row in enumerate(dets):
        t0 = clock()
        preds.append(dec.decode(row))
        times[i] = (clock() - t0) / 1e3
    return np.array(preds, dtype=bool).reshape(len(dets), -1), times


def first_bench2_points(ds=(3, 5, 7)):
    """fresh test set for main result 2 (data/first_test/MANIFEST_bench2.json; scripts/eval_pipeline.py bench2_points): the
    benchmark's lukin points at d = 3, 5, 7 with its shots and MLE prefix, new seeds."""
    run = "fresh_test" if os.environ.get("BANK_SET") == "fresh_bench2" else "first_test"
    with open(os.path.join(ROOT, "data", run, "MANIFEST_bench2.json")) as f:
        pts = [pt for pt in json.load(f)["points"] if pt["d"] in ds]
    return sorted(pts, key=lambda pt: (ds.index(pt["d"]), pt["depth"]))


def points(ds=(3, 5, 7)):
    """The bank's points (duet.bench_sets' manifest entries): one per (d, D), its 20 circuits i = 20-39; with BANK_SET=test
    the test set's (test_points)."""
    if TEST:
        return test_points(tuple(ds))
    if FIRST_BENCH2:
        return first_bench2_points(tuple(ds))
    if FRESH:
        return fresh_points(tuple(ds))
    pts = []
    for d in ds:
        for D in DEPTHS:
            cs = [bs._c(*bs.lukin(D, i), index=i, target_slot=0, seed=str(seed(1, d, D, i))) for i in INDEX]
            pts.append(dict(name=f"lukin_D{D}_d{d}", family="lukin", d=d, noise="lukin", p=0.003, depth=D, L=2 * D, K=4,
                            shots=SHOTS[d][D - 1], circuits=cs))
    return pts


def select(a):
    rx = re.compile(a.only) if a.only else None
    return [pt for pt in points(tuple(int(x) for x in a.ds.split(","))) if rx is None or rx.search(pt["name"])]


def set_path(bank, pt):
    return os.path.join(bank, "sets", pt["name"] + ".npz")


def bench_net():
    """scripts/bench_net.py as a module (load_net, frozen, BATCH)."""
    spec = importlib.util.spec_from_file_location("bench_net", os.path.join(ROOT, "scripts", "bench_net.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------------------------ gen

def gen(a):
    os.makedirs(os.path.join(a.bank, "sets"), exist_ok=True)
    todo = [(pt, os.path.join(a.bank, "sets")) for pt in select(a) if not os.path.exists(set_path(a.bank, pt))]
    t0, rec = time.time(), {}
    with mp.get_context("fork").Pool(a.workers) as pool:
        for name, sha in pool.imap_unordered(bs._point, sorted(todo, key=lambda j: -j[0]["shots"] * j[0]["L"])):
            rec[name] = sha
            print(json.dumps(dict(point=name, sha256=sha, minutes=round((time.time() - t0) / 60, 2))), flush=True)
    path = os.path.join(a.bank, "sets", "sha256.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    json.dump(dict(old, **rec), open(path, "w"), indent=0)


# ------------------------------------------------------------------------------------------------------------ dec

def _dec_job(job):
    """Decode shots idx of circuit i of a stored point; write the part file."""
    name, path, i, idx, out = job
    if os.path.exists(out):                             # made meanwhile by a second job on the same parts
        return out, 0, 0.0
    from duet import bench_decode as bd
    z = np.load(path)
    entry, m = json.loads(str(z["entry"])), int(z["m"][i])
    c_ = entry["circuits"][i]
    rows = bs.packed(path, "dets")[i]
    dets = bs.unpack(rows[idx[0]:idx[-1] + 1] if np.array_equal(idx, np.arange(idx[0], idx[-1] + 1)) else rows[idx], m)
    t0, extra = time.time(), {}
    if name == "tess":
        from duet.decoders.tesseract_dec import Tesseract
        dec = Tesseract(bs.circuit(entry, c_))
        build = time.time() - t0
        pred, us = decode_timed(dec, dets)
        ok, sec, settings = np.ones(len(idx), bool), us / 1e6, dec.settings
    else:
        c, dec = bd.build(name, entry, c_)
        build = time.time() - t0
        if name == "mle":
            pred, ok, sec = dec.decode_checked(dets)
            xs = [np.zeros(0, np.int64) if x is None else np.asarray(x, np.int64) for x in dec.xs]
            extra = dict(x=np.concatenate(xs), x_off=np.cumsum([0] + [len(x) for x in xs]),
                         weight=np.array([np.nan if x is None else dec.D.weight(x) for x in dec.xs]))
        else:
            pred, ok, s = bd.decode(name, dec, dets)
            sec = np.full(len(idx), s / len(idx))
        settings = getattr(dec, "settings", {})
    assert pred.shape[1] <= 8
    pk = np.packbits(pred, axis=1, bitorder="little")[:, 0] if pred.shape[1] else np.zeros(len(idx), np.uint8)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out + ".tmp.npz", idx=np.asarray(idx, np.int64), pred=pk, ok=ok, sec=np.asarray(sec, np.float32),
             build=build, settings=json.dumps(settings, default=str), **extra)
    os.replace(out + ".tmp.npz", out)
    return out, len(idx), float(np.sum(sec))


def dec(a):
    sel = json.load(open(a.select)) if a.select else None
    jobs = []
    for pt in select(a):
        path = set_path(a.bank, pt)
        base = os.path.join(a.bank, "dec", a.decoder + (f"_{a.tag}" if a.tag else ""), pt["name"])
        ch = CHUNK[a.decoder][pt["d"]]
        for i in range(len(pt["circuits"])):
            if sel is not None:
                idx = np.array(sorted(sel.get(pt["name"], {}).get(str(i), [])), np.int64)
            else:
                idx = np.arange(min(pt["shots"], a.max_shots or pt["shots"]))
            for s in range(0, len(idx), ch):
                out = os.path.join(base, f"c{i}_{int(idx[s])}.npz")
                if not os.path.exists(out):
                    jobs.append((a.decoder, path, i, idx[s:s + ch], out, pt["L"] * len(idx[s:s + ch]) * pt["d"] ** 3))
    jobs.sort(key=lambda j: -j[-1])                     # the largest first, so the pool ends together
    if a.reverse:                                       # a second job on the same parts, from the other end
        jobs.reverse()
    t0, n, cpu = time.time(), 0, 0.0
    print(json.dumps(dict(decoder=a.decoder, jobs=len(jobs), shots=int(sum(len(j[3]) for j in jobs)))), flush=True)
    with mp.get_context("fork").Pool(a.workers) as pool:
        for k, (_, ns, s) in enumerate(pool.imap_unordered(_dec_job, [j[:5] for j in jobs])):
            n, cpu = n + ns, cpu + s
            if (k + 1) % max(1, len(jobs) // 20) == 0 or k + 1 == len(jobs):
                print(json.dumps(dict(done=k + 1, shots=n, decode_core_h=round(cpu / 3600, 3),
                                      minutes=round((time.time() - t0) / 60, 1))), flush=True)


def load_dec(bank, decoder, pt):
    """{circuit: (shot indices, predicted flips packed in a byte, certified, seconds)} of a decoder's parts."""
    out = {}
    for f in glob.glob(os.path.join(bank, "dec", decoder, pt["name"], "c*_*.npz")):
        if f.endswith(".tmp.npz"):     # a part a killed job left unfinished (np.savez to <part>.tmp.npz, rename)
            continue
        i = int(os.path.basename(f)[1:].split("_")[0])
        z = np.load(f)
        out.setdefault(i, []).append((z["idx"], z["pred"], z["ok"], z["sec"]))
    return {i: tuple(np.concatenate([x[j] for x in v]) for j in range(4)) for i, v in out.items()}


# ------------------------------------------------------------------------------------------------------------ net

def net(a):
    import torch
    from duet.net import train_cl
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    bn = bench_net()
    name, _, path = a.net.partition("=")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    d = ck["cfg"]["d"]
    model, rec = bn.load_net("lukin", d, "cuda", path)
    rec.update(gpu=torch.cuda.get_device_name(0), code=os.environ.get("CODE"), frac=a.frac)
    a.ds = str(d)
    os.makedirs(os.path.join(a.bank, "net", name), exist_ok=True)
    for pt in select(a):
        out = os.path.join(a.bank, "net", name, pt["name"] + ".npz")
        if os.path.exists(out):
            continue
        n = int(math.ceil(a.frac * pt["shots"]))
        D = bs.packed(set_path(a.bank, pt), "dets")
        lg = np.full((len(pt["circuits"]), n, 4), np.nan, np.float32)
        secs, t0 = [], time.time()
        for i, c_ in enumerate(pt["circuits"]):
            tab = train_cl.tables(d, c_["layers"].split(), tuple(c_["prep"]), tuple(c_["meas"]), pt["p"], pt["noise"],
                                  [list(s) for s in c_["products"]])[0]
            torch.cuda.synchronize()
            t1 = time.time()
            c, B = bn.frozen(model, tab, np.ascontiguousarray(D[i, :n, :(tab["m"] + 7) // 8]), d)
            torch.cuda.synchronize()
            secs.append(time.time() - t1)
            lg[i, :, :c.shape[1]] = c
        np.savez(out + ".tmp.npz", logits=lg, secs=np.array(secs), batch=B, info=json.dumps(rec))
        os.replace(out + ".tmp.npz", out)
        print(json.dumps(dict(point=pt["name"], shots=n * len(pt["circuits"]), seconds=round(time.time() - t0, 1),
                              read_us_per_shot=round(1e6 * sum(secs) / (n * len(pt["circuits"])), 1))), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gen", "dec", "net"])
    ap.add_argument("--bank", required=True)
    ap.add_argument("--ds", default="3,5,7")
    ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("NCPU", 4)))
    ap.add_argument("--decoder", default="lomr", choices=["lomr", "tess", "mle"])
    ap.add_argument("--select", default="")
    ap.add_argument("--max-shots", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--reverse", action="store_true", help="dec: the smallest parts first (a second job,)")
    ap.add_argument("--net", default="")
    ap.add_argument("--frac", type=float, default=1.0)
    a = ap.parse_args()
    dict(gen=gen, dec=dec, net=net)[a.mode](a)


if __name__ == "__main__":
    main()
