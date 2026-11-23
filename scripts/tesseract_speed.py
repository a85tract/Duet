"""Speed study of the pair (network plus Tesseract): Tesseract settings from the long beam
down, a per-shot time cap (the network's answer kept when the cap is hit or Tesseract gives up), f = 0.5, 1 and 2 % on
the routed shots of the d = 5 and d = 7 development bank, the
network's timed path, and the pair's time per shot one shot at a time and in batch mode.

  python scripts/tesseract_speed.py sel --bank B shot lists sel/speed_d{5,7}.json (the shots s_T routes
                                                               at f = 2 % and the MLE's routed sample) and the check's
                                                               sample speed_chk_d{5,7}.json (CHK routed shots per depth)
  python scripts/tesseract_speed.py tess --bank B --settings S,S --what full|chk [--ds 7,5] [--workers W]
                                                               decode the lists with each setting into B/dec/tess_<S>/
                                                               (full) or tess_<S>_chk/ (bank.py dec's parts, plus the
                                                               low-confidence flag as `ok` = False); one pool, largest first
  python scripts/tesseract_speed.py chk --bank B --settings S,S the check: time per routed shot and agreement with the
                                                               long beam (and the MLE where it ran) per setting
  python scripts/tesseract_speed.py net --bank B --out J [--ds 5,7] [--modes precise,onepass]
                                                               the network in the Triton engine (duet.net.engine_cl, one
                                                               CUDA graph per batch), host memory to host memory, two
                                                               circuits per (d, D); its decisions against the fp32 read
  python scripts/tesseract_speed.py rep --bank B --times J --settings S,S [--out J] tables

SPEED_NET=NAME (default ft4) runs sel, tess and rep for another network's read (B/net/NAME): lists speed_NAME_d*.json,
and tess decodes only the listed shots that no part of the setting covers yet, into parts c<i>_<first>_NAME.npz beside
ft4's; SPEED_DS=5 restricts sel and rep to the distances that network has. SPEED_TSCALE='{"5|tess": x}'
multiplies a decoder's seconds per shot at that d in rep (times converted to another CPU; the cap applies to them).

Times: Tesseract's wall time per shot on one core with every core of the node busy (bank.py dec's measure); the
network's median wall time per call of B shots. Stream mode (one GPU and one core, shots one at a time): a shot takes
the network's batch-1 call, and a routed shot its Tesseract time on top (capped). Batch mode (B shots, N cores): the
batch's wall time is the network's call of B shots plus the routed shots' Tesseract times spread over N cores in
arrival order (each to the first free core), time per shot = wall / B; batches are a depth's shots in a random order
(numpy default_rng(190 + D)), at least MINB batches per depth (shots drawn again once all are used). Means per shot are
taken per depth and averaged with each depth weighted equally (as the selection score)."""
import argparse
import importlib.util
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _mod(rel, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = sys.modules[name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


report = _mod("scripts/bank_report.py", "lukin_report")
bank = report.bank
bs = report.bs

NET = os.environ.get("SPEED_NET", "ft4")      # another network's pair: its own lists, its parts beside ft4's
DS = tuple(int(x) for x in os.environ.get("SPEED_DS", "5,7").split(","))
LIST = "speed" if NET == "ft4" else f"speed_{NET}"
FS = (0.005, 0.01, 0.02)
FS_LONG = (0.005, 0.01, 0.02, 0.05, 0.10)      # the original long-beam decode (every shot) also at / f
CAPS = (None, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01)  # seconds per routed shot
BS = tuple(int(x) for x in os.environ.get("SPEED_BS", "1024,4096,16384,65536").split(","))  # batch modes (190 (4): 256)
NS = tuple(int(x) for x in os.environ.get("SPEED_NS", "1,16,64").split(","))
MINB = 16
TSCALE = json.loads(os.environ.get("SPEED_TSCALE", "{}"))   #: {"d|decoder": factor} on Tesseract's seconds
CHK = 30
THEIR_B = {5: {"1-10": 1.34}, 7: {"1-8": 1.65, "1-10": 1.96}}   # theirs / their MLE
SETTINGS = {   # overrides of duet.decoders.tesseract_dec.SETTINGS (beam 20, climbing, 21 BFS orders, pqlimit 1e6)
    "long": {},
    "o7": dict(num_det_orders=7),
    "o3": dict(num_det_orders=3),
    "o1": dict(num_det_orders=1),
    "b10o1": dict(det_beam=10, num_det_orders=1),
    "b5o1": dict(det_beam=5, num_det_orders=1),
    "b5o3": dict(det_beam=5, num_det_orders=3),
    "o1nc": dict(num_det_orders=1, beam_climbing=False),
    "b5o1nc": dict(det_beam=5, num_det_orders=1, beam_climbing=False),
    "o1q5": dict(num_det_orders=1, pqlimit=100_000),
    "o1q4": dict(num_det_orders=1, pqlimit=10_000),
    "b5o1q4": dict(det_beam=5, num_det_orders=1, pqlimit=10_000),
    "b3o1ncq4": dict(det_beam=3, num_det_orders=1, beam_climbing=False, pqlimit=10_000),
}
COST = {"long": 21, "o7": 7, "o3": 3, "b5o3": 2}   # rough cost per shot against o1 (largest jobs first)


# ------------------------------------------------------------------------------------------------------------ sel

def _flat_to_sel(out, name, idx, n):
    for f in idx:
        out.setdefault(name, {}).setdefault(str(int(f) // n), []).append(int(f) % n)


def seldir(bank_):
    return os.path.join(os.path.dirname(os.path.abspath(bank_)), "sel")       # sel, beside bank


def sel(a):
    rng = np.random.default_rng(190)
    for d in DS:
        mle = json.load(open(os.path.join(seldir(a.bank), f"routed_d{d}.json")))
        full, chk, n_full = {}, {}, 0
        for pt in bank.points((d,)):
            r = report.load(a.bank, pt, NET)
            o, _ = report.routed(np.abs(r["logits"][:, 0]), max(FS))
            extra = [int(c) * r["n"] + s for c, ss in mle.get(pt["name"], {}).items() for s in ss]
            idx = np.union1d(o, np.array(extra, np.int64))
            _flat_to_sel(full, pt["name"], idx, r["n"])
            _flat_to_sel(chk, pt["name"], np.sort(rng.choice(o, min(CHK, len(o)), replace=False)), r["n"])
            n_full += len(idx)
        for tag, x in ((LIST, full), (LIST + "_chk", chk)):
            json.dump(x, open(os.path.join(seldir(a.bank), f"{tag}_d{d}.json"), "w"))
        print(json.dumps(dict(d=d, full=n_full, chk=sum(len(v) for c in chk.values() for v in c.values()))), flush=True)


# ------------------------------------------------------------------------------------------------------------ tess

def _tess_job(job):
    """Decode shots idx of circuit i of a stored point with Tesseract settings opts; write bank.py dec's part (plus
    ok = False where the decoder raised its low-confidence flag: no solution within its limits)."""
    path, i, idx, out, opts = job
    from duet.decoders.tesseract_dec import Tesseract
    z = np.load(path)
    entry, m = json.loads(str(z["entry"])), int(z["m"][i])
    rows = bs.packed(path, "dets")[i]
    dets = bs.unpack(rows[idx], m)
    t0 = time.time()
    dec = Tesseract(bs.circuit(entry, entry["circuits"][i]), **opts)
    build = time.time() - t0
    preds, sec, ok = [], np.empty(len(idx), np.float64), np.ones(len(idx), bool)
    clock = time.perf_counter_ns
    for k, row in enumerate(dets):
        t1 = clock()
        preds.append(dec.dec.decode(row))
        sec[k] = (clock() - t1) / 1e9
        ok[k] = not bool(getattr(dec.dec, "low_confidence_flag", False))
    pred = np.array(preds, dtype=bool).reshape(len(idx), -1)
    pk = np.packbits(pred, axis=1, bitorder="little")[:, 0] if pred.shape[1] else np.zeros(len(idx), np.uint8)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out + ".tmp.npz", idx=np.asarray(idx, np.int64), pred=pk, ok=ok, sec=sec.astype(np.float32), build=build,
             settings=json.dumps(dec.settings, default=str))
    os.replace(out + ".tmp.npz", out)
    return out, len(idx), float(sec.sum()), int((~ok).sum())


def tess(a):
    jobs = []
    for s in a.settings.split(","):
        opts = SETTINGS[s]
        for d in (int(x) for x in a.ds.split(",")):
            lst = json.load(open(os.path.join(seldir(a.bank), LIST + ("_chk" if a.what == "chk" else "") + f"_d{d}.json")))
            ch = bank.CHUNK["tess"][d]
            for pt in bank.points((d,)):
                dname = f"tess_{s}" + ("_chk" if a.what == "chk" else "")
                base = os.path.join(a.bank, "dec", dname, pt["name"])
                have = {} if NET == "ft4" else {i: set(v[0].tolist()) for i, v in bank.load_dec(a.bank, dname, pt).items()}
                for c, ss in lst.get(pt["name"], {}).items():
                    idx = np.array(sorted(set(ss) - have.get(int(c), set())), np.int64)   # another net: new shots only
                    for k in range(0, len(idx), ch):
                        out = os.path.join(base, f"c{c}_{int(idx[k])}" + ("" if NET == "ft4" else f"_{NET}") + ".npz")
                        if not os.path.exists(out):
                            cost = pt["L"] * len(idx[k:k + ch]) * d ** 3 * COST.get(s, 1)
                            jobs.append((bank.set_path(a.bank, pt), int(c), idx[k:k + ch], out, opts, cost))
    jobs.sort(key=lambda j: -j[-1])
    print(json.dumps(dict(settings=a.settings, what=a.what, jobs=len(jobs), shots=int(sum(len(j[2]) for j in jobs)))),
          flush=True)
    t0, n, cpu, low = time.time(), 0, 0.0, 0
    with mp.get_context("fork").Pool(a.workers) as pool:
        for k, (_, ns, s, lc) in enumerate(pool.imap_unordered(_tess_job, [j[:5] for j in jobs])):
            n, cpu, low = n + ns, cpu + s, low + lc
            if (k + 1) % max(1, len(jobs) // 20) == 0 or k + 1 == len(jobs):
                print(json.dumps(dict(done=k + 1, shots=n, low_conf=low, decode_core_h=round(cpu / 3600, 3),
                                      minutes=round((time.time() - t0) / 60, 1))), flush=True)


# ------------------------------------------------------------------------------------------------------------ chk

def chk(a):
    from importlib.metadata import version
    from tesseract_decoder import tesseract, utils
    from duet.decoders.tesseract_dec import Tesseract
    print("tesseract-decoder", version("tesseract-decoder"))
    print("TesseractConfig:", [k for k in dir(tesseract.TesseractConfig) if not k.startswith("_")])
    print("DetOrder:", [k for k in dir(utils.DetOrder) if not k.startswith("_")])
    pt0 = bank.points((5,))[0]
    entry = json.loads(str(np.load(bank.set_path(a.bank, pt0))["entry"]))
    print("decoder:", [k for k in dir(Tesseract(bs.circuit(entry, entry["circuits"][0]), num_det_orders=1).dec)
                       if not k.startswith("_")])
    sets = a.settings.split(",")
    print("\n| d | setting | shots | low confidence | s per shot: mean / median / p99 / max | target differs from the "
          "long beam | some product differs | on MLE shots: target differs / shots |")
    print("|---|---|---|---|---|---|---|---|")
    for d in (int(x) for x in a.ds.split(",")):
        acc = {s: dict(sec=[], low=0, td=0, pd=0, mt=0, mn=0) for s in sets}
        for pt in bank.points((d,)):
            r = report.load(a.bank, pt, None, ["tess", "mle_routed"] + [f"tess_{s}_chk" for s in sets])
            ref, mle = r["tess"], r["mle_routed"]
            for s in sets:
                x, q = r[f"tess_{s}_chk"], acc[s]
                c = x["cov"]
                if not c.any():
                    continue
                q["sec"].append(x["sec"][c])
                q["low"] += int((c & ~x["ok"]).sum())
                diff = x["dec"] ^ ref["dec"]
                q["td"] += int((diff[c] & 1).astype(bool).sum())
                q["pd"] += int((diff[c] & r["mask"][c]).astype(bool).sum())
                cm = c & mle["cov"] & mle["ok"]
                q["mt"] += int(((x["dec"] ^ mle["dec"])[cm] & 1).astype(bool).sum())
                q["mn"] += int(cm.sum())
        for s in sets:
            q = acc[s]
            if not q["sec"]:
                continue
            t = np.concatenate(q["sec"])
            print(f"| {d} | {s} | {len(t)} | {q['low']} | {t.mean():.3f} / {np.median(t):.3f} / "
                  f"{np.percentile(t, 99):.2f} / {t.max():.2f} | {q['td']} | {q['pd']} | {q['mt']} / {q['mn']} |",
                  flush=True)


# ------------------------------------------------------------------------------------------------------------ net

def net(a):
    import torch
    from duet.net import engine_cl, spec_fast
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    bn = bank.bench_net()
    pr = torch.cuda.get_device_properties(0)
    rec = dict(gpu=torch.cuda.get_device_name(0), sm=pr.multi_processor_count, mem=pr.total_memory,
               code=os.environ.get("CODE"), rows=[], checks=[])
    Bs = [int(x) for x in a.batches.split(",")]
    for d in (int(x) for x in a.ds.split(",")):
        model, info = bn.load_net("lukin", d, "cuda")       # the ft4 network, its SHA-256 recorded
        rec[f"net_d{d}"] = info
        for pt in bank.points((d,)):
            path = bank.set_path(a.bank, pt)
            Dp = bs.packed(path, "dets")
            lg = np.load(os.path.join(a.bank, "net", NET, pt["name"] + ".npz"))["logits"]
            t1 = report.routed(np.abs(lg[:, :, 0]).reshape(-1), 0.01)[1]
            for i in range(a.circuits):
                c_ = pt["circuits"][i]
                lay, prep, meas = c_["layers"].split(), tuple(c_["prep"]), tuple(c_["meas"])
                prods = [list(s) for s in c_["products"]]
                for mode in a.modes.split(","):
                    eng = engine_cl.EngineCL(model, mode)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    s = spec_fast.build(d, lay, prep, meas, pt["p"], pt["noise"], prods)
                    t_spec = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    circ = eng.circuit(lay, prep, meas, pt["noise"], pt["p"], spec=s)
                    torch.cuda.synchronize()
                    t_circ = time.perf_counter() - t0
                    nb = int(circ.get("nb", (circ["m"] + 7) // 8))
                    det = np.ascontiguousarray(Dp[i, :, :nb])
                    nchk = min(len(det), a.check)
                    cE, _, dE = eng.decode(circ, det[:nchk], B=1024)
                    k = cE.shape[1]
                    ref = lg[i, :nchk, :k]
                    rec["checks"].append(dict(
                        d=d, D=pt["depth"], circuit=i, mode=mode, shots=nchk,
                        dec_differs=int(((ref > 0) != dE).any(1).sum()),
                        max_dlogit=float(np.abs(cE - ref).max()),
                        route_differs=int(((np.abs(ref[:, 0]) <= t1) != (np.abs(cE[:, 0]) <= t1)).sum())))
                    need = max(B if B <= 16384 else 4096 for B in Bs)
                    hin = torch.from_numpy(np.ascontiguousarray(np.resize(det, (max(need, len(det)), nb)))).pin_memory()
                    for B in Bs:
                        Bg = B if B <= 16384 else 4096           # larger batches: replays of the 4,096 graph
                        g, xin, out, dec = eng.graph(Bg)
                        x2 = xin[:Bg * nb].view(Bg, nb)
                        hdec = torch.zeros(Bg, dtype=torch.uint8).pin_memory()
                        hlog = torch.zeros(Bg, dtype=torch.float32).pin_memory()

                        def call(s0, Bg=Bg, x2=x2, g=g, out=out, dec=dec, hdec=hdec, hlog=hlog, reps=B // Bg):
                            for q in range(reps):
                                s = (s0 + q * Bg) % (len(hin) - Bg + 1)
                                x2.copy_(hin[s:s + Bg], non_blocking=True)
                                g.replay()
                                hdec.copy_(dec, non_blocking=True)
                                hlog.copy_(out[:, 0], non_blocking=True)
                            torch.cuda.synchronize()
                        nrep = max(8, min(64, len(hin) // Bg)) if B <= 16384 else 8
                        ts = []
                        for q in range(-3, nrep):
                            t0 = time.perf_counter()
                            call(max(q, 0) * Bg)
                            if q >= 0:
                                ts.append(time.perf_counter() - t0)
                        t = float(np.median(ts))
                        row = dict(d=d, D=pt["depth"], circuit=i, mode=mode, batch=B, call_us=round(t * 1e6, 2),
                                   us_per_shot=round(t / B * 1e6, 4), p90_call_us=round(float(np.percentile(ts, 90)) * 1e6, 2),
                                   spec_ms=round(1e3 * t_spec, 1), circuit_ms=round(1e3 * t_circ, 1))
                        rec["rows"].append(row)
                        print(json.dumps(row), flush=True)
                    print(json.dumps(rec["checks"][-1]), flush=True)
                    del eng
                    torch.cuda.empty_cache()
        json.dump(rec, open(a.out, "w"), indent=0)
    json.dump(rec, open(a.out, "w"), indent=0)


# ------------------------------------------------------------------------------------------------------------ rep

def greedy_makespan(T, N):
    """T: (batches, k) Tesseract times of each batch's routed shots in arrival order (0 = none) -> each batch's wall
    time with N cores, every shot to the first free core (list scheduling)."""
    if T.shape[1] == 0:
        return np.zeros(len(T))
    if N == 1:
        return T.sum(1)
    load = np.zeros((len(T), N))
    rows = np.arange(len(T))
    for j in range(T.shape[1]):
        c = load.argmin(1)
        load[rows, c] += T[:, j]
    return load.max(1)


def batches(seq_routed_pos, times, B, nb):
    """Pad each batch's routed times into a (nb, k) matrix in arrival order; seq_routed_pos: the routed shots'
    positions in the batch sequence (sorted), times their Tesseract times."""
    b = seq_routed_pos // B
    cnt = np.bincount(b, minlength=nb)
    k = int(cnt.max()) if len(cnt) else 0
    T = np.zeros((nb, k))
    first = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    T[b, np.arange(len(b)) - first[b]] = times
    return T


def wquantile(vals, w, q):
    o = np.argsort(vals, kind="stable")
    cw = np.cumsum(w[o])
    return float(vals[o][min(np.searchsorted(cw, q * cw[-1]), len(o) - 1)])


def net_times(path):
    """{(d, D, mode, B): median µs per shot over the circuits} of speed.py net."""
    rec = json.load(open(path))
    out = {}
    for r in rec["rows"]:
        out.setdefault((r["d"], r["D"], r["mode"], r["batch"]), []).append(r["us_per_shot"])
    return {k: float(np.median(v)) for k, v in out.items()}, rec


def rep(a):
    nt, nrec = net_times(a.times)
    th = report.theirs()
    sets = a.settings.split(",")
    combos, agree = {}, {}
    for d in DS:
        rs = [report.load(a.bank, pt, NET, ["tess", "mle_routed"] + [f"tess_{s}{a.suffix}" for s in sets])
              for pt in bank.points((d,))]
        for r in rs:                          # the check's parts (--suffix _chk) under the settings' names
            for s in sets:
                r[f"tess_{s}"] = r.pop(f"tess_{s}{a.suffix}")
        for r in rs:                          #: each time converted to another CPU (the cap then applies to it)
            for k, v in TSCALE.items():
                if int(k.split("|")[0]) == d:
                    r[k.split("|")[1]]["sec"] = r[k.split("|")[1]]["sec"] * v
        Ds = [r["D"] for r in rs]
        fn = [report.fails(r, r["dec_net"])[0] for r in rs]
        ftl = [report.fails(r, r["tess"]["dec"])[0] for r in rs]
        sT = [np.abs(r["logits"][:, 0]) for r in rs]

        def ntime(D, B):        # the network's seconds per shot at batch B (the largest timed batch at most B)
            Bk = max(b for b in [1] + list(BS) if b <= B and (d, D, a.mode, b) in nt)
            return nt[(d, D, a.mode, Bk)] * 1e-6
        seqs = {}
        for r, D in zip(rs, Ds):              # the batch sequences (shot orders) of each depth and batch size
            n = len(r["obs"])
            g = np.random.default_rng(190 + D)
            for B in BS:
                L = -(-max(n, MINB * B) // B) * B
                seqs[(D, B)] = np.concatenate([g.permutation(n) for _ in range(-(-L // n))])[:L]
        for s in sets:                        # agreement on every decoded shot: the long beam, the certified MLE
            q = dict(shots=0, low=0, t_long=0, p_long=0, mle_shots=0, t_mle=0, f_x=0, f_mle=0)
            for r in rs:
                x, ref, mle = r[f"tess_{s}"], r["tess"], r["mle_routed"]
                c = x["cov"]
                q["shots"] += int(c.sum())
                q["low"] += int((c & ~x["ok"]).sum())
                df = x["dec"] ^ ref["dec"]
                q["t_long"] += int((df[c] & 1).astype(bool).sum())
                q["p_long"] += int((df[c] & r["mask"][c]).astype(bool).sum())
                cm = c & mle["cov"] & mle["ok"]
                q["mle_shots"] += int(cm.sum())
                q["t_mle"] += int(((x["dec"] ^ mle["dec"])[cm] & 1).astype(bool).sum())
                q["f_x"] += int(report.fails(r, x["dec"])[0][cm].sum())
                q["f_mle"] += int(report.fails(r, mle["dec"])[0][cm].sum())
            agree[(d, s)] = q
        ref_pair = {}
        for dec_name in ["tess"] + [f"tess_{s}" for s in sets]:
            for f in (FS_LONG if dec_name == "tess" else FS):
                for cap in CAPS:
                    rows, per, miss = [], [], 0
                    for r, D, fnet, sc in zip(rs, Ds, fn, sT):
                        o, _ = report.routed(sc, f)
                        x = r[dec_name]
                        miss += int((~x["cov"][o]).sum())
                        sec = np.where(x["cov"][o], x["sec"][o], 0.0)
                        use = x["cov"][o] & x["ok"][o] & ((sec <= cap) if cap else True)
                        fp = fnet.copy()
                        fp[o] = np.where(use, report.fails(r, x["dec"])[0][o], fnet[o])
                        rows.append(fp)
                        per.append((D, len(fp), o, np.minimum(sec, cap) if cap else sec))
                    if cap is None and dec_name == "tess":
                        ref_pair[f] = rows
                    X, _, R, ci = report.ratio(list(zip(rows, fn)))
                    v = dict(score=X, ratio_net=R, ci_net=ci, missing=miss,
                             vs_long=report.ratio(list(zip(rows, ref_pair[f])))[2:],
                             published={k: list(x) for k, x in report.published(d, rs, rows, th).items()})
                    for lab, Dmax in (("1-10", 10), ("1-8", 8)):
                        v[f"b_{lab}"] = report.ratio([(p_, q_) for p_, q_, D in zip(rows, ftl, Ds) if D <= Dmax])[2:]
                    # stream mode: one GPU (batch-1 calls) and one core
                    means, vals, ws, mx = [], [], [], 0.0
                    for D, n, o, tt in per:
                        n1 = ntime(D, 1)
                        means.append(n1 + tt.sum() / n)
                        vals += [np.array([n1]), n1 + tt]
                        ws += [np.array([(n - len(tt)) / (len(per) * n)]), np.full(len(tt), 1 / (len(per) * n))]
                        mx = max(mx, n1 + (tt.max() if len(tt) else 0.0))
                    allt = np.concatenate([p[3] for p in per])
                    v["stream"] = dict(mean=float(np.mean(means)), p99=wquantile(np.concatenate(vals), np.concatenate(ws),
                                                                                  0.99), max=mx,
                                       tess_mean=float(allt.mean()) if len(allt) else 0.0,
                                       tess_p99=float(np.percentile(allt, 99)) if len(allt) else 0.0)
                    # batch mode: B shots through the network, the routed ones over N cores
                    acc = {(B, N): ([], []) for B in BS for N in NS}
                    for D, n, o, tt in per:
                        tfull = np.zeros(n)
                        tfull[o] = tt
                        rm = np.zeros(n, bool)
                        rm[o] = True
                        for B in BS:
                            seq = seqs[(D, B)]
                            pos = np.flatnonzero(rm[seq])
                            T = batches(pos, tfull[seq[pos]], B, len(seq) // B)
                            for N in NS:
                                wall = ntime(D, B) * B + greedy_makespan(T, N)
                                acc[(B, N)][0].append(wall.mean() / B)
                                acc[(B, N)][1].append(wall)
                    v["batch"] = {f"B{B}_N{N}": dict(mean=float(np.mean(m_)), lat_p99=float(np.percentile(np.concatenate(l_), 99)),
                                                     lat_max=float(np.concatenate(l_).max()),
                                                     lat_median=float(np.median(np.concatenate(l_))),   # 
                                                     per_D=[float(x) for x in m_], net_D=[ntime(D, B) for D, *_ in per])
                                  for (B, N), (m_, l_) in acc.items()}
                    combos[(d, dec_name, f, cap)] = v
            print(json.dumps(dict(d=d, decoder=dec_name, done=True)), flush=True)
    _write(a, combos, agree, nrec, sets)


def _key(k):
    d, dec, f, cap = k
    return f"d{d}|{dec}|{f}|{cap}"


def _write(a, combos, agree, nrec, sets):
    out = {_key(k): v for k, v in combos.items()}
    if a.out:
        json.dump(dict(combos=out, agree={f"d{k[0]}|{k[1]}": v for k, v in agree.items()}, net=nrec, tscale=TSCALE),
                  open(a.out, "w"), default=float)
    cap_s = lambda c: "-" if c is None else f"{c:g} s"  # noqa: E731
    # the network
    print("\n### The network in the engine (median µs per shot over two circuits per depth; batch 1: per call)\n")
    rows = nrec["rows"]
    for d in DS:
        for mode in sorted({r["mode"] for r in rows}):
            line = []
            for B in [1] + list(BS):
                v = [r["us_per_shot"] for r in rows if r["d"] == d and r["mode"] == mode and r["batch"] == B]
                line.append(f"{np.mean(v):.1f}" if v else "-")
            print(f"- d = {d}, {mode}: batch 1 / " + " / ".join(map(str, BS)) + ": " + " / ".join(line) +
                  " µs per shot (mean over depths)")
    ch = nrec["checks"]
    for d in DS:
        for mode in sorted({c["mode"] for c in ch}):
            cc = [c for c in ch if c["d"] == d and c["mode"] == mode]
            if cc:
                print(f"- check d = {d}, {mode}: {sum(c['shots'] for c in cc):,} shots, decisions differ on "
                      f"{sum(c['dec_differs'] for c in cc)}, routing (f = 1 %) on {sum(c['route_differs'] for c in cc)}, "
                      f"max |logit difference| {max(c['max_dlogit'] for c in cc):.2e}")
    # settings at f = 1 %, no cap
    for d in DS:
        print(f"\n### d = {d}: Tesseract settings on the routed shots (f = 1 %, no cap)\n")
        print("| decoder | pair score | pair / network | pair / long beam | (a) pair / their network | (b) 1-10 | "
              "s per routed shot: mean / p99 | stream ms per shot: mean / p99 / max | missing | low confidence / "
              "decoded | target / product differs from the long beam | MLE shots: target differs, failures x / MLE |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for dec in ["tess"] + [f"tess_{s}" for s in sets]:
            v = combos.get((d, dec, 0.01, None))
            if v is None:
                continue
            pub = v["published"].get("their network")
            st = v["stream"]
            vl = v.get("vs_long", (None, None))
            print(f"| {dec} | {v['score']:.3e} | {report.fmt(v['ratio_net'], v['ci_net'])} | {report.fmt(*vl)} | "
                  f"{report.fmt(pub[3], pub[4]) if pub else '-'} | {report.fmt(*v['b_1-10'])} | {st['tess_mean']:.3f} / "
                  f"{st['tess_p99']:.2f} | {1e3 * st['mean']:.2f} / {1e3 * st['p99']:.1f} / {1e3 * st['max']:.0f} | "
                  f"{v['missing']} | " + (lambda q: "-" if q is None else
                  f"{q['low']} / {q['shots']:,} | {q['t_long']} / {q['p_long']} | {q['t_mle']} / {q['mle_shots']:,}, "
                  f"{q['f_x']} / {q['f_mle']}")(agree.get((d, dec[5:]))) + " |")
    # Pareto sets (stream mean time, pair score) and the reported setting
    for d in DS:
        pts = sorted(((v["stream"]["mean"], v["score"], k) for k, v in combos.items() if k[0] == d and k[1] != "tess"),
                     key=lambda x: (x[0], x[1]))
        par, best = [], math.inf
        for t, sc, k in pts:
            if sc < best:
                par.append(k)
                best = sc
        print(f"\n### d = {d}: Pareto set (stream mean time per shot, pair score), {len(par)} of {len(pts)}\n")
        print("| decoder | f | cap | pair score | (a) pair / their network | stream ms: mean / p99 / max | "
              f"batch ms per shot B = {max(BS):,}: N = " + " / ".join(map(str, NS)) + " |")
        print("|---|---|---|---|---|---|---|")
        for k in par:
            v = combos[k]
            pub = v["published"]["their network"]
            b = v["batch"]
            print(f"| {k[1]} | {k[2]:.1%} | {cap_s(k[3])} | {v['score']:.3e} | {report.fmt(pub[3], pub[4])} | "
                  f"{1e3 * v['stream']['mean']:.3f} / {1e3 * v['stream']['p99']:.2f} / {1e3 * v['stream']['max']:.0f} | "
                  + " / ".join(f"{1e3 * b[f'B{max(BS)}_N{N}']['mean']:.3f}" for N in NS) + " |")
    # the rule: the fastest (decoder, f, cap) whose pair's (a) upper bound is below 1 at d = 5 and 7
    keys = {k[1:] for k in combos if k[1] != "tess"}       # every setting timed on the speed job's node
    ok = [k for k in keys if all((d,) + k in combos and combos[(d,) + k]["published"]["their network"][4][1] < 1
                                 for d in DS)]
    print("\n### The reported setting (stream mode, d = 7's mean time; ties by d = 5's) and per batch mode\n")
    if set(DS) != {5, 7}:
        print("- not applied: the rule needs d = 5 and 7 (SPEED_DS)")
    elif ok:
        k = min(ok, key=lambda k: (combos[(7,) + k]["stream"]["mean"], combos[(5,) + k]["stream"]["mean"]))
        for d in DS:
            v = combos[(d,) + k]
            pub = v["published"]["their network"]
            print(f"- stream: {k[0]}, f = {k[1]:.1%}, cap {cap_s(k[2])}; d = {d}: pair / their network "
                  f"{report.fmt(pub[3], pub[4])}, mean {1e3 * v['stream']['mean']:.3f} ms, p99 "
                  f"{1e3 * v['stream']['p99']:.2f} ms, max {1e3 * v['stream']['max']:.0f} ms")
        for B in BS:
            for N in NS:
                k = min(ok, key=lambda k: (combos[(7,) + k]["batch"][f"B{B}_N{N}"]["mean"],
                                           combos[(5,) + k]["batch"][f"B{B}_N{N}"]["mean"]))
                v7, v5 = combos[(7,) + k], combos[(5,) + k]
                print(f"- B = {B:,}, N = {N}: {k[0]}, f = {k[1]:.1%}, cap {cap_s(k[2])}; ms per shot d = 5 / 7: "
                      f"{1e3 * v5['batch'][f'B{B}_N{N}']['mean']:.3f} / {1e3 * v7['batch'][f'B{B}_N{N}']['mean']:.3f}; "
                      f"batch latency p99 d = 7 {v7['batch'][f'B{B}_N{N}']['lat_p99']:.2f} s; (a) d = 5 / 7: "
                      f"{report.fmt(*v5['published']['their network'][3:])} / "
                      f"{report.fmt(*v7['published']['their network'][3:])}")
    else:
        print("- none")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode_", choices=["sel", "tess", "chk", "net", "rep"])
    ap.add_argument("--bank", required=True)
    ap.add_argument("--settings", default="long")
    ap.add_argument("--what", default="full", choices=["full", "chk"])
    ap.add_argument("--ds", default="7,5")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("NCPU", 4)))
    ap.add_argument("--modes", default="precise,onepass")
    ap.add_argument("--batches", default="1,1024,4096,16384,65536")
    ap.add_argument("--circuits", type=int, default=2)
    ap.add_argument("--check", type=int, default=4096)
    ap.add_argument("--times", default="")
    ap.add_argument("--mode", default="precise")
    ap.add_argument("--suffix", default="", help="rep on the check's parts: _chk")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dict(sel=sel, tess=tess, chk=chk, net=net, rep=rep)[a.mode_](a)


if __name__ == "__main__":
    main()
