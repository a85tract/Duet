"""Routing on Benchmark I: network reads, confidence thresholds, operating points and the simulated paced stream.
Nothing here has read a test set. The development bank comes first; the test sets only after the choices are frozen.

  read --manifest F --dir SETS --out NET --name N --ckpt CK [--path engine|frozen] [--only RE] [--part j/P]
         (GPU) a network's read of stored points: per circuit the tables with the stored product basis, then the engine's
         precise mode or the
         frozen forward (train_cl.collate and forward; fp32, TF32 off): NET/<N>/<point>.npz with logits (circuits, shots,
         4) float32 (NaN past the products), the seconds per circuit and the read's record (network SHA-256, path, GPU)
  check --manifest F --dir SETS --ckpt CK [--circuits 24 --max-shots 2048] (GPU) the engine's decisions against the
         frozen fp32 forward's on a sample of circuits (every K and depth): logits' largest difference, decision
         mismatches (run before the first engine read of a set, as scripts/bench_net.py dev --check)
  cmp --manifest DEV --dir SETS --net NET --a NEW --b START --d D --out J [--boot 10000] 's check:
         per K and pooled, failures of A over B on the same development shots of every point of d (a shot fails if any
         product is wrong), with circuit-level bootstrap draws (circuits resampled within each point); the 95 % and
         97.5 % quantiles of the ratio (the one-sided and two-sided 95 % upper bounds). Reported, never a trigger.
  thr --manifest DEV --net NET --name N --d D --out THR [--by depth|point]
         thresholds from the development bank (every point of d, each read whole by one network file, or it stops):
         score s = the smallest |logit| over the circuit's products; per depth (d, L, p; K pooled) or per point, t_f =
         the largest score among the ceil(f n) shots with the smallest scores (f in FS; exact decimal ceil; one
         partition for every f); a shot is routed iff s <= t_f
  route --manifest F --net NET --name N --thr THR --d D --out LISTS [--fs 0.2]
         the routed shots of a set at every f (nested: f' < f routes a subset), as {point: {circuit: [shot...]}} per
         f; every point of d read whole by the thresholds' network, or it stops
  gaps --manifest F --lists LISTS --f 0.2 --tess TESS --out GAPS [--exclude LISTS...]
         the routed shots at share f that have no trial part yet (nor in --exclude: shots another job is decoding), as
         scripts/eval_pipeline.py tess's lists file: then `ep.py tess --lists GAPS` fills them (parts merge per point in sim)
  sim --manifest F --dir SETS --net NET --name N --thr THR --tess TESS --d D --basis grace|spark|local --out J
         [--cal DIR] [--caps...] [--fac CPU=x...] [--fail target]
         per point of d and (f, c): the composition's failures (a shot fails if any product is wrong): kept shots take
         the network's decisions; a routed shot takes the lowest-cost trial among its 21 trials that is neither flagged,
         dropped nor past its cap c x L (basis seconds; the lower trial on a tie), and keeps the network's answer if none
         is left; a part run with a per-trial deadline serves only caps up to it; routed shots without a
         usable part are counted (missing) and keep the network's answer
  curve --manifest F --sim J --net-us US --out MD [--N 20,64] [--B 256] [--max-batches 400]
         per d (and per K): failures against time per layer for every (f, c) on the sim's basis and parts: / 
         batch simulation and the paced stream, batches by the circuit-major plan of stream.py live (every
         circuit, R = max-batches // circuits batches of B consecutive shots each); every point's network time required;
         per layer = seconds per shot / L, the mean over the points; the geometric mean of the composition's failure
         rate (0.5 zero rule); the plan's bar (the Lukin group's per gate on an H100, Fig. 5a: 47.4, 103.3, 185.2,
         302.2 us at d = 3, 5, 7, 9)
  nettime --manifest DEV --dir SETS --ckpt CK --out US [--grow 7] [--B 256 --Bs 256,1024,4096] [--circuits 4]
         (GPU) the network's seconds per batch of B shots of one circuit through the engine, per
         (K, L) at p = 0.3 %: US = {net_key: us per shot} for curve's --net-us (d<d>_K<K>_L<L>; lukin_d<d>_K4_L<L> on the
         lukin bank), US.full.json with the batch curve and the GPU; --grow 7: the checkpoint grown to 7 trunk blocks
         with zero-output new blocks
  choose --curve CJ --out MD [--N 64] 's rule per d: among
         the (f, c) whose paced-stream time per layer is below the bar, the smallest geometric mean over the d's points
         of the composition's failure rate (0.5 zero rule; exact comparison); ties to fewer routed shots, then the
         smaller f, then the larger cap; f = 0 when none qualifies (with the nearest point); it needs the grace or spark
         basis (spark: the GB10's network times), every network time and no routed shot without trials
(per CPU model: that CPU / the bank's 9534 or 7713P, median per shot) and converts a trial's seconds x GRACE[d'] /
median, d' = 5 at d = 3, 5 and d' = 7 at d = 7, 9 (borrowed at d = 3, 9: an estimate); every CPU model of the parts
needs its factor (no identity default); spark: the same x 1.124; local: no conversion (diagnostics; choose refuses
it). --fac 'CPU=x' or 'd<d>:CPU=x' adds or overrides factors.
--fail target (main result 2): a shot fails when the target (product slot 0) is wrong, else when any product is.
FS = 0, 0.5, 1, 2, 5, 10, 20 %; CAPS c = 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1 s and none (seconds per trial per
layer, on the reference CPU after --fac). The trial parts are scripts/eval_pipeline.py tess's (21 trials per shot, each timed alone
the simulation's: 'batch' is / (one batch at a time), 'stream' the paced stream as _stream: one first-in-first-out queue over N cores for every batch of a point; batch k + 1's network starts
when the GPU is free and batch k's first trial has started; the time is the last result's (dispatch and the queue's
drain inside), per layer = wall / (batches x B) / L; each batch's latency = its network's start to its last result."""
import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys
import time
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_pipeline as ep  # noqa: E402

from duet import bench_sets as bs  # noqa: E402

FS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
CAPS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, None)
BAR_US = {3: 47.4, 5: 103.3, 7: 185.2, 9: 302.2}          # their network's time per gate on an H100 (Fig. 5a)
K21 = ep.K_TRIALS
GRACE = {7: 0.884923, 5: 0.577412}   # Grace factors on the bank's long-beam parts: Grace / 9534 (d = 7), / 7713P (d = 5)


def fkey(f):
    return f"{f:g}"


def ckey(c):
    return "none" if c is None else f"{c:g}"


# ----------------------------------------------------------------------------------------------------------- read

def cmd_read(a):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    bn = ep.bench_net()
    m = ep.load_manifest(a.manifest)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    d = ck["cfg"]["d"]
    del ck
    assert a.device == "cuda" or a.path == "frozen", "the engine needs a GPU"
    net, rec = bn.load_net("lukin", d, a.device, a.ckpt)
    rec.update(gpu=torch.cuda.get_device_name(0) if a.device == "cuda" else ep.cpu(), path=a.path,
               code=os.environ.get("CODE"), manifest=a.manifest, purpose=m.get("purpose"))
    sync = torch.cuda.synchronize if a.device == "cuda" else (lambda: None)
    os.makedirs(os.path.join(a.out, a.name), exist_ok=True)
    pts = [pt for pt in ep.select(m, a.only) if pt["d"] == d]
    pts = ep.part_of(pts, a.part, key=lambda pt: pt["name"], weight=ep.net_weight)   # "bj/P": balanced
    pts.sort(key=lambda pt: (pt["K"], pt["L"]))                       # one engine per shape: few rebuilds
    rd = bn.Reader(net, d, "lukin", None, path=a.path)
    for pt in pts:
        out = os.path.join(a.out, a.name, pt["name"] + ".npz")
        if os.path.exists(out):
            continue
        rd.p = pt["p"]
        path = ep.set_path(a.dir, pt)
        D = bs.packed(path, "dets")
        n = min(pt["shots"], a.max_shots or pt["shots"])
        lg = np.full((len(pt["circuits"]), n, 4), np.nan, np.float32)
        secs, setup = [], []
        for i, c_ in enumerate(pt["circuits"]):
            info, s0 = rd.load(c_["layers"].split(), tuple(c_["prep"]), tuple(c_["meas"]),
                               [list(s) for s in c_["products"]])
            sync()
            t0 = time.time()
            c, dec = rd.logits(np.ascontiguousarray(D[i, :n, :info["nb"]]))
            sync()
            secs.append(time.time() - t0)
            setup.append(s0)
            c = np.asarray(c, np.float32)
            assert c.shape == (n, info["n_obs"]) and np.array_equal(np.asarray(dec, bool), c > 0)
            lg[i, :, :info["n_obs"]] = c
        np.savez(out + ".tmp.npz", logits=lg, secs=np.array(secs), setup=np.array(setup),
                 settings=json.dumps(rd.settings(), default=str), info=json.dumps(rec, default=str))
        os.replace(out + ".tmp.npz", out)
        print(json.dumps(dict(point=pt["name"], shots=n * len(pt["circuits"]), seconds=round(sum(secs), 1),
                              us_per_shot=round(1e6 * sum(secs) / (n * len(pt["circuits"])), 2))), flush=True)


def cmd_check(a):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    bn = ep.bench_net()
    m = ep.load_manifest(a.manifest)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    d = ck["cfg"]["d"]
    del ck
    net, rec = bn.load_net("lukin", d, "cuda", a.ckpt)
    pts = [pt for pt in m["points"] if pt["d"] == d]
    g = np.random.default_rng(2026111)
    pick = [(pts[int(i)], int(g.integers(len(pts[int(i)]["circuits"]))))
            for i in g.choice(len(pts), min(a.circuits, len(pts)), replace=False)]
    eng = bn.Reader(net, d, "lukin", None, path="engine")
    fro = bn.Reader(net, d, "lukin", None, path="frozen")
    out = []
    for pt, i in sorted(pick, key=lambda x: (x[0]["K"], x[0]["L"])):
        c_ = pt["circuits"][i]
        D = bs.packed(ep.set_path(a.dir, pt), "dets")
        n = min(pt["shots"], a.max_shots or 2048)
        res = {}
        for nm, rd in (("engine", eng), ("frozen", fro)):
            rd.p = pt["p"]
            info, _ = rd.load(c_["layers"].split(), tuple(c_["prep"]), tuple(c_["meas"]), [list(x) for x in c_["products"]])
            res[nm] = np.asarray(rd.logits(np.ascontiguousarray(D[i, :n, :info["nb"]]))[0], np.float32)
        diff = np.abs(res["engine"] - res["frozen"])
        r = dict(point=pt["name"], circuit=i, shots=n, max_abs_diff=float(diff.max()),
                 decision_mismatch=int(((res["engine"] > 0) != (res["frozen"] > 0)).sum()))
        out.append(r)
        print(json.dumps(r), flush=True)
    json.dump(dict(net=rec, rows=out), open(a.out, "w"), indent=0)
    print(json.dumps(dict(circuits=len(out), mismatches=sum(r["decision_mismatch"] for r in out))))


def fails_of(a, pt, name):
    """(circuits, n) bool: a shot fails when any product's decision (logit > 0) differs from the stored flips."""
    lg = load_logits(a.net, name, pt)
    truth, netp = point_truth(a, pt, lg)
    return (truth != netp).reshape(lg.shape[0], lg.shape[1])


def cmd_cmp(a):
    m = ep.load_manifest(a.manifest)
    assert m.get("purpose") in ("dev", "lukin_dev"), "the check reads a development bank"
    g = np.random.default_rng(3)
    per = {}
    pts, sha_a = complete_points(m, a.net, a.a, a.d)          # every point of d read by both
    _, sha_b = complete_points(m, a.net, a.b, a.d)
    for pt in pts:
        A, B = fails_of(a, pt, a.a), fails_of(a, pt, a.b)
        assert A.shape == B.shape, (pt["name"], A.shape, B.shape)
        n = A.shape[1]
        ca, cb = A[:, :n].sum(1), B[:, :n].sum(1)              # per-circuit failures on the same shots
        idx = g.integers(len(ca), size=(a.boot, len(ca)))      # circuit-level bootstrap within the point
        per[pt["name"]] = dict(d=pt["d"], K=pt["K"], a=int(ca.sum()), b=int(cb.sum()), ba=ca[idx].sum(1),
                               bb=cb[idx].sum(1))
    rows = []
    for d in sorted({v["d"] for v in per.values()}):
        for K in ("pooled", 2, 3, 4):
            v = [x for x in per.values() if x["d"] == d and (K == "pooled" or x["K"] == K)]
            if not v:
                continue
            sa, sb = sum(x["a"] for x in v), sum(x["b"] for x in v)
            ra = sum(x["ba"] for x in v) / np.maximum(sum(x["bb"] for x in v), 1)
            rows.append(dict(d=d, K=K, points=len(v), fails_a=sa, fails_b=sb, ratio=sa / max(sb, 1),
                             q95=float(np.quantile(ra, 0.95)), q975=float(np.quantile(ra, 0.975)),
                             passes_1_05_one_sided=bool(np.quantile(ra, 0.95) <= 1.05),
                             passes_1_05_two_sided=bool(np.quantile(ra, 0.975) <= 1.05)))
            print(json.dumps(rows[-1]), flush=True)
    json.dump(dict(a=a.a, b=a.b, sha256_a=sha_a, sha256_b=sha_b, d=a.d, boot=a.boot, rows=rows), open(a.out, "w"),
              indent=0)


# ------------------------------------------------------------------------------------------------------------ thr

def load_logits(net_dir, name, pt):
    return np.load(os.path.join(net_dir, name, pt["name"] + ".npz"))["logits"]


def read_sha(net_dir, name, pt):
    """The SHA-256 of the network file that made a point's read (route.py read's or ep.py net's record)."""
    with np.load(os.path.join(net_dir, name, pt["name"] + ".npz")) as z:
        return json.loads(str(z["info"])).get("sha256")


def complete_points(m, net_dir, name, d):
    """Every point of distance d in manifest m, each with its whole read (logits (circuits, shots, 4)) by one network
    file; an incomplete bank stops the run. -> (points, SHA-256)."""
    pts = [pt for pt in m["points"] if pt["d"] == d]
    assert pts, ("no point of d in the manifest", d)
    miss = [pt["name"] for pt in pts if not os.path.exists(os.path.join(net_dir, name, pt["name"] + ".npz"))]
    assert not miss, (f"d = {d}: {len(miss)} of {len(pts)} points have no read", miss[:4])
    shas = set()
    for pt in pts:
        lg = load_logits(net_dir, name, pt)
        assert lg.shape[:2] == (len(pt["circuits"]), pt["shots"]), (pt["name"], lg.shape, "a partial read")
        shas.add(read_sha(net_dir, name, pt))
    assert len(shas) == 1, ("the reads of one d come from different network files", shas)
    return pts, shas.pop()


def score_of(lg):
    s = np.nanmin(np.abs(lg), axis=2)
    assert not np.isnan(s).any(), "a shot without a product's logit"
    return s


def kth_of(f, n):
    """ceil(f n) in exact decimal arithmetic (f as written in FS: 0.1 x 30 is 3, not 4)."""
    q = Fraction(fkey(f))
    return -((-q.numerator * n) // q.denominator)


def thresholds(s, fs=FS):
    """{fkey(f): t_f} on the flat scores s: t_f = the largest score among the ceil(f n) smallest (the k-th smallest;
    -inf at f = 0), from one partition at every k; a shot is routed iff s <= t_f,
    so ties at t_f are routed (they may route more than ceil(f n))."""
    ks = {fkey(f): kth_of(f, s.size) for f in fs}
    kth = sorted({k - 1 for k in ks.values() if k > 0})
    part = np.partition(s, kth) if kth else s
    return {f: (float(part[k - 1]) if k > 0 else -np.inf) for f, k in ks.items()}


def group_key(pt, by):
    return f"d{pt['d']}_L{pt['L']}_p{ep.PCODE[pt['p']]}" + ("" if by == "depth" else f"_K{pt['K']}")


def cmd_thr(a):
    m = ep.load_manifest(a.manifest)
    assert m.get("purpose") in ("dev", "lukin_dev"), "thresholds come from a development bank"
    pts, sha = complete_points(m, a.net, a.name, a.d)
    groups = {}
    for pt in pts:
        groups.setdefault(group_key(pt, a.by), []).append(score_of(load_logits(a.net, a.name, pt)).reshape(-1))
    out = dict(score="the smallest |logit| over the circuit's products (fp32 read)", by=a.by, net=a.name, d=a.d,
               net_sha256=sha, manifest=a.manifest, points=[pt["name"] for pt in pts], f=[fkey(f) for f in FS], t={},
               n={}, k={})
    for key, v in sorted(groups.items()):
        s = np.concatenate(v)
        out["n"][key] = int(s.size)
        out["k"][key] = {fkey(f): kth_of(f, s.size) for f in FS}
        out["t"][key] = thresholds(s)
    json.dump(out, open(a.out, "w"), indent=0)
    print(json.dumps(dict(groups=len(groups), by=a.by, d=a.d, points=len(pts), net_sha256=sha)))


def is_routed(s, thr, pt, f):
    t = thr["t"][group_key(pt, thr["by"])][fkey(f)]
    return s <= t


def cmd_route(a):
    m = ep.load_manifest(a.manifest)
    thr = json.load(open(a.thr))
    assert thr.get("d") == a.d, ("thresholds of another d", thr.get("d"), a.d)
    pts, sha = complete_points(m, a.net, a.name, a.d)
    assert sha == thr["net_sha256"], ("the set's read and the thresholds come from different networks", sha)
    out = {fkey(f): {} for f in [float(x) for x in a.fs.split(",")]}
    tot = {k: 0 for k in out}
    for pt in pts:
        s = score_of(load_logits(a.net, a.name, pt))
        for fk in out:
            r = is_routed(s, thr, pt, float(fk))
            c, sh = np.nonzero(r)
            tot[fk] += int(r.sum())
            out[fk][pt["name"]] = {str(int(ci)): sh[c == ci].tolist() for ci in np.unique(c)}
    json.dump(dict(net=a.name, thr=a.thr, d=a.d, net_sha256=sha, routed=tot, lists=out), open(a.out, "w"))
    print(json.dumps(dict(routed=tot)))


def list_flat(lists, pts):
    """{point: set of flat indices (circuit * shots + shot)} of a lists file's {point: {circuit: [shot...]}}."""
    return {nm: {int(c) * pts[nm]["shots"] + int(s_) for c, ss in cs.items() for s_ in ss} for nm, cs in lists.items()}


def cmd_gaps(a):
    """The routed shots at share f that have no trial part in TESS yet; --exclude LISTS... also leaves out the shots
    another job is already decoding (the groundwork's lists), so the gap fill never repeats its work."""
    m = ep.load_manifest(a.manifest)
    pts = {pt["name"]: pt for pt in m["points"]}
    L = json.load(open(a.lists))["lists"][fkey(float(a.f))]
    excl = {}
    for f_ in (a.exclude.split(",") if a.exclude else []):
        for nm, v in list_flat(json.load(open(f_))["lists"], pts).items():
            excl.setdefault(nm, set()).update(v)
    out, tot, have, skip = {}, 0, 0, 0
    for nm, cs in L.items():
        n = pts[nm]["shots"]
        done = set(load_trials(a.tess, pts[nm], n)["q"].tolist())
        ex = excl.get(nm, set())
        for c, ss in cs.items():
            miss = [s_ for s_ in ss if int(c) * n + s_ not in done]
            have += len(ss) - len(miss)
            k0 = len(miss)
            miss = [s_ for s_ in miss if int(c) * n + s_ not in ex]
            skip += k0 - len(miss)
            if miss:
                out.setdefault(nm, {})[c] = miss
                tot += len(miss)
    json.dump(dict(f=a.f, rule="routed at share f without a trial part" + (" nor in --exclude" if excl else ""),
                   exclude=a.exclude, shots=tot, lists=out), open(a.out, "w"))
    print(json.dumps(dict(missing=tot, covered=have, excluded=skip)))


# ------------------------------------------------------------------------------------------------------------ sim

def load_trials(tess_dir, pt, n, fac=None, convert=True):
    """A point's trial parts as arrays: flat index (circuit * n + shot) sorted, and per row sec, flag, drop, cost, pred
    (each (R, 21)), the CPU model and the part's per-trial deadline dl (inf: none), in fac's basis (to_basis: seconds on
    the part's CPU x its factor at d; fac None: the local basis, factor 1; convert=False: seconds and dl as recorded).
    A shot decoded twice keeps its part with the largest deadline in basis seconds, then the last in file order. Each part's arrays are read once."""
    parts = []
    for f in sorted(glob.glob(os.path.join(tess_dir, pt["name"], "c*_*.npz"))):
        if f.endswith(".tmp.npz"):
            continue
        i = int(os.path.basename(f)[1:].split("_")[0])
        with np.load(f) as z:
            idx = np.asarray(z["idx"], np.int64)
            dl = float(z["deadline"]) if "deadline" in z.files else math.nan
            parts.append(dict(q=i * n + idx, sec=np.asarray(z["sec"], np.float64), flag=np.asarray(z["flag"], bool),
                              drop=np.asarray(z["drop"], bool), cost=np.asarray(z["cost"], np.float64),
                              pred=np.asarray(z["pred"], np.uint8),
                              cpu=np.full(len(idx), str(z["cpu"]), object),
                              dl=np.full(len(idx), math.inf if math.isnan(dl) else dl)))
    if not parts:
        z0 = np.zeros((0, K21))
        return dict(q=np.zeros(0, np.int64), sec=z0, flag=z0.astype(bool), drop=z0.astype(bool), cost=z0,
                    pred=z0.astype(np.uint8), cpu=np.zeros(0, object), dl=np.zeros(0))
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    bdl = cat["dl"] * row_factors(cat, fac, pt["d"])        # basis seconds (inf stays inf)
    o = np.lexsort((np.arange(len(cat["q"])), bdl, cat["q"]))
    last = np.r_[cat["q"][o][1:] != cat["q"][o][:-1], True]
    keep = o[last]
    out = {k: v[keep] for k, v in cat.items()}
    assert np.all(np.diff(out["q"]) > 0)
    return to_basis(out, fac, pt["d"]) if convert else out


def answers(T, rows, lim):
    """ rule on trial rows: (answer byte, has answer) with the lowest-cost trial neither flagged, dropped nor
    past lim (np.argmin keeps the lower trial on a tie); no answer when every trial is excluded."""
    ok = ~T["flag"][rows] & ~T["drop"][rows]
    if lim is not None:
        ok &= T["sec"][rows] <= lim
    c = np.where(ok, T["cost"][rows], np.inf)
    j = np.argmin(c, axis=1)
    has = np.isfinite(c[np.arange(len(rows)), j])
    return T["pred"][rows, j], has


BASES = ("grace", "spark", "local")
SPARK_CPU = 1.124
                         # d = 7; an assumption at the other d)


def parse_fac(spec, cal="", mult=1.0, cal_ds=(3, 5, 7, 9)):
    """{d or None: {CPU model: factor}}, seconds on that CPU x factor = the basis's seconds (module docstring: --cal from
    distances cal_ds borrow from (d = 3, 5: d5/; d = 7, 9: d7/) must exist."""
    out = {}
    if cal:
        for dref, ds in ((5, (3, 5)), (7, (7, 9))):
            if not set(ds) & set(cal_ds):
                continue
            f = os.path.join(cal, f"d{dref}", "factor.json")
            assert os.path.exists(f), (f, "the Grace basis needs the calibration's factors of d =", dref)
            js = json.load(open(f))
            assert js and all(float(v["median"]) > 0 for v in js.values()), (f, "an empty or invalid calibration")
            for cpu_, v in js.items():
                for d in ds:
                    out.setdefault(d, {})[cpu_] = GRACE[dref] / float(v["median"]) * mult
    for x in (spec.split(",") if spec else []):
        k, v = x.rsplit("=", 1)
        d = None
        if k.startswith("d") and ":" in k and k[1:k.index(":")].isdigit():
            d, k = int(k[1:k.index(":")]), k[k.index(":") + 1:]
        out.setdefault(d, {})[k] = float(v)
    return out


def basis_fac(a, d):
    """The conversion of --basis at distance d: grace = cal.sh's factors (--cal required), spark = the same x
    SPARK_CPU, local = none (the recorded seconds: diagnostics only, choose refuses it)."""
    assert a.basis in BASES, a.basis
    if a.basis == "local":
        assert not a.cal and not a.fac, "the local basis converts nothing"
        return None
    assert a.cal, f"the {a.basis} basis needs --cal (the calibration's factors)"
    fac = parse_fac(a.fac, a.cal, SPARK_CPU if a.basis == "spark" else 1.0, cal_ds=(d,))
    assert fac.get(d), (d, "no CPU factor at this d: the basis cannot be built")
    return fac


def row_factors(T, fac, d=None):
    """Per trial row the factor of (d, CPU model), else of the CPU model for every d; a CPU model without one stops the
    run (no silent mix of clocks, an empty factor set included); fac None (the local basis only): 1."""
    if fac is None:
        return np.ones(len(T["cpu"]))
    if not len(T["cpu"]):
        return np.ones(0)
    f = []
    for c in T["cpu"]:
        v = fac.get(d, {}).get(c, fac.get(None, {}).get(c))
        assert v is not None, (d, c, "no conversion factor for this CPU model")
        f.append(v)
    return np.array(f, np.float64)


def scaled(T, fac, d=None):
    """Trial seconds converted to the basis (row_factors)."""
    return T["sec"] * row_factors(T, fac, d)[:, None]


def to_basis(T, fac, d):
    """T's seconds and part deadlines converted to the basis, in place."""
    r = row_factors(T, fac, d)
    T["sec"] = T["sec"] * r[:, None]
    T["dl"] = T["dl"] * r
    return T


def fac_json(fac):
    return None if fac is None else {str(k): v for k, v in fac.items()}


def point_truth(a, pt, lg):
    """ per shot, flat (circuit * n + shot); --fail target keeps bit 0 (the
    target's product slot) only."""
    n = lg.shape[1]
    path = ep.set_path(a.dir, pt)
    z = np.load(path)
    O, k = bs.packed(path, "obs"), z["k"]
    truth = np.zeros((len(pt["circuits"]), n), np.uint8)
    netp = np.zeros((len(pt["circuits"]), n), np.uint8)
    for i in range(len(pt["circuits"])):
        if k[i]:
            truth[i] = np.packbits(bs.unpack(O[i][:n], k[i]), axis=1, bitorder="little")[:, 0]
            netp[i] = np.packbits(lg[i, :, :k[i]] > 0, axis=1, bitorder="little")[:, 0]
    if getattr(a, "fail", "any") == "target":
        truth, netp = truth & 1, netp & 1
    return truth.reshape(-1), netp.reshape(-1)


def lookup(T, q):
    """(has a trial row, row position) of flat shot indices q in T's sorted q."""
    pos = np.searchsorted(T["q"], q)
    if not len(T["q"]):
        return np.zeros(len(q), bool), pos
    return (pos < len(T["q"])) & (T["q"][np.minimum(pos, len(T["q"]) - 1)] == q), pos


def valid_rows(T, rows, lim):
    """Trial rows usable at cap lim (basis seconds): a part run with a per-trial deadline serves only caps up to that
    deadline, a part without one every cap."""
    return T["dl"][rows] >= (math.inf if lim is None else lim * (1 - 1e-9))


def cmd_sim(a):
    """Per point of --d and (f, c): the composition's failures on every shot of the point (module docstring). Every
    point of d must have its whole read (complete_points); a routed shot without a usable trial part is counted in
    'missing' (choose refuses any)."""
    m = ep.load_manifest(a.manifest)
    thr = json.load(open(a.thr))
    assert thr.get("d") == a.d, ("thresholds of another d", thr.get("d"), a.d)
    pts, sha = complete_points(m, a.net, a.name, a.d)
    assert sha == thr["net_sha256"], ("the read and the thresholds come from different networks", sha)
    fac = basis_fac(a, a.d)
    caps = [None if c == "none" else float(c) for c in a.caps.split(",")]
    rows, cpus = [], set()
    for pt in pts:
        lg = load_logits(a.net, a.name, pt)
        s_ = score_of(lg).reshape(-1)
        n, L = lg.shape[1], pt["L"]
        truth, netp = point_truth(a, pt, lg)
        net_fail = netp != truth
        T = load_trials(a.tess, pt, n, fac)
        cpus |= set(T["cpu"].tolist())
        if a.fail == "target":
            T["pred"] = T["pred"] & 1
        r = dict(point=pt["name"], d=pt["d"], K=pt["K"], L=L, p=pt["p"], shots=int(net_fail.size),
                 net_fails=int(net_fail.sum()), cells={})
        for f in FS:
            q = np.flatnonzero(is_routed(s_, thr, pt, f))
            have0, pos = lookup(T, q)
            for c in caps:
                lim = None if c is None else c * L
                have = have0.copy()
                have[have0] = valid_rows(T, pos[have0], lim)
                rq, rr = q[have], pos[have]
                fails = net_fail.copy()
                ans, ok = answers(T, rr, lim)
                fails[rq[ok]] = ans[ok] != truth[rq[ok]]
                sec = T["sec"][rr] if lim is None else np.minimum(T["sec"][rr], lim)
                r["cells"][f"{fkey(f)}|{ckey(c)}"] = dict(
                    routed=int(len(q)), fails=int(fails.sum()), missing=int((~have).sum()), kept_net=int((~ok).sum()),
                    trial_s_sum=float(sec.sum()), trial_s_max_mean=float(sec.max(1).mean()) if len(rr) else 0.0)
        rows.append(r)
        print(json.dumps(dict(point=r["point"], net_fails=r["net_fails"],
                              f20_none=r["cells"].get("0.2|none"))), flush=True)
    json.dump(dict(net=a.name, net_dir=a.net, net_sha256=sha, d=a.d, manifest=a.manifest, thr=a.thr, tess=a.tess,
                   basis=a.basis,
                   cal=a.cal, fac=fac_json(fac), cpus=sorted(cpus), fail=a.fail, caps=[ckey(c) for c in caps],
                   rows=rows), open(a.out, "w"))


# ---------------------------------------------------------------------------------------------------------- curve

def greedy_makespan(T, N):
    """T: (batches, k) task times in arrival order (0 = none) -> each batch's wall time on N cores (list scheduling)."""
    if T.shape[1] == 0:
        return np.zeros(len(T))
    load = np.zeros((len(T), N))
    rows = np.arange(len(T))
    for j in range(T.shape[1]):
        c = load.argmin(1)
        load[rows, c] += T[:, j]
    return load.max(1)


def paced(nr, dur, net, N):
    """_stream, 'paced': a point's batches on N cores with one first-in-first-out
    queue; nr (batches,) routed shots per batch, dur their trials' seconds in queue order (each shot's 21 trials, largest
    beam first), net the network's seconds per batch (a number, or one per batch: stream.py cmp replays the recorded
    GPU times). Batch k + 1's network starts when the GPU is free and batch k's first trial has started. -> per batch its
    network's start and its last result (its network's end if it routes no shot)."""
    import heapq
    nets = np.broadcast_to(np.asarray(net, np.float64), (len(nr),)).tolist()
    F, g, i = [0.0] * N, 0.0, 0
    st, en = np.zeros(len(nr)), np.zeros(len(nr))
    for b, m in enumerate(nr.tolist()):
        r = g + nets[b]
        last, first = r, max(F[0], r)
        for q in range(i, i + m * K21):
            e = max(F[0], r) + dur[q]
            heapq.heapreplace(F, e)
            if e > last:
                last = e
        i += m * K21
        st[b], en[b] = g, last
        g = max(g + nets[b], first) if m else g + nets[b]
    return st, en


def batch_plan(C, R, n, B):
    """The circuit-major batch plan of the timed stream, shared by curve and stream.py live: circuit 0's batches r = 0.. R - 1 (shots
    [r B, (r + 1) B)), then circuit 1's... -> ([(circuit, r)], flat shot indices circuit * n + shot in arrival
    order)."""
    assert 1 <= R <= n // B, (R, n, B)
    plan = [(c, r) for c in range(C) for r in range(R)]
    order = np.concatenate([c * n + np.arange(r * B, (r + 1) * B) for c, r in plan]).astype(np.int64)
    return plan, order


def curve_R(C, n, B, max_batches):
    """Batches per circuit in curve: every circuit of the point, R = min(n // B, max(1, max_batches // C))."""
    return min(n // B, max(1, max_batches // C)) if max_batches else n // B


def net_key(pt):
    """The network-time key of a point (route.py nettime): d<d>_K<K>_L<L> for main result 1's family, lukin_d<d>_K4_L<L>
    for the lukin circuits of main result 2."""
    pre = "" if pt.get("family", "general_mix") == "general_mix" else "lukin_"
    return f"{pre}d{pt['d']}_K{pt['K']}_L{pt['L']}"


CUR = {}


def _curve_point(job):
    """One point's rows of curve: per (f, c, N) the per-layer time one batch at a time and in the paced stream, and
    the batch latencies of the stream."""
    a, pt = job
    thr, net_us, fac, caps, Ns, simrow = CUR["thr"], CUR["net_us"], CUR["fac"], CUR["caps"], CUR["Ns"], CUR["simrow"]
    lg = load_logits(a.net, a.name, pt)
    s_ = score_of(lg).reshape(-1)
    C, n, L, B = lg.shape[0], lg.shape[1], pt["L"], a.B
    T = load_trials(a.tess, pt, n, fac)
    plan, order = batch_plan(C, curve_R(C, n, B, a.max_batches), n, B)
    nb = len(plan)
    tnet = net_us[net_key(pt)] * 1e-6 * B           # the network's seconds per batch (every key required)
    out = []
    for f in FS:
        rt = is_routed(s_, thr, pt, f)[order]
        pos = np.flatnonzero(rt)                         # positions in the arrival sequence
        q = order[pos]
        have0, j = lookup(T, q)
        b = pos // B
        cnt = np.bincount(b, minlength=nb)
        width = int(cnt.max()) * K21 if len(cnt) else 0
        for c in caps:
            lim = None if c is None else c * L
            have = have0.copy()
            have[have0] = valid_rows(T, j[have0], lim)
            sec = np.zeros((len(q), K21))
            if have.any():
                v = T["sec"][j[have]]
                sec[have] = (np.minimum(v, lim) if lim is not None else v)[:, ::-1]   # largest beam first
            Tm = np.zeros((nb, width))
            first = np.concatenate([[0], np.cumsum(cnt)[:-1]])
            slot = (np.arange(len(q)) - first[b]) * K21
            for t in range(K21):
                Tm[b, slot + t] = sec[:, t]
            cell = simrow["cells"][f"{fkey(f)}|{ckey(c)}"]
            dur = sec.reshape(-1).tolist()
            for N in Ns:
                mk = greedy_makespan(Tm, N)
                st, en = paced(cnt, dur, tnet, N)
                wall = float(en.max())
                out.append(dict(key=(pt["d"], pt["K"], fkey(f), ckey(c), N), point=pt["name"], L=L,
                                batch=float(np.mean(tnet + mk)) / B / L * 1e6, stream=wall / (nb * B) / L * 1e6,
                                lat=(en - st).tolist(), fails=cell["fails"], shots=simrow["shots"],
                                routed=cell["routed"], missing=cell["missing"], net_fails=simrow["net_fails"],
                                batches=nb, timed_missing=int((~have).sum()),
                                busy=float(sec.sum()) / N / wall if wall > 0 else 0.0))
    return out


def lrate(fails, shots):
    """log of a point's failure rate with the 0.5 zero rule."""
    return math.log((fails if fails > 0 else 0.5) / shots)


def cmd_curve(a):
    """Per point the routed shots' 21 capped trial times (converted to the sim's basis), batches of B shots by the
    circuit-major plan (batch_plan: every circuit, R batches each), each shot's trials largest beam first onto the first
    free of N cores; the network's batch time from --net-us. Means per layer over the points of a (d, K) and over K for d (equal weight per point); batch p99 over
    every batch of those points; the accuracy aggregate: the geometric mean over the points of the
    composition's failure rate (0.5 zero rule), computed directly on rates."""
    import multiprocessing as mp
    m = ep.load_manifest(a.manifest)
    sim = json.load(open(a.sim))
    assert not a.thr or os.path.abspath(a.thr) == os.path.abspath(sim["thr"]), "curve routes with the sim's thresholds"
    assert not (a.cal or a.fac), "curve converts with the sim's basis (its fac)"
    assert a.net_us, "curve needs --net-us (the network's B = 256 times of every point)"
    fac = None if sim["fac"] is None else {(None if k == "None" else int(k)): v for k, v in sim["fac"].items()}
    net_us = json.load(open(a.net_us))
    assert os.path.exists(a.net_us + ".full.json"), "the network times need their record (US.full.json: B, GPU)"
    net_full = json.load(open(a.net_us + ".full.json"))
    net_info = net_full.get("info", {})
    net_B = int(net_full["B"])
    assert net_B == a.B, ("the network times are of another batch", net_B, a.B)
    CUR.update(thr=json.load(open(sim["thr"])), fac=fac, Ns=[int(x) for x in a.N.split(",")], net_us=net_us,
               caps=[None if c == "none" else float(c) for c in sim["caps"]])
    simrow = {r["point"]: r for r in sim["rows"]}
    pts = [pt for pt in m["points"] if pt["name"] in simrow]
    assert len(pts) == len(simrow), "the sim's points are not the manifest's"
    nokey = sorted({net_key(pt) for pt in pts} - set(net_us))
    assert not nokey, ("no network time for", nokey)
    for k_, v in (("net", sim["net_dir"]), ("name", sim["net"]), ("tess", sim["tess"])):
        assert not getattr(a, k_) or getattr(a, k_) == v, (k_, "curve reads the sim's", v)
        setattr(a, k_, v)
    acc = {}
    CUR["simrows"] = simrow                            # before the fork: the workers see CUR
    with mp.get_context("fork").Pool(a.workers) as pool:
        res = pool.map(_curve_job, [(a, pt) for pt in pts], chunksize=1)
    for rows in res:
        for x in rows:
            d, K, f, c, N = x["key"]
            for key in ((d, K, f, c, N), (d, "all", f, c, N)):
                y = acc.setdefault(key, dict(batch=[], stream=[], lat=[], fails=0, shots=0, missing=0, net_fails=0,
                                             routed=0, ratios=[], lrates=[], nlrates=[], busy=[], timed_missing=0))
                y["batch"].append(x["batch"])
                y["stream"].append(x["stream"])
                y["lat"].extend(x["lat"])
                y["busy"].append(x["busy"])
                for k_ in ("fails", "shots", "missing", "net_fails", "routed", "timed_missing"):
                    y[k_] += x[k_]
                y["lrates"].append(lrate(x["fails"], x["shots"]))
                y["nlrates"].append(lrate(x["net_fails"], x["shots"]))
                fz, nz = x["fails"], x["net_fails"]
                if fz == 0 or nz == 0:
                    fz, nz = fz + 0.5, nz + 0.5
                y["ratios"].append(math.log(fz / nz))
    out = []
    for (d, K, f, c, N), x in sorted(acc.items(), key=lambda kv: tuple(str(v) for v in kv[0])):
        out.append(dict(d=d, K=K, f=f, cap=c, N=N, fails=x["fails"], net_fails=x["net_fails"], shots=x["shots"],
                        routed=x["routed"], missing=x["missing"], timed_missing=x["timed_missing"],
                        points=len(x["batch"]), mlog_rate=float(np.mean(x["lrates"])),
                        gmean_rate=math.exp(float(np.mean(x["lrates"]))),
                        gmean_rate_net=math.exp(float(np.mean(x["nlrates"]))),
                        gmean_ratio=math.exp(float(np.mean(x["ratios"]))),
                        us_per_layer_batch=float(np.mean(x["batch"])), us_per_layer_stream=float(np.mean(x["stream"])),
                        p99_ms_stream=1e3 * float(np.percentile(x["lat"], 99)) if x["lat"] else 0.0,
                        busy_stream=float(np.mean(x["busy"]))))
    json.dump(dict(rows=out, sim=a.sim, basis=sim["basis"], cal=sim.get("cal"), cpus=sim.get("cpus"), net_us=a.net_us,
                   net_info=net_info, net_B=net_B, fac=fac_json(fac), B=a.B, max_batches=a.max_batches,
                   plan="circuit-major",
                   fail=sim["fail"], d=sim["d"], net_sha256=sim["net_sha256"]), open(a.out + ".json", "w"), indent=0)
    with open(a.out, "w") as fh:
        fh.write(f"Basis {sim['basis']} (factors {json.dumps(fac_json(fac))}); network times {a.net_us} "
                 f"({net_info.get('gpu', '?')}); B = {a.B}, circuit-major batches.\n\n")
        fh.write("| d | K | f | cap c (s x L) | N | failures / shots | geo. mean failure rate (network alone) | "
                 "routed | routed without trials | us per layer, one batch at a time | paced stream | stream batch p99 "
                 "ms | their bar |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in out:
            fh.write(f"| {r['d']} | {r['K']} | {r['f']} | {r['cap']} | {r['N']} | {r['fails']} / {r['shots']} | "
                     f"{r['gmean_rate']:.4g} ({r['gmean_rate_net']:.4g}) | {r['routed']} | {r['missing']} | "
                     f"{r['us_per_layer_batch']:.1f} | {r['us_per_layer_stream']:.1f} | {r['p99_ms_stream']:.0f} | "
                     f"{BAR_US[r['d']]} |\n")
    print(json.dumps(dict(rows=len(out))))


def _curve_job(job):
    a, pt = job
    CUR["simrow"] = CUR["simrows"][pt["name"]]
    return _curve_point(job)


def cmd_choose(a):
    """The operating point per d and N on the development bank:
    among the (f, c) whose paced-stream time per layer (curve, K pooled) is below the bar, the smallest geometric mean
    over the d's points of the composition's failure rate (0.5 zero rule; compared exactly, no rounding); ties to fewer
    routed shots, then the smaller f, then the larger cap (none the largest). With no candidate below the bar: f = 0
    (the network alone) and the nearest point. A choice needs the grace or spark basis, every network time (curve) and
    every routed shot's trials (no candidate with a missing trial part)."""
    cj = json.load(open(a.curve + ".json" if not a.curve.endswith(".json") else a.curve))
    assert cj["basis"] in ("grace", "spark"), ("a choice needs the grace or the spark basis", cj["basis"])
    assert cj.get("net_B") == 256 and cj.get("B") == 256, ("the bar compares B = 256", cj.get("net_B"), cj.get("B"))
    gpu = str(cj.get("net_info", {}).get("gpu", ""))
    want = {"grace": "RTX PRO 6000", "spark": "GB10"}[cj["basis"]]
    assert want in gpu, (f"the {cj['basis']} basis needs the {want}'s network times", gpu)
    rows = [r for r in cj["rows"] if r["K"] == "all"]
    Ns = [int(x) for x in a.N.split(",")]
    capv = lambda c: math.inf if c == "none" else float(c)
    res = []
    for d in sorted({r["d"] for r in rows}):
        for N in Ns:
            cand = [r for r in rows if r["d"] == d and r["N"] == N]
            assert cand, (d, N, "no curve rows")
            miss = [r for r in cand if r["missing"] or r["timed_missing"]]
            assert not miss, (d, N, f"{len(miss)} candidates have routed shots without trials: fill the gaps first")
            ok = [r for r in cand if r["us_per_layer_stream"] < BAR_US[d]]
            alone = [r for r in cand if float(r["f"]) == 0.0 and r["cap"] == "none"][0]
            if ok:
                best = min(ok, key=lambda r: (r["mlog_rate"], r["routed"], float(r["f"]), -capv(r["cap"])))
                res.append(dict(d=d, N=N, basis=cj["basis"], chosen=best, qualifying=len(ok), candidates=len(cand),
                                network_alone=alone, rule="most accurate below the bar"))
            else:
                near = min(cand, key=lambda r: r["us_per_layer_stream"])
                res.append(dict(d=d, N=N, basis=cj["basis"], chosen=alone, nearest=near, qualifying=0,
                                candidates=len(cand), network_alone=alone,
                                rule="no candidate below the bar: f = 0, the network alone"))
    json.dump(dict(curve=a.curve, basis=cj["basis"], net_sha256=cj.get("net_sha256"), res=res),
              open(a.out + ".json", "w"), indent=0)
    with open(a.out, "w") as fh:
        fh.write("| d | basis | N | chosen f | cap c (s x L) | failures / shots | geo. mean failure rate (network "
                 "alone) | routed | us per layer (paced stream) | bar | qualifying |\n"
                 "|---|---|---|---|---|---|---|---|---|---|---|\n")
        for x in res:
            r = x["chosen"]
            fh.write(f"| {x['d']} | {x['basis']} | {x['N']} | {r['f']} | {r['cap'] if float(r['f']) else '-'} | "
                     f"{r['fails']} / {r['shots']} | {r['gmean_rate']:.4g} ({r['gmean_rate_net']:.4g}) | "
                     f"{r['routed']} | {r['us_per_layer_stream']:.1f} | {BAR_US[x['d']]} | {x['qualifying']} / "
                     f"{x['candidates']} |\n")
            if x.get("nearest"):
                n_ = x["nearest"]
                fh.write(f"| {x['d']} | nearest | {x['N']} | {n_['f']} | {n_['cap']} | {n_['fails']} / {n_['shots']} "
                         f"| {n_['gmean_rate']:.4g} | {n_['routed']} | {n_['us_per_layer_stream']:.1f} | "
                         f"{BAR_US[x['d']]} | - |\n")
    print(json.dumps([dict(d=x["d"], N=x["N"], f=x["chosen"]["f"], cap=x["chosen"]["cap"], rule=x["rule"])
                      for x in res]))


def cmd_nettime(a):
    """The network's seconds per batch through the engine (module docstring)."""
    import torch
    from duet.net import model_cl, train_cl
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    bn = ep.bench_net()
    m = ep.load_manifest(a.manifest)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    d = ck["cfg"]["d"]
    cfg = dict(ck["cfg"], blocks=a.grow) if a.grow else dict(ck["cfg"])
    net = model_cl.NetCL(**cfg)
    added = model_cl.load_grown(net, ck) if a.grow else net.load_state_dict(ck["model"])
    net = net.to("cuda").eval()
    rec = dict(path=a.ckpt, sha256=hashlib.sha256(open(a.ckpt, "rb").read()).hexdigest(), cfg=net.cfg,
               grown=bool(a.grow), added=str(added) if a.grow else None, gpu=torch.cuda.get_device_name(0),
               code=os.environ.get("CODE"))
    del ck
    rd = bn.Reader(net, d, "lukin", None, path="engine")
    Bs = sorted(set([int(x) for x in a.Bs.split(",")] + [a.B]))
    out = dict(info=rec, d=d, B=a.B, us={}, curve={}, setup={})
    pts = [pt for pt in ep.select(m, a.only) if pt["d"] == d and pt["p"] == 0.003]
    for pt in sorted(pts, key=lambda pt: (pt["K"], pt["L"])):
        rd.p = pt["p"]
        D = bs.packed(ep.set_path(a.dir, pt), "dets")
        per, setup = {}, []
        for i, c_ in enumerate(pt["circuits"][:a.circuits]):
            info, s0 = rd.load(c_["layers"].split(), tuple(c_["prep"]), tuple(c_["meas"]), [list(x) for x in c_["products"]])
            setup.append(s0)
            x = np.ascontiguousarray(D[i, :, :info["nb"]])
            for B in Bs:
                xb = x[:B] if len(x) >= B else np.ascontiguousarray(np.resize(x, (B, x.shape[1])))
                for _ in range(2):
                    rd.e.decode(rd.info, xb, B=B)
                torch.cuda.synchronize()
                ts = []
                for _ in range(a.reps):
                    t0 = time.perf_counter()
                    rd.e.decode(rd.info, xb, B=B)
                    torch.cuda.synchronize()
                    ts.append(time.perf_counter() - t0)
                per.setdefault(B, []).append(float(np.median(ts)))
        key = net_key(pt)
        out["us"][key] = 1e6 * float(np.median(per[a.B])) / a.B
        out["curve"][key] = {str(B): 1e6 * float(np.median(v)) / B for B, v in per.items()}
        out["setup"][key] = float(np.median(setup))
        print(json.dumps(dict(point=key, us_per_shot=round(out["us"][key], 3),
                              us_per_layer=round(out["us"][key] / pt["L"], 3))), flush=True)
    json.dump(out["us"], open(a.out, "w"), indent=0)
    json.dump(out, open(a.out + ".full.json", "w"), indent=0, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["read", "check", "cmp", "thr", "route", "gaps", "sim", "curve", "choose",
                                     "nettime"])
    ap.add_argument("--manifest", default="")
    ap.add_argument("--dir", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--name", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--path", default="engine", choices=["engine", "frozen"])
    ap.add_argument("--only", default="")
    ap.add_argument("--part", default="")
    ap.add_argument("--net", default="")
    ap.add_argument("--by", default="depth", choices=["depth", "point"])
    ap.add_argument("--thr", default="")
    ap.add_argument("--fs", default="0.2")
    ap.add_argument("--f", default="0.2")
    ap.add_argument("--lists", default="")
    ap.add_argument("--device", default="cuda", help="read: cuda, or cpu for a frozen-path smoke test")
    ap.add_argument("--tess", default="")
    ap.add_argument("--caps", default="0.001,0.002,0.005,0.01,0.02,0.05,0.1,none")
    ap.add_argument("--fac", default="", help="CPU model=factor (or d<d>:CPU model=factor) to the basis's seconds, "
                    "comma list; added to --cal's")
    ap.add_argument("--basis", default="", help="sim: grace (--cal), spark (--cal x 1.124) or local (no conversion)")
    ap.add_argument("--d", type=int, default=0, help="thr, route, sim: the distance (every point of it required)")
    ap.add_argument("--exclude", default="", help="gaps: lists files whose shots another job decodes")
    ap.add_argument("--sim", default="")
    ap.add_argument("--N", default="20,64")
    ap.add_argument("--a", default="")
    ap.add_argument("--b", default="")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--circuits", type=int, default=24)
    ap.add_argument("--max-shots", type=int, default=0, help="read: the first N shots of each circuit (smoke tests)")
    ap.add_argument("--B", type=int, default=256)
    ap.add_argument("--net-us", default="", help="JSON {d<d>_K<K>_L<L>: the network's us per shot at batch B}")
    ap.add_argument("--cal", default="", help="the calibration's directory (d7/, d5/ factor.json): the Grace basis")
    ap.add_argument("--fail", default="any", choices=["any", "target"])
    ap.add_argument("--max-batches", type=int, default=400, help="curve: the first batches of each point's stream")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("NCPU", 1)))
    ap.add_argument("--curve", default="")
    ap.add_argument("--grow", type=int, default=0)
    ap.add_argument("--Bs", default="256,1024,4096")
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    dict(read=cmd_read, check=cmd_check, cmp=cmd_cmp, thr=cmd_thr, route=cmd_route, gaps=cmd_gaps, sim=cmd_sim,
         curve=cmd_curve, choose=cmd_choose, nettime=cmd_nettime)[a.mode](a)


if __name__ == "__main__":
    main()
