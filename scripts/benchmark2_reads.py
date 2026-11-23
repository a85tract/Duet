"""Test-set reads of Benchmark II: the open benchmark's
lukin family under the Lukin model at d = 3, 5, 7 (D = 1-10, the test set's circuits i = 0-19), read once by each
chosen network and its pairs, and at d = 5 and 7 the test set's other families, the network against its start.

  python scripts/benchmark2_reads.py thr --bank B --out J t(d, D, f) of each chosen network on the development bank, the dev decode seconds per depth
  python scripts/benchmark2_reads.py conv --release R --bench P --test T [--ds 3,5,7] [--noise lukin] [--workers W]
                               the lukin points: R's b8 files, each SHA-256 checked against R's SHA256SUMS, into
                               T/sets/<point>.npz (duet.bench_sets' format), and the benchmark's merged predictions
                               P/<dec>/<point>.npz (LoMatching, the exact MLE with its certified shots, the start's phase-4
                               read) into bank.py dec's parts T/dec/{lomr,mle,start}/; a point whose manifest entry has
                               no MLE prefix (mle_shots 0: d = 7, D = 9, 10) gets no MLE parts
  python scripts/benchmark2_reads.py check --release R --bench P [--thr J] every input file of the test jobs exists (t_conv
                               runs it first): the release's files and SHA256SUMS lines (lukin; the other families at
                               d = 5, 7), the benchmark's LoMatching and phase-4 reads, its MLE where the manifest has a
                               prefix, the networks (SHA-256), the thresholds and Fig. 1b's values
  python scripts/benchmark2_reads.py verify --release R [--ds 5,7] [--workers W] SHA-256 of the other families' b8 files
  python scripts/benchmark2_reads.py route --test T --thr J --ds D [--sel-dir DIR] (BANK_SET=test) the shots with s_T <= t(d, D, f),
                                                         per f: T/sel/route_d<d>_f<f>.json (bank.py dec's --select)
  python scripts/benchmark2_reads.py tess --test T --setting S --sel FILE --ds D [--dec-dir DIR] [--workers W] (BANK_SET=test)
                                                          timed decode of the listed shots with Tesseract setting
                                                         S (speed.py SETTINGS) into T/dec/tess_S/
  python scripts/benchmark2_reads.py rep --test T --thr J [--ds 3,5,7] [--mle mle] [--start start | --start-net NAME] [--out J]
                               the statistics of 
  python scripts/benchmark2_reads.py guard --release R --bench P --new P [--ds 5,7] [--out J] the other families: the chosen
                               network's merged read P/net/ (scripts/bench_net.py read --release --net-path, decode.py
                               merge) against the start's phase-4 read (--bench P/net/), any-product failures pooled
                               per family, paired bootstrap (scripts/bench_stats.py's draws and ratio)

Statistics. Target parity, product slot 0. (a) A = K / E: K our target failures over D = 1-10, E = sum_D n_D p_D
(n_D our shots, p_D their published network value); 95 % interval log-normal with Var(log A) = V_K + V_E: V_K the
variance of log K* over BOOT draws that redraw each depth's circuits with replacement and each drawn circuit's
failures binomially at its own rate and shots; V_E = sum_D n_D^2 s_D^2 / E^2
from their published bars s_D. Sensitivity ((7) 4): V_E plus their circuit-variation term
sum_D n_D^2 p_D^2 c_D / E^2, c_D the relative variance of a 20-circuit mean between circuits at depth D, from our
network's per-circuit rates (binomial part removed, at least 0). (b) B = (sum_D x_D / sum_D m_D) / (sum_D p_D /
sum_D q_D) over the depths where our MLE ran (the manifest's MLE prefix: d = 3, 5 D = 1-10, d = 7 D = 1-8; asserted on
the test set) (x_D, m_D: the system's and the MLE's rates on the MLE's certified shots;
q_D their MLE value); Var(log B) = V_ours + V_theirs: V_ours the variance of the log of ours / our MLE over BOOT draws
redrawing each depth's circuits and each drawn circuit's paired shots (both fail, system only, MLE only, neither);
V_theirs = sum s_p^2 / (sum p)^2 + sum s_q^2 / (sum q)^2 ((7) 5). Each comparison holds when its upper bound is below
1. Pairs: routed iff s_T <= t(d, D, f) of the development bank; the long beam's decision on a routed
shot; o1nc's when its wall time is at most the cap and it raised no low-confidence flag, else the network's."""
import argparse
import glob
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import sys
import time
import zlib

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

S_LUKIN = os.environ.get("DUET_BENCH2_DIR", "benchmark2")    # working directory of the development bank and the test reads
NETDIR = os.environ.get("DUET_NETWORKS", "networks")
NETS = {   # the chosen networks: name in B/net/, file, SHA-256
    3: ("ft4", NETDIR + "/lukin/d3_ft4_s8423/final_ema.pt",
        "388def833b586967d359dcfb28ca97d647bc25e6ad9f7767cfcebdec68187e27"),
    5: ("MH_q4", S_LUKIN + "/arms/MH/final_ema.pt", "785b2a09d270dd0747a44aa07ec70e881146b183174cba456b8a94b2d8e02902"),
    7: ("MH7_q4", S_LUKIN + "/arms/MH7/final_ema.pt",
        "cb3a6876720942049913601852b8393ebbadd7d7e276f7bba846f43ab3378614"),
}
FS = (0.005, 0.01, 0.02, 0.05, 0.10)   # the long beam's f
FAST = dict(setting="o1nc", f=0.005, cap=0.1, ds=(5, 7))   # speed setting
BOOT = 10_000
SEED = 186_014                          # the bootstrap's numpy seed (+ d)
OTHER = ("lom_rep", "lom_2q", "lom_deep", "mccd", "cain_ghz", "ours_k")   # the test set's other families


def fkey(f):
    return f"{f:g}"


# ------------------------------------------------------------------------------------------------------------ thr

def thr(a):
    assert not bank.TEST, "thresholds come from the development bank (BANK_SET unset)"
    out = dict(score="s_T = |target logit| of the fp32 read", f=list(FS), nets={}, t={}, dev={})
    for d, (name, path, sha) in NETS.items():
        out["nets"][str(d)] = dict(name=name, path=path, sha256=sha)
        out["t"][str(d)] = {fkey(f): [] for f in FS}
        dev = out["dev"][str(d)] = {fkey(f): dict(routed=[], le_t=[], long_s=[], o1nc_s=[], o1nc_cov=[]) for f in FS}
        decs = ["tess"] + (["tess_o1nc"] if d in FAST["ds"] else [])
        for pt in bank.points((d,)):
            r = report.load(a.bank, pt, name, decs)
            s = np.abs(r["logits"][:, 0])
            for f in FS:
                o, t = report.routed(s, f)
                out["t"][str(d)][fkey(f)].append(t)
                q = dev[fkey(f)]
                q["routed"].append(len(o))
                q["le_t"].append(int((s <= t).sum()))      # the test rule s <= t on the bank itself
                q["long_s"].append(float(np.nanmean(r["tess"]["sec"][o])))
                if "tess_o1nc" in r:
                    c = r["tess_o1nc"]["cov"][o]
                    q["o1nc_cov"].append(int(c.sum()))
                    q["o1nc_s"].append(float(np.nanmean(r["tess_o1nc"]["sec"][o][c])) if c.any() else None)
            print(json.dumps(dict(d=d, D=pt["depth"], t={fkey(f): out["t"][str(d)][fkey(f)][-1] for f in FS})),
                  flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    # what the test set costs at these rates (routed shots ~ f times its shots)
    tp = {(p["d"], p["depth"]): p["shots"] * len(p["circuits"]) for p in bank.test_points((3, 5, 7))}
    cost = {}
    for d in NETS:
        for f in FS:
            q = out["dev"][str(d)][fkey(f)]
            cost[f"d{d}|long|{fkey(f)}"] = sum(f * tp[(d, D)] * q["long_s"][D - 1] for D in range(1, 11)) / 3600
            if q["o1nc_s"] and None not in q["o1nc_s"]:
                cost[f"d{d}|o1nc|{fkey(f)}"] = sum(f * tp[(d, D)] * q["o1nc_s"][D - 1] for D in range(1, 11)) / 3600
    out["test_core_hours"] = cost
    print(json.dumps(cost), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


# ------------------------------------------------------------------------------------------------------------ conv

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as g:
        for block in iter(lambda: g.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def sums(release):
    out = {}
    for ln in open(os.path.join(release, "SHA256SUMS")):
        h, p = ln.rstrip("\n").split(None, 1)
        out[p.lstrip("*")] = h
    return out


def _conv_point(job):
    a, pt, S = job
    name, C = pt["name"], len(pt["circuits"])
    dd = os.path.join(a.release, "points", name)
    meta = json.load(open(os.path.join(dd, "meta.json")))
    assert [c["seed"] for c in meta["entry"]["circuits"]] == [c["seed"] for c in pt["circuits"]], name
    assert meta["entry"]["shots"] == pt["shots"], name
    rec = dict(point=name, files={})
    for i in range(C):
        for kind in ("dets", "obs"):
            rel = f"points/{name}/c{i}.{kind}.b8"
            h = sha256_file(os.path.join(a.release, rel))
            assert h == S[rel], ("checksum", rel)
            rec["files"][rel] = h
    m = np.array([int(c["detectors"]) for c in meta["circuits"]])
    k = np.array([int(c["observables"]) for c in meta["circuits"]])
    for i, c_ in enumerate(pt["circuits"]):      # the circuit the decoders and tables rebuild has the same sizes
        c = bs.circuit(pt, c_)
        assert (c.num_detectors, c.num_observables) == (m[i], k[i]), (name, i)
    n = pt["shots"]
    D = np.zeros((C, n, (m.max() + 7) // 8), np.uint8)
    O = np.zeros((C, n, (k.max() + 7) // 8), np.uint8)
    for i in range(C):
        x = np.fromfile(os.path.join(dd, f"c{i}.dets.b8"), np.uint8).reshape(n, -1)
        y = np.fromfile(os.path.join(dd, f"c{i}.obs.b8"), np.uint8).reshape(n, -1)
        D[i, :, :x.shape[1]], O[i, :, :y.shape[1]] = x, y
    os.makedirs(os.path.join(a.test, "sets"), exist_ok=True)
    path = os.path.join(a.test, "sets", name + ".npz")
    np.savez(path + ".tmp.npz", dets=D, obs=O, m=m, k=k, entry=json.dumps(pt))
    os.replace(path + ".tmp.npz", path)
    assert k.max() <= 8
    rec["mle_shots"] = bs.prefix(pt, "mle")
    for dec, src in (("lomr", "lomr"), ("mle", "mle"), ("start", "net")):   # the benchmark's reads of these shots
        if dec == "mle" and not rec["mle_shots"]:     # no exact MLE here (the manifest's prefix 0: d = 7, D = 9, 10;
            continue                                  # 186 (2)): no MLE parts, so rep's (b) leaves the depth out
        f = os.path.join(a.bench, src, name + ".npz")
        z = np.load(f)
        rec["files"][f] = sha256_file(f)
        pred, ok, decoded = z["pred"], z["ok"], z["decoded"]
        assert len(decoded) == C, (name, dec)
        for i in range(C):
            nd = int(decoded[i])
            assert (nd == n) if dec != "mle" else (0 <= nd <= n), (name, dec, i, nd)
            out = os.path.join(a.test, "dec", dec, name, f"c{i}_0.npz")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            np.savez(out + ".tmp.npz", idx=np.arange(nd, dtype=np.int64), pred=np.asarray(pred[i, :nd, 0], np.uint8),
                     ok=np.asarray(ok[i, :nd]).astype(bool), sec=np.full(nd, np.nan, np.float32), build=0.0,
                     settings=json.dumps(dict(source=f)))
            os.replace(out + ".tmp.npz", out)
    return rec


def conv(a):
    S = sums(a.release)
    ds = tuple(int(x) for x in a.ds.split(","))
    pts = [p for p in json.load(open(os.path.join(ROOT, "data", "open_benchmark", "MANIFEST.json")))["points"]
           if p["family"] == "lukin" and p["noise"] == a.noise and p["d"] in ds]     # --noise uniform: the dry run
    assert a.noise != "lukin" or {p["name"] for p in pts} == {p["name"] for p in bank.test_points(ds)}
    t0, recs = time.time(), []
    with mp.get_context("fork").Pool(a.workers) as pool:
        jobs = [(a, pt, {r: h for r, h in S.items() if r.startswith(f"points/{pt['name']}/")}) for pt in pts]
        for rec in pool.imap_unordered(_conv_point, jobs):
            recs.append(rec)
            print(json.dumps(dict(point=rec["point"], files=len(rec["files"]),
                                  minutes=round((time.time() - t0) / 60, 2))), flush=True)
    json.dump(dict(release=a.release, bench=a.bench, points=sorted(recs, key=lambda r: r["point"])),
              open(os.path.join(a.test, "conv.json"), "w"), indent=0)


def _verify_file(job):
    root, rel, h = job
    return rel, sha256_file(os.path.join(root, rel)) == h


def verify(a):
    S = sums(a.release)
    ds = tuple(int(x) for x in a.ds.split(","))
    pts = [p for p in json.load(open(os.path.join(ROOT, "data", "open_benchmark", "MANIFEST.json")))["points"]
           if p["noise"] == "lukin" and p["d"] in ds and p["family"] in OTHER]
    jobs = [(a.release, rel, S[rel]) for p in pts for i in range(len(p["circuits"]))
            for rel in (f"points/{p['name']}/c{i}.dets.b8", f"points/{p['name']}/c{i}.obs.b8")]
    jobs.sort(key=lambda j: -os.path.getsize(os.path.join(a.release, j[1])))
    bad = []
    with mp.get_context("fork").Pool(a.workers) as pool:
        for rel, ok in pool.imap_unordered(_verify_file, jobs, chunksize=4):
            if not ok:
                bad.append(rel)
    print(json.dumps(dict(points=len(pts), files=len(jobs), differ=bad[:50], n_differ=len(bad))), flush=True)
    assert not bad, "release files differ from SHA256SUMS"


def check(a):
    """Every input file the test jobs read, before any of them runs: the release's meta.json and b8 files with their
    SHA256SUMS lines (conv and the fp32 reads: the lukin points at d = 3, 5, 7; verify, the engine reads and guard: the
    other families at d = 5, 7), the benchmark's LoMatching and phase-4 reads of the lukin points (conv) and its phase-4
    reads of the other families (guard's start), its exact MLE where the manifest has a prefix (conv; none at d = 7,
    D = 9, 10), the networks with their SHA-256 (the reads; route and rep assert it), the thresholds and Fig. 1b's
    values (route, rep). Reads no shot."""
    S = sums(a.release)
    man = json.load(open(os.path.join(ROOT, "data", "open_benchmark", "MANIFEST.json")))["points"]
    lukin_pts = [p for p in man if p["family"] == "lukin" and p["noise"] == "lukin" and p["d"] in NETS]
    ot = [p for p in man if p["noise"] == "lukin" and p["d"] in (5, 7) and p["family"] in OTHER]
    assert {p["name"] for p in lukin_pts} == {p["name"] for p in bank.test_points(tuple(NETS))}
    need, nosum = [], []
    for p in lukin_pts + ot:
        need.append(os.path.join(a.release, "points", p["name"], "meta.json"))
        for i in range(len(p["circuits"])):
            for kind in ("dets", "obs"):
                rel = f"points/{p['name']}/c{i}.{kind}.b8"
                need.append(os.path.join(a.release, rel))
                if rel not in S:
                    nosum.append(rel)
        decs = (("lomr", "net") + (("mle",) if bs.prefix(p, "mle") else ())) if p["family"] == "lukin" else ("net",)
        need += [os.path.join(a.bench, x, p["name"] + ".npz") for x in decs]
    need += [x[1] for x in NETS.values()] + [a.thr, os.path.join(ROOT, "data", "open_benchmark", "fig1b_values.csv")]
    miss = [f for f in need if not os.path.exists(f)]
    bad_sha = [x[1] for x in NETS.values() if os.path.exists(x[1]) and sha256_file(x[1]) != x[2]]
    no_mle = [p["name"] for p in lukin_pts if not bs.prefix(p, "mle")]
    stray = [n for n in no_mle if os.path.exists(os.path.join(a.bench, "mle", n + ".npz"))]
    print(json.dumps(dict(lukin_points=len(lukin_pts), other_points=len(ot), files=len(need), n_missing=len(miss),
                          missing=miss[:50], n_no_sha256sums_line=len(nosum), no_sha256sums_line=nosum[:50],
                          network_sha_differs=bad_sha, no_mle=no_mle, mle_file_without_prefix=stray)), flush=True)
    assert not (miss or nosum or bad_sha), "test inputs missing or networks changed"


# ------------------------------------------------------------------------------------------------------------ route

def route(a):
    th = json.load(open(a.thr))
    sd = a.sel_dir or os.path.join(a.test, "sel")
    os.makedirs(sd, exist_ok=True)
    for d in (int(x) for x in a.ds.split(",")):
        name = th["nets"][str(d)]["name"]
        sel, cnt = {f: {} for f in FS}, {f: [] for f in FS}
        for pt in bank.points((d,)):
            r = report.load(a.test, pt, name)
            assert r["net_info"]["sha256"] == th["nets"][str(d)]["sha256"], (pt["name"], "not the fixed network")
            s = np.abs(r["logits"][:, 0])
            for f in FS:
                o = np.flatnonzero(s <= th["t"][str(d)][fkey(f)][pt["depth"] - 1])
                speed()._flat_to_sel(sel[f], pt["name"], o, r["n"])
                cnt[f].append(len(o))
        for f in FS:
            json.dump(sel[f], open(os.path.join(sd, f"route_d{d}_f{fkey(f)}.json"), "w"))
        print(json.dumps(dict(d=d, routed={fkey(f): cnt[f] for f in FS})), flush=True)


_SPEED = []


def speed():
    if not _SPEED:
        _SPEED.append(_mod("scripts/tesseract_speed.py", "lukin_speed"))
    return _SPEED[0]


def tess(a):
    sp = speed()
    opts, lst = sp.SETTINGS[a.setting], json.load(open(a.sel))
    jobs = []
    for pt in bank.points(tuple(int(x) for x in a.ds.split(","))):
        ch = bank.CHUNK["tess"][pt["d"]]
        base = os.path.join(a.dec_dir or os.path.join(a.test, "dec"), f"tess_{a.setting}", pt["name"])
        for c, ss in lst.get(pt["name"], {}).items():
            idx = np.array(sorted(ss), np.int64)
            for k in range(0, len(idx), ch):
                out = os.path.join(base, f"c{c}_{int(idx[k])}.npz")
                if not os.path.exists(out):
                    jobs.append((bank.set_path(a.test, pt), int(c), idx[k:k + ch], out, opts,
                                 pt["L"] * len(idx[k:k + ch]) * pt["d"] ** 3))
    jobs.sort(key=lambda j: -j[-1])
    print(json.dumps(dict(setting=a.setting, jobs=len(jobs), shots=int(sum(len(j[2]) for j in jobs)))), flush=True)
    t0, n, cpu, low = time.time(), 0, 0.0, 0
    with mp.get_context("fork").Pool(a.workers) as pool:
        for k, (_, ns, s, lc) in enumerate(pool.imap_unordered(sp._tess_job, [j[:5] for j in jobs])):
            n, cpu, low = n + ns, cpu + s, low + lc
            if (k + 1) % max(1, len(jobs) // 20) == 0 or k + 1 == len(jobs):
                print(json.dumps(dict(done=k + 1, shots=n, low_conf=low, decode_core_h=round(cpu / 3600, 3),
                                      minutes=round((time.time() - t0) / 60, 1))), flush=True)


# ------------------------------------------------------------------------------------------------------------ rep

def theirs_sd():
    """({(d, D): (value, bar)} of their network ("ML"), the same of their MLE) from Fig. 1b (report.theirs gives the
    variance, the plotted bar squared, or the binomial one at their shots where the figure has no bar)."""
    return tuple({k: (v, math.sqrt(s2)) for k, (v, s2) in t.items()} for t in report.theirs())


def boot_k(rng, per_depth):
    """Draws of the summed failures K*: per depth, its circuits redrawn with replacement, each drawn circuit's failures
    redrawn binomially at its rate and shots. per_depth: [(failures per circuit, shots per circuit)]."""
    K = np.zeros(BOOT)
    for k, n in per_depth:
        idx = rng.integers(len(k), size=(BOOT, len(k)))
        K += rng.binomial(n[idx], k[idx] / n[idx]).sum(1)
    return K


def boot_b(rng, per_depth):
    """Draws of (sum_D x*_D, sum_D m*_D): per depth its circuits redrawn and each drawn circuit's paired shots redrawn
    (both fail, system only, MLE only, neither; binomial steps of the multinomial). per_depth: [(rows per circuit
    [shots, both, x only, m only])]."""
    X, M = np.zeros(BOOT), np.zeros(BOOT)
    for rows in per_depth:
        rows = np.asarray(rows, np.int64)
        idx = rng.integers(len(rows), size=(BOOT, len(rows)))
        n, b, xo, mo = (rows[:, j][idx] for j in range(4))
        b_ = rng.binomial(n, np.where(n > 0, b / np.maximum(n, 1), 0.0))
        r1 = n - b_
        p1 = np.where(n - b > 0, xo / np.maximum(n - b, 1), 0.0)
        x_ = rng.binomial(r1, p1)
        r2 = r1 - x_
        p2 = np.where(n - b - xo > 0, mo / np.maximum(n - b - xo, 1), 0.0)
        m_ = rng.binomial(r2, p2)
        nt = np.maximum(n.sum(1), 1)
        X += (b_ + x_).sum(1) / nt
        M += (b_ + m_).sum(1) / nt
    return X, M


def lognorm(x, v):
    s = math.sqrt(max(v, 0.0))
    return [x * math.exp(-1.96 * s), x * math.exp(1.96 * s)]


def systems(d, extra=()):
    out = [("network", 0.0, None, None)] + [(f"long {fkey(100 * f)} %", f, "tess", None) for f in FS]
    if d in FAST["ds"]:
        out.append((f"{FAST['setting']} {fkey(100 * FAST['f'])} % cap {FAST['cap']:g} s", FAST["f"],
                    f"tess_{FAST['setting']}", FAST["cap"]))
    for e in extra:   # "LABEL|F|DEC": a decoder whose parts already apply their cap
        lab, f, dec = e.split("|")
        out.append((lab, float(f), dec, None))
    return out


def rep(a):
    th = json.load(open(a.thr))
    pnet, pmle = theirs_sd()
    rec = {}
    for d in (int(x) for x in a.ds.split(",")):
        name = th["nets"][str(d)]["name"]
        sysl = systems(d, a.extra)
        decs = sorted({x[2] for x in sysl if x[2]} | {"lomr", a.mle} | ({a.start} if a.start else set()))
        pts = bank.points((d,))
        rs = [report.load(a.test, pt, name, decs) for pt in pts]
        for r, pt in zip(rs, pts):
            assert r["net_info"]["sha256"] == th["nets"][str(d)]["sha256"], (pt["name"], "not the fixed network")
            if a.start_net:
                r["start_dec"] = report.load(a.test, pt, a.start_net)["dec_net"]
            else:
                assert r[a.start]["cov"].all(), "the start's read covers every shot"
                r["start_dec"] = r[a.start]["dec"]
            assert r["lomr"]["cov"].all(), "LoMatching decoded every shot"
            if (bank.TEST or bank.FRESH) and a.mle == "mle":   # the benchmark's exact MLE has shots exactly where the manifest has a
                assert r[a.mle]["cov"].any() == bool(bs.prefix(pt, "mle")), (pt["name"], "MLE")   # prefix (d = 7: 1-8)
        cid =[np.repeat(np.arange(r["C"]), r["n"]) for r in rs]
        fnet = [report.fails(r, r["dec_net"]) for r in rs]
        flom = [report.fails(r, r["lomr"]["dec"])[0] for r in rs]
        fst = [report.fails(r, r["start_dec"])[0] for r in rs]
        mcov = [r[a.mle]["cov"] & r[a.mle]["ok"] for r in rs]
        fmle = [report.fails(r, r[a.mle]["dec"])[0] for r in rs]
        cD = []              # c_D: the circuit variation of a 20-circuit mean (relative), from the network's rates
        for (ft, _), c_, r in zip(fnet, cid, rs):
            p = np.bincount(c_, ft, r["C"]) / r["n"]
            cD.append(max(0.0, p.var(ddof=1) - np.mean(p * (1 - p) / r["n"])) / (r["C"] * p.mean() ** 2)
                      if p.mean() > 0 else 0.0)
        out, F = {}, {}
        for lab, f, dec, cap in sysl:
            depth, kc, per, Fx = [], [], [], []
            for r, (ft, fa), c_, fl, fs_, mc, fm in zip(rs, fnet, cid, flom, fst, mcov, fmle):
                D = r["D"]
                q = dict(D=D, shots=len(ft))
                if dec:
                    s = np.abs(r["logits"][:, 0])
                    o = np.flatnonzero(s <= th["t"][str(d)][fkey(f)][D - 1])
                    x = r[dec]
                    use = x["cov"][o] & x["ok"][o]
                    q.update(routed=len(o), frac=len(o) / len(ft), missing=int((~x["cov"][o]).sum()),
                             low_conf=int((x["cov"][o] & ~x["ok"][o]).sum()))
                    if cap is not None:
                        sec = np.where(x["cov"][o], x["sec"][o], np.inf)
                        q["capped"] = int((x["cov"][o] & (sec > cap)).sum())
                        use &= sec <= cap
                    secs = x["sec"][o][x["cov"][o]]
                    q["dec_s_mean"] = float(np.nanmean(secs)) if len(secs) and not np.isnan(secs).all() else None
                    dx = r["dec_net"].copy()
                    dx[o] = np.where(use, x["dec"][o], r["dec_net"][o])
                    fx, fax = report.fails(r, dx)
                else:
                    fx, fax = ft, fa
                Fx.append(fx)
                q.update(fails=int(fx.sum()), rate=float(fx.mean()), any_fails=int(fax.sum()), lomr=int(fl.sum()),
                         vs_lomr=report.ratio([(fx, fl)])[2:], start=int(fs_.sum()),
                         vs_start=report.ratio([(fx, fs_)])[2:],
                         net_differs_start=int(((r["dec_net"] ^ r["start_dec"]) & r["mask"]).astype(bool).sum()))
                if (d, D) in pnet:
                    q.update(their_net=pnet[(d, D)][0], expected=len(ft) * pnet[(d, D)][0],
                             vs_their_net=float(fx.sum() / (len(ft) * pnet[(d, D)][0])))
                if mc.any() and (d, D) in pmle:
                    xm, mm, cm = fx[mc], fm[mc], c_[mc]
                    q.update(mle_shots=int(mc.sum()), x_on_mle=int(xm.sum()), mle=int(mm.sum()),
                             vs_mle=report.ratio([(xm, mm)])[2:], their_mle=pmle[(d, D)][0])
                    per.append(np.stack([np.bincount(cm, minlength=r["C"]), np.bincount(cm, xm & mm, r["C"]),
                                         np.bincount(cm, xm & ~mm, r["C"]), np.bincount(cm, ~xm & mm, r["C"])],
                                        1).astype(np.int64))
                else:
                    per.append(None)
                kc.append((np.bincount(c_, fx, r["C"]).astype(np.int64), np.full(r["C"], r["n"], np.int64)))
                depth.append(q)
            F[lab] = Fx
            v = dict(depths=depth, score=sum(q["rate"] for q in depth),
                     vs_lomr=report.ratio(list(zip(Fx, flom)))[2:], vs_start=report.ratio(list(zip(Fx, fst)))[2:],
                     vs_network=report.ratio(list(zip(Fx, F["network"])))[2:],
                     routed=sum(q.get("routed", 0) for q in depth),
                     frac=sum(q.get("routed", 0) for q in depth) / sum(q["shots"] for q in depth),
                     missing=sum(q.get("missing", 0) for q in depth), capped=sum(q.get("capped", 0) for q in depth),
                     low_conf=sum(q.get("low_conf", 0) for q in depth))
            # (a): ours / their network, their per-depth values times our shots
            got = [(q, pnet[(d, q["D"])]) for q in depth if (d, q["D"]) in pnet]
            K, E = sum(q["fails"] for q, _ in got), sum(q["shots"] * p for q, (p, _) in got)
            VE = sum((q["shots"] * sd) ** 2 for q, (_, sd) in got) / E ** 2
            VC = sum((q["shots"] * p) ** 2 * cD[q["D"] - 1] for q, (p, _) in got) / E ** 2
            Kb = boot_k(np.random.default_rng(SEED + d), [kc[j] for j, q in enumerate(depth) if (d, q["D"]) in pnet])
            VK = float(np.var(np.log(np.maximum(Kb, 0.5))))
            A = K / E
            v["a"] = dict(depths=f"{got[0][0]['D']}-{got[-1][0]['D']}", fails=K, expected=E, ratio=A,
                          ci=lognorm(A, VK + VE), V_K=VK, V_E=VE, V_circ=VC, ci_sens=lognorm(A, VK + VE + VC))
            v["a"]["holds"] = bool(v["a"]["ci"][1] < 1)
            # (b): (ours / our MLE) / (theirs / their MLE), the depths where our MLE ran
            Db = [q["D"] for q, pr in zip(depth, per) if pr is not None]
            if (bank.TEST or bank.FRESH) and a.mle == "mle":   # (5): (b) over the depths where our MLE ran (d = 3, 5: 1-10, 7: 1-8)
                assert Db == [pt["depth"] for pt in pts if bs.prefix(pt, "mle")], (d, lab, Db)
            if Db:
                qs = [q for q in depth if q["D"] in Db]
                ours = sum(q["x_on_mle"] / q["mle_shots"] for q in qs) / sum(q["mle"] / q["mle_shots"] for q in qs)
                ps, qq = sum(pnet[(d, D)][0] for D in Db), sum(pmle[(d, D)][0] for D in Db)
                Xb, Mb = boot_b(np.random.default_rng(SEED + 100 + d), [pr for pr in per if pr is not None])
                ok = (Xb > 0) & (Mb > 0)
                VO = float(np.var(np.log(Xb[ok] / Mb[ok])))
                VT = sum(pnet[(d, D)][1] ** 2 for D in Db) / ps ** 2 + sum(pmle[(d, D)][1] ** 2 for D in Db) / qq ** 2
                B = ours / (ps / qq)
                v["b"] = dict(depths=f"{Db[0]}-{Db[-1]}", ours=ours, ours_ci=lognorm(ours, VO), theirs=ps / qq,
                              ratio=B, ci=lognorm(B, VO + VT), V_ours=VO, V_theirs=VT, draws_dropped=int((~ok).sum()))
                v["b"]["holds"] = bool(v["b"]["ci"][1] < 1)
            out[lab] = v
        lom = dict(score=sum(float(f_.mean()) for f_ in flom), depths=[dict(D=r["D"], fails=int(f_.sum()),
                                                                           shots=len(f_)) for r, f_ in zip(rs, flom)])
        rec[str(d)] = dict(net=th["nets"][str(d)], c_D=cD, systems=out, lomr=lom, mle=a.mle,
                           start=a.start or a.start_net)
        _print_d(d, out, lom)
    if a.out:
        json.dump(rec, open(a.out, "w"), indent=1, default=float)


def _print_d(d, out, lom):
    fm = report.fmt
    print(f"\n### d = {d}\n")
    print(f"LoMatching: selection score {lom['score']:.4e}.\n")
    print("| system | routed (fraction) | target failures / shots | any product | selection score | / network | "
          "/ LoMatching | / start | (a) / their network [95 %] | sensitivity | (b) depths: ours / our MLE; / theirs "
          "[95 %] | (a), (b) hold |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for lab, v in out.items():
        dp, b = v["depths"], v.get("b")
        print(f"| {lab} | " + (f"{v['routed']:,} ({v['frac']:.4f})" if v["routed"] else "-") +
              f" | {sum(q['fails'] for q in dp)} / {sum(q['shots'] for q in dp):,} | "
              f"{sum(q['any_fails'] for q in dp)} | {v['score']:.4e} | {fm(*v['vs_network'])} | {fm(*v['vs_lomr'])} | "
              f"{fm(*v['vs_start'])} | {fm(v['a']['ratio'], v['a']['ci'])} | {fm(v['a']['ratio'], v['a']['ci_sens'])} | "
              + (f"{b['depths']}: {fm(b['ours'], b['ours_ci'])}; {fm(b['ratio'], b['ci'])} | " if b else "- | ")
              + f"{v['a']['holds']}, {b['holds'] if b else '-'} |")
    for lab, v in out.items():
        print(f"\n{lab}, per depth" + (f" (missing {v['missing']}, capped {v['capped']}, low confidence "
                                       f"{v['low_conf']})" if v["routed"] else "") + ":\n")
        print("| D | shots | routed (fraction) | target failures: rate | any product | / their network | MLE shots: "
              "system / MLE | / MLE | LoMatching | / LoMatching | / start | network differs from start |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for q in v["depths"]:
            print(f"| {q['D']} | {q['shots']:,} | " + (f"{q['routed']:,} ({q['frac']:.4f})" if "routed" in q else "-") +
                  f" | {q['fails']}: {q['rate']:.3e} | {q['any_fails']} | " +
                  (f"{q['vs_their_net']:.3f}" if "vs_their_net" in q else "-") + " | " +
                  (f"{q['mle_shots']:,}: {q['x_on_mle']} / {q['mle']} | {fm(*q['vs_mle'])}" if "mle_shots" in q
                   else "- | -") + f" | {q['lomr']} | {fm(*q['vs_lomr'])} | {fm(*q['vs_start'])} | "
                  f"{q['net_differs_start']:,} |")


# ------------------------------------------------------------------------------------------------------------ guard

def _guard_point(job):
    a, pt = job
    res = _mod("scripts/bench_stats.py", "bench_results")
    name, C = pt["name"], len(pt["circuits"])
    zs = np.load(os.path.join(a.bench, "net", name + ".npz"))          # the start's phase-4 read
    zn = np.load(os.path.join(a.new, "net", name + ".npz"))            # the chosen network's read
    meta = json.load(open(os.path.join(a.release, "points", name, "meta.json")))
    rows = []
    for i in range(C):
        k = int(meta["circuits"][i]["observables"])
        n = int(zs["decoded"][i])
        assert n == int(zn["decoded"][i]) == pt["shots"], (name, i)
        y = bs.unpack(np.fromfile(os.path.join(a.release, "points", name, f"c{i}.obs.b8"), np.uint8).reshape(n, -1), k)
        fs_ = (bs.unpack(zs["pred"][i, :n], k) != y).any(1)
        fn_ = (bs.unpack(zn["pred"][i, :n], k) != y).any(1)
        rows.append([n, int((fn_ & fs_).sum()), int((fn_ & ~fs_).sum()), int((~fn_ & fs_).sum())])
    rows = np.array(rows, np.int64)
    bx, by = res.draws(rows, zlib.crc32(name.encode()) % 2 ** 31, pt["family"] in res.EXHAUSTIVE or C == 1)
    fx, fy = int(rows[:, 1:3].sum()), int(rows[:, [1, 3]].sum())
    return name, res.ratio(fx, fy, bx, by, rows[:, 0].sum()), (fx, fy, int(rows[:, 0].sum()), bx, by)


def guard(a):
    """The network / its start on the test set's other families: any-product
    failures pooled per family by summing counts and draws (scripts/bench_stats.py's draws and ratio: circuits redrawn
    for sampled lists, each circuit's paired shots for lom_2q and one-circuit points); a family fails when its 95 %
    interval lies wholly above 1. ours_k's sub-families (mix, cnot, f1b, struct) are printed as well, descriptive."""
    res = _mod("scripts/bench_stats.py", "bench_results")
    ds = tuple(int(x) for x in a.ds.split(","))
    pts = {p["name"]: p for p in json.load(open(os.path.join(ROOT, "data", "open_benchmark", "MANIFEST.json")))["points"]
           if p["noise"] == "lukin" and p["d"] in ds and p["family"] in OTHER}
    per, acc = {}, {}
    with mp.get_context("fork").Pool(a.workers) as pool:
        for name, r, dr in pool.imap_unordered(_guard_point, [(a, p) for p in pts.values()]):
            per[name] = r
            p = pts[name]
            keys = [(p["d"], p["family"])] + ([(p["d"], "ours_k_" + name.split("_")[2])] if p["family"] == "ours_k"
                                              else [])
            for key in keys:
                x = acc.setdefault(key, [0, 0, 0, np.zeros(res.BOOT), np.zeros(res.BOOT), 0])
                x[0], x[1], x[2], x[5] = x[0] + dr[0], x[1] + dr[1], x[2] + dr[2], x[5] + 1
                x[3] += dr[3]
                x[4] += dr[4]
    out = {}
    print("| d | family | points | shots | failures: network / start | network / start [95 %] | upper end | "
          "191 (2) |")
    print("|---|---|---|---|---|---|---|---|")
    for (d, fam), (fx, fy, n, bx, by, npts) in sorted(acc.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        r = res.ratio(fx, fy, bx, by, n)
        ci = r["ratio95"]
        verdict = ("descriptive" if fam.startswith("ours_k_") else "no failures in either" if r["both_zero"] else
                   "fails (interval wholly above 1)" if ci[0] > 1 else "passes")
        out[f"d{d}|{fam}"] = dict(r, points=npts, verdict=verdict)
        print(f"| {d} | {fam} | {npts} | {n:,} | {fx} / {fy} | " +
              (f"{r['ratio']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {ci[1]:.3f}" if ci else "- | -") + f" | {verdict} |")
    if a.out:
        json.dump(dict(pooled=out, points=per), open(a.out, "w"), indent=0, default=float)


# ------------------------------------------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["thr", "conv", "verify", "check", "route", "tess", "rep", "guard"])
    ap.add_argument("--bank", default=S_LUKIN + "/bank")
    ap.add_argument("--test", default=S_LUKIN + "/test")
    ap.add_argument("--release", default="release")
    ap.add_argument("--bench", default="pred")
    ap.add_argument("--thr", default=os.path.join(ROOT, "results", "lukin", "test", "thresholds.json"))
    ap.add_argument("--ds", default="3,5,7")
    ap.add_argument("--setting", default="o1nc")
    ap.add_argument("--sel", default="")
    ap.add_argument("--mle", default="mle")
    ap.add_argument("--start", default="start")
    ap.add_argument("--start-net", default="")
    ap.add_argument("--extra", action="append", default=[], help="rep: another pair, LABEL|F|DEC (e19.py tpart's parts)")
    ap.add_argument("--new", default="")
    ap.add_argument("--noise", default="lukin", help="conv: the points' model (uniform: the dry run, no test read)")
    ap.add_argument("--sel-dir", default="", help="route: the lists' directory (default T/sel)")
    ap.add_argument("--dec-dir", default="", help="tess: the parts' root (default T/dec)")
    ap.add_argument("--out", default="")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("NCPU", 4)))
    a = ap.parse_args()
    if a.start_net:
        a.start = ""
    dict(thr=thr, conv=conv, verify=verify, check=check, route=route, tess=tess, rep=rep, guard=guard)[a.mode](a)


if __name__ == "__main__":
    main()
