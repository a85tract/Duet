"""Statistics of the open benchmark's test sets from the stored shots: a shot fails when any reliable product
is decoded wrong; the lukin family also counts its target parity (product slot 0). Shots whose MLE result is not
certified are left out for every decoder.
- Per point and decoder: failures and shots, the rate with a 95 % Wilson interval.
- Where the exact MLE ran, each matching decoder X against it on the shots both decoded (each circuit's shorter
  prefix): failures of X / failures of the MLE, with a 95 % interval from 10,000 bootstrap draws, the same draws for
  both. A sampled circuit list (lukin, mccd, ours_k) has its circuits redrawn with their counts; an exhaustive one
  (lom_2q's fixed roster, every one-circuit point) has each circuit's shots redrawn paired, a multinomial over the
  four outcomes, when every circuit holds at least MIN_SHOTS paired shots, and
  otherwise the point's paired shots redrawn together. Both counts zero: "no failures in either",
  no ratio or interval; one count zero: 0.5 added to both, flagged, the ratio then a bound. Pooled
  per (family, model, d): the counts and the draws summed over the points.
- The networks ("net", phase 4): the same comparison against the MLE, LoMatching and BP+LoMatching (net_vs), on the
  shots both decoded.
- Buildable-subset control: where a matching decoder dropped an error on some circuit of a point,
  the same counts and ratios on the circuits where none did, with the left-out circuits counted; pooled ratios with
  every other point unchanged.

  python scripts/bench_stats.py --manifest M --sets SETS --pred PRED --out results.json --workers N [--only REGEX]"""
import argparse
import json
import math
import multiprocessing as mp
import os
import re
import time
import zlib

import numpy as np

from duet import bench_sets as bs

DECODERS = ("lomr", "cain", "bplomr", "mle", "net")
MATCHING = ("lomr", "cain", "bplomr")
NET_REFS = ("mle", "lomr", "bplomr")         # the network against these
EXHAUSTIVE = ("lom_2q",)                     # families whose points hold their whole circuit roster
MIN_SHOTS = 10                               # exhaustive lists: a circuit with fewer paired shots pools its point's
BOOT = 10_000
Z95 = 1.959963984540054
A = None                                     # arguments and per-circuit dropped counts, shared with the workers by fork


def wilson(k, n, z=Z95):
    if n == 0:
        return [math.nan, math.nan]
    p, den = k / n, 1 + z * z / n
    mid, half = (p + z * z / (2 * n)) / den, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, mid - half), min(1.0, mid + half)]


def draws(tab, seed, shots, boot=BOOT):
    """Per-draw failure counts of X and Y at one point from its per-circuit rows [shots, both fail, X only, Y only]:
    the circuits redrawn with their counts, or (shots=True: an exhaustive circuit list) each circuit's shots redrawn,
    paired between X and Y. Where a circuit holds fewer than MIN_SHOTS paired shots, the point's paired shots are
    redrawn together instead: a redraw of n shots keeps (n - 1) / n of their variance, none at one shot, and the pooled
    redraw adds the spread between circuits, so it errs wide."""
    tab = np.asarray(tab, np.int64).reshape(-1, 4)
    rng = np.random.default_rng(seed)
    n, both, xo, yo = tab.T
    if not shots:
        rows = rng.integers(len(tab), size=(boot, len(tab)))
        return (both + xo)[rows].sum(1), (both + yo)[rows].sum(1)
    if (n < MIN_SHOTS).any():
        tab = tab.sum(0, keepdims=True)
    bx, by = np.zeros(boot, np.int64), np.zeros(boot, np.int64)
    for r in tab:
        if r[1] + r[2] + r[3]:
            d = rng.multinomial(int(r[0]), np.array([r[1], r[2], r[3], r[0] - r[1] - r[2] - r[3]], np.float64) / r[0],
                                size=boot)
            bx += d[:, 0] + d[:, 1]
            by += d[:, 0] + d[:, 2]
    return bx, by


def ratio(fx, fy, bx, by, shots):
    """X / Y of failure counts with its 95 % interval from the draws; 0.5 added to both where either is zero. Both
    counts zero: no ratio and no interval (both_zero)."""
    if fx == 0 and fy == 0:
        return dict(shots=int(shots), fails_x=0, fails_ref=0, ratio=None, ratio95=None, zero_count=False,
                    both_zero=True)

    def add(x, y):
        z = (x == 0) | (y == 0)
        return x + 0.5 * z, y + 0.5 * z
    rx, ry = add(np.float64(fx), np.float64(fy))
    dx, dy = add(np.asarray(bx, np.float64), np.asarray(by, np.float64))
    return dict(shots=int(shots), fails_x=int(fx), fails_ref=int(fy), ratio=float(rx / ry),
                ratio95=np.exp(np.percentile(np.log(dx / dy), [2.5, 97.5])).tolist(), zero_count=bool(fx == 0 or fy == 0),
                both_zero=False)


def point(pt):
    """(name, record, draws for pooling) of one point."""
    name = pt["name"]
    path = os.path.join(A.sets, name + ".npz")
    k = np.load(path)["k"]
    O = bs.packed(path, "obs")
    P = {}
    for dec in DECODERS:
        f = os.path.join(A.pred, dec, name + ".npz")
        if os.path.exists(f):
            z = np.load(f)
            P[dec] = {x: z[x] for x in ("pred", "ok", "decoded")}
            assert len(P[dec]["decoded"]) == len(pt["circuits"]), (name, dec)
    F, T, V = ({d: [] for d in P} for _ in range(3))      # per circuit: failure, target failure, counted shot
    for i, c_ in enumerate(pt["circuits"]):
        n = {d: int(P[d]["decoded"][i]) for d in P}
        obs = bs.unpack(O[i, :max(n.values())], k[i])
        cert = P["mle"]["ok"][i, :n["mle"]].astype(bool) if "mle" in P else np.ones(0, bool)
        for d in P:
            pred, y = bs.unpack(P[d]["pred"][i, :n[d]], k[i]), obs[:n[d]]
            valid = np.ones(n[d], bool)
            valid[:min(n[d], len(cert))] = cert[:n[d]]
            F[d].append((pred != y).any(1))
            V[d].append(valid)
            ts = c_.get("target_slot")
            T[d].append(None if ts is None else pred[:, ts] != y[:, ts])
    C = len(pt["circuits"])
    drop = A.dropped.get(name, {})
    keep_b = [i for i in range(C) if drop.get(i, 0) == 0]
    seed = zlib.crc32(name.encode()) % 2 ** 31

    def counts(d, keep):
        sh = sum(int(V[d][i].sum()) for i in keep)
        fl = sum(int((F[d][i] & V[d][i]).sum()) for i in keep)
        r = dict(shots=sh, fails=fl, rate=fl / sh if sh else None, rate95=wilson(fl, sh))
        if T[d] and T[d][0] is not None:
            tf = sum(int((T[d][i] & V[d][i]).sum()) for i in keep)
            r.update(target_fails=tf, target_rate=tf / sh if sh else None, target_rate95=wilson(tf, sh))
        return r

    def paired(x, y, keep):
        rows = []
        for i in keep:
            n = min(len(F[x][i]), len(F[y][i]))
            ok = V[x][i][:n] & V[y][i][:n]
            fx, fy = F[x][i][:n][ok], F[y][i][:n][ok]
            rows.append([len(fx), int((fx & fy).sum()), int((fx & ~fy).sum()), int((~fx & fy).sum())])
        rows = np.array(rows, np.int64).reshape(-1, 4)
        bx, by = draws(rows, seed, pt["family"] in EXHAUSTIVE or len(rows) == 1)
        fx, fy = int(rows[:, 1:3].sum()), int(rows[:, [1, 3]].sum())
        return ratio(fx, fy, bx, by, rows[:, 0].sum()), (fx, fy, int(rows[:, 0].sum()), bx.astype(np.int32),
                                                         by.astype(np.int32))

    rec = dict(family=pt["family"], d=pt["d"], noise=pt["noise"], p=pt["p"], depth=pt["depth"], L=pt["L"], K=pt["K"],
               circuits=C, shots=pt["shots"], dec={d: counts(d, range(C)) for d in P},
               uncertified=int(sum((~V["mle"][i]).sum() for i in range(C))) if "mle" in P else 0,
               circuits_dropped=C - len(keep_b), vs_mle={}, net_vs={})
    dr = {}
    if "mle" in P:
        for x in (d for d in MATCHING if d in P):
            rec["vs_mle"][x], dr[x] = paired(x, "mle", range(C))
    for y in (d for d in NET_REFS if "net" in P and d in P):
        rec["net_vs"][y], dr["net|" + y] = paired("net", y, range(C))
    if len(keep_b) < C:
        rec["buildable"] = dict(circuits=len(keep_b), dec={d: counts(d, keep_b) for d in P}, vs_mle={}, net_vs={})
        if "mle" in P and keep_b:
            for x in (d for d in MATCHING if d in P):
                rec["buildable"]["vs_mle"][x], dr["b_" + x] = paired(x, "mle", keep_b)
        for y in (d for d in NET_REFS if "net" in P and d in P and keep_b):
            rec["buildable"]["net_vs"][y], dr["b_net|" + y] = paired("net", y, keep_b)
    return name, rec, dr


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--sets", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default="")
    A = ap.parse_args()
    t0 = time.time()
    pts = [pt for pt in json.load(open(A.manifest))["points"] if not A.only or re.search(A.only, pt["name"])]
    summ = {d: json.load(open(os.path.join(A.pred, d, "summary.json"))) for d in DECODERS
            if os.path.exists(os.path.join(A.pred, d, "summary.json"))}
    A.dropped = {}                           # per point and circuit: the most errors any matching decoder dropped
    for d in MATCHING:
        for name, s in summ.get(d, {}).items():
            for c in s["per_circuit"]:
                A.dropped.setdefault(name, {})[c["i"]] = max(A.dropped.get(name, {}).get(c["i"], 0), c["dropped"])
    pts.sort(key=lambda pt: -pt["shots"] * len(pt["circuits"]) * pt["K"] * pt["L"])
    out, groups = {}, {}
    with mp.get_context("fork").Pool(A.workers) as pool:
        for name, rec, dr in pool.imap_unordered(point, pts):
            out[name] = rec
            if "mle" in summ and name in summ["mle"]:
                rec["mle_cpu"] = summ["mle"][name]["cpu"]
            if dr:
                groups.setdefault(f"{rec['family']}|{rec['noise']}|{rec['d']}", []).append((rec, dr))
    pooled = {}
    pool = lambda its: ratio(sum(t[0] for t in its), sum(t[1] for t in its), sum(t[3] for t in its),   # noqa: E731
                             sum(t[4] for t in its), sum(t[2] for t in its))
    for key, items in sorted(groups.items()):
        g = dict(points=sum("mle" in r["dec"] for r, _ in items),
                 circuits_dropped=sum(r["circuits_dropped"] for r, _ in items), vs_mle={}, buildable_vs_mle={},
                 net_vs={}, buildable_net_vs={})
        for dec in ("mle", "net"):
            cs = [r["dec"][dec] for r, _ in items if dec in r["dec"]]
            if cs:
                g[dec] = dict(points=len(cs), shots=sum(c["shots"] for c in cs), fails=sum(c["fails"] for c in cs))
                g[dec]["rate95"] = wilson(g[dec]["fails"], g[dec]["shots"])
        for kind, pairs in (("vs_mle", [(x, x, "b_" + x) for x in MATCHING]),
                            ("net_vs", [(y, "net|" + y, "b_net|" + y) for y in NET_REFS])):
            for x, k, kb in pairs:
                its = [dr[k] for _, dr in items if k in dr]
                if its:
                    g[kind][x] = pool(its)
                    if g["circuits_dropped"]:    # a point with no buildable circuit adds nothing
                        g["buildable_" + kind][x] = pool([dr[k] if not r["circuits_dropped"] else dr[kb]
                                                          for r, dr in items
                                                          if k in dr and (not r["circuits_dropped"] or kb in dr)])
        pooled[key] = g
    with open(A.out, "w") as f:
        json.dump(dict(points=dict(sorted(out.items())), pooled=pooled, boot=BOOT,
                       sets=A.sets, pred=A.pred), f, indent=0)
    print(f"{len(out)} points, {len(pooled)} pooled groups, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
