"""Budget-matched subsets: the composition against its fallback alone and against a budget-matched Tesseract preset,
on a fixed subset of Benchmark I's test set.

The subset (fixed before any test read): the first 500 shots of every circuit of the p = 0.3 % points (18 points, K = 2,
3, 4, L = 1-16) at d = 3, 5, 7 (d = 9 only if the budget allows): 576,000 shots per d.
- Tesseract alone: the long beam's 21 trials (ep.py tess, uncapped) on every subset shot, answer rule (the lowest
  cost trial not flagged; a shot whose 21 trials are all flagged has no answer and counts as a failure).
- Budget-matched Tesseract: a Tesseract preset (scripts/tesseract_speed.py SETTINGS, duet.decoders.tesseract_dec) chosen on
  DEVELOPMENT data: for each composition row (the frozen 11-10 choice and every f at its development-best cap), the
  most thorough preset (the slowest one) whose time per layer on the basis's N cores is at most the composition's
  development paced-stream time per layer. The plan's wording is "the shortest preset whose mean time per shot is at
  most the composition's"; read here as the strongest preset inside the composition's time budget (the shortest one
  that fits would be the fastest preset whatever the budget). The preset's time per layer: the mean over the
  timing sample's points of (basis seconds per shot / N / L); the sample: the first 10 shots of every circuit of the
  development bank's p = 0.3 % points of d, each shot decoded alone with every core busy, seconds converted to the

  lists --manifest F --d D --first 500 --out LISTS [--p 0.003] the subset (or the timing sample, --first 10) as
         ep.py tess's lists file
  preset --manifest F --dir SETS --lists L --settings S,S --out DIR [--part j/P] [--workers W] each listed shot
         decoded by every preset: DIR/<setting>/<point>/c<i>_<first>.npz (idx, pred, ok, sec, cpu)
  pick --manifest DEV --timing DIR --curves CJ... --devbest J --frozen J --cal CAL --d D --out J the budget match
         per (d, basis, row) on development data
  score --manifest F --dir SETS --net NET --thr THR --tess TESS --subtess TESS2 --preset DIR --pick J --cal CAL
         --d D --out J on the subset: the composition at every run cell, Tesseract alone and the picked presets:
         failures, paired ratios (score.py's statistics) and time per layer"""
import argparse
import glob
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grid_scores as grid  # noqa: E402
import operating_points as route  # noqa: E402
import score_test as score  # noqa: E402
import eval_pipeline as ep  # noqa: E402

from duet import bench_sets as bs  # noqa: E402

SPEED = score._mod("scripts/tesseract_speed.py", "lukin_speed")
SETTINGS = SPEED.SETTINGS


def cmd_lists(a):
    m = ep.load_manifest(a.manifest)
    pts = [pt for pt in m["points"] if pt["d"] == a.d and pt["p"] == a.p]
    assert pts, ("no point", a.d, a.p)
    lists = {pt["name"]: {str(i): list(range(min(a.first, pt["shots"]))) for i in range(len(pt["circuits"]))}
             for pt in pts}
    json.dump(dict(rule=f"the first {a.first} shots of every circuit of the p = {a.p:g} points of d = {a.d}",
                   shots=int(sum(len(c) for v in lists.values() for c in v.values())), lists=lists), open(a.out, "w"))
    print(json.dumps(dict(points=len(lists), shots=int(sum(len(c) for v in lists.values() for c in v.values())))))


def _preset_job(job):
    """scripts/tesseract_speed.py's _tess_job on a set in the first test's layout: shots idx of circuit i decoded by one preset, each timed alone."""
    path, i, idx, out, opts = job
    if os.path.exists(out):
        return 0, 0.0, 0
    from duet.decoders.tesseract_dec import Tesseract
    z = np.load(path)
    entry, m = json.loads(str(z["entry"])), int(z["m"][i])
    dets = bs.unpack(bs.packed(path, "dets")[i][idx], m)
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
             cpu=ep.cpu(), settings=json.dumps(dec.settings, default=str))
    os.replace(out + ".tmp.npz", out)
    return len(idx), float(sec.sum()), int((~ok).sum())


def cmd_preset(a):
    m = ep.load_manifest(a.manifest)
    pts = {pt["name"]: pt for pt in m["points"]}
    Ls = json.load(open(a.lists))["lists"]
    jobs = []
    for s in a.settings.split(","):
        opts = SETTINGS[s]
        for nm, cs in Ls.items():
            pt = pts[nm]
            ch = ep.tess_chunk(pt["d"], pt["L"]) * 4
            for c, ss in cs.items():
                ss = sorted(ss)
                for k in range(0, len(ss), ch):
                    idx = np.array(ss[k:k + ch], np.int64)
                    out = os.path.join(a.out, s, nm, f"c{c}_{int(idx[0])}.npz")
                    w = pt["d"] ** 4 * pt["K"] * pt["L"] ** 1.5 * len(idx) * SPEED.COST.get(s, 1)
                    jobs.append((w, (ep.set_path(a.dir, pt), int(c), idx, out, opts)))
    jobs.sort(key=lambda j: -j[0])
    jobs = ep.part_of(jobs, a.part, key=lambda j: j[1][3], weight=lambda j: j[0])
    print(json.dumps(dict(settings=a.settings, jobs=len(jobs), shots=int(sum(len(j[1][2]) for j in jobs)),
                          cpu=ep.cpu())), flush=True)
    t0, n, s_, low = time.time(), 0, 0.0, 0
    with mp.get_context("fork").Pool(a.workers) as pool:
        for k, (ns, ss, lc) in enumerate(pool.imap_unordered(_preset_job, [j[1] for j in jobs])):
            n, s_, low = n + ns, s_ + ss, low + lc
            if (k + 1) % max(1, len(jobs) // 20) == 0 or k + 1 == len(jobs):
                print(json.dumps(dict(done=k + 1, shots=n, low_conf=low, core_h=round(s_ / 3600, 3),
                                      minutes=round((time.time() - t0) / 60, 1))), flush=True)


def load_preset(pdir, setting, pt, n):
    """A preset's parts of a point: flat index (circuit * n + shot) sorted, pred, ok, sec, cpu per shot."""
    parts = []
    for f in sorted(glob.glob(os.path.join(pdir, setting, pt["name"], "c*_*.npz"))):
        if f.endswith(".tmp.npz"):
            continue
        i = int(os.path.basename(f)[1:].split("_")[0])
        with np.load(f) as z:
            parts.append(dict(q=i * n + np.asarray(z["idx"], np.int64), pred=np.asarray(z["pred"], np.uint8),
                              ok=np.asarray(z["ok"], bool), sec=np.asarray(z["sec"], np.float64),
                              cpu=np.full(len(z["idx"]), str(z["cpu"]), object)))
    if not parts:
        return None
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    o = np.argsort(cat["q"], kind="stable")
    out = {k: v[o] for k, v in cat.items()}
    assert np.all(np.diff(out["q"]) > 0), (pt["name"], setting, "a shot decoded twice")
    return out


def basis_sec(sec, cpus, fac, d):
    return sec * route.row_factors(dict(cpu=cpus), fac, d)


def cmd_pick(a):
    """Per (d, basis) and composition row (development data only): the slowest preset whose time per layer on N cores
    is at most the composition's development paced-stream time per layer."""
    m = ep.load_manifest(a.manifest)
    pts = [pt for pt in m["points"] if pt["d"] == a.d and pt["p"] == 0.003]
    facs = {b: route.parse_fac("", a.cal, route.SPARK_CPU if b == "spark" else 1.0, cal_ds=(a.d,)) for b in grid.BASES}
    tim = {}
    for s in SETTINGS:
        per = []
        for pt in pts:
            P = load_preset(a.timing, s, pt, pt["shots"])
            if P is None:
                break
            per.append((pt, P))
        if len(per) != len(pts):
            continue
        tim[s] = {}
        for b in grid.BASES:
            v = [float(basis_sec(P["sec"], P["cpu"], facs[b], a.d).mean()) / grid.NB[b] / pt["L"] * 1e6
                 for pt, P in per]
            tim[s][b] = dict(us_per_layer=float(np.mean(v)), shots=int(sum(len(P["q"]) for _, P in per)),
                             no_solution=int(sum((~P["ok"]).sum() for _, P in per)))
    assert tim, "no preset has a complete timing sample"
    rows = []
    frozen = json.load(open(a.frozen))["rows"]
    dbest = json.load(open(a.devbest))["rows"]
    curves = {}
    for p in a.curves.split(","):
        cj = json.load(open(p if p.endswith(".json") else p + ".json"))
        if cj["fail"] == "any" and cj["d"] == a.d:
            curves[cj["basis"]] = {(r["f"], r["cap"]): r for r in cj["rows"] if r["K"] == "all" and r["N"] == grid.NB[cj["basis"]]}
    for b in grid.BASES:
        want = [("frozen", route.fkey(float(x["f"])), x["cap"]) for x in frozen
                if x["d"] == a.d and x["result"] == 1 and x["basis"] == b]
        want += [("devbest", r["f"], r["cap"]) for r in dbest if r["d"] == a.d and r["result"] == 1 and r["basis"] == b
                 and r["N"] == grid.NB[b] and float(r["f"]) > 0]
        for kind, f, c in want:
            us = curves[b][(f, c)]["us_per_layer_stream"]
            fit = [s for s in tim if tim[s][b]["us_per_layer"] <= us]
            pick = max(fit, key=lambda s: tim[s][b]["us_per_layer"]) if fit else None
            rows.append(dict(d=a.d, basis=b, kind=kind, f=f, cap=c, comp_us_per_layer=us, preset=pick,
                             preset_us_per_layer=tim[pick][b]["us_per_layer"] if pick else None))
    json.dump(dict(rule="the slowest preset whose time per layer (basis seconds per shot / N / L, mean over the "
                        "development timing sample's p = 0.3 % points) is at most the composition's development "
                        "paced-stream time per layer; development data only", d=a.d, timing=tim, rows=rows),
              open(a.out, "w"), indent=0)
    for r in rows:
        print(json.dumps(r))


def cmd_score(a):
    """The subset's comparisons: per point the composition at every run cell, Tesseract alone and each picked preset (no solution = failure), paired ratios with score.py's
    statistics (pooled over the subset's points), and time per layer on each basis."""
    m, thr, pts0, sha = grid.frozen_inputs(a)
    pts = [pt for pt in pts0 if pt["p"] == 0.003]
    a.fail = "any"
    facs = grid.facs_of(a, a.d)
    picks = sorted({r["preset"] for r in json.load(open(a.pick))["rows"] if r["preset"]}) if a.pick else []
    run = [x for x in grid.grid_cells() if grid.exact(x[1], x[2])]
    pool, rows = grid.Pool(), []
    sub = json.load(open(a.sublists))["lists"]
    pts = [pt for pt in pts if pt["name"] in sub]
    assert pts, "no subset point in the manifest"
    for pt in pts:
        lg = route.load_logits(a.net, a.name, pt)
        C, n = lg.shape[:2]
        truth, netp = route.point_truth(a, pt, lg)
        S = sub[pt["name"]]
        k = len(S["0"])
        assert all(v == list(range(k)) for v in S.values()), (pt["name"], "the subset is a prefix of every circuit")
        q = (np.arange(C)[:, None] * n + np.arange(k)[None, :]).reshape(-1)
        nf = (netp != truth)[q].reshape(C, k)
        F = {"network": nf}
        tim = {}
        Tl = route.load_trials(a.subtess, pt, n, None, convert=False)
        have, pos = route.lookup(Tl, q)
        assert have.all() and np.isinf(Tl["dl"][pos]).all(), (pt["name"], "Tesseract alone: every subset shot uncapped")
        ans, ok = route.answers(Tl, pos, None)
        F["tess_alone"] = ~ok.reshape(C, k) | (ans != truth[q]).reshape(C, k)
        for b in grid.BASES:
            T = route.load_trials(a.subtess, pt, n, facs[b])
            tim[f"tess_alone|{b}"] = float(T["sec"][pos].sum(1).mean()) / grid.NB[b] / pt["L"] * 1e6
            Tr = route.load_trials(a.tess, pt, n, facs[b])
            for _, f, c in [x for x in run if x[0] == b]:
                dec, rt, an, miss = score.comp_dec(pt, lg, netp, dict(f=f, cap=c, thr=thr), Tr)
                assert a.dry or miss == 0, (pt["name"], f, c, miss)
                F[grid.lab(b, f, c)] = (dec != truth)[q].reshape(C, k)
            del Tr
        for s in picks:
            P = load_preset(a.preset, s, pt, n)
            assert P is not None, (pt["name"], s, "no preset parts")
            hv, ps = route.lookup(dict(q=P["q"]), q)
            assert hv.all(), (pt["name"], s, "the preset decoded every subset shot")
            F[f"preset_{s}"] = (~P["ok"][ps] | (P["pred"][ps] != truth[q])).reshape(C, k)
            for b in grid.BASES:
                tim[f"preset_{s}|{b}"] = float(basis_sec(P["sec"][ps], P["cpu"][ps], facs[b], a.d).mean()) / \
                    grid.NB[b] / pt["L"] * 1e6
        R = grid.boot_rows(pt["name"] + "|subset", C)
        ns = np.full(C, k)
        r = dict(point=pt["name"], K=pt["K"], L=pt["L"], shots=C * k, fails={s: int(v.sum()) for s, v in F.items()},
                 us_per_layer=tim, pairs={})
        refs = ["tess_alone"] + [f"preset_{s}" for s in picks]
        for x in [s for s in F if s not in refs]:
            for y in refs:
                X, Y = F[x], F[y]
                st, logs = grid.paired_z(ns, X.sum(1), Y.sum(1), (int((X & ~Y).sum()), int((~X & Y).sum())), R, 0.5)
                r["pairs"][f"{x}/{y}"] = st
                pool.add((f"{x}/{y}", f"d{a.d}_K{pt['K']}"), st, logs)
                pool.add((f"{x}/{y}", f"d{a.d}_all"), st, logs)
        rows.append(r)
        print(json.dumps(dict(point=pt["name"], fails={s: v for s, v in r["fails"].items() if "|" not in s})),
              flush=True)
    pooled = {}
    for (pair, g), v in pool.out().items():
        pooled.setdefault(pair, {})[g] = v
    times = {key: float(np.mean([r["us_per_layer"][key] for r in rows])) for key in rows[0]["us_per_layer"]}
    json.dump(dict(d=a.d, net_sha256=sha, subset=a.sublists, presets=picks, points=rows, pooled=pooled,
                   us_per_layer=times, dry=a.dry), open(a.out, "w"), indent=0, default=float)
    print(json.dumps(dict(us_per_layer=times)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["lists", "preset", "pick", "score"])
    ap.add_argument("--manifest", default="")
    ap.add_argument("--dir", default="")
    ap.add_argument("--d", type=int, default=0)
    ap.add_argument("--p", type=float, default=0.003)
    ap.add_argument("--first", type=int, default=500)
    ap.add_argument("--lists", default="")
    ap.add_argument("--sublists", default="")
    ap.add_argument("--settings", default=",".join(SETTINGS))
    ap.add_argument("--part", default="")
    ap.add_argument("--timing", default="")
    ap.add_argument("--curves", default="")
    ap.add_argument("--devbest", default="")
    ap.add_argument("--frozen", default="")
    ap.add_argument("--cal", default="")
    ap.add_argument("--fac", default="")
    ap.add_argument("--net", default="")
    ap.add_argument("--name", default="main")
    ap.add_argument("--thr", default="")
    ap.add_argument("--tess", default="")
    ap.add_argument("--subtess", default="")
    ap.add_argument("--preset", default="")
    ap.add_argument("--pick", default="")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("NCPU", 1)))
    ap.add_argument("--unc", type=float, default=0.02, help="score: the routed share run uncapped (grid.py --unc)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    grid.UNC_F = a.unc
    dict(lists=cmd_lists, preset=cmd_preset, pick=cmd_pick, score=cmd_score)[a.mode](a)


if __name__ == "__main__":
    main()
