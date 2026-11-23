"""Routed-trial lists and full-grid scores. Nothing here chooses anything on a test set: the routing thresholds are the frozen development ones, the answer is rule under every
(route.py thr on MANIFEST_dev.json / MANIFEST_lukin_dev.json, data/fresh_test/frozen/), the answer is rule under every
simulated cap, and every cell of the grid is reported.

The grid: f in {0.5, 1, 2, 5, 10, 20 %} (f = 0: the network alone) x c in {0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
the least confident 2 % (the shots routed at f = 2 %) uncapped, the rest with the local per-trial deadline
2 x 0.1 x L / F_grace = 0.2 s x L Grace seconds. So every cap up
to 0.1 is exact at every f (the Spark basis: 0.2 x 1.124 L >= 0.1 L), and "none" only at f <= 2 %; the other
(f, none) cells are listed as not run.

  devbest --curve CJ[,CJ...] --out J per (d, basis N, f) the development curve's best cap: the smallest mean log
          failure rate (0.5 zero rule) among the caps whose paced-stream time per layer is below the bar, ties to the
          larger cap; with none below the bar the fastest cap, flagged. Development data only; route.py choose's rule restricted to one f.
  bench1 --manifest F --dir SETS --net NET --thr THR --tess TESS --dec DEC --cal CAL --d D --out J [--opps] [--dry]
          main result 1 for the network and every run cell: failures, the routing table, paired
          ratios against LoMatching, Cain et al., BP+LoMatching (its matched shots) and the network (score.py bench1's
          statistics: eval_stats.paired's circuit draws, seed crc32 of the point; pooled per (d, K) and d over the p =
          0.3 % points and over the p = 0.1, 0.2 % points), per-depth absolute rates with circuit-bootstrap intervals,
          and the zero rule's sensitivity (the constant 0.5 replaced by 0.1, 0.25, 1, or the zero points dropped)
  bench2 --manifest F --bank B --net NET --thr THR --tess TESS --cal CAL --d D --out J [--dry] [--mle-uncertified]
          main result 2: score.py bench2 for every run cell,
          then the routing table and the per-depth absolute rates (target parity) of every system
  report --mr J --curve CJ_grace,CJ_spark --devbest J --frozen J --out MD one result and d's tables

Routing table, per cell: shots, routed, answered by the fallback, routed but kept the network's
answer (no trial within the cap), the network's failures on kept and on routed shots, the composition's failures on
routed shots, corrections (network wrong, composition right) and introduced errors (network right, composition wrong);
the composition's failures = network failures kept + composition failures routed."""
import argparse
import json
import math
import multiprocessing as mp
import os
import re
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import operating_points as route  # noqa: E402
import score_test as score  # noqa: E402
import eval_pipeline as ep  # noqa: E402

from duet import bench_sets as bs  # noqa: E402

ES = score._mod("scripts/paired_stats.py", "paired_stats")
BOOT = ES.BOOT
BASES = ("grace", "spark")
NB = {"grace": 64, "spark": 20}
GRID_FS = route.FS[1:]
UNC_F, DL_C, F_MAX = 0.02, 0.1, 0.2
Z0S = (0.5, 0.1, 0.25, 1.0)                     # 0.5 first: the frozen rule
SENS_GROUPS = ("all_p3", "lowp")
TABLE = ("shots", "routed", "answered", "kept_network", "missing", "net_fail_kept", "net_fail_routed",
         "comp_fail_routed", "corrected", "introduced")


def exact(f, c):
    """The trial plan serves cell (f, c) exactly: every finite cap (<= 0.1) at every f <= 20 %, none at f <= UNC_F
    (the plan's 2 %; --unc sets it, the same value for tlist and the scores)."""
    return f <= UNC_F or c is not None


def lab(b, f, c):
    return f"{b}|{route.fkey(f)}|{route.ckey(c)}"


def grid_cells():
    return [(b, f, c) for b in BASES for f in GRID_FS for c in route.CAPS]


def facs_of(a, d):
    """Per basis the trial seconds' conversion at d (route.basis_fac's: cal.sh's factors, x 1.124 on the Spark basis)."""
    out = {}
    for b in BASES:
        out[b] = route.parse_fac(a.fac, a.cal, route.SPARK_CPU if b == "spark" else 1.0, cal_ds=(d,))
        assert out[b].get(d), (d, b, "no CPU factor at this d")
    return out


def frozen_inputs(a):
    """The manifest's points of d (each read whole by one network) and the frozen thresholds of that network."""
    m = ep.load_manifest(a.manifest)
    thr = json.load(open(a.thr))
    pts, sha = route.complete_points(m, a.net, a.name, a.d)
    assert thr.get("d") == a.d and thr.get("net_sha256") == sha, \
        ("thresholds of another d or network", thr.get("d"), thr.get("net_sha256"), sha)
    return m, thr, pts, sha


# ----------------------------------------------------------------------------------------------------------- tlist

def cmd_tlist(a):
    m, thr, pts, sha = frozen_inputs(a)
    assert UNC_F in route.FS and 0 < UNC_F <= F_MAX, ("--unc must be one of the grid's f", UNC_F)
    lists, req, tot, unc = {}, {}, 0, 0
    for pt in pts:
        lg = route.load_logits(a.net, a.name, pt)
        n = lg.shape[1]
        s_ = route.score_of(lg).reshape(-1)
        r20, r2 = route.is_routed(s_, thr, pt, F_MAX), route.is_routed(s_, thr, pt, UNC_F)
        assert not (r2 & ~r20).any(), (pt["name"], "routing is not nested")
        qs = np.flatnonzero(r20)
        if not len(qs):
            continue
        cs = qs // n
        lists[pt["name"]] = {str(int(c)): (qs[cs == c] % n).tolist() for c in np.unique(cs)}
        req[pt["name"]] = {str(int(c)): ["none" if r2[x] else f"grace:{DL_C:g}" for x in qs[cs == c]]
                           for c in np.unique(cs)}
        tot, unc = tot + len(qs), unc + int(r2.sum())
    json.dump(dict(rule=f"every shot routed at f = {F_MAX:g} by the frozen development thresholds; "
                        f"routed at f = {UNC_F:g}: uncapped (none), the rest grace:{DL_C:g} (ep.py tess --deadline req: "
                        f"local deadline 2 x {DL_C:g} x L / F_grace)", thr=a.thr, net_sha256=sha, basis=["grace"],
                   unc_f=UNC_F, shots=tot, uncapped=unc, lists=lists, req=req), open(a.out, "w"))
    print(json.dumps(dict(d=a.d, points=len(lists), shots=tot, uncapped=unc, unc_f=UNC_F)))


# --------------------------------------------------------------------------------------------------------- devbest

def cmd_devbest(a):
    capv = lambda c: math.inf if c == "none" else float(c)
    out = []
    for path in a.curve.split(","):
        cj = json.load(open(path if path.endswith(".json") else path + ".json"))
        rows = [r for r in cj["rows"] if r["K"] == "all"]
        for d in sorted({r["d"] for r in rows}):
            for N in sorted({r["N"] for r in rows if r["d"] == d}):
                for f in route.FS:
                    cand = [r for r in rows if r["d"] == d and r["N"] == N and float(r["f"]) == f]
                    assert len(cand) == len(route.CAPS), (path, d, N, f, len(cand))
                    assert not any(r["missing"] or r["timed_missing"] for r in cand), (path, d, N, f, "missing trials")
                    ok = [r for r in cand if r["us_per_layer_stream"] < route.BAR_US[d]]
                    best = (min(ok, key=lambda r: (r["mlog_rate"], -capv(r["cap"]))) if ok
                            else min(cand, key=lambda r: (r["us_per_layer_stream"], -capv(r["cap"]))))
                    out.append(dict(curve=path, result=1 if cj["fail"] == "any" else 2, basis=cj["basis"], d=d, N=N,
                                    f=route.fkey(f), cap=best["cap"],
                                    below_bar=bool(ok), fails=best["fails"], shots=best["shots"],
                                    gmean_rate=best["gmean_rate"], us_per_layer_stream=best["us_per_layer_stream"],
                                    p99_ms_stream=best["p99_ms_stream"], bar=route.BAR_US[d]))
    json.dump(dict(rule="per f: the smallest mean log failure rate (0.5 zero rule) among caps below the bar, ties to "
                        "the larger cap; none below the bar: the fastest cap (below_bar false); development curves only",
                   rows=out), open(a.out, "w"), indent=0)
    for r in out:
        print(f"result {r['result']} {r['basis']} d={r['d']} N={r['N']} f={r['f']}: cap {r['cap']} "
              f"{'' if r['below_bar'] else '(none below the bar) '}{r['fails']}/{r['shots']} {r['us_per_layer_stream']:.1f} us")


# ---------------------------------------------------------------------------------------------------------- stats

def rates_z(n, fx, fy, z0):
    """eval_stats.rates with the zero rule's constant z0 (0.5: the frozen rule; the same floating-point steps)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (fx.sum(-1) == 0) | (fy.sum(-1) == 0)
        add = np.where(z, z0 / n.sum(-1), 0.0)
    return (fx / n).mean(-1) + add, (fy / n).mean(-1) + add, z


def boot_rows(name, C):
    """eval_stats.paired's circuit draws of a point (mode circuits, seed crc32 of its name)."""
    return np.random.default_rng(zlib.crc32(name.encode()) % 2 ** 31).integers(C, size=(BOOT, C))


def draw_stats(n, f, rows):
    """Per draw of rows a system's equal-weight mean rate, zero flag and summed shots, by rates_z's own steps, so one
    system's draws serve every pair it enters (the same numbers as rates_z on the pair's gathered arrays)."""
    n, f = np.asarray(n, np.float64), np.asarray(f, np.float64)
    nr, fr = n[rows], f[rows]
    with np.errstate(divide="ignore", invalid="ignore"):
        return (fr / nr).mean(-1), fr.sum(-1) == 0, nr.sum(-1)


def paired_z(n, fx, fy, disc, rows, z0, dx=None, dy=None):
    """eval_stats.paired (mode circuits) from per-circuit shots n and failures fx, fy on the pair's shots, the draws
    rows and the zero constant z0; disc = (X-only, Y-only) totals for the degenerate flag; dx, dy: draw_stats of X and
    Y on the same shots (optional: computed here otherwise). -> (stats, log draws)."""
    n, fx, fy = (np.asarray(v, np.float64) for v in (n, fx, fy))
    rx, ry, z = rates_z(n, fx, fy, z0)
    if dx is None:
        bx, by, zb = rates_z(n[rows], fx[rows], fy[rows], z0)
    else:
        zb = dx[1] | dy[1]
        with np.errstate(divide="ignore", invalid="ignore"):
            add = np.where(zb, z0 / dx[2], 0.0)
        bx, by = dx[0] + add, dy[0] + add
    logs = np.log(bx / by)
    return dict(rate_x=float((fx / n).mean()), rate_y=float((fy / n).mean()), ratio=float(rx / ry),
                ratio95=np.exp(np.percentile(logs, [2.5, 97.5])).tolist(), fails_x=int(fx.sum()),
                fails_y=int(fy.sum()), shots=int(n.sum()), bootstrap="circuits", zero_count=bool(z),
                draws_zero_count=float(zb.mean()), degenerate=bool(disc[0] == 0 and disc[1] == 0)), logs


class Pool:
    """Pooled rows (score.pooled): the geometric mean of point ratios and its interval from each draw's mean log."""

    def __init__(self):
        self.v = {}

    def add(self, key, r, logs):
        x = self.v.setdefault(key, dict(lr=0.0, logs=np.zeros(BOOT), n=0, zero=0, degen=0))
        x["lr"] += math.log(r["ratio"])
        x["logs"] += logs
        x["n"] += 1
        x["zero"] += int(r["zero_count"])
        x["degen"] += int(r["degenerate"])

    def out(self):
        res = {}
        for key, x in self.v.items():
            res[key] = dict(ratio=math.exp(x["lr"] / x["n"]),
                            ratio95=np.exp(np.percentile(x["logs"] / x["n"], [2.5, 97.5])).tolist(), points=x["n"],
                            zero_count=x["zero"], degenerate=x["degen"])
        return res


def groups_of(pt):
    d = pt["d"]
    return ([f"d{d}_K{pt['K']}_p3", f"d{d}_all_p3"] if pt["p"] == 0.003 else [f"d{d}_lowp"])


def depth_key(pt):
    return f"d{pt['d']}_L{pt['L']}_p{ep.PCODE[pt['p']]}"


class Rates:
    """Absolute failure rates per (system, key): the equal-weight mean over the key's points of each point's rate, its
    95 % interval from circuit draws (each point's own rows), and the summed counts."""

    def __init__(self):
        self.v = {}

    def add(self, key, fails_c, shots_c, rows):
        f, s = np.asarray(fails_c, np.float64), np.asarray(shots_c, np.float64)
        x = self.v.setdefault(key, dict(r=0.0, draws=np.zeros(BOOT), n=0, fails=0, shots=0))
        x["r"] += float(f.sum() / s.sum())
        x["draws"] += f[rows].sum(1) / s[rows].sum(1)
        x["n"] += 1
        x["fails"] += int(f.sum())
        x["shots"] += int(s.sum())

    def out(self):
        return {k: dict(fails=x["fails"], shots=x["shots"], points=x["n"], rate=x["r"] / x["n"],
                        rate95=np.percentile(x["draws"] / x["n"], [2.5, 97.5]).tolist()) for k, x in self.v.items()}


def table_of(nf, cf, rt, ans, miss):
    """A cell's routing table on one point (flat bools: network failure, composition failure, routed, answered)."""
    return dict(shots=int(nf.size), routed=int(rt.sum()), answered=int(ans.sum()), kept_network=int(rt.sum() - ans.sum()),
                missing=int(miss), net_fail_kept=int((nf & ~rt).sum()), net_fail_routed=int((nf & rt).sum()),
                comp_fail_routed=int((cf & rt).sum()), corrected=int((nf & ~cf).sum()), introduced=int((~nf & cf).sum()))


# ------------------------------------------------------------------------------------------------------------ bench1

G = {}


def _mr1_point(pt):
    """One point of main result 1: per system its per-circuit failures on all shots and on the BP prefix, per pair the
    discordant totals, per cell the routing table (no draws here: the parent does them)."""
    a, thr, facs, opps = G["a"], G["thr"], G["facs"], G["opps"]
    lg = route.load_logits(a.net, a.name, pt)
    C, n = lg.shape[:2]
    truth, netp = route.point_truth(a, pt, lg)
    path = ep.set_path(a.dir, pt)
    z = np.load(path)
    O, k = bs.packed(path, "obs"), z["k"]
    nb = pt["bplomr_shots"]
    nf = netp != truth
    sysF = {"network": nf.reshape(C, n)}
    for o in opps:
        sysF[o] = np.stack(score.opp_fails(a.dec, o, pt, O, k, nb if o == "bplomr" else n))
    cnt = {}                                        # system -> (fails per circuit on all shots, on the first nb)
    disc = {}                                       # pair -> (X only, Y only) totals on the pair's shots

    def add_sys(s, X):
        cnt[s] = (X.sum(1), X[:, :nb].sum(1))
        for o in opps:
            Y = sysF[o]
            Xo = X[:, :Y.shape[1]]
            disc[f"{s}/{o}"] = (int((Xo & ~Y).sum()), int((~Xo & Y).sum()))
        if s != "network":
            Y = sysF["network"]
            disc[f"{s}/network"] = (int((X & ~Y).sum()), int((~X & Y).sum()))

    for o in opps:
        cnt[o] = (sysF[o].sum(1), sysF[o][:, :nb].sum(1))
    add_sys("network", sysF["network"])
    cells, miss_tot = {}, 0
    for b in BASES:
        T = route.load_trials(a.tess, pt, n, facs[b])
        for f in GRID_FS:
            for c in route.CAPS:
                if not exact(f, c):
                    continue
                dec, rt, ans, miss = score.comp_dec(pt, lg, netp, dict(f=f, cap=c, thr=thr), T)
                assert a.dry or miss == 0, (pt["name"], lab(b, f, c), miss, "routed shots without a usable trial part")
                miss_tot += miss
                cf = dec != truth
                s = lab(b, f, c)
                cells[s] = table_of(nf, cf, rt, ans, miss)
                add_sys(s, cf.reshape(C, n))
        del T
    return dict(point=pt["name"], d=pt["d"], K=pt["K"], L=pt["L"], p=pt["p"], C=int(C), n=int(n), nb=int(nb),
                cnt={s: (v[0].tolist(), v[1].tolist()) for s, v in cnt.items()}, disc=disc, cells=cells,
                missing=miss_tot)


def cmd_bench1(a):
    m, thr, pts, sha = frozen_inputs(a)
    opps = a.opps.split(",")
    assert a.dry or opps == ["lomr", "cain", "bplomr"] or (a.provisional and set(opps) < {"lomr", "cain", "bplomr"}), \
        "the test set's scores need every opponent (--provisional: a score without the left-out ones, redone when they land)"
    facs = facs_of(a, a.d)
    a.fail = "any"
    run = [x for x in grid_cells() if exact(x[1], x[2])]
    comps = [lab(*x) for x in run]
    G.update(a=a, thr=thr, facs=facs, opps=opps)
    pool, rates, rows = Pool(), Rates(), []
    with mp.get_context("fork").Pool(min(a.workers, len(pts))) as pp:
        for res in pp.imap(_mr1_point, pts, chunksize=1):
            pt = next(p for p in pts if p["name"] == res["point"])
            C, n, nb = res["C"], res["n"], res["nb"]
            R = boot_rows(res["point"], C)
            ns_all, ns_bp = np.full(C, n), np.full(C, nb)
            r = dict(point=res["point"], d=res["d"], K=res["K"], L=res["L"], p=res["p"], shots=C * n, circuits=C,
                     fails={s: int(sum(v[0])) for s, v in res["cnt"].items() if s not in opps},
                     opp_fails={o: int(sum(res["cnt"][o][1 if o == "bplomr" else 0])) for o in opps},
                     cells=res["cells"], missing=res["missing"], pairs={})
            for s, (fa, fb) in res["cnt"].items():
                if s == "bplomr":
                    rates.add((s, depth_key(pt)), fb, ns_bp, R)
                    rates.add((s, res["point"]), fb, ns_bp, R)
                else:
                    rates.add((s, depth_key(pt)), fa, ns_all, R)
                    rates.add((s, res["point"]), fa, ns_all, R)
            dcache = {}

            def dst(s, bp):
                if (s, bp) not in dcache:
                    dcache[(s, bp)] = draw_stats(ns_bp if bp else ns_all, res["cnt"][s][1 if bp else 0], R)
                return dcache[(s, bp)]

            for pair, dsc in res["disc"].items():
                x, y = pair.split("/")
                bp = y == "bplomr"
                fx = res["cnt"][x][1 if bp else 0]
                fy = res["cnt"][y][1 if bp else 0]
                ns = ns_bp if bp else ns_all
                for z0 in Z0S:
                    st, logs = paired_z(ns, fx, fy, dsc, R, z0, dst(x, bp), dst(y, bp))
                    if z0 == 0.5:
                        r["pairs"][pair] = st
                        for g in groups_of(pt):
                            pool.add((pair, "0.5", g), st, logs)
                            if not st["zero_count"]:
                                pool.add((pair, "drop", g), st, logs)
                    else:
                        for g in groups_of(pt):
                            if g.split("_", 1)[1] in SENS_GROUPS:
                                pool.add((pair, f"{z0:g}", g), st, logs)
            rows.append(r)
            print(json.dumps(dict(point=r["point"], net=r["fails"]["network"], opp=r["opp_fails"],
                                  missing=r["missing"])), flush=True)
    pooled = {}
    for (pair, var, g), v in pool.out().items():
        pooled.setdefault(pair, {}).setdefault(var, {})[g] = v
    fresh_tab = {}
    for r in rows:
        for s, t in r["cells"].items():
            for key in (f"d{r['d']}_all", f"d{r['d']}_{'p3' if r['p'] == 0.003 else 'lowp'}"):
                x = fresh_tab.setdefault(s, {}).setdefault(key, {k: 0 for k in TABLE})
                for k in TABLE:
                    x[k] += t[k]
    rts = {}
    for (s, key), v in rates.out().items():
        rts.setdefault(s, {})[key] = v
    json.dump(dict(result=1, unc_f=UNC_F, manifest=a.manifest, d=a.d, net_sha256=sha, thr=a.thr, tess=a.tess, cal=a.cal,
                   facs={b: route.fac_json(v) for b, v in facs.items()}, cells=comps,
                   not_run=[lab(*x) for x in grid_cells() if not exact(x[1], x[2])], opps=opps, dry=a.dry,
                   zero_rule=dict(main=0.5, sensitivity=[z for z in Z0S if z != 0.5] + ["drop"],
                                  sensitivity_groups=list(SENS_GROUPS)),
                   points=rows, pooled=pooled, routing=fresh_tab, rates=rts), open(a.out, "w"), default=float)
    for pair in ("network/lomr", f"{lab('grace', 0.1, 0.001)}/lomr"):
        if pair in pooled:
            print(pair, {g: round(v["ratio"], 4) for g, v in pooled[pair]["0.5"].items()})


# ------------------------------------------------------------------------------------------------------------ bench2

def cmd_bench2(a):
    """score.py bench2 for the network and every run cell (its code unchanged: the cells enter as choices whose dicts are
    built here instead of read from route.py choose's files; each point's trial parts load once per basis), then the
    routing table and the per-depth absolute rates (target parity) in a second pass."""
    m, thr, pts, sha = frozen_inputs(a)
    facs = facs_of(a, a.d)
    run = [x for x in grid_cells() if exact(x[1], x[2])]
    specs = {lab(b, f, c): dict(f=f, cap=c, basis=b, d=a.d, N=NB[b], thr=thr, thr_path=a.thr, fac=facs[b], cal=a.cal,
                                label=lab(b, f, c), net_sha256=sha, path=lab(b, f, c)) for b, f, c in run}
    score.load_choice = lambda p: specs[p]
    cache, orig = {}, route.load_trials

    def cached(tess_dir, pt, n, fac=None, convert=True):
        key = (tess_dir, pt["name"], n, id(fac), convert)
        if key not in cache:
            cache[key] = orig(tess_dir, pt, n, fac, convert)
        return cache[key]

    route.load_trials = cached
    e14 = a.out + ".e14.json"
    sa = argparse.Namespace(manifest=a.manifest, net=a.net, name=a.name, choice=list(specs), tess=a.tess, d=a.d,
                            out=e14, bank=a.bank, mle=a.mle, dry=a.dry, mle_uncertified=a.mle_uncertified)
    score.cmd_bench2(sa)
    # second pass: routing tables and absolute rates per depth (target parity), LoMatching and the MLE beside
    tst = score._mod("scripts/benchmark2_reads.py", "lukin_test")
    report = tst.report
    a.fail, a.dir = "target", os.path.join(a.bank, "sets")
    rates, tabs = Rates(), {}
    for pt in pts:
        lg = route.load_logits(a.net, a.name, pt)
        C, n = lg.shape[:2]
        truth, netp = route.point_truth(a, pt, lg)
        nf = netp != truth
        R = boot_rows(pt["name"], C)
        ns = np.full(C, n)
        key = f"d{pt['d']}_D{pt['depth']}"
        rates.add(("network", key), nf.reshape(C, n).sum(1), ns, R)
        r = report.load(a.bank, pt, None, ("lomr", a.mle))
        fl = report.fails(r, r["lomr"]["dec"])[0].reshape(C, n)
        rates.add(("lomr", key), fl.sum(1), ns, R)
        mc = (r[a.mle]["cov"] & r[a.mle]["ok"]).reshape(C, n)
        if mc.any():
            fm = report.fails(r, r[a.mle]["dec"])[0].reshape(C, n) & mc
            rates.add(("mle_prefix", key), fm.sum(1), mc.sum(1), R)
            rates.add(("network_on_mle_prefix", key), (nf.reshape(C, n) & mc).sum(1), mc.sum(1), R)
        for b, f, c in run:
            T = cached(a.tess, pt, n, facs[b])
            dec, rt, ans, miss = score.comp_dec(pt, lg, netp, dict(f=f, cap=c, thr=thr), T)
            cf = (dec & 1) != truth
            s = lab(b, f, c)
            tabs.setdefault(s, {})[key] = table_of(nf, cf, rt, ans, miss)
            rates.add((s, key), cf.reshape(C, n).sum(1), ns, R)
            if mc.any():
                rates.add((s + "|on_mle_prefix", key), (cf.reshape(C, n) & mc).sum(1), mc.sum(1), R)
    for s, v in tabs.items():
        tot = {k: sum(t[k] for t in v.values()) for k in TABLE}
        v[f"d{a.d}_all"] = tot
    rts = {}
    for (s, key), v in rates.out().items():
        rts.setdefault(s, {})[key] = v
    json.dump(dict(result=2, unc_f=UNC_F, manifest=a.manifest, d=a.d, net_sha256=sha, thr=a.thr, tess=a.tess, cal=a.cal,
                   facs={b: route.fac_json(v) for b, v in facs.items()}, cells=list(specs),
                   not_run=[lab(*x) for x in grid_cells() if not exact(x[1], x[2])], e14=e14, dry=a.dry,
                   routing=tabs, rates=rts), open(a.out, "w"), default=float)
    print(json.dumps(dict(d=a.d, cells=len(specs), e14=e14)))


# --------------------------------------------------------------------------------------------------------- report

def lrate_z(fails, shots, z0):
    return math.log((fails if fails > 0 else z0) / shots)


def nice(s):
    """A cell label for Markdown: grace|0.1|0.05 -> grace f=0.1 c=0.05 (a pair keeps its "/")."""
    return "/".join(" ".join([x.split("|")[0], f"f={x.split('|')[1]}", f"c={x.split('|')[2]}"]) if x.count("|") == 2
                    else x for x in s.split("/"))


def cmd_report(a):
    """One result and d's tables (Markdown): per basis every cell's failures, geometric-mean rate, paced-stream time per
    layer and p99 (the test set's curve), paired ratios and the routing table; the frozen 11-10 choice and each f's
    development-best cap marked; the per-depth absolute rates; the zero rule's sensitivity."""
    J = json.load(open(a.mr))
    curves = {}
    for p in a.curve.split(","):
        cj = json.load(open(p if p.endswith(".json") else p + ".json"))
        curves[cj["basis"]] = {(r["f"], r["cap"]): r for r in cj["rows"] if r["K"] == "all" and r["d"] == J["d"]}
    dbest = json.load(open(a.devbest))["rows"] if a.devbest else []
    best = {(r["basis"], r["f"]): r["cap"] for r in dbest if r["d"] == J["d"] and r["N"] == NB[r["basis"]]
            and (r.get("result", J["result"]) == J["result"])}
    frozen = {}
    for x in (json.load(open(a.frozen))["rows"] if a.frozen else []):
        if x["d"] == J["d"] and x["result"] == J["result"]:
            frozen[x["basis"]] = (route.fkey(float(x["f"])), x["cap"])
    d, res = J["d"], J["result"]
    e14 = json.load(open(J["e14"])) if res == 2 else None
    left = [o for o in ("lomr", "cain", "bplomr") if o not in J.get("opps", ["lomr", "cain", "bplomr"])]
    L = [f"# Re-test, main result {res}, d = {d}{' (DRY RUN on development data)' if J.get('dry') else ''}"
         f"{' (PROVISIONAL: without ' + ', '.join(left) + ')' if left else ''}\n",
         f"Network SHA-256 {J['net_sha256'][:16]}...; thresholds {J['thr']}; cells not run (uncapped beyond f = "
         f"{J.get('unc_f', 0.02) * 100:g} %): "
         f"{len(J['not_run'])}. Marks: F = the frozen 11-10 choice, B = the development-best cap of that f.\n"]
    for b in BASES:
        cur = curves.get(b, {})
        L.append(f"\n## Basis {b} (N = {NB[b]})\n")
        if res == 1:
            L.append("| f | cap | mark | failures / shots | geo. mean rate | us per layer (p99 ms) | vs LoMatching | "
                     "vs Cain | vs BP+LoM | vs network | routed | corrected | introduced |\n|" + "---|" * 13 + "\n")
        else:
            L.append("| f | cap | mark | target failures | (a) ratio (95 %) | (b) ratio (95 %) | vs LoMatching | "
                     "vs network | us per layer (p99 ms) | routed | corrected | introduced |\n|" + "---|" * 12 + "\n")
        rows = [("network", "0", "none")] + [(lab(b, f, c), route.fkey(f), route.ckey(c)) for f in GRID_FS
                                             for c in route.CAPS]
        for s, f, c in rows:
            mark = ("F" if frozen.get(b) == (f, c) else "") + ("B" if f != "0" and best.get((b, f)) == c else "")
            cv = cur.get((f, c)) or (cur.get(("0", "none")) if s == "network" else None)
            us = f"{cv['us_per_layer_stream']:.1f} ({cv['p99_ms_stream']:.0f})" if cv else "-"
            if s != "network" and s not in J["cells"]:
                L.append(f"| {f} | {c} | {mark} | not run | | | | | | | | | |\n" if res == 1 else
                         f"| {f} | {c} | {mark} | not run | | | | | | | | |\n")
                continue
            tb = J["routing"].get(s, {}).get(f"d{d}_all", {}) if s != "network" else {}
            if res == 1:
                pr = lambda y: (lambda v: f"{v['ratio']:.4f} ({v['ratio95'][0]:.4f}-{v['ratio95'][1]:.4f})" if v else "-")(
                    J["pooled"].get(f"{s}/{y}", {}).get("0.5", {}).get(f"d{d}_all_p3"))
                fails = sum(p["fails"][s] for p in J["points"])
                shots = sum(p["shots"] for p in J["points"])
                gm = math.exp(np.mean([lrate_z(p["fails"][s], p["shots"], 0.5) for p in J["points"]]))
                L.append(f"| {f} | {c} | {mark} | {fails} / {shots} | {gm:.4g} | {us} | {pr('lomr')} | {pr('cain')} | "
                         f"{pr('bplomr')} | {pr('network') if s != 'network' else '-'} | {tb.get('routed', 0)} | "
                         f"{tb.get('corrected', 0)} | {tb.get('introduced', 0)} |\n")
            else:
                v = e14["systems"]["network" if s == "network" else s]
                ci = lambda x: f"{x['ratio']:.3f} ({x['ci'][0]:.3f}-{x['ci'][1]:.3f})" if x else "-"
                rr = lambda x: (f"{x[0]:.3f} ({x[1][0]:.3f}-{x[1][1]:.3f})" if x and x[0] is not None else "-")
                L.append(f"| {f} | {c} | {mark} | {sum(q['fails'] for q in v['depths'])} | {ci(v.get('a'))} | "
                         f"{ci(v.get('b'))} | {rr(v['vs_lomr'])} | {rr(v['vs_network']) if s != 'network' else '-'}"
                         f" | {us} | {tb.get('routed', 0)} | {tb.get('corrected', 0)} | {tb.get('introduced', 0)} |\n")
    L.append("\n## Routing table (every run cell, summed over the points)\n\n| cell | " + " | ".join(TABLE) +
             " |\n|" + "---|" * (len(TABLE) + 1) + "\n")
    for s in J["cells"]:
        t = J["routing"][s][f"d{d}_all"]
        L.append(f"| {nice(s)} | " + " | ".join(str(t[k]) for k in TABLE) + " |\n")
    L.append("\n## Absolute failure rates per depth (95 % circuit-bootstrap interval)\n\n| system | depth | failures / "
             "shots | rate (95 %) |\n|---|---|---|---|\n")
    show = ["network", "lomr", "cain", "bplomr", "mle_prefix", "network_on_mle_prefix"]
    show += [s for s in J["cells"] if frozen.get(s.split("|")[0]) == tuple(s.split("|")[1:])
             or best.get((s.split("|")[0], s.split("|")[1])) == s.split("|")[2]]
    dnum = lambda k: tuple(int(x) if x.isdigit() else x for x in re.split(r"[A-Za-z_]+", k) if x)
    for s in show:
        for key, v in sorted(J["rates"].get(s, {}).items(), key=lambda kv: dnum(kv[0])):
            if key.startswith("fresh") or key.startswith("b1") or key.startswith("lukin") or key.startswith("lukin_"):
                continue                                   # per point: in the JSON only
            L.append(f"| {nice(s)} | {key} | {v['fails']} / {v['shots']} | {v['rate']:.3e} ({v['rate95'][0]:.3e}-"
                     f"{v['rate95'][1]:.3e}) |\n")
    if res == 1:
        L.append("\n## Zero-rule sensitivity (pooled ratio, all K at p = 0.3 % / the p = 0.1, 0.2 % points)\n\n| pair | " +
                 " | ".join(["0.5 (frozen)", "0.1", "0.25", "1", "zero points dropped"]) + " |\n|---|---|---|---|---|---|\n")
        keys = ["network/lomr", "network/cain", "network/bplomr"] + [f"{s}/{y}" for s in show if s in J["cells"]
                                                                     for y in ("lomr", "network")]
        for pair in keys:
            pv = J["pooled"].get(pair, {})
            cell = []
            for var in ("0.5", "0.1", "0.25", "1", "drop"):
                g = pv.get(var, {})
                cell.append(" / ".join(f"{g[k]['ratio']:.4f}" if k in g else "-" for k in (f"d{d}_all_p3", f"d{d}_lowp")))
            L.append(f"| {nice(pair)} | " + " | ".join(cell) + " |\n")
    with open(a.out, "w") as fh:
        fh.writelines(L)
    print(json.dumps(dict(out=a.out, lines=len(L))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tlist", "devbest", "bench1", "bench2", "report"])
    ap.add_argument("--manifest", default="")
    ap.add_argument("--dir", default="")
    ap.add_argument("--net", default="")
    ap.add_argument("--name", default="main")
    ap.add_argument("--thr", default="")
    ap.add_argument("--tess", default="")
    ap.add_argument("--dec", default="")
    ap.add_argument("--bank", default="")
    ap.add_argument("--mle", default="mle")
    ap.add_argument("--mle-uncertified", choices=["stop", "leave-out"], default="stop")
    ap.add_argument("--cal", default="")
    ap.add_argument("--fac", default="")
    ap.add_argument("--d", type=int, default=0)
    ap.add_argument("--opps", default="lomr,cain,bplomr")
    ap.add_argument("--dry", action="store_true", help="development data standing in for a test set")
    ap.add_argument("--provisional", action="store_true", help="bench1: allow --opps to leave opponents out (marked in the report)")
    ap.add_argument("--curve", default="")
    ap.add_argument("--mr", default="")
    ap.add_argument("--devbest", default="")
    ap.add_argument("--frozen", default="")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("NCPU", 1)))
    ap.add_argument("--unc", type=float, default=0.02, help="tlist, bench1, bench2: the routed share run uncapped (the plan: "
                    "0.02); the same value for the lists and the scores")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    global UNC_F
    UNC_F = a.unc
    dict(tlist=cmd_tlist, devbest=cmd_devbest, bench1=cmd_bench1, bench2=cmd_bench2, report=cmd_report)[a.mode](a)


if __name__ == "__main__":
    main()
