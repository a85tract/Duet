"""Scores of a frozen operating point on a test set, the trial lists for it, and the scores of both benchmarks. The
development bank can stand in for a test set (a dry run); nothing here chooses anything.

  tlist --manifest F --net NET --name N --choice CJ [--choice CJ2] --d D --out LISTS
        the shots a frozen choice routes (route.py choose's JSON, its basis, f and cap; the thresholds of its curve's
        sim), per shot the local per-trial deadline of: 2 c L / F, F converting this CPU's seconds to
        the choice's basis (looked up when the trials run: ep.py tess --deadline basis), the larger one when two choices
        route a shot, none for an uncapped choice: LISTS = {"lists": {point: {circuit: [shot...]}}, "cap": {point:
        {circuit: [c or null...]}}, "basis"...}
  bench1 --manifest F --dir SETS --net NET --name N --choice CJ[,CJ2] --tess TESS --dec DEC --d D --out J
        main result 1: per point the network alone, the composition at each choice, LoMatching,
        Cain et al. (every shot) and BP+LoMatching (the first 1,000 shots of each circuit); paired ratios on matched
        shots (scripts/paired_stats.py paired, mode circuits, the seed from the point's name, unchanged; zero rule there;
        a point with no discordant shot is labelled degenerate); geometric means per (d, K) over the six p = 0.3 %
        depths, per d over K (18 points), and the p = 0.1 / 0.2 % points as their own row, intervals from each draw's
        mean log ratio
  bench2 --manifest MANIFEST_bench2 --net NET --name N --choice CJ --tess TESS --d D --out J (BANK_SET=first_bench2) main result
        2: (a) and (b) with scripts/benchmark2_reads.py's own draws (boot_k, boot_b, lognorm, Fig. 1b's
        values and bars, seeds 186014 + d and 186114 + d), the network alone and the composition, LoMatching on every
        shot, the exact MLE on its certified prefix shots, target parity
A composition's answer (route.py sim's rule): a shot is routed iff its score (the smallest |logit| over the circuit's
products) is at most t_f of its depth (the frozen development thresholds); a routed shot takes the lowest-cost trial
neither flagged, dropped nor past c x L (basis seconds; parts serve caps up to their deadline), else the network's."""
import argparse
import importlib.util
import json
import math
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import operating_points as route  # noqa: E402
import eval_pipeline as ep  # noqa: E402

from duet import bench_sets as bs  # noqa: E402

ROOT = ep.ROOT


def _mod(rel, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = sys.modules[name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_choice(path):
    """A frozen choice: route.py choose's JSON with one d and one N -> dict(f, cap (None: uncapped), basis, d, thr,
    fac, label)."""
    ch = json.load(open(path))
    assert len(ch["res"]) == 1, (path, "one (d, N) per choice file")
    x = ch["res"][0]
    cj = json.load(open(ch["curve"] + (".json" if not ch["curve"].endswith(".json") else "")))
    sim = json.load(open(cj["sim"]))
    c = x["chosen"]
    fac = {(None if k == "None" else int(k)): v for k, v in sim["fac"].items()}
    return dict(f=float(c["f"]), cap=None if c["cap"] == "none" else float(c["cap"]), basis=ch["basis"], d=x["d"],
                N=x["N"], thr=json.load(open(sim["thr"])), thr_path=sim["thr"], fac=fac, cal=sim.get("cal"),
                label=f"{ch['basis']} N={x['N']} f={c['f']} c={c['cap']}", net_sha256=sim["net_sha256"], path=path)


def routed_mask(lg, thr, pt, f):
    return route.is_routed(route.score_of(lg).reshape(-1), thr, pt, f)


def comp_dec(pt, lg, netp, ch, T, fail="any"):
    """ of the composition at
    choice ch; netp the network's decision bytes (flat), T the point's trial parts in the choice's basis."""
    rt = routed_mask(lg, ch["thr"], pt, ch["f"])
    q = np.flatnonzero(rt)
    have, pos = route.lookup(T, q)
    lim = None if ch["cap"] is None else ch["cap"] * pt["L"]
    have[have] = route.valid_rows(T, pos[have], lim)
    ans, ok = route.answers(T, pos[have], lim)
    dec = netp.copy()
    qq = q[have]
    dec[qq[ok]] = ans[ok]
    answered = np.zeros(len(dec), bool)
    answered[qq[ok]] = True
    return dec, rt, answered, int((~have).sum())


# ----------------------------------------------------------------------------------------------------------- tlist

def cmd_tlist(a):
    """The routed shots of the frozen choices and, per shot, its requirement: the sorted (basis, cap) of every choice
    that routes it ("none" when one of them is uncapped); ep.py tess --deadline req turns it into the local deadline
    max over (basis, c) of 2 c L / F_basis."""
    m = ep.load_manifest(a.manifest)
    chs = [load_choice(p) for p in a.choice]
    pts, sha = route.complete_points(m, a.net, a.name, a.d)
    for ch in chs:
        assert ch["d"] == a.d and ch["net_sha256"] == sha, (ch["path"], "another d or network")
    lists, reqs, tot = {}, {}, 0
    for pt in pts:
        lg = route.load_logits(a.net, a.name, pt)
        n = lg.shape[1]
        need = {}                     # flat shot -> {(basis, cap)} of the choices routing it
        for ch in chs:
            if ch["f"] == 0:
                continue
            for q in np.flatnonzero(routed_mask(lg, ch["thr"], pt, ch["f"])).tolist():
                need.setdefault(q, set()).add((ch["basis"], "none" if ch["cap"] is None else f"{ch['cap']:g}"))
        if not need:
            continue
        qs = np.array(sorted(need), np.int64)
        tot += len(qs)
        req = lambda x: "none" if any(c == "none" for _, c in need[x]) else ";".join(f"{b}:{c}" for b, c in
                                                                                        sorted(need[x]))
        lists[pt["name"]] = {str(int(c)): (qs[qs // n == c] % n).tolist() for c in np.unique(qs // n)}
        reqs[pt["name"]] = {str(int(c)): [req(int(x)) for x in qs[qs // n == c]] for c in np.unique(qs // n)}
    json.dump(dict(rule=": the shots the frozen choices route; per shot req = its choices' (basis, cap): "
                        "ep.py tess --deadline req runs each trial with the local deadline max 2 c L / F_basis, none "
                        "when a choice is uncapped", choices=[ch["path"] for ch in chs],
                   basis=[ch["basis"] for ch in chs], shots=tot, lists=lists, req=reqs), open(a.out, "w"))
    print(json.dumps(dict(shots=tot, points=len(lists), choices=[ch["label"] for ch in chs])))


# ------------------------------------------------------------------------------------------------------------- bench1

def per_circuit(v, C, n):
    return [v[i * n:(i + 1) * n] for i in range(C)]


def opp_fails(dec_dir, dec_, pt, O, k, first):
    """Per circuit the failure bools of an opponent on shots 0.. first - 1, its coverage required exactly."""
    got = ep.load_dec(dec_dir, dec_, pt)
    out = []
    for i in range(len(pt["circuits"])):
        assert i in got, (pt["name"], dec_, i, "no part")
        idx, pred, ok, _ = got[i]
        o = np.argsort(idx, kind="stable")
        assert np.array_equal(idx[o], np.arange(first)) and ok.all(), (pt["name"], dec_, i, "coverage")
        y = np.packbits(bs.unpack(O[i][idx[o]], k[i]), axis=1, bitorder="little")[:, 0] if k[i] else \
            np.zeros(first, np.uint8)
        out.append(pred[o] != y)
    return out


def pair_stats(name, fx, fy):
    es = _mod("scripts/paired_stats.py", "paired_stats")
    tab = es.table(fx, fy)
    r, logs = es.paired(tab, zlib.crc32(name.encode()) % 2 ** 31, "circuits")
    r["degenerate"] = bool(tab[:, 2].sum() == 0 and tab[:, 3].sum() == 0)
    return r, logs


def pooled(items):
    """[(ratio, logs)] -> geometric mean and its interval from each draw's mean log ratio."""
    return dict(ratio=float(np.exp(np.mean([math.log(r["ratio"]) for r, _ in items]))),
                ratio95=np.exp(np.percentile(np.mean([lg for _, lg in items], 0), [2.5, 97.5])).tolist(),
                points=len(items), zero_count=sum(r["zero_count"] for r, _ in items),
                degenerate=sum(r["degenerate"] for r, _ in items))


def cmd_bench1(a):
    m = ep.load_manifest(a.manifest)
    chs = [load_choice(p) for p in a.choice]
    pts, sha = route.complete_points(m, a.net, a.name, a.d)
    opps = a.opps.split(",")
    assert a.dry or opps == ["lomr", "cain", "bplomr"], "the test set's scores need every opponent"
    pairs = [("network", o) for o in opps]
    for j in range(len(chs)):
        pairs += [(f"comp{j}", o) for o in opps] + [(f"comp{j}", "network")]
    rows, draws = [], {}
    for pt in pts:
        lg = route.load_logits(a.net, a.name, pt)
        C, n = lg.shape[:2]
        truth, netp = route.point_truth(a, pt, lg)
        path = ep.set_path(a.dir, pt)
        z = np.load(path)
        O, k = bs.packed(path, "obs"), z["k"]
        F = dict(network=per_circuit(netp != truth, C, n))
        r = dict(point=pt["name"], d=pt["d"], K=pt["K"], L=pt["L"], p=pt["p"], shots=int(C * n), circuits=int(C))
        for j, ch in enumerate(chs):
            T = route.load_trials(a.tess, pt, n, ch["fac"])
            dec, rt, ans, miss = comp_dec(pt, lg, netp, ch, T)
            assert miss == 0, (pt["name"], ch["label"], miss, "routed test shots without a usable trial part")
            F[f"comp{j}"] = per_circuit(dec != truth, C, n)
            r[f"comp{j}"] = dict(choice=ch["label"], routed=int(rt.sum()), answered=int(ans.sum()),
                                 kept_network=int(rt.sum() - ans.sum()))
        for dec_ in [o for o in opps if o != "bplomr"]:
            F[dec_] = opp_fails(a.dec, dec_, pt, O, k, n)
        nb = pt["bplomr_shots"]
        if "bplomr" in opps:
            F["bplomr"] = opp_fails(a.dec, "bplomr", pt, O, k, nb)
        r["fails"] = {s: int(sum(v.sum() for v in x)) for s, x in F.items()}
        r["pairs"] = {}
        for x, y in pairs:
            fx, fy = F[x], F[y]
            if y == "bplomr":                                   # matched shots: the first 1,000 of each circuit
                fx = [v[:nb] for v in fx]
            st, logs = pair_stats(pt["name"], fx, fy)
            r["pairs"][f"{x}/{y}"] = st
            draws.setdefault(f"{x}/{y}", {})[pt["name"]] = (st, logs)
        rows.append(r)
        print(json.dumps(dict(point=pt["name"], fails=r["fails"],
                              **{k_: round(v["ratio"], 4) for k_, v in r["pairs"].items()})), flush=True)
    groups = {}
    for pt in pts:
        g = [f"d{pt['d']}_K{pt['K']}_p3"] if pt["p"] == 0.003 else [f"d{pt['d']}_lowp"]
        if pt["p"] == 0.003:
            g.append(f"d{pt['d']}_all_p3")
        for key in g:
            groups.setdefault(key, []).append(pt["name"])
    pool = {pr: {g: pooled([draws[pr][nm] for nm in names]) for g, names in groups.items()} for pr in draws}
    json.dump(dict(manifest=a.manifest, d=a.d, net_sha256=sha, choices=[ch["label"] for ch in chs],
                   choice_files=a.choice, points=rows, pooled=pool), open(a.out, "w"), indent=0, default=float)
    print(json.dumps({pr: {g: [round(v["ratio"], 4)] + [round(x, 4) for x in v["ratio95"]] for g, v in gp.items()}
                      for pr, gp in pool.items()}))


# ------------------------------------------------------------------------------------------------------------- bench2

def check_mle_prefix(name, r, mle, P, leave_out=False):
    """the MLE decisions are exactly the manifest's prefix P of every circuit (report.load's flat,
    circuit-major arrays), each of its shots decoded and successful (certified), none beyond it; else stop. leave_out (--mle-uncertified leave-out, never the default): every
    prefix shot must still be decoded and none beyond it, but a decoded prefix shot without a certificate is left out
    of x_D and m_D. Returns the uncertified prefix shots as [circuit, shot] pairs."""
    cv = np.asarray(r[mle]["cov"], bool).reshape(r["C"], r["n"])
    okm = cv & np.asarray(r[mle]["ok"], bool).reshape(r["C"], r["n"])
    assert cv[:, :P].all() and not cv[:, P:].any(), \
        (name, "MLE: not exactly the prefix", P, int(cv[:, :P].sum()), int(cv[:, P:].sum()))
    unc = np.argwhere(~okm[:, :P]).tolist()
    assert leave_out or not unc, (name, "MLE: not exactly the successful prefix (uncertified prefix shots)", P, unc[:10])
    return unc


def cmd_bench2(a):
    assert os.environ.get("BANK_SET") == "first_bench2" or a.dry, "main result 2's set: BANK_SET=first_bench2"
    tst = _mod("scripts/benchmark2_reads.py", "lukin_test")
    report, bank = tst.report, tst.bank
    m = ep.load_manifest(a.manifest)
    chs = [load_choice(p) for p in a.choice]
    pts, sha = route.complete_points(m, a.net, a.name, a.d)
    pnet, pmle = tst.theirs_sd()
    d = a.d
    sysl = [("network", None)] + [(ch["label"], ch) for ch in chs]
    rs, unc = [], {}
    for pt in pts:
        r = report.load(a.bank, pt, None, ("lomr", a.mle))
        lg = route.load_logits(a.net, a.name, pt)
        assert lg.shape[1] == r["n"] and lg.shape[0] == r["C"], (pt["name"], lg.shape)
        r["lg"] = lg
        r["dec_net"] = ((lg.reshape(-1, 4) > 0) * (1 << np.arange(4))).sum(-1).astype(np.uint8)
        assert r["lomr"]["cov"].all(), (pt["name"], "LoMatching decoded every shot")
        if not a.dry:
            u = check_mle_prefix(pt["name"], r, a.mle, bs.prefix(pt, "mle"), a.mle_uncertified == "leave-out")
            if u:
                unc[pt["name"]] = u
                print(f"{pt['name']}: {len(u)} uncertified MLE prefix shot(s) left out of x_D and m_D: {u}", flush=True)
        rs.append(r)
    cid = [np.repeat(np.arange(r["C"]), r["n"]) for r in rs]
    fnet = [report.fails(r, r["dec_net"]) for r in rs]
    flom = [report.fails(r, r["lomr"]["dec"])[0] for r in rs]
    mcov = [r[a.mle]["cov"] & r[a.mle]["ok"] for r in rs]
    fmle = [report.fails(r, r[a.mle]["dec"])[0] for r in rs]
    cD = []                  # c_D as test.py rep: the circuit variation of a 20-circuit mean, from the network's rates
    for (ft, _), c_, r in zip(fnet, cid, rs):
        p = np.bincount(c_, ft, r["C"]) / r["n"]
        cD.append(max(0.0, p.var(ddof=1) - np.mean(p * (1 - p) / r["n"])) / (r["C"] * p.mean() ** 2)
                  if p.mean() > 0 else 0.0)
    out, F = {}, {}
    for lab, ch in sysl:
        depth, kc, per, Fx = [], [], [], []
        for pt, r, (ft, fa), c_, fl, mc, fm in zip(pts, rs, fnet, cid, flom, mcov, fmle):
            D = r["D"]
            q = dict(D=D, shots=len(ft))
            if ch is not None:
                T = route.load_trials(a.tess, pt, r["n"], ch["fac"])
                dx, rt, ans, miss = comp_dec(pt, r["lg"], r["dec_net"], ch, T)
                assert miss == 0, (pt["name"], lab, miss, "routed shots without a usable trial part")
                q.update(routed=int(rt.sum()), frac=float(rt.mean()), kept_network=int(rt.sum() - ans.sum()))
                fx, fax = report.fails(r, dx)
            else:
                fx, fax = ft, fa
            Fx.append(fx)
            q.update(fails=int(fx.sum()), rate=float(fx.mean()), any_fails=int(fax.sum()), lomr=int(fl.sum()),
                     vs_lomr=report.ratio([(fx, fl)])[2:])
            if (d, D) in pnet:
                q.update(their_net=pnet[(d, D)][0], expected=len(ft) * pnet[(d, D)][0],
                         vs_their_net=float(fx.sum() / (len(ft) * pnet[(d, D)][0])))
            if mc.any() and (d, D) in pmle:
                xm, mm, cm = fx[mc], fm[mc], c_[mc]
                q.update(mle_shots=int(mc.sum()), mle_uncertified=len(unc.get(pt["name"], [])),
                         x_on_mle=int(xm.sum()), mle=int(mm.sum()),
                         vs_mle=report.ratio([(xm, mm)])[2:], their_mle=pmle[(d, D)][0])
                per.append(np.stack([np.bincount(cm, minlength=r["C"]), np.bincount(cm, xm & mm, r["C"]),
                                     np.bincount(cm, xm & ~mm, r["C"]), np.bincount(cm, ~xm & mm, r["C"])],
                                    1).astype(np.int64))
            else:
                per.append(None)
            kc.append((np.bincount(c_, fx, r["C"]).astype(np.int64), np.full(r["C"], r["n"], np.int64)))
            depth.append(q)
        F[lab] = Fx
        v = dict(depths=depth, score=sum(q["rate"] for q in depth), vs_lomr=report.ratio(list(zip(Fx, flom)))[2:],
                 vs_network=report.ratio(list(zip(Fx, F["network"])))[2:],
                 routed=sum(q.get("routed", 0) for q in depth),
                 frac=sum(q.get("routed", 0) for q in depth) / sum(q["shots"] for q in depth))
        got = [(q, pnet[(d, q["D"])]) for q in depth if (d, q["D"]) in pnet]
        K, E = sum(q["fails"] for q, _ in got), sum(q["shots"] * p for q, (p, _) in got)
        VE = sum((q["shots"] * sd) ** 2 for q, (_, sd) in got) / E ** 2
        VC = sum((q["shots"] * p) ** 2 * cD[q["D"] - 1] for q, (p, _) in got) / E ** 2
        Kb = tst.boot_k(np.random.default_rng(tst.SEED + d), [kc[j] for j, q in enumerate(depth) if (d, q["D"]) in pnet])
        VK = float(np.var(np.log(np.maximum(Kb, 0.5))))
        A = K / E
        v["a"] = dict(depths=f"{got[0][0]['D']}-{got[-1][0]['D']}", fails=K, expected=E, ratio=A,
                      ci=tst.lognorm(A, VK + VE), V_K=VK, V_E=VE, V_circ=VC, ci_sens=tst.lognorm(A, VK + VE + VC))
        v["a"]["holds"] = bool(v["a"]["ci"][1] < 1)
        Db = [q["D"] for q, pr in zip(depth, per) if pr is not None]
        if not a.dry:
            assert Db == [pt["depth"] for pt in pts if bs.prefix(pt, "mle")], (d, lab, Db)
        if Db:
            qs = [q for q in depth if q["D"] in Db]
            ours = sum(q["x_on_mle"] / q["mle_shots"] for q in qs) / sum(q["mle"] / q["mle_shots"] for q in qs)
            ps, qq = sum(pnet[(d, D)][0] for D in Db), sum(pmle[(d, D)][0] for D in Db)
            Xb, Mb = tst.boot_b(np.random.default_rng(tst.SEED + 100 + d), [pr for pr in per if pr is not None])
            ok = (Xb > 0) & (Mb > 0)
            VO = float(np.var(np.log(Xb[ok] / Mb[ok])))
            VT = sum(pnet[(d, D)][1] ** 2 for D in Db) / ps ** 2 + sum(pmle[(d, D)][1] ** 2 for D in Db) / qq ** 2
            B = ours / (ps / qq)
            v["b"] = dict(depths=f"{Db[0]}-{Db[-1]}", ours=ours, ours_ci=tst.lognorm(ours, VO), theirs=ps / qq,
                          ratio=B, ci=tst.lognorm(B, VO + VT), V_ours=VO, V_theirs=VT, draws_dropped=int((~ok).sum()))
            v["b"]["holds"] = bool(v["b"]["ci"][1] < 1)
        out[lab] = v
        print(json.dumps(dict(system=lab, a=[round(v["a"]["ratio"], 4)] + [round(x, 4) for x in v["a"]["ci"]],
                              b=[round(v["b"]["ratio"], 4)] + [round(x, 4) for x in v["b"]["ci"]] if "b" in v else None,
                              vs_lomr=v["vs_lomr"], routed=v["routed"])), flush=True)
    lom = dict(score=sum(float(f_.mean()) for f_ in flom),
               depths=[dict(D=r["D"], fails=int(f_.sum()), shots=len(f_)) for r, f_ in zip(rs, flom)])
    json.dump(dict(d=d, net_sha256=sha, c_D=cD, systems=out, lomr=lom, mle=a.mle, choices=a.choice, dry=a.dry,
                   mle_uncertified_rule=a.mle_uncertified, mle_uncertified=unc),
              open(a.out, "w"), indent=1, default=float)


def cmd_row(a):
    """ row of a frozen choice (route.py choose's JSON): result, d, basis, N, f, c, the development bank's
    composition failures and geometric-mean rate (and the network alone's), routed shots, the paced-stream time per
    layer and batch p99, the bar, the curve and the rule's commit."""
    ch = json.load(open(a.choice[0]))
    cj = json.load(open(ch["curve"] + (".json" if not ch["curve"].endswith(".json") else "")))
    for x in ch["res"]:
        r, al = x["chosen"], x["network_alone"]
        cap = (r["cap"] if float(r["f"]) else "-")
        print(f"| {a.result} | {x['d']} | {x['basis']} (N = {x['N']}) | {float(r['f']) * 100:g} % | {cap} | "
              f"{r['fails']} / {r['shots']} (network alone {al['fails']}) | {r['gmean_rate']:.4g} "
              f"({al['gmean_rate']:.4g}) | {r['routed']} | {r['us_per_layer_stream']:.1f} (p99 {r['p99_ms_stream']:.0f} ms)"
              f" | {route.BAR_US[x['d']]} | {x['qualifying']} / {x['candidates']} | {x['rule']} | {a.commit} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tlist", "bench1", "bench2", "row"])
    ap.add_argument("--result", default="1")
    ap.add_argument("--commit", default="")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--dir", default="")
    ap.add_argument("--net", default="")
    ap.add_argument("--name", default="main")
    ap.add_argument("--choice", action="append", default=[])
    ap.add_argument("--d", type=int, default=0)
    ap.add_argument("--tess", default="")
    ap.add_argument("--dec", default="")
    ap.add_argument("--bank", default="", help="bench2: the set's root (sets/, dec/lomr, dec/mle; scripts/dev_bank.py layout)")
    ap.add_argument("--mle", default="mle")
    ap.add_argument("--opps", default="lomr,cain,bplomr", help="bench1: the opponents (a dry run may leave BP out)")
    ap.add_argument("--dry", action="store_true", help="bench2 on a development bank (no prefix assertions)")
    ap.add_argument("--mle-uncertified", choices=["stop", "leave-out"], default="stop",
                    help="bench2: a decoded but uncertified MLE prefix shot stops the score (default) or is left out of "
                         "x_D and m_D ( certified prefix shots; listed in the output)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dict(tlist=cmd_tlist, bench1=cmd_bench1, bench2=cmd_bench2, row=cmd_row)[a.mode](a)


if __name__ == "__main__":
    main()
