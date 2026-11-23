"""Reads of the development bank: A0 (a network against LoMatching, Tesseract
and the MLE per depth and pooled) and Track B (the pair at f = 0, 1, 2, 5, 10 % under both routing scores), and the
shot lists for the MLE and Tesseract.

  python scripts/bank_report.py a0 --bank B --net NAME [--ds 3,5,7] [--decs lomr,tess,mle] [--out J]
  python scripts/bank_report.py pair --bank B --net NAME [--ds 3,5,7] [--tess tess] [--out J]
  python scripts/bank_report.py vs --bank B --x NAME --y NAME [--ds 5] two networks, paired (Track A's screen)
  python scripts/bank_report.py select --bank B --net NAME --ds 5 --what fails|routed [--cap N] --out FILE
  python scripts/bank_report.py agree --bank B --x tess --y mle_routed [--ds 5,7] [--out J] rule 6

Failures: the target (product slot 0) or any product decoded wrong; a network decodes a product flipped when its logit
is positive. Pooled = the selection score, the depths' rates summed (each depth weighted equally); ratios of two
decoders on the same shots with a 95 % log-normal interval from the paired variance (delta method; the covariance from
the shots both fail). Track B: per depth the ceil(f n) shots of least score (stable order) go to Tesseract; scores
s_T = |target logit| and s_min = the least |logit| over the circuit's products; t(d, D, f) = the largest routed score."""
import argparse
import collections
import importlib.util
import json
import math
import os
import sys

import numpy as np

from duet import bench_sets as bs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FS = (0.0, 0.01, 0.02, 0.05, 0.10)


def _mod(rel, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = sys.modules[name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


bank = _mod("scripts/dev_bank.py", "lukin_bank")


def load(B, pt, net=None, decs=()):
    """A point's shots, circuits concatenated (flat index circuit * n + shot) over the first n shots of each circuit (n
    the network's read when one is given): obs and the network's decisions as bytes (bit j = product j), the product
    mask, the network's logits, and per decoder its decisions, coverage, certification and seconds."""
    path = bank.set_path(B, pt)
    z = np.load(path)
    k = z["k"].astype(int)
    out = dict(D=pt["depth"], d=pt["d"])
    if net:
        g = np.load(os.path.join(B, "net", net, pt["name"] + ".npz"))
        lg = g["logits"]
        out.update(logits=lg.reshape(-1, 4), net_secs=float(g["secs"].sum()), net_batch=int(g["batch"]),
                   net_info=json.loads(str(g["info"])))
        n = lg.shape[1]
        out["dec_net"] = ((lg > 0) * (1 << np.arange(4))).sum(-1).astype(np.uint8).reshape(-1)
    else:
        n = pt["shots"]
    out["n"], out["C"] = n, len(k)
    out["obs"] = np.asarray(bs.packed(path, "obs")[:, :n, 0]).reshape(-1)
    out["mask"] = np.repeat(((1 << k) - 1).astype(np.uint8), n)
    for name in decs:
        P, cov = np.zeros(len(k) * n, np.uint8), np.zeros(len(k) * n, bool)
        ok, sec = np.zeros(len(k) * n, bool), np.full(len(k) * n, np.nan)
        for i, (idx, pr, okk, s) in bank.load_dec(B, name, pt).items():
            keep = idx < n
            f = i * n + idx[keep]
            P[f], cov[f], ok[f], sec[f] = pr[keep], True, okk[keep], s[keep]
        out[name] = dict(dec=P, cov=cov, ok=ok, sec=sec)
    return out


def fails(r, dec):
    """(target, any-product) failure bools of decisions `dec` (bytes) on the point's shots."""
    x = dec ^ r["obs"]
    return (x & 1).astype(bool), (x & r["mask"]).astype(bool)


def ratio(rows):
    """rows: [(fails of A, fails of B)] per depth on the same shots -> (score A, score B, A / B, [lo, hi])."""
    X = sum(a.sum() / len(a) for a, _ in rows)
    Y = sum(b.sum() / len(b) for _, b in rows)
    vX = sum(a.sum() / len(a) ** 2 for a, _ in rows)
    vY = sum(b.sum() / len(b) ** 2 for _, b in rows)
    cv = sum((a & b).sum() / len(a) ** 2 for a, b in rows)
    if X == 0 or Y == 0:
        return X, Y, None, None
    R = X / Y
    s = math.sqrt(max(vX / X ** 2 + vY / Y ** 2 - 2 * cv / (X * Y), 0.0))
    return X, Y, R, [R * math.exp(-1.96 * s), R * math.exp(1.96 * s)]


def fmt(x, ci=None):
    if x is None:
        return "-"
    return f"{x:.3f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")


def theirs():
    rt = _mod("scripts/bench_tables.py", "bench_results_table")
    return rt.theirs("ML"), rt.theirs("MLE")


def published(d, rs, fl, th):
    """{who: (depths, our target failures, expected, ratio, 95 % interval)}: target failures fl (one bool array per
    point of rs) against their published per-depth rates (their network, their MLE) times these shots."""
    out = {}
    for who, t in (("their network", th[0]), ("their MLE", th[1])):
        got = [(r, f, t[(d, r["D"])]) for r, f in zip(rs, fl) if (d, r["D"]) in t]
        e = sum(len(r["obs"]) * v[0] for r, _, v in got)
        v = sum(len(r["obs"]) ** 2 * v[1] for r, _, v in got)
        kk = int(sum(f.sum() for _, f, _ in got))
        if e:
            x, s = kk / e, math.sqrt(1 / max(kk, 0.5) + v / e ** 2)
            out[who] = (f"{got[0][0]['D']}-{got[-1][0]['D']}", kk, e, x, [x * math.exp(-1.96 * s), x * math.exp(1.96 * s)])
    return out


def a0(a):
    rec, th = {}, theirs()
    for d in (int(x) for x in a.ds.split(",")):
        pts = [p for p in bank.points((d,))]
        rs = [load(a.bank, p, a.net, a.decs.split(",")) for p in pts]
        print(f"\n### d = {d}: {a.net} against the other decoders (target parity; any product in the second table)\n")
        for kind, j in (("target", 0), ("any product", 1)):
            print(f"| D | shots | network | " + " | ".join(f"{x} | network / {x}" for x in a.decs.split(",")) + " |")
            print("|---|---|---|" + "---|---|" * len(a.decs.split(",")))
            per = {x: [] for x in a.decs.split(",")}
            for r in rs:
                fn = fails(r, r["dec_net"])[j]
                row = [str(r["D"]), f"{len(fn):,}", f"{fn.sum()}: {fn.mean():.2e}"]
                for x in a.decs.split(","):
                    c = r[x]["cov"] & (r[x]["ok"] if x == "mle" else True)
                    if not c.any():
                        row += ["-", "-"]
                        continue
                    fx = fails(r, r[x]["dec"])[j][c]
                    per[x].append((fn[c], fx))
                    _, _, R, ci = ratio([(fn[c], fx)])
                    row += [f"{fx.sum()} / {c.sum():,}", fmt(R, ci)]
                print("| " + " | ".join(row) + " |")
            pooled = ["pooled", "", ""]
            for x in a.decs.split(","):
                if len(per[x]) == len(rs):
                    X, Y, R, ci = ratio(per[x])
                    pooled += [f"{Y:.3e}", fmt(R, ci)]
                    rec[f"d{d}_{kind}_{x}"] = dict(net=X, other=Y, ratio=R, ci=ci)
                else:
                    pooled += ["-", "-"]
            print("| " + " | ".join(pooled) + " |\n")
        # against their published per-depth rates (Fig. 1b), on these shots: (a) network, and their MLE
        k = sum(fails(r, r["dec_net"])[0].sum() for r in rs)
        for who, (Ds, kk, e, x, ci) in published(d, rs, [fails(r, r["dec_net"])[0] for r in rs], th).items():
            print(f"- ours / {who} predicted on these shots (D = {Ds}): {kk} / {e:.1f} = {fmt(x, ci)}")
            rec[f"d{d}_vs_{who.replace(' ', '_')}"] = dict(fails=int(kk), expected=e, ratio=x, ci=ci)
        rec[f"d{d}_net_target_fails"] = int(k)
    if a.out:
        json.dump(rec, open(a.out, "w"), indent=1)


def scores(r):
    L = np.abs(r["logits"])
    return dict(T=L[:, 0], min=np.nanmin(L, axis=1))


def routed(s, f):
    m = int(math.ceil(f * len(s)))
    o = np.argsort(s, kind="stable")[:m]
    return o, (float(s[o[-1]]) if m else None)


def pair(a):
    rec = {}
    for d in (int(x) for x in a.ds.split(",")):
        rs = [load(a.bank, p, a.net, (a.tess,)) for p in bank.points((d,))]
        print(f"\n### d = {d}: the pair ({a.net} with {a.tess} on its least confident shots), target parity\n")
        out = {}
        for sc in ("T", "min"):
            for f in FS:
                rows, fix, brk, t, tsec, miss = [], 0.0, 0.0, {}, [], 0
                for r in rs:
                    fn = fails(r, r["dec_net"])[0]
                    o, t[r["D"]] = routed(scores(r)[sc], f)
                    fp = fn.copy()
                    if len(o):
                        miss += int((~r[a.tess]["cov"][o]).sum())
                        ft = fails(r, r[a.tess]["dec"])[0]
                        fp[o] = ft[o]
                        fix += (fn[o] & ~ft[o]).sum() / len(fn)
                        brk += (~fn[o] & ft[o]).sum() / len(fn)
                        tsec.append(r[a.tess]["sec"][o])
                    rows.append((fp, fn))
                X, Y, R, ci = ratio(rows)
                ts = np.concatenate(tsec) if tsec else np.zeros(0)
                net_us = sum(r["net_secs"] for r in rs) / sum(len(r["obs"]) for r in rs)
                out[(sc, f)] = dict(score=X, net_score=Y, ratio_to_net=R, ci=ci, fixed=fix, broken=brk, t=t,
                                    missing_tess=miss, fails=[fp for fp, _ in rows],
                                    tess_mean_s=float(np.nanmean(ts)) if len(ts) else 0.0,
                                    tess_p99_s=float(np.nanpercentile(ts, 99)) if len(ts) else 0.0,
                                    tess_max_s=float(np.nanmax(ts)) if len(ts) else 0.0, net_s=net_us)
        net_corr = {sc: sum(out[(sc, f)]["fixed"] - out[(sc, f)]["broken"] for f in FS[1:]) for sc in ("T", "min")}
        sc = max(net_corr, key=net_corr.get)
        best = min(FS, key=lambda f: out[(sc, f)]["score"])
        head = None
        for f in FS:
            fb = out[(sc, best)]["fails"]
            _, _, R, ci = ratio(list(zip(out[(sc, f)]["fails"], fb)))
            out[(sc, f)]["vs_best"] = (R, ci)
            if head is None and (f == best or (ci is not None and ci[1] <= 1.02)):
                head = f
        print(f"Net target corrections summed over f = 1-10 % (selection score units): s_T {net_corr['T']:.3e}, "
              f"s_min {net_corr['min']:.3e} -> score {sc}; best f {best:.0%}, headline f {head:.0%}\n")
        print("| score | f | pair score | pair / network [95 %] | pair / best f [95 %] | fixed | broken | Tesseract s per "
              "routed shot: mean / p99 / max | pair s per shot: mean | missing Tesseract |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for (s_, f), v in out.items():
            R, ci = v.get("vs_best", (None, None))
            print(f"| {s_} | {f:.0%} | {v['score']:.3e} | {fmt(v['ratio_to_net'], v['ci'])} | "
                  f"{fmt(R, ci) if s_ == sc else '-'} | {v['fixed']:.2e} | {v['broken']:.2e} | {v['tess_mean_s']:.3f} / "
                  f"{v['tess_p99_s']:.3f} / {v['tess_max_s']:.3f} | {v['net_s'] + f * v['tess_mean_s']:.2e} | "
                  f"{v['missing_tess']} |")
        th = theirs()
        print(f"\nWith score {sc}, target failures against their published per-depth rates times these shots:\n")
        for f in FS:
            pub = published(d, rs, out[(sc, f)]["fails"], th)
            out[(sc, f)]["published"] = pub
            print(f"- f = {f:.0%}: " + "; ".join(f"pair / {who} (D = {Ds}) {kk} / {e:.1f} = {fmt(x, ci)}"
                                                for who, (Ds, kk, e, x, ci) in pub.items()))
        rec[f"d{d}"] = dict(score=sc, best=best, headline=head, net_corr=net_corr,
                            rows={f"{s_}_{f}": {k_: v_ for k_, v_ in v.items() if k_ != "fails"}
                                  for (s_, f), v in out.items()})
    if a.out:
        json.dump(rec, open(a.out, "w"), indent=1, default=float)


def vs(a):
    """Two networks on the same shots (Track A's screen): per depth and pooled, target parity."""
    for d in (int(x) for x in a.ds.split(",")):
        rows = []
        print(f"\n### d = {d}: {a.x} / {a.y}, target parity\n\n| D | shots | {a.x} | {a.y} | ratio [95 %] |\n"
              "|---|---|---|---|---|")
        for p in bank.points((d,)):
            rx, ry = load(a.bank, p, a.x), load(a.bank, p, a.y)
            n = min(rx["n"], ry["n"])
            fx = fails(rx, rx["dec_net"])[0].reshape(rx["C"], -1)[:, :n].reshape(-1)
            fy = fails(ry, ry["dec_net"])[0].reshape(ry["C"], -1)[:, :n].reshape(-1)
            rows.append((fx, fy))
            _, _, R, ci = ratio([(fx, fy)])
            print(f"| {p['depth']} | {len(fx):,} | {fx.sum()} | {fy.sum()} | {fmt(R, ci)} |")
        X, Y, R, ci = ratio(rows)
        print(f"| pooled | | {X:.4e} | {Y:.4e} | {fmt(R, ci)} |")


def select(a):
    """Shot lists ({point: {circuit: [shot...]}}): the network's any-product failures, or the shots routed at
    f = 10 % by either score; at most --cap shots (a uniform sample, numpy default_rng(186))."""
    out, n = {}, 0
    for d in (int(x) for x in a.ds.split(",")):
        for p in bank.points((d,)):
            r = load(a.bank, p, a.net)
            if a.what == "fails":
                idx = np.flatnonzero(fails(r, r["dec_net"])[1])
            else:
                s = scores(r)
                idx = np.union1d(routed(s["T"], 0.10)[0], routed(s["min"], 0.10)[0])
            out[p["name"]] = (idx, r["n"])
            n += len(idx)
    if a.cap and n > a.cap:
        rng = np.random.default_rng(186)
        keep = set(rng.choice(n, a.cap, replace=False).tolist())
        j, sel = 0, {}
        for name, (idx, nn) in out.items():
            sel[name] = (np.array([x for q, x in enumerate(idx) if j + q in keep], np.int64), nn)
            j += len(idx)
        out = sel
    res = {}
    for name, (idx, nn) in out.items():              # flat index = circuit * (the read's shots per circuit) + shot
        for f in idx:
            res.setdefault(name, {}).setdefault(str(int(f) // nn), []).append(int(f) % nn)
    json.dump(res, open(a.out, "w"))
    print(json.dumps(dict(shots=sum(len(v) for c in res.values() for v in c.values()), points=len(res))))


def agree(a):
    """ rule 6: decoder --x (Tesseract) against the certified MLE --y on --y's shots, per depth and pooled. The
    uncertified shots are counted apart; on the certified ones (that --x decoded): the shots whose predicted class of
    the target, or of some product, differs, and each decoder's target and any-product failures there."""
    rec = {}
    for d in (int(x) for x in a.ds.split(",")):
        print(f"\n### d = {d}: {a.x} against the certified {a.y} on its shots\n")
        print(f"| D | shots | uncertified | {a.x} missing | target differs | some product differs | target failures "
              f"{a.x} / {a.y} | any-product failures {a.x} / {a.y} |")
        print("|---|---|---|---|---|---|---|---|")
        tot = collections.Counter()
        for p in bank.points((d,)):
            r = load(a.bank, p, None, (a.x, a.y))
            x, y = r[a.x], r[a.y]
            if not y["cov"].any():
                continue
            c = y["cov"] & y["ok"] & x["cov"]
            diff = x["dec"][c] ^ y["dec"][c]
            fx, fy = fails(r, x["dec"]), fails(r, y["dec"])
            row = dict(shots=int(y["cov"].sum()), unc=int((y["cov"] & ~y["ok"]).sum()),
                       miss=int((y["cov"] & y["ok"] & ~x["cov"]).sum()), tdiff=int((diff & 1).astype(bool).sum()),
                       pdiff=int((diff & r["mask"][c]).astype(bool).sum()), tx=int(fx[0][c].sum()),
                       ty=int(fy[0][c].sum()), ax=int(fx[1][c].sum()), ay=int(fy[1][c].sum()))
            tot.update(row)
            rec[f"d{d}_D{p['depth']}"] = row
            print(f"| {p['depth']} | {row['shots']:,} | {row['unc']} | {row['miss']} | {row['tdiff']} | {row['pdiff']} | "
                  f"{row['tx']} / {row['ty']} | {row['ax']} / {row['ay']} |")
        t = dict(tot)
        rec[f"d{d}"] = t
        if t:
            print(f"| all | {t['shots']:,} | {t['unc']} | {t['miss']} | {t['tdiff']} | {t['pdiff']} | {t['tx']} / "
                  f"{t['ty']} | {t['ax']} / {t['ay']} |")
    if a.out:
        json.dump(rec, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["a0", "pair", "vs", "select", "agree"])
    ap.add_argument("--bank", required=True)
    ap.add_argument("--net", default="")
    ap.add_argument("--x", default="")
    ap.add_argument("--y", default="")
    ap.add_argument("--ds", default="3,5,7")
    ap.add_argument("--decs", default="lomr,tess")
    ap.add_argument("--tess", default="tess")
    ap.add_argument("--what", default="fails", choices=["fails", "routed"])
    ap.add_argument("--cap", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dict(a0=a0, pair=pair, vs=vs, select=select, agree=agree)[a.mode](a)


if __name__ == "__main__":
    main()
