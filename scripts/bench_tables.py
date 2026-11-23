"""Tables of the open benchmark's test-set results, from the record of scripts/bench_stats.py and the Lukin group's
published Fig. 1b values; with --net, the tables of the networks, with the development
checks of scripts/bench_net.py devstats (--dev). Each part is a written summary followed by this output.

  python scripts/bench_tables.py --record results/bench/records/results.json >> 
  python scripts/bench_tables.py --record R --net [--dev devstats.json] >> """
import argparse
import csv
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAMILIES = ("lom_rep", "lom_2q", "lom_deep", "lukin", "mccd", "cain_ghz", "ours_k")
MODELS = ("lukin", "uniform", "si1000", "cain")
NAMES = {"lomr": "LoMatching", "cain": "Cain et al.", "bplomr": "BP+LoMatching", "mle": "MLE", "net": "network"}
MATCHING = ("lomr", "cain", "bplomr")
NET_REFS = ("mle", "lomr", "bplomr")
PHYSICAL = ("lukin", "mccd", "lom_deep")      # external families with physical Paulis: the networks read them zero-shot
FROZEN = {("uniform", 5)}                     # cells read on the frozen path, the rest on the engine
Z = "(z): one count zero, 0.5 added to both, the ratio then a bound; \"no failures in either\": both counts zero, no " \
    "ratio (). Intervals: circuits redrawn for sampled circuit lists (lukin, mccd, ours_k), each " \
    "circuit's shots redrawn paired for exhaustive ones (lom_2q's roster, one-circuit points;) " \
    "when every circuit holds at least 10 paired shots, otherwise the point's paired shots redrawn together " \
    "(). (zero width): every draw gave the same ratio, so the interval says nothing."


def label(name, r):
    s = name[len(r["family"]) + 1:]
    return s[:s.rindex(f"_{r['noise']}_")].replace(f"_d{r['d']}", "") or "-"


def rate(c):
    if not c or not c["shots"]:
        return "-"
    lo, hi = c["rate95"]
    return f"{c['fails']:,}: {c['fails'] / c['shots']:.2e} [{lo:.1e}, {hi:.1e}]"


def ratio(v):
    if not v:
        return "-"
    if v.get("both_zero"):
        return "no failures in either"
    lo, hi = v["ratio95"]
    return (f"{v['ratio']:.2f} [{lo:.2f}, {hi:.2f}]" + (" (z)" if v.get("zero_count") else "")
            + (" (zero width)" if hi <= lo * (1 + 1e-9) else ""))


def fmt(v):
    return f"{v['ratio']:.2f} [{v['ratio95'][0]:.2f}, {v['ratio95'][1]:.2f}]"


FIG1B = ("Ours / theirs = our target failures / the failures their per-depth rates predict on our shots, summed over "
         "the depths. The interval is log-normal with variance 1 / (our failures) + sum(n^2 sigma^2) / (sum n p)^2, "
         "with n our shots, p their value and sigma its plotted error bar at each depth (data/open_benchmark/fig1b_values.csv; "
         "the binomial sigma at their shots where the figure has no bar), so their points' noise is weighted as their "
         "rates are (). The comparison is descriptive ().")


def pooled(cs):
    """Summed failures and shots of several count records, with the 95 % Wilson interval."""
    k, n = sum(c["fails"] for c in cs), sum(c["shots"] for c in cs)
    z = 1.959963984540054
    if not n:
        return dict(fails=0, shots=0, rate95=[math.nan, math.nan])
    p, den = k / n, 1 + z * z / n
    mid, half = (p + z * z / (2 * n)) / den, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return dict(fails=k, shots=n, rate95=[max(0.0, mid - half), min(1.0, mid + half)])


def order(item):
    name, r = item
    return (FAMILIES.index(r["family"]), MODELS.index(r["noise"]), r["p"], r["d"], name)


def theirs(decoder="MLE"):
    """Fig. 1b's values (target parity) of their MLE or their network ("ML") per (d, D): (value, variance), the
    variance the plotted error bar squared. Where the figure has no error bar (MLE at d = 7, D = 1), the binomial
    variance at the shots of that depth (equal across d, from d = 3)."""
    rows = [r for r in csv.DictReader(ln for ln in open(os.path.join(ROOT, "data/open_benchmark/fig1b_values.csv"))
                                      if not ln.startswith("#")) if r["decoder"] == decoder]
    shots = {int(r["depth"]): float(r["N_implied"]) for r in rows if r["d"] == "3"}
    return {(int(r["d"]), int(r["depth"])): (float(r["value"]), float(r["std"]) ** 2 if r["std"] else
                                             float(r["value"]) * (1 - float(r["value"])) / shots[int(r["depth"])])
            for r in rows}


def vs_theirs(P, dec, th, d):
    """Our target failures of decoder `dec` at the lukin points (Lukin model, distance d) / the failures their per-depth
    rates th predict on our shots, summed over the depths, with a log-normal 95 % interval of variance
    1 / ours + sum(n^2 sigma^2) / (sum n p)^2. None where no depth has both."""
    rs = sorted((r for r in P.values() if r["family"] == "lukin" and r["noise"] == "lukin" and r["d"] == d
                 and dec in r["dec"] and (d, r["depth"]) in th), key=lambda r: r["depth"])
    if not rs:
        return None
    k = sum(r["dec"][dec]["target_fails"] for r in rs)
    n = [r["dec"][dec]["shots"] for r in rs]
    e = sum(ni * th[(d, r["depth"])][0] for ni, r in zip(n, rs))
    v = sum(ni * ni * th[(d, r["depth"])][1] for ni, r in zip(n, rs))
    x, s = k / e, math.sqrt(1 / max(k, 0.5) + v / e ** 2)
    return dict(depths=f"{rs[0]['depth']}-{rs[-1]['depth']}", fails=k, shots=sum(n), expected=e, ratio=x,
                ratio95=[x * math.exp(-1.96 * s), x * math.exp(1.96 * s)])


def net_mle_target(P, d):
    """Our network / our MLE on the lukin family's target parity at distance d under the Lukin model, pooled over the
    depths where the MLE ran with each depth weighted equally, as Fig. 1b's own ratio is; a log-normal 95 % interval
    from the binomial variances of the rates. None where no depth has both."""
    rs = sorted((r for r in P.values() if r["family"] == "lukin" and r["noise"] == "lukin" and r["d"] == d
                 and "net" in r["dec"] and "mle" in r["dec"]), key=lambda r: r["depth"])
    if not rs:
        return None
    rate = lambda c: c["target_fails"] / c["shots"]                       # noqa: E731
    var = lambda c: max(c["target_fails"], 0.5) / c["shots"] ** 2         # noqa: E731
    a, b = sum(rate(r["dec"]["net"]) for r in rs), sum(rate(r["dec"]["mle"]) for r in rs)
    x, s = a / b, math.sqrt(sum(var(r["dec"]["net"]) for r in rs) / a ** 2 + sum(var(r["dec"]["mle"]) for r in rs) / b ** 2)
    return dict(D=[r["depth"] for r in rs], depths=f"{rs[0]['depth']}-{rs[-1]['depth']}", ratio=x,
                ratio95=[x * math.exp(-1.96 * s), x * math.exp(1.96 * s)])


def gkey(k):
    f, m, d = k.split("|")
    return FAMILIES.index(f), MODELS.index(m), int(d)


def net_tables(rec, P, G, dev):
    """Phase 4's tables: the networks against the other decoders."""
    k1 = (dev or {}).get("k1") or {}
    form = k1.get("form", "native")
    pen = (dev or {}).get("twin", {})
    mark = lambda f: "physical, zero-shot" if f in PHYSICAL else "software" if f == "ours_k" else "none"   # noqa: E731
    print("## The networks against the other decoders, pooled per family, model and d\n")
    print("Failures of the network / failures of each other decoder on the shots both decoded (every shot for "
          "LoMatching, the first BP and MLE shots of each circuit for those two; uncertified MLE shots left out for "
          f"every decoder), summed over the points; 95 % bootstrap intervals ({rec['boot']:,} draws). {Z} Paulis: what "
          "the family's circuits apply; the networks were trained with software Paulis, so the physical ones are a "
          "zero-shot read, with the twin's penalty of section 13.3 beside it.\n")
    print("| family | model | d | Paulis | points | network failures / shots | network rate [95 %] | network / MLE | "
          "network / LoMatching | network / BP+LoMatching |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted((k for k in G if "net" in G[k]), key=gkey):
        f, m, d = key.split("|")
        g = G[key]
        dd = f"{d} (frozen path)" if (m, int(d)) in FROZEN else d
        print(f"| {f} | {m} | {dd} | {mark(f)} | {g['net']['points']} | {g['net']['fails']:,} / {g['net']['shots']:,} | "
              f"{rate(g['net']).split(': ')[1]} | " + " | ".join(ratio(g["net_vs"].get(y)) for y in NET_REFS) + " |")
    if pen:
        print("\nThe twin's penalty (section 13.3; development shots): (network / MLE failures, physical Paulis) / (the "
              "same, software Paulis), 95 % interval from circuits redrawn within family and depth, the same circuits in "
              "both arms. It is the network's relative performance under the changed noise, not the causal effect of "
              "the missing Pauli input.\n")
        print("| cell | circuits | network / MLE failures, physical | network / MLE failures, software | "
              "penalty [95 %] | fine-tune called |")
        print("|---|---|---|---|---|---|")
        for cell, t in sorted(pen.items()):
            print(f"| {cell.split('_d')[0]}, d = {cell.split('_d')[1]} | {t['circuits']} | {t['net_gate']} / {t['mle_gate']} | {t['net_frame']} / "
                  f"{t['mle_frame']} | {t['penalty']:.2f} [{t['penalty95'][0]:.2f}, {t['penalty95'][1]:.2f}] | "
                  f"{'yes' if t['fine_tune_called'] else 'no'} |")

    print("\n## The d = 7 claim: lukin family, Lukin model, D = 1-8\n")
    g = G.get("lukin|lukin|7", {})
    print(f"Network / MLE, pooled over the depths where the MLE ran: {ratio(g.get('net_vs', {}).get('mle'))}.\n")

    print("## lukin family, target parity, against the Lukin group's Fig. 1b\n")
    print(FIG1B + " Theirs: their network (ML) and their MLE.\n")
    print("| d | ours | depths | our target failures / shots | predicted by their network | ours / their network "
          "[95 %] | predicted by their MLE | ours / their MLE [95 %] |")
    print("|---|---|---|---|---|---|---|---|")
    ml, mle = theirs("ML"), theirs("MLE")
    for d in (3, 5, 7):
        for dec in ("net", "mle"):
            a, b = vs_theirs(P, dec, ml, d), vs_theirs(P, dec, mle, d)
            if a or b:
                v = a or b
                print(f"| {d} | {NAMES[dec]} | {v['depths']} | {v['fails']} / {v['shots']:,} | "
                      + (f"{a['expected']:.1f} | {fmt(a)}" if a else "- | -") + " | "
                      + (f"{b['expected']:.1f} | {fmt(b)}" if b else "- | -") + " |")
    print("\nNetwork / MLE pooled over the depths where our MLE ran (; descriptive). Ours on target parity "
          "from our rates per depth, each depth weighted equally as Fig. 1b's own ratio is (95 % log-normal interval from "
          "the binomial variances), and, as the evidence, on the same shots with every product scored (paired); theirs "
          "from Fig. 1b's values, over the same depths and over D = 1-10.\n")
    print("| d | depths | ours: network / MLE, target parity [95 %] | ours: network / MLE, same shots, any product "
          "[95 %] | theirs: network / MLE, same depths | theirs, D = 1-10 |")
    print("|---|---|---|---|---|---|")
    for d in (3, 5, 7):
        v = net_mle_target(P, d)
        if v:
            th = sum(ml[(d, D)][0] for D in v["D"]) / sum(mle[(d, D)][0] for D in v["D"])
            th10 = sum(ml[(d, D)][0] for D in range(1, 11)) / sum(mle[(d, D)][0] for D in range(1, 11))
            print(f"| {d} | {v['depths']} | {fmt(v)} | {ratio(G.get(f'lukin|lukin|{d}', {}).get('net_vs', {}).get('mle'))} "
                  f"| {th:.2f} | {th10:.2f} |")

    print("\n## Buildable-subset control\n")
    keys = sorted((k for k in G if G[k].get("buildable_net_vs")), key=gkey)
    if not keys:
        print("No circuit of these points has an error that the matching decoders leave out.")
    else:
        print("| family | model | d | circuits left out | network / MLE | network / LoMatching | "
              "network / BP+LoMatching |")
        print("|---|---|---|---|---|---|---|")
    for key in keys:
        f, m, d = key.split("|")
        print(f"| {f} | {m} | {d} | {G[key]['circuits_dropped']} | "
              + " | ".join(ratio(G[key]["buildable_net_vs"].get(y)) for y in NET_REFS) + " |")

    if dev:
        print("\n## Development checks (section 13.4)\n")
        print("| cell | circuits | shots | decisions that differ | largest logit difference | detectors equal | "
              "passed |")
        print("|---|---|---|---|---|---|---|")
        for cell, c in sorted(dev.get("check", {}).items()):
            m, d = cell.split("_d")
            print(f"| {m}, d = {d} | {c['circuits']} | {c['shots']:,} | {c['differ']} | {c['max_abs_dc']:.1e} | "
                  f"{'yes' if c['detectors_equal'] else 'no'} | {'yes' if c['passed'] else 'no'}"
                  + (" (: read on the frozen path)" if (m, int(d)) in FROZEN else "") + " |")
        if k1:
            print(f"\nK = 1 form: idle-block / native failures {k1['idle_fails']:,} / {k1['native_fails']:,} = "
                  f"{k1['ratio']:.3f} [{k1['ratio95'][0]:.3f}, {k1['ratio95'][1]:.3f}]; read in the {form} form.")

    print("\n## Per point\n")
    print(f"Cells: failures: rate per shot [95 % Wilson interval]. K = 1 points are read in the {form} form.\n")
    for fam in FAMILIES:
        rows = sorted(((n, r) for n, r in P.items() if r["family"] == fam and "net" in r["dec"]),
                      key=lambda x: (*order(x)[:4], x[1]["depth"], x[0]))
        if not rows:
            continue
        print(f"### {fam} ({len(rows)} points; Paulis {mark(fam)})\n")
        print("| label | d | model | depth | circuits | network | LoMatching | BP+LoMatching | MLE | network / MLE | "
              "network / LoMatching |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for n, r in rows:
            dec = r["dec"]
            print(f"| {label(n, r)} | {r['d']} | {r['noise']} | {r['depth']} | {r['circuits']} | "
                  + " | ".join(rate(dec.get(x)) for x in ("net", "lomr", "bplomr", "mle")) + " | "
                  + " | ".join(ratio(r["net_vs"].get(y)) for y in ("mle", "lomr")) + " |")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--net", action="store_true", help="phase 4's tables (the networks)")
    ap.add_argument("--dev", help="scripts/bench_net.py devstats' devstats.json")
    a = ap.parse_args()
    rec = json.load(open(a.record))
    P, G = rec["points"], rec["pooled"]
    if a.net:
        return net_tables(rec, P, G, json.load(open(a.dev)) if a.dev else None)
    man = {pt["name"]: pt for pt in json.load(open(os.path.join(ROOT, "data/open_benchmark/MANIFEST.json")))["points"]}

    print("## Failures relative to the exact MLE, pooled per family, model and d\n")
    print("Failures of each decoder / failures of the MLE on the shots both decoded (each circuit's shorter prefix, "
          "certified MLE results only), summed over the points where the MLE ran; 95 % bootstrap intervals "
          f"({rec['boot']:,} draws). {Z}\n")
    print("| family | model | d | points | MLE failures / shots | MLE rate [95 %] | LoMatching / MLE | "
          "Cain et al. / MLE | BP+LoMatching / MLE |")
    print("|---|---|---|---|---|---|---|---|---|")
    for key in sorted((k for k in G if "mle" in G[k]), key=lambda k: (FAMILIES.index(k.split("|")[0]),
                                                                     MODELS.index(k.split("|")[1]), int(k.split("|")[2]))):
        f, m, d = key.split("|")
        g = G[key]
        lo, hi = g["mle"]["rate95"]
        print(f"| {f} | {m} | {d} | {g['points']} | {g['mle']['fails']:,} / {g['mle']['shots']:,} | "
              f"{g['mle']['fails'] / g['mle']['shots']:.2e} [{lo:.1e}, {hi:.1e}] | "
              + " | ".join(ratio(g["vs_mle"].get(x)) for x in MATCHING) + " |")

    print("\n## The d = 7 pooled MLE claim: lukin family, Lukin model, D = 1-8\n")
    th = theirs()
    rows = sorted(((n, r) for n, r in P.items() if r["family"] == "lukin" and r["noise"] == "lukin" and r["d"] == 7
                   and "mle" in r["dec"]), key=lambda x: x[1]["depth"])
    print("| D | MLE shots | MLE failures: rate [95 %] | target parity failures: rate [95 %] | Fig. 1b MLE (target) | "
          "LoMatching / MLE | Cain et al. / MLE | BP+LoMatching / MLE | MLE s/shot |")
    print("|---|---|---|---|---|---|---|---|---|")
    for n, r in rows:
        m = r["dec"]["mle"]
        t = dict(shots=m["shots"], fails=m["target_fails"], rate95=m["target_rate95"])
        print(f"| {r['depth']} | {m['shots']:,} | {rate(m)} | {rate(t)} | {th.get((7, r['depth']), ('-',))[0]} | "
              + " | ".join(ratio(r["vs_mle"].get(x)) for x in MATCHING)
              + f" | {r['mle_cpu'] / max(m['shots'] + r['uncertified'], 1):.2f} |")
    g = G.get("lukin|lukin|7")
    if g:
        lo, hi = g["mle"]["rate95"]
        print(f"| 1-8 pooled | {g['mle']['shots']:,} | {g['mle']['fails']}: {g['mle']['fails'] / g['mle']['shots']:.2e} "
              f"[{lo:.1e}, {hi:.1e}] | | | " + " | ".join(ratio(g["vs_mle"].get(x)) for x in MATCHING) + " | |")

    print("\n## lukin family, target parity: our exact MLE against the Lukin group's (Fig. 1b), pooled over depths\n")
    print(FIG1B + "\n")
    print("| d | depths | our target failures / shots | failures their rates predict on our shots | "
          "ours / theirs [95 %] |")
    print("|---|---|---|---|---|")
    for d in (3, 5, 7):
        v = vs_theirs(P, "mle", th, d)
        if v:
            print(f"| {d} | {v['depths']} | {v['fails']} / {v['shots']:,} | {v['expected']:.1f} | {fmt(v)} |")

    print("\n## Buildable-subset control\n")
    print("Points where a matching decoder dropped an error (stim could not split it into graphlike pieces) on some "
          "circuit; the MLE covers none of them. Each decoder's failures over those points, pooled: on every circuit, "
          "then on the circuits where no matching decoder dropped an error (the same shots for every decoder).\n")
    groups = {}
    for r in P.values():
        if r["circuits_dropped"]:
            groups.setdefault((r["family"], r["noise"], r["d"]), []).append(r)
    print("| family | model | d | points | circuits left out / of those points / at this family, model and d | "
          + " | ".join(f"{NAMES[x]}: all; buildable" for x in MATCHING) + " |")
    print("|---|---|---|---|---|---|---|---|")
    for (f, m, d), rs in sorted(groups.items(), key=lambda x: (FAMILIES.index(x[0][0]), MODELS.index(x[0][1]),
                                                                 x[0][2])):
        allc = sum(r["circuits"] for r in P.values() if (r["family"], r["noise"], r["d"]) == (f, m, d))
        cells = []
        for x in MATCHING:
            c = [pooled([r["dec"][x] for r in rs]), pooled([r["buildable"]["dec"][x] for r in rs])]
            cells.append(f"{rate(c[0])}; {rate(c[1])}")
        print(f"| {f} | {m} | {d} | {len(rs)} | {sum(r['circuits_dropped'] for r in rs)} / "
              f"{sum(r['circuits'] for r in rs)} / {allc} | " + " | ".join(cells) + " |")
    print("\n| point | d | model | circuits left out / all | " + " | ".join(
        f"{NAMES[x]}: all; buildable" for x in ("lomr", "cain", "bplomr", "mle")) + " |")
    print("|---|---|---|---|---|---|---|---|")
    for name, r in sorted(((n, r) for n, r in P.items() if r["circuits_dropped"]), key=order):
        b = r["buildable"]["dec"]
        print(f"| {name} | {r['d']} | {r['noise']} | {r['circuits_dropped']} / {r['circuits']} | "
              + " | ".join(f"{rate(r['dec'].get(x))}; {rate(b.get(x))}" if x in r["dec"] else "-"
                           for x in ("lomr", "cain", "bplomr", "mle")) + " |")

    print("\n## Per point and decoder\n")
    print("Cells: failures: rate per shot [95 % Wilson interval]. Shots per circuit: every shot for LoMatching and Cain "
          "et al., the first BP and MLE shots of each circuit for those two; uncertified MLE shots are left out for "
          "every decoder. Label: the point's name without family, d, model and p.\n")
    for fam in FAMILIES:
        rows = sorted(((n, r) for n, r in P.items() if r["family"] == fam), key=order)
        if not rows:
            continue
        print(f"### {fam} ({len(rows)} points)\n")
        print("| label | d | model | p | depth | circuits | shots per circuit (all / BP / MLE) | LoMatching | "
              "Cain et al. | BP+LoMatching | MLE |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for n, r in rows:
            dec, pt = r["dec"], man[n]
            per = lambda x: f"{pt.get(x + '_shots', 0):,}" if x in dec else "-"
            print(f"| {label(n, r)} | {r['d']} | {r['noise']} | {r['p'] * 100:g} % | {r['depth']} | {r['circuits']} | "
                  f"{r['shots']:,} / {per('bplomr')} / {per('mle')} | "
                  + " | ".join(rate(dec.get(x)) for x in ("lomr", "cain", "bplomr", "mle")) + " |")
        print()


if __name__ == "__main__":
    main()
