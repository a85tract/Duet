"""Main-text tables on the fresh test set at the operating points (the paper's "value points").
  python3 paper/scripts/fresh_main.py [--out DIR] [--only tab_x.tex ...]       (run before make_data.py and rt_tables.py)
Value point per (benchmark, platform, d), on development data only (data/fresh_test/development_best.json): the smallest
f > 0 whose development failures are within 1 % of the most accurate development row on that platform (the rule of
value_point.py). Writes DIR/tables/tab_benchmark1.tex, tab_benchmark2.tex, tab_routing.tex, tab_app_value.tex,
DIR/data/benchmark1_ratios.csv and DIR/data/value_points.csv (DIR defaults to paper/out) from
data/fresh_test/d<d>/benchmark<1,2>/ (grid.json.gz, mle_measures.json.gz, report.md). --only limits the tables written."""
import argparse
import csv
import gzip
import json
import re
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent
BENCH = {1: "I", 2: "II"}   # the paper's names of the two results: Benchmark I (general circuits), II (the Lukin group's)
_ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
_ap.add_argument("--out", default=str(PAPER / "out"), help="output directory (tables/ and data/ below it)")
_ap.add_argument("--only", nargs="*", default=None, help="table file names to write (default: all)")
ARGS = _ap.parse_args()
OUT = Path(ARGS.out)
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "data").mkdir(parents=True, exist_ok=True)
FRESH = PAPER / "data" / "fresh_test"
DELTA = 0.01


def bdir(d, res):
    return FRESH / f"d{d}" / f"benchmark{res}"


def jgz(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def num(n):
    """Integers with a thin-space thousands separator from 10,000 up, as make_data.py writes them."""
    return f"{n:,}".replace(",", "\\,") if n >= 10000 else str(n)


def pct(f):
    return f"{float(f) * 100:g}\\,\\%"


def capl(c):
    return "none" if c == "none" else f"{c}\\,s"


def iv(x, fmt):
    return (fmt.format(x[0]), f"{{\\tiny[{fmt.format(x[1][0])}, {fmt.format(x[1][1])}]}}")


def tabular(name, spec, head, lines):
    if ARGS.only is not None and name not in ARGS.only:
        return
    (OUT / "tables" / name).write_text("% fresh_main.py from data/fresh_test\n\\begin{tabular}{" + spec + "}\n\\toprule\n"
                                       + head.strip() + "\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    print(f"tables/{name}")


# ------------------------------------------------------------------ value points (development data only)
DEV = json.loads((FRESH / "development_best.json").read_text())["rows"]
VP = {}
for key in sorted({(r["result"], r["basis"], r["d"]) for r in DEV}):
    rs = sorted([r for r in DEV if (r["result"], r["basis"], r["d"]) == key and r["f"] != "0"], key=lambda r: float(r["f"]))
    best = min(rs, key=lambda r: r["fails"])
    pick = next(r for r in rs if r["fails"] <= best["fails"] * (1 + DELTA))
    VP[key] = {"pick": pick, "best": best, "f20": next(r for r in rs if r["f"] == "0.2"),
               "cell": f"{key[1]}|{pick['f']}|{pick['cap']}"}
    print(f"  value point Benchmark {BENCH[key[0]]} {key[1]:5s} d = {key[2]}: f = {pct(pick['f'])}, cap {pick['cap']} "
          f"(development {pick['fails']} against the best {best['fails']} at f = {pct(best['f'])})")


def report_rows(res, d):
    """{(basis, f, cap): row dict} of report.md's basis tables (rows marked B, the development-best cap of each f)."""
    out, basis, head = {}, None, None
    for ln in (bdir(d, res) / "report.md").read_text().splitlines():
        m = re.match(r"## Basis (\w+) \(N", ln)
        if ln.startswith("## "):
            basis, head = (m.group(1), None) if m else (None, None)
        elif basis and ln.startswith("| f |"):
            head = [c.strip() for c in ln.strip().strip("|").split("|")]
        elif basis and head and ln.startswith("| ") and not ln.startswith("|---"):
            r = dict(zip(head, [c.strip() for c in ln.strip().strip("|").split("|")]))
            out[(basis, r["f"], r["cap"])] = r
    return out


# ------------------------------------------------------------------ value-point rows of the fresh test, with f = 20 %
vrows = []
for (res, basis, d), v in VP.items():
    rep = report_rows(res, d)
    def test(r):
        x = rep[(basis, r["f"], r["cap"])]
        fails = int((x.get("failures / shots") or x["target failures"]).split("/")[0].replace(",", ""))
        return fails, float(x["us per layer (p99 ms)"].split()[0])
    tf, tu = test(v["pick"])
    f20, u20 = test(v["f20"])
    vrows.append({"result": res, "basis": basis, "d": d, "f": v["pick"]["f"], "cap": v["pick"]["cap"],
                  "dev_fails": v["pick"]["fails"], "best_fails": v["best"]["fails"], "best_f": v["best"]["f"],
                  "dev_us": round(v["pick"]["us_per_layer_stream"], 1), "test_fails": tf, "test_us": tu,
                  "f20_cap": v["f20"]["cap"], "f20_fails": f20, "f20_us": u20, "bar": v["pick"]["bar"]})
with open(OUT / "data" / "value_points.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(vrows[0]))
    w.writeheader()
    w.writerows(vrows)
lines = []
for i, x in enumerate(vrows):
    if i and x["result"] != vrows[i - 1]["result"]:
        lines.append("\\midrule")
    lines.append(" & ".join([BENCH[x["result"]], str(x["d"]), x["basis"].capitalize(), pct(x["f"]), capl(x["cap"]),
                             num(x["dev_fails"]), f"{num(x['best_fails'])} ({pct(x['best_f'])})", num(x["test_fails"]),
                             f"{x['test_us']:.1f}", num(x["f20_fails"]), f"{x['f20_us']:.1f}"]) + " \\\\")
    print(f"  fresh Benchmark {BENCH[x['result']]} {x['basis']:5s} d = {x['d']}: {x['test_fails']} failures at {x['test_us']} us "
          f"(f = 20 %: {x['f20_fails']} at {x['f20_us']} us, cap {x['f20_cap']}; bar {x['bar']})")
tabular("tab_app_value.tex", "@{}cclcrrrrrrr@{}", r"""
& & & & & \multicolumn{2}{c}{development failures} & \multicolumn{2}{c}{fresh test} & \multicolumn{2}{c}{fresh test, $f=20\,\%$} \\
\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}
benchmark & $d$ & platform & $f$ & cap per layer & this row & best row & failures & $\mu$s per layer & failures & $\mu$s per layer \\""",
         lines)

# ------------------------------------------------------------------ Benchmark I (Table I) and Fig. 3's data
GR = {"p3": "_all_p3", "plow": "_lowp"}
lines, fig, gap = [], [], 0.0
for d in (3, 5, 7, 9):
    g = jgz(bdir(d, 1) / "grid.json.gz")
    P = lambda key, grp: (g["pooled"][key]["0.5"][f"d{d}{GR[grp]}"]["ratio"], g["pooled"][key]["0.5"][f"d{d}{GR[grp]}"]["ratio95"])
    gc, sc = VP[(1, "grace", d)], VP[(1, "spark", d)]
    comps = [(gc, "")]
    if (gc["pick"]["f"], gc["pick"]["cap"]) != (sc["pick"]["f"], sc["pick"]["cap"]):
        comps.append((sc, "Spark: "))
    else:
        for opp in ("lomr", "cain", "bplomr", "network"):
            for grp in GR:
                gap = max(gap, abs(P(f"{gc['cell']}/{opp}", grp)[0] - P(f"{sc['cell']}/{opp}", grp)[0]))
    for j, (c, lab) in enumerate(comps):
        for grp in GR:
            top = [str(d) if j == 0 and grp == "p3" else "", lab + pct(c["pick"]["f"]) if grp == "p3" else "",
                   "0.3" if grp == "p3" else "0.1, 0.2"]
            bot = ["", "", ""]
            for opp in ("lomr", "cain", "bplomr"):
                for key in (f"network/{opp}", f"{c['cell']}/{opp}"):
                    if j and key.startswith("network"):
                        top.append("")
                        bot.append("")
                        continue
                    t, b = iv(P(key, grp), "{:.4f}")
                    top.append(t)
                    bot.append(b)
            t, b = iv(P(f"{c['cell']}/network", grp), "{:.4f}")
            lines.append(" & ".join(top + [t]) + " \\\\")
            lines.append(" & ".join(bot + [b]) + " \\\\" + ("[2pt]" if grp == "p3" else ""))
            if j == 0:
                for key, s in [("network/lomr", "net_lom"), (f"{c['cell']}/lomr", "comp_lom"), ("network/bplomr", "net_bp"),
                               (f"{c['cell']}/bplomr", "comp_bp"), ("network/cain", "net_cain"),
                               (f"{c['cell']}/cain", "comp_cain"), (f"{c['cell']}/network", "comp_net")]:
                    r, (lo, hi) = P(key, grp)
                    fig.append({"d": d, "group": grp, "series": s, "r": f"{r:.4f}", "lo": f"{lo:.4f}", "hi": f"{hi:.4f}"})
    if d != 9:
        lines.append("\\midrule")
tabular("tab_benchmark1.tex", "@{}cccccccccc@{}", r"""
& & & \multicolumn{2}{c}{against LoMatching} & \multicolumn{2}{c}{against Cain et al.} & \multicolumn{2}{c}{against \bplom} & composition \\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
$d$ & $f$ & $p$ (\%) & network & composition & network & composition & network & composition & / network \\""", lines)
print(f"  Benchmark I: largest |Grace - Spark| ratio gap where both platforms share (f, cap): {gap:.4f}")
with open(OUT / "data" / "benchmark1_ratios.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["d", "group", "series", "r", "lo", "hi"])
    w.writeheader()
    w.writerows(fig)

# ------------------------------------------------------------------ Benchmark II (Table II)
lines = []
for d in (3, 5, 7):
    e = jgz(bdir(d, 2) / "mle_measures.json.gz")["systems"]
    gc, sc = VP[(2, "grace", d)], VP[(2, "spark", d)]
    same = (gc["pick"]["f"], gc["pick"]["cap"]) == (sc["pick"]["f"], sc["pick"]["cap"]) \
        and e[gc["cell"]]["a"]["fails"] == e[sc["cell"]]["a"]["fails"]
    rows = [("network alone", "network")] + ([(f"composition, both, {pct(gc['pick']['f'])}", gc["cell"])] if same else
                                             [(f"composition, Grace, {pct(gc['pick']['f'])}", gc["cell"]),
                                              (f"composition, Spark, {pct(sc['pick']['f'])}", sc["cell"])])
    for j, (lab, cell) in enumerate(rows):
        s = e[cell]
        a, b = s["a"], s["b"]
        ta, ba = iv((a["ratio"], a["ci"]), "{:.3f}")
        to, bo = iv((b["ours"], b["ours_ci"]), "{:.3f}")
        tb, bb = iv((b["ratio"], b["ci"]), "{:.3f}")
        tl, bl = iv(s["vs_lomr"], "{:.3f}")
        tn, bn = iv(s["vs_network"], "{:.3f}") if cell != "network" else ("---", "")
        lines.append(" & ".join([str(d) if j == 0 else "", lab, f"{a['fails']} / {a['expected']:.1f}", ta, to,
                                 f"{b['theirs']:.3f}", tb, tl, tn]) + " \\\\")
        lines.append(" & ".join(["", "", "", ba, bo, "", bb, bl, bn]) + " \\\\")
        mle = sum(x["mle"] for x in s["depths"] if x.get("mle") is not None)
        xs = sum(x["x_on_mle"] for x in s["depths"] if x.get("x_on_mle") is not None)
        print(f"  Benchmark II d = {d} {lab}: (a) {a['fails']}/{a['expected']:.1f} = {a['ratio']:.3f} {a['ci']} sens {a.get('ci_sens')}; "
              f"(b) {b['ratio']:.3f} {b['ci']}, ours/MLE {b['ours']:.3f} {b['ours_ci']}; certified prefix {xs} against MLE {mle}")
    if d != 7:
        lines.append("\\midrule")
tabular("tab_benchmark2.tex", "@{}clccccccc@{}", r"""
& & \multicolumn{2}{c}{measure (a)} & \multicolumn{3}{c}{measure (b)} & & \\
\cmidrule(lr){3-4}\cmidrule(lr){5-7}
$d$ & system & failures / expected & ratio & ours / MLE & theirs / MLE & ratio & vs LoMatching & vs network \\""", lines)

# ------------------------------------------------------------------ where the failures go (Table III)
lines, rgap = [], 0
K = ("net_fail_kept", "net_fail_routed", "comp_fail_routed", "corrected", "introduced", "routed", "shots")
for res, ds in ((1, (3, 5, 7, 9)), (2, (3, 5, 7))):
    if res == 2:
        lines.append("\\midrule")
    for d in ds:
        routed_by_cell = jgz(bdir(d, res) / "grid.json.gz")["routing"]
        def tot(cell):
            if res == 1:
                return {k: routed_by_cell[cell][f"d{d}_p3"][k] for k in K}
            return {k: sum(v[k] for kk, v in routed_by_cell[cell].items() if re.fullmatch(rf"d{d}_D[0-9]+", kk)) for k in K}
        gc, sc = VP[(res, "grace", d)], VP[(res, "spark", d)]
        pts = [("Grace", gc)]
        if (gc["pick"]["f"], gc["pick"]["cap"]) != (sc["pick"]["f"], sc["pick"]["cap"]):
            pts.append(("Spark", sc))
        else:
            a, b = tot(gc["cell"]), tot(sc["cell"])
            rgap = max(rgap, max(abs(a[k] - b[k]) for k in K))
            pts = [("both" if a == b else "Grace", gc)]
        for plat, c in pts:
            x = tot(c["cell"])
            nk, nr, cr = x["net_fail_kept"], x["net_fail_routed"], x["comp_fail_routed"]
            assert cr == nr - x["corrected"] + x["introduced"], (res, d, plat)
            n0, c0 = nk + nr, nk + cr
            lines.append(" & ".join([BENCH[res], str(d), f"{plat}, {pct(c['pick']['f'])}, {capl(c['pick']['cap'])}",
                                     f"{100 * x['routed'] / x['shots']:.2f}\\,\\%", num(nk),
                                     f"{num(nr)} ({100 * nr / n0:.2f}\\,\\%)", f"{num(cr)} ({cr / nr:.3f})",
                                     num(x["corrected"]), num(x["introduced"]),
                                     f"{num(n0)} $\\to$ {num(c0)} ({c0 / n0:.3f})"]) + " \\\\")
            print(f"  routing Benchmark {BENCH[res]} d = {d} {plat}: routed {x['routed']}/{x['shots']}, network {n0} (kept {nk}, routed {nr}), "
                  f"composition {c0} (routed {cr}), corrected {x['corrected']}, introduced {x['introduced']}")
tabular("tab_routing.tex", "@{}cclrrrrrrr@{}", r"""
& & & & \multicolumn{2}{c}{network failures} & composition on & & & failures, network \\
\cmidrule(lr){5-6}
benchmark & $d$ & platform, $f$, cap & routed & kept & routed (share) & routed (ratio) & corrected & introduced & $\to$ composition (ratio) \\""",
         lines)
print("  routing: largest Grace-Spark gap of a row listed once:", rgap)
