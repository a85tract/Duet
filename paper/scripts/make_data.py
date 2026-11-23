"""Tables of the paper (Tables IV, V, VII-XII, XX, XXI) and the speed figure's data (speed.csv) from the result files in
paper/data/first_test, paper/data/training and paper/data/fresh_test.
  python3 paper/scripts/make_data.py [--out DIR] [--only tab_x.tex ...]       (run fresh_main.py first)
Writes DIR/tables/*.tex and DIR/data/speed.csv (DIR defaults to paper/out). It reads DIR/data/value_points.csv, which
fresh_main.py writes. --only limits the tables that are written; the computation is the same.
A value that does not exist yet is written as \\placeholder{...}."""
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

BENCH = {1: "I", 2: "II"}   # the paper's names of the two results: Benchmark I (general circuits), II (the Lukin group's)
PAPER = Path(__file__).resolve().parent.parent
_ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
_ap.add_argument("--out", default=str(PAPER / "out"), help="output directory (tables/ and data/ below it)")
_ap.add_argument("--only", nargs="*", default=None, help="table file names to write (default: all)")
ARGS = _ap.parse_args()
OUT = Path(ARGS.out)
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "data").mkdir(parents=True, exist_ok=True)
FIRST = PAPER / "data" / "first_test"       # first test: results.md, choices.md, appendix_tables.md, network times
TRAIN = PAPER / "data" / "training"         # the network variants and their parameter counts
FRESH = PAPER / "data" / "fresh_test"       # fresh test: live-stream tables
DS = (3, 5, 7, 9)
NUM = r"[0-9][0-9,]*(?:\.[0-9]+)?"


def read(path):
    return Path(path).read_text()


def md_tables(text):
    """Every markdown table of a document: the enclosing ## and ### headings, the header and the rows (dicts)."""
    out, h2, h3, cur = [], "", "", None
    for line in text.splitlines():
        if line.startswith("## "):
            h2, h3 = line[3:].strip(), ""
        elif line.startswith("### "):
            h3 = line[4:].strip()
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cur is None:
                cur = {"h2": h2, "h3": h3, "header": cells, "rows": []}
                out.append(cur)
            elif not set("".join(cells)) <= set("-: "):
                cur["rows"].append(dict(zip(cur["header"], cells)))
        else:
            cur = None
    return out


def find(tabs, h2=None, h3=None, first=None, has=None):
    hits = [t for t in tabs if (h2 is None or t["h2"] == h2) and (h3 is None or t["h3"] == h3)
            and (first is None or t["header"][0] == first) and (has is None or has in t["header"])]
    assert len(hits) == 1, (h2, h3, first, len(hits))
    return hits[0]


def ci(cell):
    """'0.5645 (0.5585-0.5710)' -> ('0.5645', '0.5585', '0.5710'); None when the cell holds no interval."""
    m = re.match(rf"\s*({NUM})\s*\(({NUM})-({NUM})\)", cell)
    return m.groups() if m else None


def ci_sq(cell):
    """'0.976 [0.942, 1.010] (1673 / 1700)' -> ('0.976', '0.942', '1.010')."""
    m = re.match(rf"\s*({NUM})\s*\[({NUM}),\s*({NUM})\]", cell)
    return m.groups() if m else None


def num(s):
    """Integers with a thin-space thousands separator from 10,000 up (so 1722 stays 1722), other strings unchanged."""
    t = s.replace(",", "")
    if t.isdigit():
        return f"{int(t):,}".replace(",", "\\,") if int(t) >= 10000 else t
    return s


def two(c):
    """A value with its interval as two table lines: value, then [lo, hi] in a smaller size."""
    return (c[0], f"{{\\tiny[{c[1]}, {c[2]}]}}")


def write(name, text):
    if ARGS.only is not None and name not in ARGS.only:
        return
    (OUT / "tables" / name).write_text(text)
    print(f"tables/{name}")


def tabular(name, body_lines, foot=""):
    """A whole tabular, the header from HEAD and the rows generated here, so no \\input sits inside an alignment."""
    spec, head = HEAD[name]
    write(name, "\\begin{tabular}{" + spec + "}\n\\toprule\n" + head.strip() + "\n\\midrule\n" + "\n".join(body_lines)
          + "\n" + foot + "\\bottomrule\n\\end{tabular}\n")


HEAD = {
    "tab_first_benchmark1.tex": ("@{}cccccccccc@{}", r"""
& & & \multicolumn{2}{c}{against LoMatching} & \multicolumn{2}{c}{against Cain et al.} & \multicolumn{2}{c}{against \bplom} & composition \\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
$d$ & $f$ & $p$ (\%) & network & composition & network & composition & network & composition & / network \\"""),
    "tab_app_k.tex": ("@{}clccc@{}", r"""$d$ & ratio & $K=2$ & $K=3$ & $K=4$ \\"""),
    "tab_app_counts.tex": ("@{}ccrrrrrrr@{}", r"""
$d$ & $p$ (\%) & shots & network & composition, Grace & composition, Spark & LoMatching & Cain et al. & \bplom (matched shots) \\"""),
    "tab_first_benchmark2.tex": ("@{}clccccccc@{}", r"""
& & \multicolumn{2}{c}{measure (a)} & \multicolumn{3}{c}{measure (b)} & & \\
\cmidrule(lr){3-4}\cmidrule(lr){5-7}
$d$ & system & failures / expected & ratio & ours / MLE & theirs / MLE & ratio & vs LoMatching & vs network \\"""),
    "tab_sweep.tex": ("@{}cccrrcccc@{}", r"""
& & & \multicolumn{2}{c}{development} & \multicolumn{4}{c}{test} \\
\cmidrule(lr){4-5}\cmidrule(lr){6-9}
$d$ & $f$ & cap per layer & failures & $\mu$s per layer & (a) & (b) ours / MLE & (b) & vs network \\"""),
    "tab_speed.tex": ("@{}cccccccccc@{}", r"""
& their network & \multicolumn{4}{c}{network alone} & \multicolumn{2}{c}{composition, Benchmark I} & \multicolumn{2}{c}{composition, Benchmark II} \\
\cmidrule(lr){3-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}
& & \multicolumn{2}{c}{Benchmark I} & \multicolumn{2}{c}{Benchmark II} & & & & \\
$d$ & H100 & RTX PRO 6000 & GB10 & RTX PRO 6000 & GB10 & Grace platform & Spark platform & Grace platform & Spark platform \\"""),
    "tab_app_choices.tex": ("@{}cccccrrrrrrc@{}", r"""
benchmark & $d$ & platform & $f$ & cap per layer & composition & network & shots & routed & $\mu$s per layer & p99 (ms) & below budget \\"""),
    "tab_abl.tex": ("@{}lrcccccc@{}", r"""
& & \multicolumn{5}{c}{Benchmark I} & Benchmark II \\
\cmidrule(lr){3-7}
variant & parameters & $K=2$ & $K=3$ & $K=4$ & pooled & $p=0.1, 0.2\,\%$ & pooled \\"""),
}
SPEED_FOOT = ""
PH = "\\placeholder"
RES = md_tables(read(FIRST / "results.md"))
CHO = md_tables(read(FIRST / "choices.md"))
ABL = md_tables(read(TRAIN / "ablations.md"))
TRN = md_tables(read(TRAIN / "networks.md"))

# ------------------------------------------------------------------ Benchmark I: pooled ratios per d
PAIRS = [("network / LoMatching", "composition (Grace) / LoMatching"),
         ("network / Cain et al.", "composition (Grace) / Cain et al."),
         ("network / BP+LoMatching", "composition (Grace) / BP+LoMatching")]
COLS = {"p3": "p = 0.3 %, all K", "plow": "p = 0.1, 0.2 %"}
FCHO = {}   # (result, d, basis) -> choice row
for r in find(CHO, h2="Choices")["rows"]:
    FCHO[(int(r["result"]), int(r["d"]), r["basis"].split()[0])] = r


def fc(r):
    c = r["c (s x L)"]
    return r["f"].replace(" %", "\\,\\%"), ("none" if c == "none" else f"{c}\\,s")


bench1, fig_rows, spark_gap = {}, [], 0.0
lines = []
for d in DS:
    t = find(RES, h2=f"d = {d}", h3="Benchmark I", first="pair")
    rows = {r["pair"]: r for r in t["rows"]}
    bench1[d] = rows
    for g, col in COLS.items():
        for k, v in rows.items():
            if k.startswith("composition (Grace)"):
                s = rows[k.replace("(Grace)", "(Spark)")]
                for kk in v:
                    if kk != "pair":
                        spark_gap = max(spark_gap, abs(float(ci(v[kk])[0]) - float(ci(s[kk])[0])))
        f, cap = fc(FCHO[(1, d, "grace")])
        top = [str(d) if g == "p3" else "", f if g == "p3" else "", "0.3" if g == "p3" else "0.1, 0.2"]
        bot = ["", "", ""]
        for a, b in PAIRS:
            for k in (a, b):
                v = two(ci(rows[k][col]))
                top.append(v[0])
                bot.append(v[1])
        v = two(ci(rows["composition (Grace) / network"][col]))
        top.append(v[0])
        bot.append(v[1])
        lines.append(" & ".join(top) + " \\\\")
        lines.append(" & ".join(bot) + " \\\\" + ("[2pt]" if g == "p3" else ""))
        for k, key in [("network / LoMatching", "net_lom"), ("composition (Grace) / LoMatching", "comp_lom"),
                       ("network / BP+LoMatching", "net_bp"), ("composition (Grace) / BP+LoMatching", "comp_bp"),
                       ("network / Cain et al.", "net_cain"), ("composition (Grace) / Cain et al.", "comp_cain"),
                       ("composition (Grace) / network", "comp_net")]:
            x = ci(rows[k][col])
            fig_rows.append({"d": d, "group": g, "series": key, "r": x[0], "lo": x[1], "hi": x[2]})
    if d != DS[-1]:
        lines.append("\\midrule")
tabular("tab_first_benchmark1.tex", lines)
print(f"  largest |Grace - Spark| over every Benchmark I ratio cell: {spark_gap:.4f}")

# per-K breakdown (appendix) and the failure counts
lines = []
for d in DS:
    rows = bench1[d]
    for k, lab in [("network / LoMatching", "network / LoMatching"),
                   ("composition (Grace) / LoMatching", "composition / LoMatching"),
                   ("network / BP+LoMatching", "network / BP+LoMatching"),
                   ("composition (Grace) / BP+LoMatching", "composition / BP+LoMatching"),
                   ("composition (Grace) / network", "composition / network")]:
        r = rows[k]
        cells = [str(d) if lab.startswith("network / Lo") else "", lab]
        for col in ("K = 2", "K = 3", "K = 4"):
            x = ci(r[col])
            cells.append(f"{x[0]} [{x[1]}, {x[2]}]")
        lines.append(" & ".join(cells) + " \\\\")
    if d != DS[-1]:
        lines.append("\\midrule")
tabular("tab_app_k.tex", lines)

lines = []
for d in DS:
    t = find(RES, h2=f"d = {d}", h3="Benchmark I", first="points")
    for r in t["rows"]:
        g = "0.3" if r["points"].startswith("p = 0.3") else "0.1, 0.2"
        bp = re.match(rf"({NUM}) of ({NUM})", r["BP+LoMatching (matched shots)"]).groups()
        cells = [str(d) if g == "0.3" else "", g, num(r["shots"]), num(r["network"]), num(r["composition, Grace"]),
                 num(r["composition, Spark"]), num(r["LoMatching"]), num(r["Cain et al."]), f"{num(bp[0])} of {num(bp[1])}"]
        lines.append(" & ".join(cells) + " \\\\")
tabular("tab_app_counts.tex", lines)


# ------------------------------------------------------------------ Benchmark II
SYS = [("network alone", "network alone"),
       ("composition (Grace and Spark: both f = 10 %, uncapped)", "composition, both"),
       ("composition, Grace (f = 10 %, cap 0.02 s x L)", "composition, Grace"),
       ("composition, Spark (f = 10 %, cap 0.02 s x L)", "composition, Spark"),
       ("composition, Grace (f = 20 %, cap 0.1 s x L)", "composition, Grace"),
       ("composition, Spark (f = 2 %, uncapped)", "composition, Spark")]
lines, bench2 = [], {}
for d in (3, 5, 7):
    t = find(RES, h2=f"d = {d}", h3="Benchmark II", first="system")
    first = True
    for r in t["rows"]:
        lab = dict(SYS)[r["system"]]
        a = ci(r["(a) ratio (95 %)"])
        bench2[(d, lab)] = r
        head = [str(d) if first else "", lab]
        first = False
        if a is None:   # the row exists but its value has not landed
            lines.append(" & ".join(head) + f" & \\multicolumn{{6}}{{l}}{{{PH}{{late block: $d=7$ composition, Benchmark II}}}} \\\\")
            lines.append("\\\\[-4pt]")
            continue
        fe = r["(a) failures / expected"].split(" / ")
        b1, b2 = ci(r["(b) ours / MLE (95 %)"]), ci(r["(b) ratio (95 %)"])
        lom = ci(r["vs LoMatching (95 %)"])
        vn = ci(r["vs network (95 %)"])
        top = head + [f"{num(fe[0])} / {fe[1]}", a[0], b1[0], r["(b) theirs / MLE"], b2[0], lom[0], vn[0] if vn else "---"]
        bot = ["", "", "", two(a)[1], two(b1)[1], "", two(b2)[1], two(lom)[1], two(vn)[1] if vn else ""]
        lines.append(" & ".join(top) + " \\\\")
        lines.append(" & ".join(bot) + " \\\\")
    if d != 7:
        lines.append("\\midrule")
tabular("tab_first_benchmark2.tex", lines)

# ------------------------------------------------------------------ routed-fraction table (Benchmark II, Grace basis)
sw = find(RES, h2="Grace sweep (reported only)", first="d")
DEV = [c for c in sw["header"] if c.startswith("development: failures")][0]
lines, rows = [], []
for r in sw["rows"]:   # the frozen rows are in this table too, marked "(frozen ...)"
    s = r["system (Grace basis)"]
    m = re.match(r"f = ([0-9.]+) %, (uncapped|cap ([0-9.]+) s x L)", s)
    f, cap = float(m.group(1)), (m.group(3) or "none")
    dev = r[DEV].split(", ")
    rows.append({"d": int(r["d"]), "f": f, "cap": cap, "frozen": "(frozen" in s, "dev_fail": dev[0], "dev_us": dev[1], "row": r})
rows.sort(key=lambda x: (x["d"], x["f"], x["cap"] != "none", x["cap"]))
for d in (3, 5, 7):
    first = True
    for x in [x for x in rows if x["d"] == d]:
        r = x["row"]
        sp = FCHO[(2, d, "spark")]   # the Spark platform's operating point, when it is not the Grace one
        spark_pt = not x["frozen"] and x["f"] == float(sp["f"].split()[0]) and x["cap"] == sp["c (s x L)"]
        mark = "$^{\\ast}$" if x["frozen"] else ("$^{\\dagger}$" if spark_pt else "")
        f = f"{x['f']:g}\\,\\%{mark}"
        cap = "none" if x["cap"] == "none" else f"{x['cap']}\\,s"
        head = [str(d) if first else "", f, cap, num(x["dev_fail"]), x["dev_us"]]
        first = False
        a = ci(r["(a) ratio (95 %)"])
        if a is None:
            lines.append(" & ".join(head) + f" & \\multicolumn{{4}}{{l}}{{{PH}{{late block: $d=7$ test values}}}} \\\\")
            continue
        b1, b2, vn = ci(r["(b) ours / MLE (95 %)"]), ci(r["(b) ratio (95 %)"]), ci(r["vs network (95 %)"])
        routed = r.get("test shots: all / MLE prefix / routed", "").split(" / ")[-1] if not x["frozen"] else ""
        lines.append(" & ".join(head + [f"{a[0]} [{a[1]}, {a[2]}]", f"{b1[0]} [{b1[1]}, {b1[2]}]", f"{b2[0]} [{b2[1]}, {b2[2]}]",
                                        f"{vn[0]} [{vn[1]}, {vn[2]}]"]) + " \\\\")
    if d != 7:
        lines.append("\\midrule")
tabular("tab_sweep.tex", lines)

# ------------------------------------------------------------------ speed: choices, network alone, the bar
BAR = {int(r["d"]): r["bar"] for r in find(CHO, h2="Choices")["rows"]}


GPUDIR = {"rtxpro": "rtx_pro_6000", "gb10": "gb10"}


def per_layer(gpu, bench, d):
    """Equal-weight mean over the benchmark's points of the network's time per shot at B = 256 divided by L (us)."""
    t = json.loads(read(FIRST / "network_time_batch256" / GPUDIR[gpu] / f"benchmark{bench}_d{d}.json"))
    if bench == 1:   # 24 points: K = 2, 3, 4 x L = 1, 2, 4, 8, 12, 16 at p = 0.3 % and L = 8 at p = 0.1, 0.2 %
        pts = [(K, L) for K in (2, 3, 4) for L in (1, 2, 4, 8, 12, 16)] + [(K, 8) for K in (2, 3, 4) for _ in range(2)]
        return sum(t[f"d{d}_K{K}_L{L}"] / L for K, L in pts) / len(pts)
    return sum(t[f"benchmark2_d{d}_K4_L{2 * D}"] / (2 * D) for D in range(1, 11)) / 10


if not (OUT / "data" / "value_points.csv").exists():
    sys.exit("run fresh_main.py first (it writes value_points.csv)")
VPT = {(int(r["result"]), int(r["d"]), r["basis"]): r for r in csv.DictReader(open(OUT / "data" / "value_points.csv"))}
net = {}
for d in DS:
    for gpu in ("rtxpro", "gb10"):
        net[(1, d, gpu)] = per_layer(gpu, 1, d)
        if d != 9:
            net[(2, d, gpu)] = per_layer(gpu, 2, d)
lines, srows = [], []
for d in DS:
    cells, below = [str(d), BAR[d]], ["", ""]
    for res in (1, 2):
        for gpu in ("rtxpro", "gb10"):
            cells.append(f"{net[(res, d, gpu)]:.2f}" if (res, d, gpu) in net else "---")
            below.append("")
    srow = {"d": d, "bar": BAR[d]}
    for res in (1, 2):
        for basis in ("grace", "spark"):
            v = VPT.get((res, d, basis))
            if v is None:
                cells.append("---")
                below.append("")
                continue
            us = f"{float(v['test_us']):.1f}"
            cap = "none" if v["cap"] == "none" else f"{v['cap']}\\,s"
            cells.append(us)
            below.append(f"{{\\tiny {float(v['f']) * 100:g}\\,\\%, {cap}}}")
            srow[f"c{res}_{basis}"] = us
    for res in (1, 2):
        for gpu in ("rtxpro", "gb10"):
            if (res, d, gpu) in net:
                srow[f"n{res}_{gpu}"] = f"{net[(res, d, gpu)]:.3f}"
    srows.append(srow)
    lines.append(" & ".join(cells) + " \\\\")
    lines.append(" & ".join(below) + " \\\\")


def live_tables():
    """Every live table of fresh_test/live_*.md, keyed by host: 'rtx' (rows at N = 32, an RTX PRO 6000 with EPYC cores) or
    'spark' (rows at N <= 20, or a ## / ### heading or title naming the DGX Spark or GB10)."""
    out = {}
    for path in sorted(FRESH.glob("live_*.md")):
        text = read(path)
        title = next((ln for ln in text.splitlines() if ln.startswith("# ")), "")
        for t in md_tables(text):
            if t["header"][:5] != ["row", "result", "d", "f", "cap"]:
                continue
            ns = {int(m.group(1)) for r in t["rows"] for m in [re.search(r"\(N = ([0-9]+);", list(r.values())[-1])] if m}
            label = (t["h2"] + " " + t["h3"]).strip() or title
            host = ("rtx" if ns == {32} else "spark" if (ns and max(ns) <= 20) or re.search(r"spark|gb10", label, re.I)
                    else None)
            if host is None:
                print(f"  live table in {path.name} under '{label}' not recognised (N = {ns}); not used")
                continue
            for r in t["rows"]:
                cap = r["cap"] if r["cap"] == "none" else f"{float(r['cap']):g}"
                out[(host, int(r["result"]), int(r["d"]), round(float(r["f"]), 6), cap)] = r
    return out


LIVE = live_tables()
LCOL = "live in the basis / sim (N, same batches; points' range)"
LUS = "us per layer live (mean over points; worst point)"
for host in ("rtx", "spark"):   # the summary numbers that the text quotes (RQ4, Appendix E)
    rs = [r for k, r in LIVE.items() if k[0] == host]
    if not rs:
        print(f"  live {host}: no rows")
        continue
    med = lambda v: (v[(len(v) - 1) // 2] + v[len(v) // 2]) / 2
    q = sorted(float(m.group(1)) for r in rs for m in [re.match(rf"({NUM}) \(N", r[LCOL])] if m)
    rep = sorted(float(r["live / replay (this CPU)"]) for r in rs)
    print(f"  live {host}: {len(rs)} rows; live / sim (basis) median {med(q):.3f} (lower median {q[(len(q) - 1) // 2]:.3f}) "
          f"over {len(q)} rows with trials "
          f"({q[0]:.3f}-{q[-1]:.3f}); live / replay median {med(rep):.4f} ({rep[0]:.3f}-{rep[-1]:.3f}); answers changed "
          f"{sum(int(r['answers changed (to fail / to correct)'].split()[0]) for r in rs)}; rows routing differently "
          f"{sum('!=' in r['routed live = sim'] for r in rs)}; failures differ in "
          f"{sum('!=' in r['failures live = sim'] for r in rs)}")
tabular("tab_speed.tex", lines, SPEED_FOOT)
lines = []
for res, ds in ((1, DS), (2, (3, 5, 7))):
    if res == 2:
        lines.append("\\midrule")
    for d in ds:
        for k, lab in ((0, "main"), (1, "first test")):
            cells = [BENCH[res] if (d == ds[0] and k == 0) else "", str(d) if k == 0 else "", lab]
            for host, basis in (("rtx", "grace"), ("spark", "spark")):   # each live host runs that platform's points
                if k == 0:
                    v = VPT[(res, d, basis)]
                    f, cap = float(v["f"]), v["cap"]
                else:
                    c = FCHO[(res, d, basis)]
                    f, cap = float(c["f"].split()[0]) / 100, c["c (s x L)"]
                cap = cap if cap == "none" else f"{float(cap):g}"
                r = LIVE[(host, res, d, round(f, 6), cap)]
                us = r[LUS].split(";")[0]
                m = re.match(rf"({NUM}) \(N = ([0-9]+);", r[LCOL])
                fl = r["failures live = sim"].replace(" != ", " / ").replace(" = ", " / ")
                cells += [f"{f * 100:g}\\,\\%, " + ("none" if cap == "none" else f"{cap}\\,s"), us, m.group(1) if m else "---", fl]
            lines.append(" & ".join(cells) + " \\\\")
HEAD["tab_app_live.tex"] = ("@{}cclcrcrcrcr@{}", r"""
& & & \multicolumn{4}{c}{RTX PRO 6000, 32 EPYC 9555 cores, $N=32$} & \multicolumn{4}{c}{DGX Spark, 20 cores, $N=20$} \\
\cmidrule(lr){4-7}\cmidrule(lr){8-11}
benchmark & $d$ & point & $f$, cap & $\mu$s & live / sim & failures & $f$, cap & $\mu$s & live / sim & failures \\""")
tabular("tab_app_live.tex", lines)
with open(OUT / "data" / "speed.csv", "w", newline="") as f:
    keys = ["d", "bar", "n1_rtxpro", "n1_gb10", "n2_rtxpro", "n2_gb10", "c1_grace", "c1_spark", "c2_grace", "c2_spark"]
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(srows)
print("  network alone, us per layer at B = 256:", {f"r{k[0]} d{k[1]} {k[2]}": round(v, 3) for k, v in net.items()})

# the choices in full (appendix)
lines = []
for (res, d, basis), r in sorted(FCHO.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
    f, cap = fc(r)
    fail = r["failures / shots, development (network alone)"]
    m = re.match(rf"({NUM}) / ({NUM}) \(network alone ({NUM})\)", fail)
    us, p99 = re.match(rf"({NUM}) \(p99 ({NUM}) ms\)", r["us per layer, paced stream (batch p99)"]).groups()
    lines.append(" & ".join([BENCH[res], str(d), "Grace" if basis == "grace" else "Spark", f, cap, num(m.group(1)),
                             num(m.group(3)), num(m.group(2)), num(r["routed shots"]), us, num(p99),
                             r["qualifying"].replace(" / ", "/")]) + " \\\\")
tabular("tab_app_choices.tex", lines)

# ------------------------------------------------------------------ ablations at d = 5
params = {r["network"]: r["params"] for r in find(TRN, h2="Verified networks")["rows"]}
ARMS = [("abl_noexch", "no exchange between blocks"), ("abl_late", "exchange only at the read-out"),
        ("abl_pool", "pooled read-out"), ("abl_small", "small trunk (3 blocks)"),
        ("abl_k2", "specialist, $K=2$ only"), ("abl_k3", "specialist, $K=3$ only"), ("abl_k4", "specialist, $K=4$ only"),
        ("main_d5", "main network (fine-tuned)")]
sc = {r["arm"]: r for r in find(ABL, h2="Scores", first="arm", has="dev K = 2")["rows"]}
cols = ["dev K = 2", "dev K = 3", "dev K = 4", "dev pooled (18 points)", "dev p = 0.1 / 0.2 %", "Benchmark II pooled (D = 1-10)"]
lines = []
for arm, lab in ARMS:
    if arm == "main_d5":
        lines.append("\\midrule")
    top, bot = [lab, num(params[arm].split()[0])], ["", ""]
    for c in cols:
        x = ci_sq(sc[arm][c])
        top.append("---" if x is None else x[0])
        bot.append("" if x is None else two(x)[1])
    lines.append(" & ".join(top) + " \\\\")
    lines.append(" & ".join(bot) + " \\\\")
tabular("tab_abl.tex", lines)
print("  full arm parameters:", params["abl_full"])


# ------------------------------------------------------------------ appendix table heads
HEAD.update({
    "tab_app_pt_counts_a.tex": ("@{}ccccrrrrrrrrr@{}", r"""
& & & & & & & & & & \multicolumn{3}{c}{matched shots of \bplom} \\
\cmidrule(lr){11-13}
$d$ & $K$ & $L$ & $p$ (\%) & shots & routed & network & composition & LoMatching & Cain et al. & \bplom & network & composition \\"""),
    "tab_app_pt_ratios_a.tex": ("@{}cccclllll@{}", r"""
& & & & \multicolumn{2}{c}{against LoMatching} & \multicolumn{2}{c}{against \bplom} & composition \\
\cmidrule(lr){5-6}\cmidrule(lr){7-8}
$d$ & $K$ & $L$ & $p$ (\%) & network & composition & network & composition & / network \\"""),
    "tab_app_benchmark2_depth.tex": ("@{}ccrlrrrrrrrrl@{}", r"""
& & \multicolumn{5}{c}{every test shot, measure (a)} & \multicolumn{6}{c}{certified prefix, measure (b)} \\
\cmidrule(lr){3-7}\cmidrule(lr){8-13}
& & & & & \multicolumn{2}{c}{composition} & & & \multicolumn{2}{c}{composition} & exact & \\
\cmidrule(lr){6-7}\cmidrule(lr){10-11}
$d$ & $D$ & shots & their $p_D$ & network & Grace & Spark & shots & network & Grace & Spark & MLE & their $q_D$ \\"""),
})
HEAD["tab_app_pt_counts_b.tex"] = HEAD["tab_app_pt_counts_a.tex"]
HEAD["tab_app_pt_ratios_b.tex"] = HEAD["tab_app_pt_ratios_a.tex"]
I = lambda s: int(s.replace(",", ""))

# ------------------------------------------------------------------ appendix: Benchmark I per point, Benchmark II per depth
AT = md_tables(read(FIRST / "appendix_tables.md"))


def flagged(cell):
    v, lo, hi, fl = re.match(rf"\s*({NUM})\s*\(({NUM})-({NUM})\)\s*([zd]?)", cell).groups()
    return v + (f"$^{{\\mathrm{{{fl}}}}}$" if fl else "") + f" {{\\tiny[{lo}, {hi}]}}"


gsp = 0
for part, dd in (("a", (3, 5)), ("b", (7, 9))):
    cl, rl = [], []
    for d in dd:
        cnt = find(AT, h2="Benchmark I: per point", h3=f"d = {d}", first="K", has="shots")
        rat = find(AT, h2="Benchmark I: per point", h3=f"d = {d}", first="K", has="network / LoMatching")
        H = cnt["header"]
        cg = [c for c in H if c.startswith("comp.") and "(matched)" not in c][0]
        cs = [c for c in H if c.startswith("comp. Spark") and "(matched)" not in c]
        cm = [c for c in H if c.startswith("comp.") and c.endswith("(matched)") and "Spark" not in c.split("(")[0]][0]
        for i, r in enumerate(cnt["rows"]):
            assert r["matched shots"] == "64,000", r["matched shots"]
            if cs:
                gsp = max(gsp, abs(I(r[cg]) - I(r[cs[0]])))
            cl.append(" & ".join([str(d) if i == 0 else "", r["K"], r["L"], r["p (%)"]] + [num(r[c]) for c in
                      ("shots", "routed", "network", cg, "LoMatching", "Cain et al.", "BP+LoMatching", "network (matched)", cm)]) + " \\\\")
        RH = rat["header"]
        col = lambda s: [c for c in RH if c.startswith("comp.") and c.endswith(s) and "Spark" not in c][0]
        for i, r in enumerate(rat["rows"]):
            rl.append(" & ".join([str(d) if i == 0 else "", r["K"], r["L"], r["p (%)"]] + [flagged(r[c]) for c in
                      ("network / LoMatching", col("/ LoMatching"), "network / BP+LoMatching", col("/ BP+LoMatching"), col("/ network"))]) + " \\\\")
        if d == dd[0]:
            cl.append("\\midrule")
            rl.append("\\midrule")
    tabular(f"tab_app_pt_counts_{part}.tex", cl)
    tabular(f"tab_app_pt_ratios_{part}.tex", rl)
print("  Benchmark I per point: largest Grace-Spark failure gap:", gsp)


def dec(x):
    """'7.8e-05' -> '0.000078'; plain decimals unchanged."""
    return f"{float(x):.10f}".rstrip("0") if "e" in x else x


ml = []
for d in (3, 5, 7):
    ta = find(AT, h2="Benchmark II: per depth", h3=f"d = {d}", first="D", has="test shots n_D")
    tb = find(AT, h2="Benchmark II: per depth", h3=f"d = {d}", first="D", has="certified prefix shots (left out)")
    ca = [c for c in ta["header"] if c.startswith("comp.")]
    cb = [c for c in tb["header"] if c.startswith("comp.") and c.endswith(" x")]
    ca, cb = (ca if len(ca) == 2 else ca * 2), (cb if len(cb) == 2 else cb * 2)
    B = {r["D"]: r for r in tb["rows"]}
    for i, r in enumerate(ta["rows"]):
        if r["D"] == "all":   # the sum line: E = sum_D n_D p_D, (a)'s denominator
            row = ["", "all", num(r["test shots n_D"]), f"$E={r['expected n_D p_D']}$", num(r["network"]), num(r[ca[0]]), num(r[ca[1]])]
        else:
            p, pb = re.match(r"([0-9.e-]+) \(([0-9.e-]+)\)", r["their network p_D (bar)"]).groups()
            row = [str(d) if i == 0 else "", r["D"], num(r["test shots n_D"]), f"{dec(p)} {{\\tiny$\\pm${dec(pb)}}}",
                   num(r["network"]), num(r[ca[0]]), num(r[ca[1]])]
        b = B.get(r["D"])
        if r["D"] == "all" and "all" not in B:   # the prefix side has no sum line: add its depths
            pre = [x for k, x in B.items() if k != "all"]
            n = sum(I(re.match(r"([0-9,]+)", x["certified prefix shots (left out)"]).group(1)) for x in pre)
            lo = sum(int(re.search(r"\(([0-9]+)\)", x["certified prefix shots (left out)"]).group(1)) for x in pre)
            row += [num(str(n)) + (f" ({lo})" if lo else "")] + [num(str(sum(I(x[c]) for x in pre))) for c in
                    ("network x", cb[0], cb[1], "exact MLE m")] + [""]
        elif b:
            n, lo = re.match(r"([0-9,]+) \(([0-9]+)\)", b["certified prefix shots (left out)"]).groups()
            qm = re.match(r"([0-9.e-]+) \(([0-9.e-]+)\)(\*?)", b["their MLE q_D (bar)"])   # * after the bar: no bar in their figure
            qcell = "" if qm is None else f"{dec(qm.group(1))}{'$^{\\ast}$' if qm.group(3) else ''} {{\\tiny$\\pm${dec(qm.group(2))}}}"
            row += [num(n) + (f" ({lo})" if lo != "0" else ""), num(b["network x"]), num(b[cb[0]]), num(b[cb[1]]),
                    num(b["exact MLE m"]), qcell]
        else:
            row += ["---"] * 6
        ml.append(" & ".join(row) + " \\\\")
    if d != 7:
        ml.append("\\midrule")
tabular("tab_app_benchmark2_depth.tex", ml)
