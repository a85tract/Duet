"""Tables of every routed fraction on the fresh test sets (Tables XVIII and XIX): tab_fresh_benchmark1.tex and
tab_fresh_benchmark2.tex.
  python3 paper/scripts/fresh_fractions.py [--out DIR] [--only tab_x.tex ...]      (run fresh_main.py first)
Reads data/fresh_test/d<d>/benchmark<1,2>/report.md and DIR/data/value_points.csv (written by fresh_main.py); DIR defaults
to paper/out. Rows: the network alone (f = 0) and every f in {0.5, 1, 2, 5, 10, 20 %} at the development-best cap of each
platform (mark B in report.md); a star marks each platform's first-test operating point (mark F). A frozen row whose cap
is not the development-best one gets a row of its own. A distance whose report.md is missing prints a red "pending" row.
The printout gives the values that the text quotes."""
import argparse
import csv
import re
from pathlib import Path

BENCH = {1: "I", 2: "II"}   # the paper's names of the two results: Benchmark I (general circuits), II (the Lukin group's)
PAPER = Path(__file__).resolve().parent.parent
_ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
_ap.add_argument("--out", default=str(PAPER / "out"), help="output directory (tables/ and data/ below it)")
_ap.add_argument("--only", nargs="*", default=None, help="table file names to write (default: all)")
ARGS = _ap.parse_args()
OUT = Path(ARGS.out)
(OUT / "tables").mkdir(parents=True, exist_ok=True)
FRESH = PAPER / "data" / "fresh_test"
FS = ["0.005", "0.01", "0.02", "0.05", "0.1", "0.2"]
DS = {1: (3, 5, 7, 9), 2: (3, 5, 7)}
PH = "\\placeholder"
NUM = r"[0-9][0-9,]*(?:\.[0-9]+)?"


def show(d, res):
    p = FRESH / f"d{d}" / f"benchmark{res}" / "report.md"
    return p.read_text() if p.exists() else None


def bases(text):
    """{basis: [row dicts]} of report.md's '## Basis <name> (N = ..)' tables."""
    out, basis, head = {}, None, None
    for ln in text.splitlines():
        if ln.startswith("## "):
            m = re.match(r"## Basis (\w+) \(N = ([0-9]+)\)", ln)
            basis, head = (m.group(1), None) if m else (None, None)
        elif basis and ln.startswith("| f |"):
            head = [c.strip() for c in ln.strip().strip("|").split("|")]
            out[basis] = []
        elif basis and head and ln.startswith("| ") and not ln.startswith("|---"):
            out[basis].append(dict(zip(head, [c.strip() for c in ln.strip().strip("|").split("|")])))
    return out


def ci(cell):
    """'0.9595 (0.9551-0.9640)' -> '0.9595 [0.9551, 0.9640]'; '-' -> '---'."""
    m = re.match(rf"({NUM}) \(({NUM})-({NUM})\)", cell)
    return f"{m.group(1)} [{m.group(2)}, {m.group(3)}]" if m else "---"


def num(s):
    """Integers with a thin-space thousands separator from 10,000 up, as make_data.py writes them."""
    t = s.replace(",", "")
    return f"{int(t):,}".replace(",", "\\,") if t.isdigit() and int(t) >= 10000 else t


def fails(r):
    return (r.get("failures / shots") or r["target failures"]).split(" / ")[0]


def pick(rows, f):
    """The rows of f to show: the development-best one (B) and the frozen one (F) when it is another row."""
    rs = [r for r in rows if r["f"] == f]
    if f == "0":
        assert len(rs) == 1, f
        return rs
    b = [r for r in rs if "B" in r["mark"]]
    fz = [r for r in rs if "F" in r["mark"] and "B" not in r["mark"]]
    assert len(b) == 1, (f, len(b))
    return b + fz


def pct(f):
    return f"{100 * float(f):g}\\,\\%"


VPS = {(int(r["result"]), r["basis"], int(r["d"]), r["f"], r["cap"])   # the operating points (fresh_main.py)
       for r in csv.DictReader(open(OUT / "data" / "value_points.csv"))}
LIVEBAD = {(2, "spark", 7, "0.2", "0.001")}   # live check: 28 failures against 11 simulated (live_dgx_spark.md)


def cap(r, res, d, basis):
    """The cap, marked: circle = operating point, star = the first test's operating point (mark F), dagger = LIVEBAD."""
    c = "---" if r["f"] == "0" else ("none" if r["cap"] == "none" else f"{r['cap']}\\,s")
    k = (res, basis, d, r["f"], r["cap"])
    m = ("\\circ" if k in VPS else "") + ("\\ast" if "F" in r["mark"] else "") + ("\\dagger" if k in LIVEBAD else "")
    return c + (f"$^{{{m}}}$" if m else "")


def us(r):
    return r["us per layer (p99 ms)"].split(" (")[0]


HEAD = {
    1: ("@{}cccrcccccrc@{}", r"""
& & \multicolumn{5}{c}{Grace platform} & & \multicolumn{3}{c}{Spark platform} \\
\cmidrule(lr){3-7}\cmidrule(lr){9-11}
$d$ & $f$ & cap & failures & vs network & vs \lom & $\mu$s & & cap & failures & $\mu$s \\"""),
    2: ("@{}cccrccccccrc@{}", r"""
& & \multicolumn{6}{c}{Grace platform} & & \multicolumn{3}{c}{Spark platform} \\
\cmidrule(lr){3-8}\cmidrule(lr){10-12}
$d$ & $f$ & cap & failures & measure (a) & vs network & vs \lom & $\mu$s & & cap & failures & $\mu$s \\"""),
}
RATIOS = {1: ["vs network", "vs LoMatching"], 2: ["(a) ratio (95 %)", "vs network", "vs LoMatching"]}


def table(res):
    spec, head = HEAD[res]
    ncol = spec.count("c") + spec.count("r")
    lines, done = [], []
    for d in DS[res]:
        text = show(d, res)
        if text is None:
            lines.append(f"{d} & \\multicolumn{{{ncol - 1}}}{{l}}{{{PH}{{pending}}}} \\\\")
            print(f"  Benchmark {BENCH[res]}, d = {d}: pending")
            continue
        done.append(d)
        b = bases(text)
        g, s = b["grace"], b["spark"]
        if res == 1:
            shots = {r["failures / shots"].split(" / ")[1] for rows in (g, s) for r in rows}
            assert len(shots) == 1, shots
            print(f"  Benchmark I, d = {d}: {shots.pop()} shots per row")
        first = True
        gap = 0
        for f in ["0"] + FS:
            gr, sr = pick(g, f), pick(s, f)
            for i in range(max(len(gr), len(sr))):
                a = gr[i] if i < len(gr) else None
                c = sr[i] if i < len(sr) else None
                cells = [str(d) if first else "", pct(f) if i == 0 else ""]
                first = False
                if a:
                    cells += [cap(a, res, d, "grace"), num(fails(a))] + [ci(a[k]) for k in RATIOS[res]] + [us(a)]
                else:
                    cells += [""] * (3 + len(RATIOS[res]))
                cells.append("")
                cells += [cap(c, res, d, "spark"), num(fails(c)), us(c)] if c else ["", "", ""]
                lines.append(" & ".join(cells) + " \\\\")
                if a and c:
                    gap = max(gap, abs(int(fails(a).replace(",", "")) - int(fails(c).replace(",", ""))))
        summary(res, d, g, s, gap)
        if d != DS[res][-1]:
            lines.append("\\midrule")
    name = f"tab_fresh_benchmark{res}.tex"
    if ARGS.only is not None and name not in ARGS.only:
        return
    (OUT / "tables" / name).write_text(
        f"% fresh_fractions.py from data/fresh_test/d<d>/benchmark{res}/report.md; scored d: "
        f"{', '.join(map(str, done)) or 'none'}\n\\begin{{tabular}}{{{spec}}}\n\\toprule\n{head.strip()}\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    print(f"tables/{name}")


def summary(res, d, g, s, gap):
    """The values that the text quotes, per basis: f = 0, every development-best row, the frozen row."""
    for name, rows in (("grace", g), ("spark", s)):
        sel = pick(rows, "0") + [r for f in FS for r in pick(rows, f)]
        def short(r):
            a = f", (a) {r['(a) ratio (95 %)'].split()[0]}" if res == 2 else ""
            return (f"f {float(r['f']) * 100:g}%{'*' if 'F' in r['mark'] else ''} cap {r['cap']}: {fails(r)} fails, "
                    f"vs net {r['vs network'].split()[0]}{a}, vs LoM {r['vs LoMatching'].split()[0]}, {us(r)} us")
        print(f"  Benchmark {BENCH[res]}, d = {d}, {name}:")
        for r in sel:
            print("    " + short(r))
    print(f"  Benchmark {BENCH[res]}, d = {d}: largest Grace - Spark failure gap within a row: {gap}")


for res in (1, 2):
    table(res)
