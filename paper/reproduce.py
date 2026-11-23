#!/usr/bin/env python3
"""Regenerate the tables and figures of the paper from the data in paper/data, one command per item.

  python paper/reproduce.py --list                  every table and figure, its script and its data
  python paper/reproduce.py table 3 [--check]       Table III (the paper's numbering; Roman numerals work too)
  python paper/reproduce.py figure 4 [--check]      Figure 4
  python paper/reproduce.py all [--check]           everything
  python paper/reproduce.py check                   the same as: all --check
  python paper/reproduce.py value-points            the independent check of the operating points

Output goes to paper/out/{tables,data,figures} (--out DIR to change). --check compares each regenerated file with the
copy of the paper's in paper/reference and prints OK or DIFF per item. Python 3 standard library only, except the two
data-driven figures, which need matplotlib (see requirements.txt)."""
import argparse
import difflib
import subprocess
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
SCRIPTS = PAPER / "scripts"
ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII", 9: "IX", 10: "X", 11: "XI", 12: "XII",
         13: "XIII", 14: "XIV", 15: "XV", 16: "XVI", 17: "XVII", 18: "XVIII", 19: "XIX", 20: "XX", 21: "XXI"}

# (kind, number, file, script, first words of the caption, data files)
FRESH, FIRST, TRAIN = "fresh_test/", "first_test/", "training/"
ITEMS = [
    ("figure", 1, "fig_circuit.tex", None, "The circuits and how a CNOT spreads an error.", "figures/fig_circuit.tex (static TikZ)"),
    ("figure", 2, "fig_network.tex", None, "The network and the composition.", "figures/fig_network.tex (static TikZ)"),
    ("table", 1, "tab_benchmark1.tex", "fresh_main.py", "Benchmark I on the fresh test set at the operating points",
     FRESH + "development_best.json, d3/d5/d7/d9/benchmark1/{report.md,grid.json.gz}"),
    ("figure", 3, "fig_benchmark1.pdf", "figures.py", "Benchmark I at p=0.3 % on the fresh test set",
     "out/data/benchmark1_ratios.csv, from " + FRESH + "d*/benchmark1/grid.json.gz"),
    ("table", 2, "tab_benchmark2.tex", "fresh_main.py", "Benchmark II on the fresh test set at the operating points",
     FRESH + "development_best.json, d3/d5/d7/benchmark2/{report.md,mle_measures.json.gz}"),
    ("table", 3, "tab_routing.tex", "fresh_main.py", "Where the failures go at the operating points",
     FRESH + "development_best.json, d*/benchmark{1,2}/{report.md,grid.json.gz}"),
    ("figure", 4, "fig_speed.pdf", "figures.py", "Time per layer at batch 256 against the distance.",
     "out/data/speed.csv, from " + FIRST + "{choices.md,network_time_batch256/}, " + FRESH + "(value points)"),
    ("table", 4, "tab_speed.tex", "make_data.py", "Time per layer in microseconds at batch 256.",
     FIRST + "choices.md, " + FIRST + "network_time_batch256/*/benchmark{1,2}_d*.json, " + FRESH + "(value points, live_*.md)"),
    ("table", 5, "tab_abl.tex", "make_data.py", "Variants of the network at d=5, each trained from scratch",
     TRAIN + "ablations.md, " + TRAIN + "networks.md"),
    ("table", 6, "tab_starts.tex", None, "The start of each network and the training that preceded it",
     "tables_static/tab_starts.tex (typed training settings, no data file)"),
    ("table", 7, "tab_app_benchmark2_depth.tex", "make_data.py", "The first test, Benchmark II per depth: the inputs of measures (a) and (b).",
     FIRST + "appendix_tables.md"),
    ("table", 8, "tab_first_benchmark1.tex", "make_data.py", "The first test, Benchmark I: failures of the network alone",
     FIRST + "results.md, " + FIRST + "choices.md"),
    ("table", 9, "tab_first_benchmark2.tex", "make_data.py", "The first test, Benchmark II, at p=0.3 %, as Table II",
     FIRST + "results.md, " + FIRST + "choices.md"),
    ("table", 10, "tab_sweep.tex", "make_data.py", "The first test: Benchmark II on the Grace platform at several routed fractions",
     FIRST + "results.md (Grace sweep), " + FIRST + "choices.md"),
    ("table", 11, "tab_app_counts.tex", "make_data.py", "The first test: failures on the test shots of Benchmark I.", FIRST + "results.md"),
    ("table", 12, "tab_app_k.tex", "make_data.py", "The first test: pooled ratios of Benchmark I at p=0.3 % per block count",
     FIRST + "results.md"),
    ("table", 13, "tab_app_pt_counts_a.tex", "make_data.py", "The first test, Benchmark I per point at d=3 and 5: failures",
     FIRST + "appendix_tables.md"),
    ("table", 14, "tab_app_pt_counts_b.tex", "make_data.py", "The first test, Benchmark I per point at d=7 and 9: failures",
     FIRST + "appendix_tables.md"),
    ("table", 15, "tab_app_pt_ratios_a.tex", "make_data.py", "The first test, Benchmark I per point at d=3 and 5: paired failure ratios",
     FIRST + "appendix_tables.md"),
    ("table", 16, "tab_app_pt_ratios_b.tex", "make_data.py", "The first test, Benchmark I per point at d=7 and 9: paired failure ratios",
     FIRST + "appendix_tables.md"),
    ("table", 17, "tab_app_value.tex", "fresh_main.py", "The operating points: for each benchmark, distance and platform",
     FRESH + "development_best.json, d*/benchmark{1,2}/report.md"),
    ("table", 18, "tab_fresh_benchmark1.tex", "fresh_fractions.py", "Every routed fraction on the fresh test set, Benchmark I",
     FRESH + "d3/d5/d7/d9/benchmark1/report.md"),
    ("table", 19, "tab_fresh_benchmark2.tex", "fresh_fractions.py", "Every routed fraction on the fresh test set, Benchmark II",
     FRESH + "d3/d5/d7/benchmark2/report.md"),
    ("table", 20, "tab_app_live.tex", "make_data.py", "Live paced stream on development shots at the operating points",
     FRESH + "live_rtx_pro_6000.md, " + FRESH + "live_dgx_spark.md, " + FIRST + "choices.md"),
    ("table", 21, "tab_app_choices.tex", "make_data.py", "The first test's operating points. Development failures of the composition",
     FIRST + "choices.md"),
]
CSV = [("benchmark1_ratios.csv", "fresh_main.py"), ("value_points.csv", "fresh_main.py"), ("speed.csv", "make_data.py")]


def sub(kind):
    return "tables" if kind == "table" else "figures"


def run(script, out, *extra):
    """One of the scripts, with --out (figures.py: --data and --out); the printout is kept off the screen unless it fails."""
    cmd = [sys.executable, str(SCRIPTS / script)]
    if script == "figures.py":
        cmd += ["--data", str(out / "data"), "--out", str(out / "figures")]
    else:
        cmd += ["--out", str(out)]
    cmd += list(extra)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout + r.stderr, file=sys.stderr)
        sys.exit(f"{script} failed")
    return r.stdout


def make(item, out):
    kind, num, name, script, _, _ = item
    (out / sub(kind)).mkdir(parents=True, exist_ok=True)
    if script is None:   # static: TikZ figures and the typed Table VI
        d = PAPER / ("figures" if kind == "figure" else "tables_static")
        # the TikZ sources come with figstyle.tex (they need it) and figures.tex (compiles Figures 1 and 2 on their own)
        for f in sorted(d.glob("*.tex")) if kind == "figure" else [d / name]:
            (out / sub(kind) / f.name).write_text(f.read_text())
        return
    if script != "fresh_main.py":
        run("fresh_main.py", out, "--only")           # value_points.csv (and benchmark1_ratios.csv), no tables
    if script == "figures.py":
        run("make_data.py", out, "--only")            # speed.csv (figures.py reads both CSVs)
        run("figures.py", out, "--only", name)
    else:
        run(script, out, "--only", name)


def compare(got, ref):
    """OK when the two files are byte-identical; otherwise DIFF and the first differing lines."""
    if not got.exists():
        return "DIFF (not written)"
    a, b = got.read_text(), ref.read_text()
    if a == b:
        return "OK"
    d = list(difflib.unified_diff(b.splitlines(), a.splitlines(), "reference", "regenerated", lineterm="", n=0))
    return "DIFF\n    " + "\n    ".join(x[:160] for x in d[:8])


def check(item, out):
    kind, num, name, script, _, _ = item
    got = out / sub(kind) / name
    if name.endswith(".pdf"):   # a PDF is not compared byte by byte: its data CSV is, and the PDF has to exist
        csv = "benchmark1_ratios.csv" if "benchmark1" in name else "speed.csv"
        r = compare(out / "data" / csv, PAPER / "reference" / "data" / csv)
        ok = got.exists() and got.stat().st_size > 1000
        return ("OK" if r == "OK" and ok else "DIFF") + f" (plotted data {csv}: {r.splitlines()[0]}; PDF {'written' if ok else 'missing'})"
    ref = PAPER / "reference" / "tables" / name if kind == "table" else PAPER / "figures" / name
    return compare(got, ref) + (" (static source, copied)" if script is None else "")


def pick(kind, n):
    key = ROMAN_INV.get(n.upper(), None) if not n.isdigit() else int(n)
    hits = [i for i in ITEMS if i[0] == kind and i[1] == key]
    if not hits:
        sys.exit(f"no {kind} {n}; see --list")
    return hits[0]


ROMAN_INV = {v: k for k, v in ROMAN.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list every table and figure with its script and data")
    ap.add_argument("--out", default=str(PAPER / "out"), help="output directory (default paper/out)")
    ap.add_argument("--check", action="store_true", help="compare with paper/reference and print OK or DIFF")
    ap.add_argument("what", nargs="?", choices=["table", "figure", "all", "check", "value-points"])
    ap.add_argument("number", nargs="?", help="the table or figure number as in the paper (3 or III)")
    a = ap.parse_args()
    out = Path(a.out)
    if a.list or not a.what:
        for kind, num, name, script, cap, src in sorted(ITEMS, key=lambda i: (i[0] != "table", i[1])):
            label = f"{kind} {num}" if kind == "figure" else f"table {num} ({ROMAN[num]})"
            print(f"{label:16s} {name:32s} {(script or 'static'):19s} {cap}\n{'':16s} data: {src}")
        return
    if a.what == "value-points":
        r = subprocess.run([sys.executable, str(SCRIPTS / "value_point.py")], capture_output=True, text=True)
        ref = (PAPER / "reference" / "value_point_output.txt").read_text()
        print("value-points", "OK" if r.stdout == ref else "DIFF", "(the picks of value_point.py against the reference printout)")
        return
    if a.what in ("all", "check"):
        items, a.check = sorted(ITEMS, key=lambda i: (i[0] != "table", i[1])), a.check or a.what == "check"
    else:
        if not a.number:
            sys.exit(f"{a.what} needs a number")
        items = [pick(a.what, a.number)]
    bad = 0
    for item in items:
        make(item, out)
        kind, num, name = item[:3]
        label = f"{kind} {num}"
        if a.check:
            r = check(item, out)
            bad += not r.startswith("OK")
            print(f"{label:10s} {name:32s} {r}")
        else:
            print(f"{label:10s} {out / sub(kind) / name}")
    if a.check and a.what in ("all", "check"):
        for name, script in CSV:
            r = compare(out / "data" / name, PAPER / "reference" / "data" / name)
            bad += r != "OK"
            print(f"{'data':10s} {name:32s} {r}")
    if a.check:
        print("all OK" if not bad else f"{bad} item(s) differ")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
