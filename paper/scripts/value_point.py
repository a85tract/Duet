"""Value point per (benchmark, platform, d): the smallest routed fraction f whose DEVELOPMENT failures are within DELTA of
the most accurate development row on that platform. Prints its fresh-test row beside the f = 20 % test row. An independent
check of the picks that fresh_main.py makes (value_points.csv).
Usage: python3 value_point.py [DATA_DIR [delta]]      DATA_DIR defaults to paper/data/fresh_test (development_best.json and
d<d>/benchmark<1,2>/report.md)."""
import json, re, sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data" / "fresh_test"
dev = json.load(open(root / "development_best.json"))["rows"]
delta = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
BENCH = {1: "I", 2: "II"}

# Test tables: track the current section (d, benchmark, basis) and read rows "| f | cap | mark | ...".
test = {}
sections = [(int(d.name[1:]), int(b.name[-1]), b / "report.md") for d in sorted(root.glob("d[0-9]*")) for b in sorted(d.glob("benchmark[12]"))]
for d, res, path in sections:
    basis = None
    for ln in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"## Basis (\w+) \(N", ln)
        if m:
            basis = m.group(1)
        if ln.startswith("| 0") and basis:
            c = [x.strip() for x in ln.strip("|").split("|")]
            f = c[0]
            if f != "0" and "B" not in c[2]:   # report.md lists every cap of an f; the pick uses the development-best one (mark B)
                continue
            if res == 1:
                fails = int(c[3].split("/")[0])
                us = float(c[5].split()[0])
                vs_lom, vs_net = c[6].split()[0], c[9].split()[0]
                extra = ""
            else:
                fails = int(c[3])
                us = float(c[8].split()[0])
                vs_lom, vs_net = c[6].split()[0], c[7].split()[0]
                extra = f" (a) {c[4].split()[0]} (b) {c[5].split()[0]}"
            test[(res, basis, d, f)] = (c[1], fails, us, vs_lom, vs_net, extra)

g = defaultdict(list)
for r in dev:
    g[(r["result"], r["basis"], r["d"])].append(r)
for k in sorted(g):
    rs = sorted(g[k], key=lambda r: float(r["f"]))
    rs = [r for r in rs if r["f"] != "0"]
    best = min(r["fails"] for r in rs)
    pick = next(r for r in rs if r["fails"] <= best * (1 + delta))
    t = test.get((k[0], k[1], k[2], pick["f"]))
    t20 = test.get((k[0], k[1], k[2], "0.2"))
    bar = pick["bar"]
    def fmt(t):
        if not t:
            return "missing"
        return f"cap {t[0]} fails {t[1]} {t[2]:.1f}us vsLoM {t[3]} vsNet {t[4]}{t[5]}"
    print(f"Benchmark {BENCH[k[0]]} {k[1]:5s} d={k[2]} bar {bar}: pick f={float(pick['f'])*100:g}% "
          f"(dev {pick['fails']} vs best {best}, {pick['us_per_layer_stream']:.1f}us) | TEST {fmt(t)} | f=20% {fmt(t20)}")
