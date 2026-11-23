"""The two data-driven figures of the paper (Fig. 3: fig_benchmark1.pdf, Fig. 4: fig_speed.pdf).
  python3 paper/scripts/figures.py [--data DIR] [--out DIR] [--only fig_x.pdf ...]
Reads benchmark1_ratios.csv (fresh_main.py) and speed.csv (make_data.py) from --data (default paper/out/data) and writes
the PDFs to --out (default paper/out/figures).
Both figures are drawn at the printed column width (246 pt) with Computer Modern at 8 pt, in the colours and line weights
of the TikZ figures (paper/figures/figstyle.tex): blue the network, rose the composition with Tesseract, black the
baseline."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, LogLocator, NullFormatter

PAPER = Path(__file__).resolve().parent.parent
_ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
_ap.add_argument("--data", default=str(PAPER / "out" / "data"))
_ap.add_argument("--out", default=str(PAPER / "out" / "figures"))
_ap.add_argument("--only", nargs="*", default=None, help="figure file names to write (default: both)")
ARGS = _ap.parse_args()
DATA, OUT = ARGS.data, ARGS.out
Path(OUT).mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False, "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "pdf.fonttype": 42,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 3.6, "lines.markeredgewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0, "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
    "xtick.major.pad": 3.0, "ytick.major.pad": 3.0,
    "legend.frameon": False, "legend.handlelength": 2.2, "legend.handletextpad": 0.6, "legend.labelspacing": 0.3,
    "legend.borderaxespad": 0.5,
})
# one meaning per colour in all four figures (Paul Tol's high-contrast scheme, as in figures/figstyle.tex)
NET, COMP, BASE, REF = "#004488", "#BB5566", "#000000", "#666666"
W, H = 246 / 72, 2.35          # printed column width and height in inches
DS = [3, 5, 7, 9]


def rd(name):
    with open(f"{DATA}/{name}") as f:
        return list(csv.DictReader(f))


def style(ax, ticks):
    """Distance axis, and a log y axis labelled at the given values with minor ticks at every integer multiple."""
    ax.set_xticks(DS)
    ax.set_xlim(2.5, 9.5)
    ax.set_xlabel("code distance $d$")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(1, 10), numticks=20))
    ax.yaxis.set_minor_formatter(NullFormatter())


def save(fig, name):
    if ARGS.only is None or name in ARGS.only:
        fig.subplots_adjust(left=0.135, right=0.975, bottom=0.165, top=0.975)
        fig.savefig(f"{OUT}/{name}", metadata={"CreationDate": None})   # no timestamp: the same data gives the same PDF
        print(f"figures/{name}")


# ---------------------------------------------------------------- Benchmark I: ratios against d
M = rd("benchmark1_ratios.csv")
fig, ax = plt.subplots(figsize=(W, H))
series = [("net_lom", "network / LoMatching", NET, "-", "o", True, -0.12),
          ("comp_lom", "composition / LoMatching", COMP, "-", "s", True, -0.04),
          ("net_bp", "network / BP+LoMatching", NET, "--", "o", False, 0.04),
          ("comp_bp", "composition / BP+LoMatching", COMP, "--", "s", False, 0.12)]
for key, lab, col, ls, mk, filled, dx in series:
    rows = sorted([r for r in M if r["series"] == key and r["group"] == "p3"], key=lambda r: int(r["d"]))
    x = [int(r["d"]) + dx for r in rows]
    y = [float(r["r"]) for r in rows]
    lo = [float(r["r"]) - float(r["lo"]) for r in rows]
    hi = [float(r["hi"]) - float(r["r"]) for r in rows]
    ax.errorbar(x, y, yerr=[lo, hi], color=col, ls=ls, marker=mk, mfc=col if filled else "white", mec=col,
                capsize=1.5, elinewidth=0.6, capthick=0.6, label=lab, zorder=3)
ax.axhline(1, color=REF, lw=0.6, ls=(0, (1, 2)), zorder=1)
ax.text(9.4, 0.93, "equal failures", ha="right", va="top", color=REF)
style(ax, [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1])
ax.set_ylim(0.006, 1.5)
ax.set_ylabel("failure ratio, $p=0.3\\,\\%$")
ax.legend(loc="lower left")
save(fig, "fig_benchmark1.pdf")

# ---------------------------------------------------------------- speed: time per layer against d
S = sorted(rd("speed.csv"), key=lambda r: int(r["d"]))
fig, ax = plt.subplots(figsize=(W, H))
x = [int(r["d"]) for r in S]
ax.plot(x, [float(r["bar"]) for r in S], color=BASE, ls="-", marker="x", ms=4.2, label="Lukin group's network, H100",
        zorder=3)
for key, lab, col, ls, mk, filled in [("c1_grace", "composition, Grace platform", COMP, "-", "s", True),
                                      ("c1_spark", "composition, Spark platform", COMP, "--", "s", False),
                                      ("n1_rtxpro", "network alone, RTX PRO 6000", NET, "-", "o", True),
                                      ("n1_gb10", "network alone, GB10", NET, "--", "o", False)]:
    ax.plot(x, [float(r[key]) for r in S], color=col, ls=ls, marker=mk, mfc=col if filled else "white", mec=col,
            label=lab, zorder=3)
style(ax, [1, 10, 100, 1000])
ax.set_ylim(0.2, 5000)
ax.set_ylabel("$\\mu$s per layer, batch 256")
# two legends in the two empty corners: the baseline and the composition above the data, the network below it
h, lab = ax.get_legend_handles_labels()
top = ax.legend(h[:3], lab[:3], loc="upper left")
ax.add_artist(top)
ax.legend(h[3:], lab[3:], loc="lower right")
save(fig, "fig_speed.pdf")
