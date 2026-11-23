# Tables and figures of "Neural Decoding of Logical Clifford Circuits with Transversal Gates" (Duet)

Every table and figure of the paper that comes from data is regenerated from the files in `data/` with one command.
No network, no git and no other repository are needed. Table and figure numbers are the paper's.

    python paper/reproduce.py --list                 all items with script and data
    python paper/reproduce.py table 3 --check        Table III, then OK or DIFF against paper/reference
    python paper/reproduce.py figure 4 --check
    python paper/reproduce.py all --check            everything (under half a minute)
    python paper/reproduce.py value-points           independent check of the operating points

Roman numerals work too (`table III`). Output goes to `paper/out/{tables,data,figures}` (`--out DIR` to change).
`--check` compares each regenerated table byte by byte with the paper's own table in `reference/tables/`, and the
CSVs with `reference/data/`. The first line of a generated table is a comment naming the script and data directory; the
tables of the paper differ from the generated ones only there, and `reference/tables/` carries the generated comment.
Setup: Python 3.12 and `pip install -r paper/requirements.txt` (matplotlib 3.11.2 and numpy 2.5.4, needed only for
Figures 3 and 4). The figure PDFs cannot be compared byte by byte; `--check` compares their data CSVs.

## Commands

| item | file in out/ | script | caption begins | source data (in data/) |
|---|---|---|---|---|
| Figure 1 | figures/fig_circuit.tex | static | The circuits and how a CNOT spreads an error. | figures/fig_circuit.tex (hand-written TikZ; needs figures/figstyle.tex) |
| Figure 2 | figures/fig_network.tex | static | The network and the composition. | figures/fig_network.tex (hand-written TikZ; needs figures/figstyle.tex) |
| Table I | tables/tab_benchmark1.tex | fresh_main.py | Benchmark I on the fresh test set at the operating points | fresh_test/development_best.json, d*/benchmark1/{report.md,grid.json.gz} |
| Figure 3 | figures/fig_benchmark1.pdf | figures.py | Benchmark I at p=0.3 % on the fresh test set | data/benchmark1_ratios.csv (from fresh_test/d*/benchmark1/grid.json.gz) |
| Table II | tables/tab_benchmark2.tex | fresh_main.py | Benchmark II on the fresh test set at the operating points | fresh_test/development_best.json, d*/benchmark2/{report.md,mle_measures.json.gz} |
| Table III | tables/tab_routing.tex | fresh_main.py | Where the failures go at the operating points | fresh_test/development_best.json, d*/benchmark{1,2}/{report.md,grid.json.gz} |
| Figure 4 | figures/fig_speed.pdf | figures.py | Time per layer at batch 256 against the distance. | data/speed.csv (from first_test/choices.md, network_time_batch256/, value points) |
| Table IV | tables/tab_speed.tex | make_data.py | Time per layer in microseconds at batch 256. | first_test/choices.md, network_time_batch256/*, fresh_test (value points, live_*.md) |
| Table V | tables/tab_abl.tex | make_data.py | Variants of the network at d=5, each trained from scratch | training/ablations.md, training/networks.md |
| Table VI | tables/tab_starts.tex | static | The start of each network and the training that preceded it | tables_static/tab_starts.tex (typed training settings) |
| Table VII | tables/tab_app_benchmark2_depth.tex | make_data.py | The first test, Benchmark II per depth | first_test/appendix_tables.md |
| Table VIII | tables/tab_first_benchmark1.tex | make_data.py | The first test, Benchmark I: failures of the network alone | first_test/results.md, choices.md |
| Table IX | tables/tab_first_benchmark2.tex | make_data.py | The first test, Benchmark II, at p=0.3 % | first_test/results.md, choices.md |
| Table X | tables/tab_sweep.tex | make_data.py | The first test: Benchmark II on the Grace platform at several routed fractions | first_test/results.md, choices.md |
| Table XI | tables/tab_app_counts.tex | make_data.py | The first test: failures on the test shots of Benchmark I. | first_test/results.md |
| Table XII | tables/tab_app_k.tex | make_data.py | The first test: pooled ratios of Benchmark I per block count | first_test/results.md |
| Table XIII | tables/tab_app_pt_counts_a.tex | make_data.py | The first test, Benchmark I per point at d=3 and 5: failures | first_test/appendix_tables.md |
| Table XIV | tables/tab_app_pt_counts_b.tex | make_data.py | The first test, Benchmark I per point at d=7 and 9: failures | first_test/appendix_tables.md |
| Table XV | tables/tab_app_pt_ratios_a.tex | make_data.py | The first test, Benchmark I per point at d=3 and 5: paired failure ratios | first_test/appendix_tables.md |
| Table XVI | tables/tab_app_pt_ratios_b.tex | make_data.py | The first test, Benchmark I per point at d=7 and 9: paired failure ratios | first_test/appendix_tables.md |
| Table XVII | tables/tab_app_value.tex | fresh_main.py | The operating points: for each benchmark, distance and platform | fresh_test/development_best.json, d*/benchmark*/report.md |
| Table XVIII | tables/tab_fresh_benchmark1.tex | fresh_fractions.py | Every routed fraction on the fresh test set, Benchmark I | fresh_test/d*/benchmark1/report.md |
| Table XIX | tables/tab_fresh_benchmark2.tex | fresh_fractions.py | Every routed fraction on the fresh test set, Benchmark II | fresh_test/d*/benchmark2/report.md |
| Table XX | tables/tab_app_live.tex | make_data.py | Live paced stream on development shots at the operating points | fresh_test/live_*.md, first_test/choices.md |
| Table XXI | tables/tab_app_choices.tex | make_data.py | The first test's operating points. Development failures of the composition | first_test/choices.md |

Command numbers are the table numbers above (`table 21` is Table XXI). Benchmark I is the general random Clifford
circuits, Benchmark II the Lukin group's circuits.

## What is not generated from data

- Figures 1 and 2 are hand-written TikZ pictures (`figures/fig_circuit.tex`, `figures/fig_network.tex`). They share one
  look, defined in `figures/figstyle.tex` (colours, line weights, box and arrow styles), which has to sit next to them.
  Both are `tikzpicture` fragments for a `figure*` environment at the width of the page. In your own document put this in
  the preamble and `\input{fig_circuit}` inside `figure*`:

      \usepackage{tikz}
      \usetikzlibrary{arrows.meta,decorations.pathreplacing,positioning,calc}
      \input{figstyle}
      \newcommand{\tess}{Tesseract}

  To see them on their own, run `pdflatex figures.tex` in `figures/` (or in `out/figures/`): `figures.tex` is a
  `standalone` document with that preamble and one page per figure. `reproduce.py figure 1` (or 2) copies the four
  `.tex` files to `out/figures/`. Figures 3 and 4 (`scripts/figures.py`) repeat the colours and
  line weights of `figstyle.tex` in matplotlib.
- Table VI lists the training settings of the networks (sample counts, block counts). It is typed, with no result file
  behind it; `tables_static/tab_starts.tex` holds it.

## Scripts

- `scripts/fresh_main.py`: Tables I, II, III, XVII; `out/data/value_points.csv` and `out/data/benchmark1_ratios.csv`.
- `scripts/make_data.py`: Tables IV, V, VII-XII, XX, XXI; `out/data/speed.csv` (needs `value_points.csv`).
- `scripts/fresh_fractions.py`: Tables XVIII, XIX (needs `value_points.csv`).
- `scripts/figures.py`: Figures 3 and 4 from the two CSVs.
- `scripts/value_point.py`: prints each operating point with its fresh-test row, as a second check of `fresh_main.py`.

Each script also runs alone, for example `python paper/scripts/make_data.py --out /tmp/x --only tab_speed.tex`, and prints
the values that the text quotes. `reproduce.py` runs `fresh_main.py` first where a CSV is needed.
