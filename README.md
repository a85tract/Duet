# Duet: neural decoding of logical Clifford circuits with transversal gates

Code and data for the paper "Neural Decoding of Logical Clifford Circuits with Transversal Gates".

Duet is a neural decoder for circuits of rotated surface-code blocks joined by transversal Clifford gates. Its network
puts the exact structure of such circuits into its design: a transversal gate moves an error to a fixed place, and a
logical outcome propagates through Clifford gates linearly, so the network learns only which errors occurred, and one
network per code distance decodes two to four blocks. Alone it fails 0.15 to 0.56 times as often as LoMatching at
distances 3 to 9 on random circuits (Benchmark I), and with the search decoder Tesseract re-decoding its least confident
shots, 0.023 to 0.55 times as often; on the circuits of the Lukin group's neural decoder (Benchmark II) it fails less often
than their network and is statistically compatible with an exact most-likely-error decoder. The network takes 1.2 to 5.1
microseconds per layer on one GPU, and the network plus Tesseract stays below the Lukin group's time per gate at every
distance.

## What is here

| path | content |
|---|---|
| `duet/` | the Python package: circuits and noise, the network, the training loop, the inference engine, the decoders |
| `scripts/` | command-line drivers: sample test sets, decode with each decoder, route to Tesseract, time the paced stream, train |
| `paper/` | scripts, data and reference output that regenerate every table and figure of the paper (`paper/README.md`) |
| `data/` | the circuits of Benchmark II (Lukin family of the open benchmark), the published Fig. 1b values, the Benchmark II manifests (`data/README.md`) |
| `networks/` | the trained network files of the paper's results with their checksums (`networks/NETWORKS.md`) |
| `tests/` | unit and smoke tests (CPU; the engine tests need a GPU and skip without one) |
| `third_party/` | code and data adapted from other projects, with their licenses |

Package map:

- `duet/circuits.py`, `duet/clifford_circuits.py`: stim circuits of K rotated surface-code blocks under H, S, CNOT and
  identity layers, with the Lukin group's noise model; `duet/kblock_sets.py`, `duet/bench_sets.py`, `duet/lukin_bench.py`: the
  circuit families of Benchmarks I and II and the seeds of their test sets.
- `duet/net/`: `model_cl.py` (the network, `NetCL`), `spec_cl.py` (per-circuit tables), `train_cl.py` (training),
  `engine_cl.py` (Triton inference engine), `fast_cl.py` (eager fast path).
- `duet/decoders/`: `tesseract_dec.py` (long-beam Tesseract), `lomatching_dec.py` (LoMatching and BP+LoMatching),
  `cain2025.py` (our reimplementation of Cain et al.); `duet/mle.py`: the exact most-likely-error decoder (SCIP).

## Install

Python 3.12 on Linux. The versions below are the ones that produced the paper's results.

    python3.12 -m venv .venv && . .venv/bin/activate
    pip install -r requirements.txt
    pip install -e .

`requirements.txt` pins numpy 2.5.4, scipy 1.18.1, stim 1.16.0, pymatching 2.4.0, lomatching 0.3.0, ldpc 2.4.1,
tesseract-decoder 0.1.1.dev20260822020007, torch 2.11.0, triton 3.6.0, pyscipopt 6.3.0, pytest 9.2.0 and
matplotlib 3.11.2. Training and the inference engine need an NVIDIA GPU; Triton also needs a C compiler and the Python
headers (`python3-dev`). Everything else runs on CPU.

    pytest                                   # about two minutes; the engine tests skip without a GPU
    bash scripts/quickstart.sh               # a tiny run of the whole pipeline at d = 3

## Reproduce the paper's tables and figures

    python paper/reproduce.py --list                 # every table and figure with its script and data
    python paper/reproduce.py table 3 --check        # Table III, then OK or DIFF against the paper's table
    python paper/reproduce.py figure 4 --check
    python paper/reproduce.py all --check            # everything, under half a minute

The tables and the data of the figures come from the result files in `paper/data/` and are checked byte for byte against
the paper's own tables in `paper/reference/`. Figures 1 and 2 are hand-written TikZ pictures and ship as sources; Table VI
(the training schedule of the starts) is typed in the paper. Details and the command for each table and figure are in
`paper/README.md`.

## Train

`scripts/train.sh NAME ARGS...` runs `python -m duet.net.train_cl --out runs/NAME ARGS...` and resumes exactly when run
again after an interruption. The main networks start from the start networks (`--init`) and train on 5 x 10^7 samples
with the settings below; only `--d`, `--seed`, `--init` and `--accum` change with the distance (d = 3, 5, 7, 9: seeds
9103, 9105, 9107, 9109, `--accum` 1, 2, 4, 8, which only splits each batch of 1,024 samples to save memory).

    bash scripts/train.sh main_d9 --d 9 --blocks 7 --samples 5e7 --accum 8 --seed 9109 --init networks/start_d9.pt \
      --noise lukin --cnot-share 0.25 --ent-share 0.1 --p-cnot-set 0,0.3333333,0.6666667,1 --block-counts 2,2,3,4 \
      --lukin-share 0.5 --p-tri 0.2 --lr 3e-4 --warmup 500 --lookup gather --prefetch 64 --compile --ckpt-minutes 2 \
      --eval-circuits 100 --eval-cnot 100 --eval-per 200 --eval-every 5e6

At d = 3, `--init networks/start_d3.pt` has 3 residual blocks and `--blocks 7` grows it to 7 (`scripts/check_grown_start.py`
checks that the grown network computes the start's function). The ablation variants of Table V train from scratch at
d = 5 on 10^8 samples (`--seed 9205 --accum 2 --lr 1e-3 --warmup 1000`, no `--init`, the mix and noise flags above) with `--couple none`, `--couple late`,
`--head pool`, `--blocks 3`, or `--block-counts 2` / `3` / `4` (with `--lukin-share 0` for K = 2, 3). Each run writes
`final_ema.pt`, the file that `networks/NETWORKS.md` lists. `scripts/verify_checkpoints.py` checks a finished run.
`train_cl` and `NetCL` also keep a few options from earlier experiments (a single-layer path, replay pools, a LoMatching
teacher loss). They are off by default and no paper network uses them.

## Decode

All drivers take a manifest of points (`d`, block count, layers, noise strength, circuits, shots, seeds) and a directory of
stored shots. The commands in order, for Benchmark I (see the header of each script for every option):

    python scripts/eval_pipeline.py manifest --purpose dev|test|bench2 --out M.json   # the points and their seeds
    python scripts/eval_pipeline.py gen --manifest M.json --dir SETS                  # sample and store the shots
    python scripts/eval_pipeline.py dec --manifest M.json --dir SETS --out DEC --decoder lomr|cain|bplomr
    python scripts/eval_pipeline.py net --manifest M.json --dir SETS --out NET --name N --ckpt networks/main_d5.pt
    python scripts/eval_pipeline.py sel --manifest M.json --net NET --name N --frac 0.25 --out LISTS
    python scripts/eval_pipeline.py tess --manifest M.json --dir SETS --lists LISTS --out TESS --deadline none
    python scripts/eval_pipeline.py score --manifest M.json --dir SETS --dec DEC --out J

`dec` runs the baselines: `lomr` is LoMatching, `cain` our reimplementation of Cain et al., `bplomr` belief propagation plus
LoMatching. `net` reads the network alone; `sel` picks the least confident shots; `tess` runs the 21 long-beam trials of
Tesseract on them, each timed alone. Benchmark II is `--purpose bench2` and `scripts/dev_bank.py` (its development bank,
with `dec --decoder mle` for the exact decoder); `scripts/benchmark2_reads.py` and `scripts/bank_report.py` hold its two
measures. The first test and the fresh test of the paper used the manifests of this code with the seed ranges recorded in
the manifests (`scripts/fresh_sets.py` makes the fresh ones); the shots are regenerated from those seeds.

The paced stream, the part that measures speed:

    python scripts/operating_points.py read ... --path engine     # the network through the Triton engine (GPU)
    python scripts/operating_points.py thr ...                    # confidence thresholds per routed fraction
    python scripts/operating_points.py nettime ...                # network time per layer at batch 256
    python scripts/operating_points.py sim ... --basis local      # the stream simulated from the stored trial times
    python scripts/operating_points.py curve ... ; choose ...     # time per layer against failures, the operating point
    python scripts/paced_stream.py live ...                       # the live stream: engine on the GPU, trials on N cores

`scripts/grid_scores.py`, `scripts/budget_matched.py` and `scripts/score_test.py` score every routed fraction and cap, the
budget-matched Tesseract presets and a frozen operating point on a test set. The conversion of trial times from one CPU to the
reference platform (Appendix E of the paper) takes a calibration factor per CPU model as `--cal`; the factors the paper used
are in the appendix text, and the calibration jobs themselves were cluster launchers that are not included.
`scripts/quickstart.sh` runs the whole chain at toy size.

## Hardware

The paper's numbers come from these machines.

- Training: one NVIDIA RTX PRO 6000 Blackwell Server Edition (8,180, 3,531, 1,902 and 1,161 samples per second at
  d = 3, 5, 7, 9), with CPU cores for the shot samplers.
- Network times and the first live stream: the same GPU (driver 595.91.07, CUDA 13.2) with 32 cores of an AMD EPYC 9555.
- Second live stream: a DGX Spark (GB10, driver 580.126.09, CUDA 13.0) with its 10 Cortex-X925 and 10 Cortex-A725 cores.
- Tesseract trial times: each trial alone on one core of an AMD EPYC 7713P or 9534 node with every core busy, converted to an
  NVIDIA Grace core (64 cores) with the factors of Appendix E.
- The exact most-likely-error decoder (SCIP, 600 s limit per shot) ran on CPU nodes.

This release was also checked on Ubuntu 24.04 with a GeForce RTX 3060 and an AMD Ryzen 9 9950X (tests, the quickstart and
the tables and figures).

## Citation

    @misc{duet2026,
      title  = {Neural Decoding of Logical Clifford Circuits with Transversal Gates},
      author = {{Duet authors}},
      year   = {2026},
      note   = {Placeholder: authors, venue and identifier to be filled in}
    }

## License

Apache License 2.0 (`LICENSE`, `NOTICE`). Code adapted from CoDecoder (Apache-2.0) and two data files from
data-lom-decoders (MIT) keep their own licenses in `third_party/`.
