#!/bin/bash
# Tiny end-to-end run of the whole pipeline at d = 3 (a few minutes on one machine, GPU optional for the last steps):
# train a small network, sample two small test points of Benchmark I, decode them with LoMatching, the Cain et al.
# reimplementation and BP+LoMatching, read the network (frozen forward and engine), route the least confident shots to
# Tesseract, score, simulate and run the paced stream. Nothing here reproduces a number of the paper: the network is
# tiny and the points are small. It shows the commands and the files they exchange.
#   bash scripts/quickstart.sh [WORKDIR]        (from the repository root, after `pip install -e .`)
# Steps 6 to 9 use the Triton engine and need an NVIDIA GPU, a C compiler and the Python headers (python3-dev).
set -euo pipefail
W=${1:-quickstart_work}; mkdir -p "$W"; cd "$W"
SC=../scripts; export OMP_NUM_THREADS=1
echo "1  train a tiny network (CPU)"
python -m duet.net.train_cl --out net --d 3 --width 64 --bott 16 --blocks 1 --batch 256 --chunk 64 --samples 3e4 \
  --noise lukin --block-counts 2,3 --lukin-share 0.25 --p-tri 0.2 --p-cnot-set 0,0.3333333,0.6666667,1 \
  --cnot-share 0.25 --ent-share 0.1 --eval-circuits 2 --eval-cnot 2 --eval-per 64 --eval-every 3e4 --workers 4 \
  --prefetch 8 --lookup gather --seed 1 --device cpu > train.log
echo "2  the development manifest of Benchmark I, cut down to two small points"
python $SC/eval_pipeline.py manifest --purpose dev --out dev.json
python - <<'PY'
import json
m = json.load(open("dev.json"))
pts = [p for p in m["points"] if p["name"] in ("b1d_K2_d3_L2_p3", "b1d_K3_d3_L2_p3")]
for p in pts:
    p["shots"], p["circuits"], p["bplomr_shots"] = 400, p["circuits"][:3], 200
json.dump(dict(m, points=pts), open("tiny.json", "w"))
PY
echo "3  sample the shots"
python $SC/eval_pipeline.py gen --manifest tiny.json --dir sets
echo "4  the baselines: LoMatching, our Cain et al. reimplementation, BP+LoMatching"
for d in lomr cain bplomr; do python $SC/eval_pipeline.py dec --manifest tiny.json --dir sets --out dec --decoder $d; done
echo "5  the network alone (frozen forward), the least confident shots, the Tesseract trials, the baselines' scores"
python $SC/eval_pipeline.py net --manifest tiny.json --dir sets --out net_reads --name tiny --ckpt net/final_ema.pt --device cpu
python $SC/eval_pipeline.py sel --manifest tiny.json --net net_reads --name tiny --frac 0.25 --out lists.json
python $SC/eval_pipeline.py tess --manifest tiny.json --dir sets --lists lists.json --out tess --deadline none
python $SC/eval_pipeline.py score --manifest tiny.json --dir sets --dec dec --out score.json
echo "6  the network alone through the Triton engine; thresholds that route a fraction of the shots"
python $SC/operating_points.py read --manifest tiny.json --dir sets --out engine_reads --name tiny --ckpt net/final_ema.pt --path engine
python $SC/operating_points.py thr --manifest tiny.json --net engine_reads --name tiny --d 3 --out thr.json
echo "7  network time per layer at batch 256"
python $SC/operating_points.py nettime --manifest tiny.json --dir sets --ckpt net/final_ema.pt --out net_us.json --B 256 --Bs 256 --circuits 2
echo "8  the paced stream simulated from the stored trials, on this machine's own times (basis local)"
python $SC/operating_points.py sim --manifest tiny.json --dir sets --net engine_reads --name tiny --thr thr.json --tess tess --d 3 --basis local --out sim.json
python $SC/operating_points.py curve --manifest tiny.json --sim sim.json --net-us net_us.json --out curve.md --N 4 --B 256
echo "9  the live paced stream: engine on the GPU, Tesseract trials on 4 cores"
python $SC/paced_stream.py live --manifest tiny.json --dir sets --net engine_reads --name tiny --thr thr.json --f 0.2 --cap none --N 4 \
  --out live --ckpt net/final_ema.pt --circuits 2 --batches 1 --cores 0,1,2,3
echo "done: see curve.md (simulated stream), live.jsonl (live stream), score.json (baselines)"
