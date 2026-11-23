#!/bin/bash
# Train one network: bash scripts/train.sh NAME [train_cl arguments...]
# Runs `python -m duet.net.train_cl --out $RUNS/NAME ARGS` (RUNS defaults to ./runs) with the log in $RUNS/NAME/train.log.
# Run the same command again after an interruption (exit 3 when --max-minutes stopped it, or a kill): it resumes exactly.
# Optional environment: PY (python, default python3), WORKERS (sampler processes, default nproc - 2), MAXMIN (0 = no limit).
set -uo pipefail
name=$1; shift
out=${RUNS:-runs}/$name; mkdir -p "$out"
echo "train.sh $name start $(date) max-minutes ${MAXMIN:-0}" >> "$out/train.log"
${PY:-python3} -m duet.net.train_cl --out "$out" "$@" --workers "${WORKERS:-$(( $(nproc) - 2 ))}" --max-minutes "${MAXMIN:-0}" >> "$out/train.log" 2>&1
rc=$?
echo "train.sh $name exit $rc $(date)" >> "$out/train.log"
exit $rc
