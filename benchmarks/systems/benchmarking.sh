#!/usr/bin/env bash
set -euo pipefail

PARAM_FILE="$(dirname "${BASH_SOURCE[0]}")/params.txt"
LOG_DIR="${CS336_LOG_DIR:-outputs/benchmark_logs}"
MAX_PARALLEL=1
PYTHON="${CS336_PYTHON:-python}"
export PYTHON

mkdir -p "$LOG_DIR"

# Fixed args as a string (exportable)
FIXED_ARGS="--vocab_size 10000 --theta 10000 --context_length 256 --batch_size 4 --warmup_steps 5 --num_measures 10 --use_mixed_precision"
export FIXED_ARGS

nl -ba "$PARAM_FILE" | xargs -P "$MAX_PARALLEL" -I{} bash -lc '
  set -euo pipefail
  line_no=$(echo "{}" | awk "{print \$1}")
  args=$(echo "{}" | cut -f2-)

  out="'"$LOG_DIR"'/run_${line_no}_mixed_precision.out"

  echo "Run ${line_no}: ${args}"
  "$PYTHON" -m cs336.systems.monitoring $FIXED_ARGS ${args} >"${out}"
'
echo "Done. Logs in: $LOG_DIR"
