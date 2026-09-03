#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"
cd "$SCRIPT_DIR"
python "$REPRO_ROOT/scripts/robust_fl_experiment.py" \
  --dataset "cifar10" \
  --method "fltrust" \
  --alpha "0.1" \
  --attack "medio" \
  --num-rounds "50" \
  --num-clients "25" \
  --server-fraction "0.2" \
  --batch-size "32" \
  --local-epochs "20" \
  --client-parallelism "0" \
  --seed "42" \
  --label-swap-fraction "0.5" \
  --malicious-probability "0.5" \
  --malicious-clients "1,3,6,7,9,10,13,15,12,19,20,23" \
  --output-dir "results"
