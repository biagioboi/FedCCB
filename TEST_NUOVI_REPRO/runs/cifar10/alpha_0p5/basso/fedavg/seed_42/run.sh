#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"
cd "$SCRIPT_DIR"
python "$REPRO_ROOT/scripts/robust_fl_experiment.py" \
  --dataset "cifar10" \
  --method "fedavg" \
  --alpha "0.5" \
  --attack "basso" \
  --num-rounds "50" \
  --num-clients "25" \
  --server-fraction "0.2" \
  --batch-size "32" \
  --local-epochs "20" \
  --client-parallelism "0" \
  --seed "42" \
  --label-swap-fraction "0.25" \
  --malicious-probability "0.5" \
  --malicious-clients "3,7,13,15,19,23" \
  --output-dir "results"
