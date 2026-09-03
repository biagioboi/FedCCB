#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"
cd "$SCRIPT_DIR"
python "$REPRO_ROOT/scripts/robust_fl_experiment.py" \
  --dataset "cifar100" \
  --method "fedavg" \
  --alpha "0.5" \
  --attack "alto" \
  --num-rounds "50" \
  --num-clients "25" \
  --server-fraction "0.2" \
  --batch-size "32" \
  --local-epochs "20" \
  --client-parallelism "0" \
  --seed "42" \
  --label-swap-fraction "0.75" \
  --malicious-probability "0.5" \
  --malicious-clients "1,2,3,5,6,7,9,10,12,13,15,17,18,19,20,21,23,24" \
  --output-dir "results"
