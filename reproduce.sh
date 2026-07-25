#!/usr/bin/env bash
# Reproduce the FQCNN reviewer-response results (suggestion #8).
#
# Regenerates: the ablation / multi-seed study (Results/experiments/summary.csv),
# a single full training run with all metrics + figures, and both noise-robustness
# curves. Everything is seed-controlled for determinism.
#
# Usage:
#   ./reproduce.sh            # full study (slow: many QCNN trainings)
#   ./reproduce.sh --quick    # fast smoke test to verify the pipeline
set -euo pipefail

# No .venv is committed; default to whatever python is on PATH and let the
# integrity gate below fail loudly if it is the wrong interpreter.
PY="${PYTHON:-python}"
QUICK="${1:-}"

echo "Using interpreter: $PY"
"$PY" --version

echo "=== [0/4] Freeze and protocol integrity gate ==="
# Nothing downstream is trustworthy if these fail: they pin the frozen circuit's
# topology and outputs, the main-path unitarity invariant, the effective-parameter
# audit, and the no-test-leakage protocol (UPGRADE_PLAN.md 0.1, 0.2, 0.3, 0.6).
"$PY" -m pytest tests/ -q

if [ "$QUICK" == "--quick" ]; then
  echo "=== [1/4] Ablation + multi-seed study (QUICK smoke) ==="
  "$PY" -m experiments.run_experiments --quick

  echo "=== [2/4] Single training run (tiny) ==="
  "$PY" main.py --dataset idx --path datasets/MNIST --classes 0 1 \
      --encoding amplitude --image-size 16 --samples 60 --epochs 2 --seed 0 --no-profile
else
  echo "=== [1/4] Ablation + multi-seed study (FULL) ==="
  # All ablations x hard MNIST pairs x 5 seeds -> Results/experiments/summary.csv
  "$PY" -m experiments.run_experiments \
      --datasets 0,1 3,5 4,9 5,8 \
      --seeds 0 1 2 3 4 \
      --samples 400 --epochs 30

  echo "=== [2/4] Single full training run (metrics + figures) ==="
  "$PY" main.py --dataset idx --path datasets/MNIST --classes 0 1 \
      --encoding amplitude --image-size 16 --samples 400 --epochs 30 --seed 42 --no-profile
fi

echo "=== [3/4] Noise robustness (depolarizing + realistic IBM-like) ==="
"$PY" noise_sim.py --noise-model depolarizing --classes 0 1 || \
    echo "  (noise sim needs trained weights from step 2)"
"$PY" noise_sim.py --noise-model realistic --classes 0 1 || \
    echo "  (noise sim needs trained weights from step 2)"

echo "Done. See Results/experiments/summary.csv and Results/Graphs/."
