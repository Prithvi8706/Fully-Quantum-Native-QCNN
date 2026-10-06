#!/usr/bin/env bash
# Reproduce every table and figure in paper/fqcnn.pdf.
#
# Usage:
#   ./reproduce.sh            # full reproduction (many trainings: slow)
#   ./reproduce.sh --quick    # tests and a three-task smoke run only
#
# PYTHON is the Python 3.9.13 training environment (requirements-lock.txt);
# QISKIT_PYTHON is the Python 3.11 Qiskit environment (requirements-qiskit-lock.txt).
set -euo pipefail

PY="${PYTHON:-python}"
QPY="${QISKIT_PYTHON:-.venv-qiskit/Scripts/python.exe}"
[ -x "$QPY" ] || QPY=".venv-qiskit/bin/python"

echo "=== Tests: architecture freeze, Theorem 1, protocol ==="
"$PY" -m pytest tests -q

if [ "${1:-}" == "--quick" ]; then
  "$PY" -m experiments.run_experiments --smoke
  exit 0
fi

echo "=== Four-task comparison (Tables VIII-XII, Fig. 4) ==="
"$PY" -m experiments.run_experiments \
    --task mnist:0,1 --task mnist:3,5 --task fashion_mnist:0,6 --task kmnist:2,3 \
    --configs proposed --seeds 0 1 2 3 4 --samples 400 --epochs 30 \
    --classical-baselines logistic mlp --quantum-baselines ttn
"$PY" -m experiments.q1_fast_track_analysis aggregate --output Results/evidence/q1_comparison.json

echo "=== Pooling-transfer ablation (Table XIII) ==="
"$PY" -m experiments.q1_pooling_transfer run
"$PY" -m experiments.q1_pooling_transfer aggregate

echo "=== Secondary pooling-arm campaign (Table XIV) ==="
"$PY" -m experiments.run_experiments --datasets 0,1 3,5 4,9 \
    --configs e3_pool_none e3_pool_measurement e3_pool_unitary e3_pool_coherent e3_pool_su4 \
    --seeds 0 1 2 3 4 --samples 400 --epochs 30 --no-baselines
"$PY" -m experiments.pooling_analysis --experiment e3

echo "=== Pooling theorem checks and information dynamics ==="
for experiment in e1 e2 e4 controls; do
  "$PY" -m experiments.pooling_analysis --experiment "$experiment"
done

echo "=== Gradient variance, DLA, expressibility, capacity ==="
"$PY" -m experiments.model_analysis --experiment all

echo "=== Qiskit resources and noise (Tables XVI-XVIII) ==="
"$PY" -m experiments.q1_local_evidence export
"$QPY" -m experiments.q1_local_evidence resources
"$QPY" -m experiments.q1_local_evidence pooling
"$QPY" -m experiments.q1_local_evidence noise

echo "=== Figures and paper ==="
"$PY" QCNN/utils/draw_fig1_architecture.py --output figs_final/fig1.png --dpi 600
"$PY" QCNN/utils/draw_fig3_kernel.py
"$PY" QCNN/utils/draw_fig_comparison.py --dpi 300
(cd paper && for pass in 1 2 3; do pdflatex -interaction=nonstopmode -halt-on-error fqcnn.tex >/dev/null; done)

echo "Done. Evidence is in Results/evidence/ and the paper in paper/fqcnn.pdf."
