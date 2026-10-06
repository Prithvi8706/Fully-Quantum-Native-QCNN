# FQCNN: A Fully Quantum-Native Convolutional Neural Network with Coherent Unitary Pooling

This repository contains the paper, code, and results for **FQCNN**, a quantum
convolutional neural network that performs state preparation, register-index
convolution, pooling, and classification with unitary operations before a single
terminal measurement.

- **Paper:** [`paper/fqcnn.pdf`](paper/fqcnn.pdf) (source: [`paper/fqcnn.tex`](paper/fqcnn.tex))
- **Authors:** Aasa Singh Bhui, Prithvi Raghu, J Jayashree, J Vijayashree
  (School of Computer Science and Engineering, Vellore Institute of Technology)

## Summary

The ten-qubit model uses global amplitude encoding (a 28×28 image, zero-padded to
1,024 amplitudes), a parameter-shared 2×2 register-index convolution kernel, and a
measurement-free pooling block built from controlled rotations. The paper proves
that this pooling block induces exactly the measure-and-condition channel of
measurement-based QCNN pooling on the retained register, without mid-circuit
measurement, reset, or classical feed-forward.

Test accuracy on the four-task comparison (mean ± standard deviation over five
seeds; 400/100/166 train/validation/test examples per task):

| Task | FQCNN | Logistic | MLP (2-unit) | TTN |
|---|---|---|---|---|
| MNIST 0-vs-1 | 0.975 ± 0.013 | 0.995 ± 0.005 | 0.988 ± 0.009 | 0.961 ± 0.016 |
| MNIST 3-vs-5 | 0.800 ± 0.030 | 0.948 ± 0.012 | 0.877 ± 0.044 | 0.730 ± 0.047 |
| Fashion-MNIST 0-vs-6 | 0.789 ± 0.019 | 0.842 ± 0.014 | 0.824 ± 0.015 | 0.787 ± 0.073 |
| KMNIST 2-vs-3 | 0.884 ± 0.023 | 0.947 ± 0.024 | 0.933 ± 0.016 | 0.867 ± 0.062 |

The FQCNN is below the classical baselines and above the tree-tensor-network (TTN)
baseline in mean accuracy on all four tasks; no paired difference is significant
after Holm correction. All results are classical simulations.

## Repository layout

```text
paper/                    Manuscript source and PDF
QCNN/                     Model package
  circuits.py             The canonical FQCNN circuit (single source of truth)
  layers/                 Convolution (QConv.py) and pooling (QPool.py) blocks
  models/QCNNModel.py     PureQuantumNativeCNN (batched default.qubit training path)
  training/Qtrainer.py    Training loop (Adam, EMA, LR schedule, validation selection)
  utils/                  Data loading, splits, metrics, figure scripts
baselines/                Logistic, two-unit MLP, and TTN baselines
experiments/              Experiment runner and analysis scripts
Results/
  q1_comparison/          Per-seed results and split manifests for the four-task comparison
  q1_pooling_transfer/    Pooling-transfer ablation (Table XIII)
  experiments/            Secondary pooling-arm campaign (Table XIV)
  evidence/               Aggregated results behind every table and diagnostic in the paper
  manifests/              Data splits
  Weights/                Reference checkpoint used by the circuit diagnostics
figs_final/               Paper figures
tests/                    Tests for the architecture, Theorem 1, and the evaluation protocol
```

## Setup

The training environment is Python 3.9.13 with exact package versions:

```bash
pip install -r requirements-lock.txt
```

The Qiskit resource and noise simulations use a separate Python 3.11 environment;
do not install Qiskit into the training environment:

```bash
py -3.11 -m venv .venv-qiskit
.venv-qiskit\Scripts\python.exe -m pip install -r requirements-qiskit-lock.txt
```

MNIST, Fashion-MNIST, and KMNIST are downloaded from their upstream sources into
`datasets/` (see [`DATASET_README.md`](DATASET_README.md)); they are not
redistributed here.

## Reproducing the paper

[`reproduce.sh`](reproduce.sh) runs the steps below in order. Each command writes
the evidence file named in its comment.

```bash
# Tests: architecture freeze, Theorem 1, effective parameters, protocol
python -m pytest tests -q

# Four-task comparison (Tables VIII-XII, Fig. 4)
python -m experiments.run_experiments \
    --task mnist:0,1 --task mnist:3,5 --task fashion_mnist:0,6 --task kmnist:2,3 \
    --configs proposed --seeds 0 1 2 3 4 --samples 400 --epochs 30 \
    --classical-baselines logistic mlp --quantum-baselines ttn
python -m experiments.q1_fast_track_analysis aggregate --output Results/evidence/q1_comparison.json

# Pooling-transfer ablation (Table XIII)
python -m experiments.q1_pooling_transfer run
python -m experiments.q1_pooling_transfer aggregate

# Secondary pooling-arm campaign (Table XIV)
python -m experiments.run_experiments --datasets 0,1 3,5 4,9 \
    --configs e3_pool_none e3_pool_measurement e3_pool_unitary e3_pool_coherent e3_pool_su4 \
    --seeds 0 1 2 3 4 --samples 400 --epochs 30 --no-baselines
python -m experiments.pooling_analysis --experiment e3

# Pooling theorem checks and information dynamics (Section IV-F, Table XV)
python -m experiments.pooling_analysis --experiment e1
python -m experiments.pooling_analysis --experiment e2
python -m experiments.pooling_analysis --experiment e4
python -m experiments.pooling_analysis --experiment controls

# Gradient variance, DLA, expressibility, capacity (Tables XIX-XXII)
python -m experiments.model_analysis --experiment all

# Qiskit resources and noise (Tables XVI-XVIII): export the canonical circuit
# from PennyLane, then run the Qiskit commands in the Qiskit environment
python -m experiments.q1_local_evidence export
.venv-qiskit\Scripts\python.exe -m experiments.q1_local_evidence resources
.venv-qiskit\Scripts\python.exe -m experiments.q1_local_evidence pooling
.venv-qiskit\Scripts\python.exe -m experiments.q1_local_evidence noise

# Figures and paper
python QCNN/utils/draw_fig1_architecture.py --output figs_final/fig1.png --dpi 600
python QCNN/utils/draw_fig3_kernel.py
python QCNN/utils/draw_fig_comparison.py --dpi 300
cd paper && pdflatex fqcnn.tex && pdflatex fqcnn.tex && pdflatex fqcnn.tex
```

The reference checkpoint in `Results/Weights/` (MNIST 0-vs-1, seed 42, learning
rate 0.005, at most 50 epochs) is produced by:

```bash
python main.py --dataset idx --path datasets/MNIST --classes 0 1 --encoding amplitude \
    --image-size 28 --learning-rate 0.005 --epochs 50 --seed 42 --no-profile
```

### Other scripts

`noise_sim.py` is an earlier noise study on a historical split and is
not manuscript evidence. `runApp.sh` / `runApp.bat` wrap `main.py` for
single training runs.

## Citation

```bibtex
@misc{fqcnn2026,
  title  = {FQCNN: A Fully Quantum-Native Convolutional Neural Network with Coherent Unitary Pooling},
  author = {Singh Bhui, Aasa and Raghu, Prithvi and Jayashree, J and Vijayashree, J},
  year   = {2026},
  url    = {https://github.com/Prithvi8706/Fully-Quantum-Native-QCNN}
}
```

Citation metadata is also in [`CITATION.cff`](CITATION.cff).
