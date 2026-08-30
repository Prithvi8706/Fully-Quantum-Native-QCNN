# FQCNN: Fully Quantum-Native QCNN 
### A Fully Quantum-Native Convolutional Neural Network with Coherent Unitary Pooling

The canonical full-paper source is [`fqcnn.tex`](fqcnn.tex), written in the
venue-neutral IEEEtran journal format. Its generated PDF is `fqcnn.pdf`; build
instructions and the verified release archive are documented in
[`docs/submission/README.md`](docs/submission/README.md). The tracked
[conference version (historical)](<FQCNN conference version (historical).pdf>) is
retained only for prior-version disclosure and must not be confused with the
current journal draft.

This repository implements **FQCNN**, a **fully quantum-native convolutional neural network (QCNN)** that performs state preparation, register-index convolution, coherence-preserving unitary pooling, and classification **entirely using unitary quantum operations before one terminal observable**. Unlike hybrid models, this architecture contains no classical convolutional layers at any stage.

The frozen headline study uses 28×28 MNIST inputs (784 flattened pixels), zero-padded to 1,024 amplitudes on n=10 qubits, with an active-wire schedule of 10→5→2→1. The Q1 evidence package is classically simulated; register-index parameter sharing is not an image-space translational-equivariance guarantee, and no quantum-advantage or hardware-performance claim is made.

---

## Key Features

- **100% Quantum-Native**: Convolution, Pooling, and Classification are all implemented as differentiable quantum circuits.
- **Audited headline circuit**: 269 allocated parameter slots, 78 read on the tape, and 74 gradient-effective slots; Caro's bound uses T=222 trainable gates.
- **Advanced Encoding**:
  - **Amplitude Encoding**: Maps full images into Hilbert space using only $\log_2(N)$ qubits.
  - **Patch-based Encoding**: Uses a sliding quantum filter (Quanvolution) to process large images efficiently.
- **Stable Training**: Integrated **Exponential Moving Average (EMA)**, **Learning Rate Warmup**, and **Gradient Clipping** to prevent oscillations and barren plateaus.
- **Strict Binary Classification**: Easily switch between any two classes (e.g., MNIST 3 vs 7) via simple CLI arguments.
- **Visualization Tools**: Built-in utilities for visualizing the quantum preprocessing pipeline and circuit architectures.
- **Optimized Simulation**: Support for `pennylane-lightning` and multiprocessing for accelerated training.

---

## Repository Structure

```text
QCNN/
├── config/Qconfig.py           # Centralized configuration (Architecture, LR, EMA)
├── encoding/QEncoder.py        # Amplitude, Feature Map, and Patch-based encoders
├── layers/
│   ├── QConv.py                # 2x2 weight-shared quantum kernels
│   ├── QPool.py                # Trainable unitary pooling layers
│   └── QuanvLayer.py           # Random quantum filters for patch preprocessing
├── models/QCNNModel.py         # The core PureQuantumNativeCNN implementation
├── training/Qtrainer.py        # Optimized training loop (Adam, EMA, Early Stopping)
└── utils/
    ├── dataset_loader.py       # Universal loader (MNIST, NPZ, CSV, Images)
    ├── visualize_preprocessing.py # Preprocessing visualization tool
    └── metadata_logger.py      # Experiment tracking and reproducibility
main.py                         # Unified entry point for training and evaluation
RUN_GUIDE.md                    # Quick-start guide and CLI examples
DATASET_README.md               # Technical details on data handling
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install pennylane pennylane-lightning numpy matplotlib scikit-learn seaborn
```

### 2. Run Standard Training
Classify MNIST digits 0 vs 1 using Amplitude Encoding:
```bash
python main.py --dataset mnist --classes 0 1 --samples 500 --encoding amplitude
```

### 3. Run with Patch-based Encoding
Useful for larger images or different feature extraction:
```bash
python main.py --dataset mnist --classes 3 7 --encoding patch
```

---

## Performance Benchmarks

The canonical Q1 comparison evaluates the proposed model, logistic regression, a compact two-unit dense MLP, and a tree-tensor-network baseline on four frozen binary tasks (five seeds, identical 666/400/100/166 manifests). The logistic baseline has 785 coefficients and the two-unit MLP has 1,573 dense parameters; the latter is the smallest supported dense comparator, not a parameter-matched 269-slot model. The proposed model is below logistic regression and the MLP on all four tasks and above TTN in mean accuracy on all four; the paired tests are not significant after the declared Holm correction. These are finite within-study results, not a claim of superiority or quantum advantage. See [`Results/evidence/q1_comparison.json`](Results/evidence/q1_comparison.json) and the manuscript table/figure.

The completed pooling-transfer lane evaluates five pooling arms on Fashion-MNIST 0-vs-6 for three seeds. It is a bounded ablation, not a universal pooling ranking. Local resource, noise, and fake-backend artifacts are similarly simulator-only and do not imply physical-device performance.

Canonical evidence under `Results/evidence/` also supports these narrower conclusions:

- E1–E4 are complete. The 75-cell pooling campaign found no significant benefit from the coherent extension; SU(4) is the remaining pooling headroom, so frozen pooling is not proven globally optimal.
- The headline DLA is full su(2^10), dimension 1,048,575; therefore there is no polynomial-DLA trainability certificate.
- Gradient evidence supports only “no barren plateau observed through n=14.”
- The selected expressibility metric is Haar-indistinguishable at the available resolution.
- Information-concentration language is not an established result.

Dataset provenance, manifests, source hashes, and the final claim ledger are recorded alongside the Q1 artifacts. The release status and exact validation counts are maintained in [`STATUS.md`](STATUS.md).

---

## Architecture Deep Dive

### Encoding Strategies
- **Amplitude Encoding**: Efficiently stores $2^n$ features in $n$ qubits. Ideal for preserving global structure with minimal hardware requirements.
- **Patch-based (Quanv)**: Applies a "Quantum Convolutional Layer" with random filters to extract local features before the main QCNN, enabling the processing of images of any size.

### Quantum Training Pipeline
We implement several stability techniques to ensure convergence in the quantum landscape:
1. **EMA (Exponential Moving Average)**: Maintains a shadow copy of weights for smoother inference.
2. **LR Warmup**: Gradually increases learning rate to avoid unstable initial gradients.
3. **Gradient Clipping**: Prevents "exploding" updates common in variational circuits.
4. **Early Stopping**: Monitors validation performance to prevent overfitting.

---

## Visualization
You can visualize how the quantum model "sees" the data:
```bash
python QCNN/utils/visualize_preprocessing.py --dataset mnist --classes 0 1
```
This generates comparisons between raw images and their quantum-encoded counterparts (e.g., amplitude maps or quanvolved patches).

---

## Reproducibility & Experiments

Experiments provide seed controls and the Q1 package records the exact manifests, per-run hashes, aggregate inputs, and bounded validation commands. The final package is simulator-only; no real-QPU job is submitted by the repository workflow. Use the frozen environment for matching supported numbers:

```bash
pip install -r requirements-lock.txt     # exact tested versions (Python 3.9.13)
# or, for a looser install:
pip install -r requirements.txt
```

Hardware tooling is isolated from training. Create its Python 3.11 environment and
validate the local fake-backend transpilation smoke with:

```bash
py -3.11 -m venv .venv-qiskit
.venv-qiskit\Scripts\python.exe -m pip install --upgrade pip
.venv-qiskit\Scripts\python.exe -m pip install -r requirements-qiskit-lock.txt
.venv-qiskit\Scripts\python.exe scripts/check_qiskit_environment.py
```

Task 6 needs no credentials or network for the smoke itself. Real-QPU submission
remains a later explicit approval gate. Do not install these Qiskit packages into
the Python 3.9.13 training environment or modify `requirements-lock.txt`.

Exercise the current reproduction scripts with:

```bash
./reproduce.sh            # full study (many QCNN trainings — slow)
./reproduce.sh --quick    # fast smoke test that exercises the whole pipeline
```

### Q1 comparison campaign
The bounded Q1 command evaluates the frozen proposed arm with logistic and MLP
classical baselines plus the TTN quantum baseline on four registered tasks and
five seeds. Every arm consumes the identical split and ordered sample IDs; the
canonical output is under `Results/q1_comparison/` and
`Results/evidence/q1_comparison.json`:

```bash
python -m experiments.run_experiments \
    --task mnist:0,1 --task mnist:3,5 \
    --task fashion_mnist:0,6 --task kmnist:2,3 \
    --configs proposed --seeds 0 1 2 3 4 --samples 400 --epochs 30 \
    --classical-baselines logistic mlp --quantum-baselines ttn --jobs 3
```

Ablation toggles (set in `QCNN/config/Qconfig.py` or via the runner):

| Component | Flag | Variants |
| :--- | :--- | :--- |
| Pooling | `pooling_mode` | `unitary` (proposed) · `none` · `measurement` |
| Conv entanglement | `conv_entanglement` | `full` · `one_diagonal` · `none` |
| Kernel rotations | `kernel_rotations` | `su2` (proposed) · `ry` |
| Encoding | `encoding_type` | `amplitude` · `feature_map` |

### Optional quantum-architecture baselines
The harness can also run additional quantum classifiers for exploratory work.
They are optional arms, not automatic Q1 evidence; only a completed aggregate
with matched manifests may be used in a manuscript comparison:

| Row | Architecture | Reference |
| :--- | :--- | :--- |
| `baseline_cong` | Canonical QCNN (translationally-invariant conv + pooling) | Cong, Choi & Lukin, *Nature Physics* **15**, 1273 (2019) |
| `baseline_hur` | QCNN with a more expressive two-qubit conv block | Hur, Kim & Park, *EPJ Quantum Technology* **9**, 1 (2022) |
| `baseline_ttn` | Tree tensor network hierarchical classifier | Grant et al., *npj Quantum Information* **4**, 65 (2018) |

```bash
# train/evaluate the quantum baselines standalone on one MNIST pair (debugging)
python -m baselines.quantum_baselines --classes 0 1 --samples 120 --epochs 15
```

### Metrics
Every run computes the full metric suite via `QCNN/utils/metrics.py` and writes a
`metrics.json`. A single training run also saves confusion-matrix, ROC, and
precision-recall figures under `Results/Graphs/`.

### Noise robustness (NISQ)
`noise_sim.py` uses a historical split and is not manuscript evidence. It provides two local noise models — uniform depolarizing and an IBM-hardware-like multi-channel model (depolarizing + amplitude/phase damping + readout error):

```bash
python noise_sim.py --noise-model depolarizing --classes 0 1
python noise_sim.py --noise-model realistic    --classes 0 1
```

### Optional: real quantum hardware
Hardware execution is pending. Its Qiskit dependencies belong in a separate isolated environment, not the frozen `requirements-lock.txt` environment. `experiments/hardware_run.py` requires `pennylane-qiskit`, `qiskit-ibm-runtime`, and an IBM Quantum token; no hardware result or quantum-advantage claim is currently supported.

---

## Citation & Contact
**Title**: FQCNN: A Fully Quantum-Native Convolutional Neural Network with Coherent Unitary Pooling  
**Authors**: Aasa Singh Bhui, Prithvi Raghu (Vellore Institute of Technology)  
**GitHub**: [AasaSingh05/Fully-Quantum-Native-QCNN](https://github.com/AasaSingh05/Fully-Quantum-Native-QCNN)

If you use this work, please cite it. Citation metadata is maintained in
[`CITATION.cff`](CITATION.cff) — GitHub's **"Cite this repository"** button
generates APA/BibTeX from it automatically.

**Cite as:**
> Singh Bhui, A., & Raghu, P. FQCNN: A Fully Quantum-Native Convolutional Neural Network with Coherent Unitary Pooling. https://github.com/AasaSingh05/Fully-Quantum-Native-QCNN

**BibTeX:**
```bibtex
@software{fqcnn,
  title     = {FQCNN: A Fully Quantum-Native Convolutional Neural Network with Coherent Unitary Pooling},
  author    = {Singh Bhui, Aasa and Raghu, Prithvi},
  year      = {2025},
  license   = {MIT},
  url       = {https://github.com/AasaSingh05/Fully-Quantum-Native-QCNN}
}
```
