# FQCNN: Fully Quantum-Native QCNN 
### A Fully Quantum-Native Convolutional Neural Network with Coherent Unitary Pooling

This repository implements **FQCNN**, a **fully quantum-native convolutional neural network (QCNN)** that performs data encoding, local convolution, coherence-preserving unitary pooling, and classification **entirely using unitary quantum operations, with no intermediate measurements**. Unlike hybrid models, this architecture contains no classical convolutional layers at any stage.

The frozen headline study uses 28×28 MNIST inputs (784 flattened pixels), zero-padded to 1,024 amplitudes on n=10 qubits, with an active-wire schedule of 10→5→2→1. The current n=10 study is classically simulated; no quantum-advantage claim is made.

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

The standalone accuracy values historically shown here were single-seed results, not distributions, and are not manuscript evidence. The protocol-clean 98.29% MNIST 0-vs-1 headline is also a single-seed result until a multi-seed distribution with confidence intervals exists. Current baseline numbers are blocked from manuscript use pending leakage and protocol repair.

Canonical evidence under `Results/evidence/` currently supports these narrower conclusions:

- E1–E4 are complete. The 75-cell pooling campaign found no significant benefit from the coherent extension; SU(4) is the remaining pooling headroom, so frozen pooling is not proven globally optimal.
- The headline DLA is full su(2^10), dimension 1,048,575; therefore there is no polynomial-DLA trainability certificate.
- Gradient evidence supports only “no barren plateau observed through n=14.”
- The selected expressibility metric is Haar-indistinguishable at the available resolution.
- Information-concentration language is not an established result.

Dataset breadth, repaired baselines, the broader ablation grid, resource scaling, noise, hardware, and manuscript claims remain pending unless a canonical artifact records them.

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

Experiments provide seed controls, but full clean-checkout reproduction is not yet claimed; that remains pending until M11 passes. Use the frozen environment for matching supported numbers:

```bash
pip install -r requirements-lock.txt     # exact tested versions (Python 3.9.13)
# or, for a looser install:
pip install -r requirements.txt
```

Exercise the current reproduction scripts with:

```bash
./reproduce.sh            # full study (many QCNN trainings — slow)
./reproduce.sh --quick    # fast smoke test that exercises the whole pipeline
```

### Ablation & multi-seed study
A single harness runs the ablations across hard MNIST digit pairs and multiple
seeds, then reports **mean ± std** for accuracy / precision / recall / F1 /
ROC-AUC / PR-AUC, alongside both classical baselines (logistic, MLP) and
published **quantum-architecture** baselines (see below), all trained on the
**identical** split for a fair comparison:

```bash
python -m experiments.run_experiments \
    --datasets 0,1 3,5 4,9 5,8 --seeds 0 1 2 3 4 --samples 400 --epochs 30
# -> Results/experiments/summary.csv  (one row per dataset × config, mean ± std)
```

Ablation toggles (set in `QCNN/config/Qconfig.py` or via the runner):

| Component | Flag | Variants |
| :--- | :--- | :--- |
| Pooling | `pooling_mode` | `unitary` (proposed) · `none` · `measurement` |
| Conv entanglement | `conv_entanglement` | `full` · `one_diagonal` · `none` |
| Kernel rotations | `kernel_rotations` | `su2` (proposed) · `ry` |
| Encoding | `encoding_type` | `amplitude` · `feature_map` |

### Quantum-architecture baselines
To compare the proposed FQCNN against *existing* quantum architectures (not only
classical models), the harness also trains three well-cited quantum classifiers
on the **identical** amplitude-encoded representation, split, seed, optimiser and
epoch budget (`baselines/quantum_baselines.py`). They appear automatically as
extra rows in `summary.csv` (`baseline_cong`, `baseline_hur`, `baseline_ttn`),
each reporting its own trainable-parameter count for a like-for-like comparison:

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