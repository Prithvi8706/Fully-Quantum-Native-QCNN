# FQCNN: Fully Quantum-Native QCNN 
### A Fully Quantum-Native Convolutional Neural Network with Coherent Unitary Pooling

This repository implements **FQCNN**, a **fully quantum-native convolutional neural network (QCNN)** that performs data encoding, local convolution, coherence-preserving unitary pooling, and classification **entirely using unitary quantum operations, with no intermediate measurements**. Unlike hybrid models, this architecture contains no classical convolutional layers at any stage.

The project has evolved from a small 4x4 prototype to a robust framework capable of classifying **high-resolution images (e.g., 28x28 MNIST)** with up to **98.86% accuracy** (and **99.30% ± 0.64%** in multi-seed evaluations) using advanced encoding strategies like **Amplitude Embedding** and **Patch-based Quanvolution**.

---

## Key Features

- **100% Quantum-Native**: Convolution, Pooling, and Classification are all implemented as differentiable quantum circuits.
- **High Performance**: Achieving **98.86% accuracy** (F1: **98.86%**, ROC-AUC: **0.9991**) on MNIST (0 vs 1) using 100% quantum-native operations.
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

### 1. Latest Verified Standard Run (MNIST 0 vs 1)
Full evaluation metrics from the primary 50-epoch training run (`Results/metrics.json`):

| Metric | Score / Value | Details |
| :--- | :---: | :--- |
| **Accuracy** | **98.86%** | 10 Qubits, Amplitude Encoding, 50 Epochs |
| **Precision** | **99.18%** | True Positives: 1,689 / False Positives: 14 |
| **Recall** | **98.54%** | True Positives: 1,689 / False Negatives: 25 |
| **F1 Score** | **98.86%** | Harmonic mean of Precision and Recall |
| **ROC-AUC** | **0.9991** | Area Under Receiver Operating Characteristic Curve |
| **PR-AUC** | **0.9992** | Area Under Precision-Recall Curve |

*Confusion Matrix: **TP: 1,689** \| **TN: 1,701** \| **FP: 14** \| **FN: 25** (3,429 test samples)*

---

### 2. Multi-Seed Study Across Hard MNIST Pairs
Evaluated across 5 independent random seeds per pair (**Mean ± Std**, `Results/experiments/summary.csv`):

| Digit Pair | Encoding | Qubits | Accuracy (Mean ± Std) | F1 Score (Mean ± Std) | ROC-AUC (Mean ± Std) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MNIST 0 vs 1** | Amplitude | 10 | **99.30% ± 0.64%** | **99.31% ± 0.63%** | **0.9994 ± 0.0006** | Verified (5 Seeds) |
| **MNIST 3 vs 5** | Amplitude | 10 | **93.37% ± 3.15%** | **93.12% ± 3.55%** | **0.9804 ± 0.0109** | Verified (5 Seeds) |
| **MNIST 4 vs 9** | Amplitude | 10 | **90.81% ± 2.07%** | **91.09% ± 1.84%** | **0.9637 ± 0.0111** | Verified (5 Seeds) |
| **MNIST 5 vs 8** | Amplitude | 10 | **90.58% ± 5.12%** | **90.88% ± 4.91%** | **0.9639 ± 0.0309** | Verified (5 Seeds) |

---

### 3. Architecture Comparison Against Published Quantum & Classical Baselines
Trained on identical dataset splits, seeds, and optimization budget on MNIST (0 vs 1):

| Model / Architecture | Type | Accuracy (Mean ± Std) | Reference / Description |
| :--- | :--- | :---: | :--- |
| **FQCNN (Proposed)** | Quantum-Native | **99.30% ± 0.64%** | Coherent Unitary Pooling & SU(2) Conv |
| **Canonical QCNN** | Quantum Baseline | **98.26% ± 0.82%** | Cong, Choi & Lukin (*Nature Physics*, 2019) |
| **Tree Tensor Network (TTN)** | Quantum Baseline | **98.14% ± 1.12%** | Grant et al. (*npj Quantum Info*, 2018) |
| **Expressive QCNN** | Quantum Baseline | **94.88% ± 3.99%** | Hur, Kim & Park (*EPJ Quantum Tech*, 2022) |
| **Logistic Regression** | Classical Baseline | **99.53% ± 0.26%** | Linear Baseline on Amplitude Features |
| **MLP Classifier** | Classical Baseline | **99.07% ± 0.52%** | 2-Layer Neural Network |

*All runs trained using Adam optimizer with EMA, LR Warmup, and Gradient Clipping.*

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

All experiments are seed-controlled and reproducible. Use the exact pinned
environment for matching numbers:

```bash
pip install -r requirements-lock.txt     # exact tested versions (Python 3.14)
# or, for a looser install:
pip install -r requirements.txt
```

Regenerate everything with one script:

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
Two free, local noise models — uniform depolarizing and a calibrated,
IBM-hardware-like multi-channel model (depolarizing + amplitude/phase damping +
readout error):

```bash
python noise_sim.py --noise-model depolarizing --classes 0 1
python noise_sim.py --noise-model realistic    --classes 0 1
```

### Optional: real quantum hardware
`experiments/hardware_run.py` runs the trained circuit on IBM Quantum's free
Open Plan. It needs `pip install pennylane-qiskit qiskit-ibm-runtime` and a free
token in `IBM_QUANTUM_TOKEN`; it skips cleanly if either is missing.

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