# Quantum-architecture baselines for rigorous comparison with the proposed FQCNN
# (reviewer suggestion #1; also strengthens #10 fair benchmarking).
#
# These are faithful re-implementations of three well-cited *quantum* classifiers,
# trained and evaluated on the EXACT same amplitude-encoded representation,
# train/test split, seed, optimiser family and epoch budget as the proposed model
# (the runner passes cfg.n_qubits / cfg.n_epochs / cfg.learning_rate through). So
# every row of the comparison table is produced under matched conditions and the
# comparison is architecture-vs-architecture rather than quantum-vs-classical:
#
#   - "cong" : Cong, Choi & Lukin, Nature Physics 15, 1273 (2019). The canonical
#              QCNN: translationally-invariant two-qubit convolution unitaries +
#              pooling that funnels information into half the qubits each stage
#              (8->4->2->1), read out as <Z> on the surviving qubit.
#   - "hur"  : Hur, Kim & Park, "Quantum convolutional neural network for image
#              classification" (2022). Same QCNN skeleton, but a distinct, more
#              expressive parameterised two-qubit convolution block.
#   - "ttn"  : Grant et al., npj Quantum Information 4, 65 (2018). A balanced tree
#              tensor network of two-qubit unitaries (one survivor per node, no
#              weight sharing), read out at the root.
#
# All three reuse PureQuantumEncoder.amplitude_encoding (identical state prep to
# the proposed model) and QCNN.utils.metrics.compute_classification_metrics
# (identical metric code / <Z>-threshold-0 decision convention).

from __future__ import annotations

import math

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp

from QCNN.encoding.QEncoder import PureQuantumEncoder
from QCNN.utils.metrics import compute_classification_metrics
from QCNN.utils.run_artifacts import TestEvaluationGuard


# ---------------------------------------------------------------------------
# Parameter cursor: hands out sequential slices of a flat param vector while
# preserving autograd connectivity (plain slicing of a pennylane array does).
# ---------------------------------------------------------------------------
class _Cursor:
    def __init__(self, params):
        self.p = params
        self.i = 0

    def take(self, n: int):
        s = self.p[self.i:self.i + n]
        self.i += n
        return s


def _amp_encode(x, n_qubits: int) -> None:
    """Identical state prep to the proposed model (handles pad/truncate/L2-norm)."""
    PureQuantumEncoder.amplitude_encoding(x, wires=list(range(n_qubits)))


def _pool_block(p, keep: int, discard: int) -> None:
    """Coherent pooling: funnel `discard` into `keep` via controlled rotations.

    A unitary stand-in for the measurement+feedforward pooling of Cong et al.,
    as used in the standard PennyLane QCNN treatment. Consumes 3 params.
    """
    qml.CRZ(p[0], wires=[discard, keep])
    qml.CRX(p[1], wires=[discard, keep])
    qml.RY(p[2], wires=keep)


# ---- convolution blocks (shared params per layer => translational invariance) ----
def _conv_block_cong(p, a: int, b: int) -> None:
    """Cong-style two-qubit conv unitary. 6 params."""
    qml.RX(p[0], wires=a); qml.RY(p[1], wires=a); qml.RZ(p[2], wires=a)
    qml.RX(p[3], wires=b); qml.RY(p[4], wires=b); qml.RZ(p[5], wires=b)
    qml.CNOT(wires=[a, b])


def _conv_block_hur(p, a: int, b: int) -> None:
    """Hur-style two-qubit conv unitary (more expressive, both CNOT directions). 8 params."""
    qml.RY(p[0], wires=a); qml.RZ(p[1], wires=a)
    qml.RY(p[2], wires=b); qml.RZ(p[3], wires=b)
    qml.CNOT(wires=[a, b])
    qml.RY(p[4], wires=a); qml.RY(p[5], wires=b)
    qml.CNOT(wires=[b, a])
    qml.RZ(p[6], wires=a); qml.RZ(p[7], wires=b)


def _ttn_node(p, keep: int, discard: int) -> None:
    """TTN two-qubit node; information consolidated onto `keep`. 6 params."""
    qml.RY(p[0], wires=keep); qml.RY(p[1], wires=discard)
    qml.CNOT(wires=[keep, discard])
    qml.RY(p[2], wires=keep); qml.RY(p[3], wires=discard)
    qml.CNOT(wires=[discard, keep])
    qml.RY(p[4], wires=keep); qml.RZ(p[5], wires=keep)


def _check_pow2(n_qubits: int) -> int:
    """These hierarchical architectures assume a power-of-two qubit count."""
    if n_qubits < 2 or (n_qubits & (n_qubits - 1)) != 0:
        raise ValueError(
            f"Quantum baselines require a power-of-two qubit count, got {n_qubits}. "
            "(The proposed amplitude/feature_map configs always yield one.)")
    return int(round(math.log2(n_qubits)))


# ---------------------------------------------------------------------------
# Architectures. Each exposes param_count(n) and body(cursor, n) -> readout wire.
# ---------------------------------------------------------------------------
def _qcnn_body(cursor: _Cursor, n_qubits: int, conv_block, conv_size: int) -> int:
    """Shared QCNN skeleton: per layer, a shared conv over all neighbour pairs
    then a shared pool that halves the active wires. Returns the readout wire."""
    active = list(range(n_qubits))
    while len(active) > 1:
        conv_p = cursor.take(conv_size)          # shared across the layer
        for i in range(len(active) - 1):
            conv_block(conv_p, active[i], active[i + 1])
        pool_p = cursor.take(3)                  # shared across the layer
        kept = []
        for j in range(0, len(active) - 1, 2):
            keep, discard = active[j], active[j + 1]
            _pool_block(pool_p, keep, discard)
            kept.append(keep)
        active = kept
    return active[0]


def _qcnn_param_count(n_qubits: int, conv_size: int) -> int:
    return _check_pow2(n_qubits) * (conv_size + 3)


def _ttn_body(cursor: _Cursor, n_qubits: int) -> int:
    """Balanced tree: pair survivors, apply a node block, keep one per pair."""
    active = list(range(n_qubits))
    while len(active) > 1:
        kept = []
        for j in range(0, len(active) - 1, 2):
            keep, discard = active[j], active[j + 1]
            _ttn_node(cursor.take(6), keep, discard)
            kept.append(keep)
        active = kept
    return active[0]


def _ttn_param_count(n_qubits: int) -> int:
    _check_pow2(n_qubits)
    return (n_qubits - 1) * 6


_ARCHITECTURES = {
    "cong": {
        "param_count": lambda n: _qcnn_param_count(n, 6),
        "body": lambda cur, n: _qcnn_body(cur, n, _conv_block_cong, 6),
    },
    "hur": {
        "param_count": lambda n: _qcnn_param_count(n, 8),
        "body": lambda cur, n: _qcnn_body(cur, n, _conv_block_hur, 8),
    },
    "ttn": {
        "param_count": _ttn_param_count,
        "body": _ttn_body,
    },
}


# ---------------------------------------------------------------------------
# Shared trainer
# ---------------------------------------------------------------------------
def _make_device(n_qubits: int):
    """default.qubit supports backprop + input broadcasting, which lets the whole
    training batch be differentiated in a single statevector pass — far faster
    than per-sample circuit calls for the small qubit counts used here."""
    return qml.device("default.qubit", wires=n_qubits)


def _bce(raw, y_pm1):
    """Binary cross-entropy on p = (raw + 1) / 2 with y mapped to {0, 1}."""
    p = (raw + 1.0) / 2.0
    p = pnp.clip(p, 1e-7, 1.0 - 1e-7)
    y01 = (y_pm1 + 1.0) / 2.0
    return -pnp.mean(y01 * pnp.log(p) + (1.0 - y01) * pnp.log(1.0 - p))


def _initial_params(*, n_params, seed):
    np.random.seed(seed)
    pnp.random.seed(seed)
    return pnp.array(np.random.uniform(-np.pi / 4, np.pi / 4, size=n_params),
                     requires_grad=True)


def _train_epoch(*, opt, cost, params):
    return opt.step(cost, params)


def _validation_loss(*, circuit, X_val, y_val, params, use_bce):
    raw = circuit(pnp.asarray(np.asarray(X_val, dtype=float)), params)
    labels = pnp.asarray(np.asarray(y_val, dtype=float))
    return float(_bce(raw, labels) if use_bce else pnp.mean((raw - labels) ** 2))


def _raw_outputs(*, circuit, X_test, params):
    return np.asarray(circuit(np.asarray(X_test, dtype=float), params), dtype=float).reshape(-1)


def _train_architecture(*, arch_name: str, X_train, y_train, X_val, y_val,
                        X_test, y_test, seed: int, n_qubits: int, n_epochs: int,
                        learning_rate: float, use_bce: bool) -> dict:
    """Select one architecture checkpoint on validation and evaluate test once."""
    arch = _ARCHITECTURES[arch_name]
    n_params = arch["param_count"](n_qubits)
    dev = _make_device(n_qubits)

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(x, params):
        _amp_encode(x, n_qubits)
        return qml.expval(qml.PauliZ(arch["body"](_Cursor(params), n_qubits)))

    params = _initial_params(n_params=n_params, seed=seed)
    X_tr = pnp.asarray(np.asarray(X_train, dtype=float))
    y_tr = pnp.asarray(np.asarray(y_train, dtype=float))

    def cost(p):
        raw = circuit(X_tr, p)
        return _bce(raw, y_tr) if use_bce else pnp.mean((raw - y_tr) ** 2)

    opt = qml.AdamOptimizer(stepsize=learning_rate)
    best = None
    for epoch in range(1, max(1, n_epochs) + 1):
        params = _train_epoch(opt=opt, cost=cost, params=params)
        value = _validation_loss(circuit=circuit, X_val=X_val, y_val=y_val,
                                 params=params, use_bce=use_bce)
        if best is None or value < best[0]:
            best = (value, epoch, pnp.array(np.asarray(params), requires_grad=True))

    best_value, best_epoch, selected = best
    guard = TestEvaluationGuard()
    raw_test = guard.evaluate(_raw_outputs, circuit=circuit, X_test=X_test,
                              params=selected)
    metrics = compute_classification_metrics(np.asarray(y_test), raw_test)
    metrics.update({"n_params": int(n_params), "architecture": arch_name})
    return {
        "metrics": metrics,
        "selection": {
            "criterion": "validation_loss", "best_epoch": best_epoch,
            "best_value": best_value, "hyperparameters": {},
            "n_validation_evaluations": max(1, n_epochs),
        },
        "test_evaluations": guard.count,
        "raw_outputs": raw_test,
        "selected_parameters": np.asarray(selected, dtype=float).copy(),
    }


def run_quantum_baselines(*, X_train, y_train, X_val, y_val, X_test, y_test,
                          seed: int = 42, n_qubits: int = 8, n_epochs: int = 30,
                          learning_rate: float = 0.02, use_bce: bool = True) -> dict:
    common = dict(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                  X_test=X_test, y_test=y_test, seed=seed, n_qubits=n_qubits,
                  n_epochs=n_epochs, learning_rate=learning_rate, use_bce=use_bce)
    return {name: _train_architecture(arch_name=name, **common)
            for name in _ARCHITECTURES}


# ---------------------------------------------------------------------------
# Standalone CLI: train the quantum baselines on one MNIST pair for debugging.
# Mirrors how the harness prepares a split (see experiments/run_experiments.py).
# ---------------------------------------------------------------------------
def _main():
    import argparse
    from sklearn.model_selection import train_test_split

    from QCNN.config.Qconfig import QuantumNativeConfig
    from QCNN.utils.dataset_loader import load_dataset

    ap = argparse.ArgumentParser(description="Quantum-architecture baselines (standalone)")
    ap.add_argument("--mnist-dir", default="datasets/MNIST")
    ap.add_argument("--classes", type=int, nargs=2, default=[0, 1])
    ap.add_argument("--samples", type=int, default=120)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-mse", action="store_true")
    args = ap.parse_args()

    cfg = QuantumNativeConfig.from_image_size(16, "amplitude")
    cfg.seed = args.seed
    np.random.seed(args.seed)
    X, y = load_dataset(source=args.mnist_dir, dataset_type="idx",
                        n_qubits=cfg.n_qubits, image_size=cfg.image_size,
                        normalization=cfg.preprocessing_mode,
                        encoding_type=cfg.encoding_type, classes=tuple(args.classes))
    total = int(args.samples / 0.7)
    if total < len(X):
        idx = np.random.choice(len(X), total, replace=False)
        X, y = X[idx], y[idx]
    X_dev, X_te, y_dev, y_te = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_dev, y_dev, test_size=0.125, random_state=args.seed, stratify=y_dev)
    X_tr, y_tr = X_tr[:args.samples], y_tr[:args.samples]

    res = run_quantum_baselines(
        X_train=X_tr, y_train=y_tr, X_val=X_val, y_val=y_val,
        X_test=X_te, y_test=y_te, seed=args.seed, n_qubits=cfg.n_qubits,
        n_epochs=args.epochs, learning_rate=cfg.learning_rate,
        use_bce=not args.use_mse)
    for name, result in res.items():
        m = result["metrics"]
        print(f"{name:5s} | n_params={m['n_params']:3d} acc={m['accuracy']:.3f} "
              f"f1={m['f1']:.3f} roc_auc={m['roc_auc']:.3f}")


if __name__ == "__main__":
    _main()
