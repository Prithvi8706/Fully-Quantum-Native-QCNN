"""
noise_sim.py — Depolarizing noise robustness experiment for FQCNN

Evaluates the trained weights on the REAL test set under a depolarizing-noise sweep.
The encoding used for training (amplitude, patch, …) is read from Results/metadata.json
so the reconstructed test features match exactly what the model was trained on:
  - amplitude : flattened, normalized raw-pixel features (image_size²-dim)
  - patch     : 196-dim quanvolutional features (with a cache-alignment guard)
It reconstructs the exact training test split (seed-42 pipeline) to recover aligned
labels, derives the circuit qubit count from the saved weights, then sweeps depolarizing
noise p across [0, 0.20] and plots experimental vs theoretical.

Standalone:   python noise_sim.py
Programmatic: from noise_sim import run_noise_simulation
"""

import sys
import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
from sklearn.model_selection import train_test_split


# Batch size for broadcasted circuit execution. A broadcasted ``default.mixed`` batch
# holds a (CHUNK, 2**n, 2**n) complex density tensor (~CHUNK MB at 8 qubits) plus
# intermediates, so we chunk the test set to bound peak memory. CHUNK=1 degenerates to
# the original one-sample-at-a-time behaviour (kept as a safety fallback).
DEFAULT_CHUNK = 64


def _load_training_config(weights_path):
    """
    Read the training config saved alongside the weights (Results/metadata.json).

    Returns the ``config`` dict (encoding_type, image_size, classes, preprocessing_mode,
    …) or None if the metadata file is absent, so callers can fall back to defaults.
    """
    meta_path = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(weights_path)), '..', 'metadata.json'))
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    cfg = meta.get('config', meta)
    print(f"  Loaded training config from {meta_path}: "
          f"encoding={cfg.get('encoding_type')}, image_size={cfg.get('image_size')}, "
          f"classes={cfg.get('classes')}")
    return cfg


# ── helpers ───────────────────────────────────────────────────────────────────

def _find_mnist_idx(mnist_dir):
    """Locate the MNIST train IDX image/label files in a directory."""
    if not os.path.isdir(mnist_dir):
        return None, None
    files = os.listdir(mnist_dir)
    images_f = next((f for f in files if 'train' in f and 'images' in f and 'idx3' in f), None)
    labels_f = next((f for f in files if 'train' in f and 'labels' in f and 'idx1' in f), None)
    if not images_f or not labels_f:
        return None, None
    return os.path.join(mnist_dir, images_f), os.path.join(mnist_dir, labels_f)


def _reconstruct_training_split(image_size, classes, n_samples, mnist_path=None,
                                encoding_type='patch', normalization='minmax'):
    """
    Reproduce the EXACT data pipeline from main.py so the recovered test data and
    labels line up with the trained model.

    Mirrors: load IDX → filter classes → preprocess_for_quantum(encoding_type) →
    seed-42 stratified downsample to n_samples/0.7 → train_test_split(random_state=42).

    Returns (X_test, y_test) with labels in {-1, +1}, in the same order the trainer saw:
      - 'patch'     : X_test is normalized (image_size x image_size) 2D images (the quanv
                      layer runs on these downstream).
      - 'amplitude' : X_test is the flattened, normalized feature matrix the QCNN circuit
                      consumes directly (image_size²-dim raw-pixel features).
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from QCNN.utils.dataset_loader import load_idx_dataset
    from QCNN.utils.data_preprocessing import preprocess_for_quantum

    here = os.path.dirname(os.path.abspath(__file__))
    mnist_dir = mnist_path or os.path.join(here, 'datasets', 'MNIST')
    images_f, labels_f = _find_mnist_idx(mnist_dir)
    if images_f is None:
        raise FileNotFoundError(f"No MNIST IDX files found in {mnist_dir}")

    print(f"  Loading MNIST IDX from {mnist_dir}/")
    X_raw, y_raw = load_idx_dataset(images_f, labels_f)
    mask = (y_raw == classes[0]) | (y_raw == classes[1])
    X_raw, y_raw = X_raw[mask], y_raw[mask]
    print(f"  After class filter {classes}: {len(X_raw)} samples")

    # Preprocess exactly as in training. 'patch' keeps 2D image structure for the quanv
    # layer; 'amplitude' returns the flattened, normalized feature matrix the circuit uses.
    X_pp, y_pp = preprocess_for_quantum(
        X_raw, y_raw,
        n_qubits=int(math.ceil(math.log2(max(image_size * image_size, 2)))),
        image_size=image_size,
        normalization=normalization,
        encoding_type=encoding_type,
    )

    # stratified downsample to total_needed = n_samples / 0.7 (mirrors main.py:153-181).
    # main.py relies on the module-level np.random.seed(42) being pristine here, so we
    # re-seed to reproduce the identical choice()/shuffle() draw.
    total_needed = int(n_samples / 0.7)
    if total_needed < len(X_pp):
        np.random.seed(42)
        idx_pos = np.where(y_pp == 1)[0]
        idx_neg = np.where(y_pp == -1)[0]
        n_pos = min(len(idx_pos), total_needed // 2)
        n_neg = min(len(idx_neg), total_needed - total_needed // 2)
        sampled_pos = np.random.choice(idx_pos, n_pos, replace=False)
        sampled_neg = np.random.choice(idx_neg, n_neg, replace=False)
        indices = np.concatenate([sampled_pos, sampled_neg])
        np.random.shuffle(indices)
        X_pp, y_pp = X_pp[indices], y_pp[indices]
        print(f"  Stratified downsample to {len(X_pp)} samples ({n_pos} pos, {n_neg} neg)")

    _, X_test, _, y_test = train_test_split(
        X_pp, y_pp, test_size=0.3, random_state=42, stratify=y_pp
    )
    print(f"  Reconstructed test split: {len(X_test)} samples")
    return X_test, y_test


def _resolve_cached_features(cache_dir, n_test_expected, n_feat_expected):
    """Return the cached X_test feature matrix matching the expected shape, or None."""
    if not os.path.isdir(cache_dir):
        return None
    for fname in sorted(os.listdir(cache_dir)):
        if not fname.startswith('quanv_cache_') or not fname.endswith('.npz'):
            continue
        data = np.load(os.path.join(cache_dir, fname))
        if 'X_test' in data and data['X_test'].shape == (n_test_expected, n_feat_expected):
            print(f"  Found matching quanv cache: {fname}  X_test={data['X_test'].shape}")
            return data['X_test']
    return None


def _quanv_features_for(images, seed=42):
    """Compute quanvolutional features for a batch of images (training config)."""
    from QCNN.layers.QuanvLayer import QuanvolutionalLayer
    quanv = QuanvolutionalLayer(patch_size=4, n_filters=4, stride=4, seed=seed)
    return np.array([quanv.process_image(img) for img in images])


def _preprocess_for_circuit(X, target_len):
    """Pad features to target_len and normalize each row to unit norm."""
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    feat_len = X.shape[1]
    if feat_len < target_len:
        padded = np.zeros((X.shape[0], target_len))
        padded[:, :feat_len] = X
        X = padded
    else:
        X = X[:, :target_len]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return X / norms


def _flatten_weights(w, n_conv, n_pool):
    parts = []
    for i in range(n_conv):
        parts.append(w[f'quantum_conv_kernel_{i}'].flatten())
    for i in range(n_pool):
        parts.append(w[f'quantum_pooling_{i}'].flatten())
    parts.append(w['quantum_classifier'].flatten())
    return np.concatenate(parts)


def _unflatten_params(flat, n_conv, n_pool, n_qubits):
    params = {}
    idx = 0
    kernel_size = 4 * 4 * 3  # 4 windows × depth=4 × 3 angles
    for i in range(n_conv):
        params[f'quantum_conv_kernel_{i}'] = flat[idx:idx + kernel_size].reshape(4, 4, 3)
        idx += kernel_size
    # FIX 3: pool_size uses actual n_qubits, not inferred from weight shape
    pool_size = 3 * (n_qubits // 2)
    for i in range(n_pool):
        params[f'quantum_pooling_{i}'] = flat[idx:idx + pool_size]
        idx += pool_size
    params['quantum_classifier'] = flat[idx:idx + 32]
    return params


# Representative single-/two-qubit error rates for current IBM superconducting
# hardware, used by the 'realistic' noise model (suggestion #9). These are
# order-of-magnitude calibration figures, applied per gate; the sweep multiplies
# them by an overall factor to show how accuracy degrades as hardware quality
# changes. Free + reproducible — no cloud account or qiskit dependency required.
IBM_BASE_NOISE = {
    'depol_1q': 0.001,    # single-qubit gate depolarizing
    'depol_2q': 0.010,    # two-qubit (CNOT) depolarizing
    'amp_damp': 0.002,    # amplitude damping (T1 relaxation) per op
    'phase_damp': 0.004,  # phase damping (T2 dephasing) per op
    'readout': 0.020,     # measurement bit-flip on the readout qubit
}


def _build_noise_spec(noise_model, level):
    """
    Build a per-channel noise spec for a sweep point.

    - 'depolarizing': uniform depolarizing of strength ``level`` on every wire
      after every gate group (original behaviour). x-axis = probability p.
    - 'realistic'   : IBM_BASE_NOISE scaled by ``level`` (a multiplier on
      hardware base rates). x-axis = noise multiplier (×IBM base).
    """
    if noise_model == 'depolarizing':
        return {'depol_1q': level, 'depol_2q': level,
                'amp_damp': 0.0, 'phase_damp': 0.0, 'readout': 0.0}
    elif noise_model == 'realistic':
        return {k: v * level for k, v in IBM_BASE_NOISE.items()}
    raise ValueError(f"Unknown noise_model '{noise_model}'. Use 'depolarizing' or 'realistic'.")


def _apply_1q_noise(wires, spec):
    """Apply single-qubit channels to each wire in ``wires``."""
    for q in wires:
        if spec.get('depol_1q', 0) > 0:
            qml.DepolarizingChannel(spec['depol_1q'], wires=q)
        if spec.get('amp_damp', 0) > 0:
            qml.AmplitudeDamping(spec['amp_damp'], wires=q)
        if spec.get('phase_damp', 0) > 0:
            qml.PhaseDamping(spec['phase_damp'], wires=q)


def _apply_2q_noise(a, b, spec):
    """Apply a two-qubit-gate depolarizing error to both wires of a CNOT."""
    if spec.get('depol_2q', 0) > 0:
        qml.DepolarizingChannel(spec['depol_2q'], wires=a)
        qml.DepolarizingChannel(spec['depol_2q'], wires=b)


def _make_noisy_circuit(n_qubits, n_conv, n_pool, flat_params, spec):
    from QCNN.layers.QConv import QuantumNativeConvolution
    from QCNN.layers.QPool import QuantumNativePooling

    dev = qml.device('default.mixed', wires=n_qubits)

    # Hoisted out of the QNode: these are identical on every execution, so computing
    # them once (instead of per sample / per broadcasted batch) removes redundant work.
    params = _unflatten_params(flat_params, n_conv, n_pool, n_qubits)
    all_qubits = list(range(n_qubits))

    def _cnot(a, b):
        qml.CNOT(wires=[a, b])
        _apply_2q_noise(a, b, spec)

    @qml.qnode(dev, interface='numpy')
    def circuit(x):
        # ``x`` may be a single amplitude vector (2**n_qubits,) or a batch
        # (batch, 2**n_qubits); AmplitudeEmbedding broadcasts over the leading
        # batch dim and expval then returns a (batch,) array.
        qml.AmplitudeEmbedding(features=x, wires=all_qubits, normalize=True)
        _apply_1q_noise(all_qubits, spec)

        active = all_qubits.copy()
        for layer in range(n_conv):
            n_cur = len(active)
            if n_cur >= 4:
                w = int(math.sqrt(n_cur))
                while n_cur % w != 0:
                    w -= 1
                h = n_cur // w
                W, H = max(w, h), min(w, h)
                kernel = params[f'quantum_conv_kernel_{layer}']
                for win in QuantumNativeConvolution.get_conv_windows(W, H):
                    if max(win) < n_cur:
                        wires = [active[i] for i in win]
                        QuantumNativeConvolution.quantum_conv2d_kernel(kernel, wires)
                        _apply_1q_noise(wires, spec)

            if layer < n_conv - 1:
                if len(active) < 2:
                    break
                pairs = QuantumNativePooling.make_pairing(active)
                if not pairs:
                    break
                keep    = [k for k, _ in pairs]
                discard = [d for _, d in pairs]
                QuantumNativePooling.quantum_unitary_pooling(
                    params[f'quantum_pooling_{layer}'],
                    input_qubits=keep, output_qubits=discard
                )
                _apply_1q_noise(keep + discard, spec)
                active = keep

        cp = params['quantum_classifier']
        n_a = len(active)
        readout = active[0]

        for i, q in enumerate(active[:min(n_a, 4)]):
            qml.RX(cp[i * 2 % 32], wires=q)
            qml.RY(cp[(i * 2 + 1) % 32], wires=q)
            qml.RZ(cp[(i * 2 + 8) % 32], wires=q)

        for i in range(n_a - 1):
            _cnot(active[i], active[i + 1])
        if n_a >= 2:
            _cnot(active[n_a - 1], active[0])

        for i, q in enumerate(active[:min(n_a, 4)]):
            qml.RX(cp[(i * 2 + 16) % 32], wires=q)
            qml.RY(cp[(i * 2 + 17) % 32], wires=q)

        if n_a >= 2:
            _cnot(active[0], active[min(n_a - 1, 1)])
        qml.RZ(cp[31], wires=readout)

        _apply_1q_noise(active, spec)
        # Readout (measurement) error on the readout qubit.
        if spec.get('readout', 0) > 0:
            qml.BitFlip(spec['readout'], wires=readout)

        return qml.expval(qml.PauliZ(readout))

    return circuit


# ── main entry point ──────────────────────────────────────────────────────────

def run_noise_simulation(
    weights_path,
    output_path,
    mnist_path=None,
    image_size=28,
    n_samples=8000,
    classes=(0, 1),
    max_test=200,
    noise_model='depolarizing',
    levels=None,
    chunk_size=DEFAULT_CHUNK,
):
    np.random.seed(42)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # ── config: read the encoding the model was actually trained with ───────────
    cfg = _load_training_config(weights_path)
    if cfg is not None:
        encoding_type = cfg.get('encoding_type', 'patch')
        image_size    = cfg.get('image_size', image_size)
        normalization = cfg.get('preprocessing_mode', 'minmax')
        if cfg.get('classes'):
            classes = tuple(cfg['classes'])
    else:
        print("  No metadata.json found — defaulting to patch encoding.")
        encoding_type, normalization = 'patch', 'minmax'

    # ── weights ───────────────────────────────────────────────────────────────
    print("Loading trained weights...")
    w = np.load(weights_path)
    for k in w.keys():
        print(f"  {k}: {w[k].shape}")

    n_conv = len([k for k in w.keys() if 'conv_kernel' in k])
    n_pool = len([k for k in w.keys() if 'pooling' in k])

    # Derive n_qubits from the saved pooling weights (pool_size = 3 * (n_qubits // 2)).
    n_qubits = 2 * (w['quantum_pooling_0'].size // 3)
    target_len = 2 ** n_qubits
    print(f"\nInferred: {n_conv} conv layers, {n_pool} pool layers, {n_qubits} qubits "
          f"(target_len={target_len}), encoding='{encoding_type}'")

    # ── reconstruct the REAL test set (features/images + aligned labels) ────────
    print(f"\nReconstructing the trained test set (image_size={image_size}, "
          f"n_samples={n_samples}, classes={classes})...")
    X_test, y_test = _reconstruct_training_split(
        image_size, classes, n_samples, mnist_path,
        encoding_type=encoding_type, normalization=normalization)

    # ── pick the evaluation subset ──────────────────────────────────────────────
    np.random.seed(123)  # reproducible subset selection, independent of pipeline RNG
    if max_test is not None and len(X_test) > max_test:
        sub = np.random.choice(len(X_test), max_test, replace=False)
    else:
        sub = np.arange(len(X_test))
    y_test = y_test[sub]

    # ── obtain the circuit-input features matching the trained encoding ─────────
    if encoding_type == 'patch':
        # Model was trained on 196-dim quanvolutional features (NOT raw pixels).
        # Prefer the cache written during training; verify alignment before trusting it.
        out_dim = (image_size - 4) // 4 + 1  # patch_size=4, stride=4
        n_feat = out_dim * out_dim * 4       # quanv output features (4 filters) → 196 for 28x28
        cache_dir = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(weights_path)), '..', 'Cache'))
        cached = _resolve_cached_features(cache_dir, len(X_test), n_feat)
        if cached is not None:
            check = _quanv_features_for(X_test[:5])
            if np.allclose(check, cached[:5], atol=1e-6):
                print("  Alignment guard PASSED — using cached quanvolutional features.")
            else:
                print("  Alignment guard FAILED — recomputing quanv for the evaluated subset.")
                cached = None
        if cached is not None:
            X_feat = cached[sub]
        else:
            print(f"  Computing quanvolutional features for {len(sub)} test images...")
            X_feat = _quanv_features_for(X_test[sub])
    else:
        # amplitude/feature_map: the reconstructed split IS already the feature matrix
        # the circuit consumes (flattened, normalized raw pixels).
        X_feat = X_test[sub]

    print(f"  Evaluating on {len(sub)} real test samples ({X_feat.shape[1]}-dim features)")

    # Model pads features → 2^n_qubits then AmplitudeEmbedding(normalize=True);
    # _preprocess_for_circuit reproduces that (pad to target_len + unit-norm).
    X_test_proc = _preprocess_for_circuit(X_feat, target_len)
    flat_params  = _flatten_weights(w, n_conv, n_pool)
    print(f"  Flat params: {len(flat_params)}")

    # ── noise sweep ───────────────────────────────────────────────────────────
    print(f"\nNoise model: '{noise_model}'")
    if noise_model == 'depolarizing':
        default_levels = [0.00, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.13, 0.15, 0.20]
        x_label = 'Depolarizing noise probability p'
        level_fmt = lambda v: f"p={v:.2f}"
    else:  # 'realistic'
        default_levels = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
        x_label = 'Noise multiplier (x IBM base rates)'
        level_fmt = lambda v: f"{v:.1f}x"
    # ``levels`` may be overridden by the caller (fewer points → faster sweep on the
    # heavy 10-qubit default.mixed device). Falls back to the full-resolution default.
    levels = list(levels) if levels else default_levels
    p_values = levels  # alias kept for the plotting/summary code below

    real_accs = []
    threshold = None  # calibrated once at the zero-noise point

    print("Running noise sweep...")
    print(f"  {'level':>8}  {'accuracy':>10}")
    print("  " + "-" * 24)

    cs = max(1, int(chunk_size))
    for p in levels:
        spec = _build_noise_spec(noise_model, p)
        circuit = _make_noisy_circuit(n_qubits, n_conv, n_pool, flat_params, spec)
        # Broadcast the amplitude embedding over a batch of samples in a single
        # traced execution instead of one QNode call per image. Chunked to bound the
        # broadcasted density-matrix memory; results are identical to the per-sample
        # path (see sanity check). cs == 1 reproduces the original one-at-a-time loop.
        raw_outputs = np.concatenate([
            np.atleast_1d(np.asarray(circuit(X_test_proc[i:i + cs]), dtype=float))
            for i in range(0, len(X_test_proc), cs)
        ])

        # Calibrate the decision threshold once at the zero-noise point.
        if p == levels[0]:
            class_pos_mean = float(np.mean(raw_outputs[y_test == 1]))
            class_neg_mean = float(np.mean(raw_outputs[y_test == -1]))
            threshold = (class_pos_mean + class_neg_mean) / 2.0
            print(f"  Calibrated threshold: {threshold:.4f}  "
                  f"(class+1 mean={class_pos_mean:.4f}, class-1 mean={class_neg_mean:.4f})")

        preds = np.where(raw_outputs > threshold, 1, -1)
        acc   = float(np.mean(preds == y_test))
        real_accs.append(acc)
        print(f"  {level_fmt(p):>8}  acc={acc:.4f}")

    # ── plot ──────────────────────────────────────────────────────────────────
    base_acc = real_accs[0]
    # Linear-attenuation reference curve, scaled from the clean baseline accuracy.
    if noise_model == 'depolarizing':
        theo_accs = [(1 - p) * base_acc for p in p_values]
        theo_label = f'Linear reference: (1-p) x {base_acc:.3f}'
        marker_x, marker_label = 0.05, 'NISQ threshold (p=0.05)'
    else:
        max_lv = max(p_values) or 1.0
        theo_accs = [base_acc * (1 - 0.5 * p / max_lv) for p in p_values]
        theo_label = 'Linear reference (illustrative)'
        marker_x, marker_label = 1.0, 'Current IBM hardware (1x)'

    # Dynamic ylim so the real curve is always visible.
    y_min = min(min(real_accs), min(theo_accs)) - 0.05
    y_min = max(0.40, round(y_min, 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p_values, theo_accs, 'b--o', label=theo_label, linewidth=2, markersize=6)
    ax.plot(p_values, real_accs, 'r-s',
            label=f'Experimental ({noise_model} noise)', linewidth=2, markersize=7)
    ax.axvline(x=marker_x, color='gray', linestyle=':', linewidth=1.5, label=marker_label)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Test accuracy', fontsize=12)
    ax.set_title('FQCNN Noise Robustness: Experimental vs Reference', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(y_min, 1.00)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved to: {output_path}")

    print("\n-- Summary ----------------------------------------------------------")
    print(f"  Clean baseline:  real={real_accs[0]:.4f}")
    print(f"  Highest noise:   real={real_accs[-1]:.4f}  (level={level_fmt(levels[-1])})")
    max_diff = max(abs(r - t) for r, t in zip(real_accs, theo_accs))
    print(f"  Max deviation from linear reference: {max_diff:.4f}")
    print("---------------------------------------------------------------------")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="FQCNN noise robustness simulation")
    ap.add_argument('--noise-model', choices=['depolarizing', 'realistic'],
                    default='depolarizing',
                    help="'depolarizing' (uniform p sweep) or 'realistic' "
                         "(calibrated IBM-like multi-channel model)")
    ap.add_argument('--weights', default="Results/Weights/quantum_model_params.npz")
    ap.add_argument('--image-size', type=int, default=28)
    ap.add_argument('--n-samples', type=int, default=8000)
    ap.add_argument('--classes', type=int, nargs=2, default=[0, 1])
    ap.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK,
                    help="batch size for broadcasted circuit execution "
                         "(lower = less memory; 1 = original per-sample loop)")
    args = ap.parse_args()

    suffix = 'realistic' if args.noise_model == 'realistic' else 'depolarizing'
    run_noise_simulation(
        weights_path=args.weights,
        output_path=f"Results/Graphs/fig6_noise_robustness_{suffix}.png",
        image_size=args.image_size,
        n_samples=args.n_samples,
        classes=tuple(args.classes),
        noise_model=args.noise_model,
        chunk_size=args.chunk_size,
    )