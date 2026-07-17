# Classical baselines for fair comparison with the QCNN (suggestions #1, #10).
#
# Two dependency-free (sklearn-only) baselines that train and evaluate on the
# EXACT same preprocessed representation and train/test split as the QCNN:
#   - LogisticRegression : a linear floor.
#   - MLPClassifier      : a small non-linear net whose hidden width can be set
#                          to roughly match the QCNN's trainable-parameter count.
# Both return the same metric dict as the QCNN via the shared metrics module, so
# every row of the comparison table is computed identically.
#
# The original TensorFlow CNN is kept as an OPTIONAL path; its import is now lazy
# so this module loads even when TensorFlow is not installed.

import numpy as np

from QCNN.utils.metrics import compute_classification_metrics


def _flatten(X: np.ndarray) -> np.ndarray:
    """Flatten any (N, ...) array to (N, features) for classical models."""
    X = np.asarray(X)
    return X.reshape(X.shape[0], -1)


def _metrics_from_proba(y_true_pm1: np.ndarray, proba_pos: np.ndarray) -> dict:
    """
    Convert a positive-class probability in [0, 1] to a centred continuous score
    (raw = 2p - 1, so the threshold at 0 matches the QCNN convention) and run it
    through the shared metric suite.
    """
    raw = 2.0 * np.asarray(proba_pos, dtype=float) - 1.0
    return compute_classification_metrics(np.asarray(y_true_pm1), raw)


def run_logistic_baseline(X_train, y_train, X_test, y_test, seed: int = 42) -> dict:
    """Logistic-regression baseline on the shared representation."""
    from sklearn.linear_model import LogisticRegression

    Xtr, Xte = _flatten(X_train), _flatten(X_test)
    ytr01 = np.where(np.asarray(y_train) == 1, 1, 0)

    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(Xtr, ytr01)
    proba = clf.predict_proba(Xte)[:, 1]
    return _metrics_from_proba(y_test, proba)


def _mlp_hidden_layers(target_params: int | None = None) -> tuple:
    """
    Hidden-layer sizes for the MLP baseline.

    NOTE: matching a classical net's parameter count to the QCNN's (~260) is
    meaningless here — on a ~256-dim input the first-layer weights alone dwarf that
    budget, so param-matching forces a ~2-neuron net that cannot learn (collapses to
    ~0.5). Instead we use a fixed, sensible non-linear capacity and report BOTH models'
    parameter counts in the results for an honest comparison (a classical net having
    more parameters than a quantum circuit is an expected, disclosed tradeoff).
    ``target_params`` is accepted for backwards-compatibility but no longer caps width.
    """
    return (128, 64)


def run_mlp_baseline(X_train, y_train, X_test, y_test, seed: int = 42,
                     target_params: int | None = None) -> dict:
    """Non-linear MLP baseline on the shared representation (standardized input)."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    Xtr, Xte = _flatten(X_train), _flatten(X_test)
    ytr01 = np.where(np.asarray(y_train) == 1, 1, 0)

    hidden = _mlp_hidden_layers(target_params)
    # StandardScaler + a real 2-layer net, trained to convergence (no early_stopping,
    # which on a hold-out split was stopping the collapsed net at its constant init).
    clf = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=hidden, max_iter=1000, alpha=1e-4,
                      random_state=seed, early_stopping=False),
    )
    clf.fit(Xtr, ytr01)
    proba = clf.predict_proba(Xte)[:, 1]
    metrics = _metrics_from_proba(y_test, proba)
    metrics["hidden_layer_sizes"] = list(hidden)
    # Report the MLP's own parameter count for transparency in the comparison table.
    mlp = clf.named_steps["mlpclassifier"]
    metrics["n_params"] = int(sum(c.size for c in mlp.coefs_)
                              + sum(b.size for b in mlp.intercepts_))
    return metrics


def run_classical_baselines(X_train, y_train, X_test, y_test, seed: int = 42,
                            target_params: int | None = None) -> dict:
    """
    Run all classical baselines on one shared split and return
    {baseline_name: metrics_dict}. This is what the experiment runner calls so
    classical and quantum numbers come from identical data (fair benchmarking).
    """
    return {
        "logistic": run_logistic_baseline(X_train, y_train, X_test, y_test, seed),
        "mlp": run_mlp_baseline(X_train, y_train, X_test, y_test, seed, target_params),
    }


# ----------------------------------------------------------------------------
# Optional TensorFlow CNN baseline (lazy import — only needed if explicitly used)
# ----------------------------------------------------------------------------
def build_baseline_cnn(input_shape=(4, 4, 1)):
    """Builds a small CNN for image-shaped inputs. Requires TensorFlow."""
    try:
        from tensorflow.keras import layers, models
    except ImportError as e:
        raise ImportError("TensorFlow not installed. Install via: pip install tensorflow") from e

    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(4, kernel_size=2, activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(4, activation='relu'))
    model.add(layers.Dense(1, activation='tanh'))  # match QCNN output range
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model


def train_baseline_cnn(X_train, y_train, X_test, y_test, epochs=40, image_size=None):
    """Trains the optional TensorFlow CNN baseline on image-shaped inputs."""
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    if image_size is None:
        # Infer square side from feature count if inputs are flat.
        feats = X_train.reshape(X_train.shape[0], -1).shape[1]
        image_size = int(round(feats ** 0.5))
    X_train_cnn = X_train.reshape(-1, image_size, image_size, 1)
    X_test_cnn = X_test.reshape(-1, image_size, image_size, 1)

    model = build_baseline_cnn(input_shape=(image_size, image_size, 1))
    history = model.fit(
        X_train_cnn, y_train,
        validation_data=(X_test_cnn, y_test),
        epochs=epochs, batch_size=16, verbose=0,
    )
    return model, history
