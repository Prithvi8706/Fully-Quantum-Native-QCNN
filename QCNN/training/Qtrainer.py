import pennylane as qml
import pennylane.numpy as pnp
import numpy as np
import time
import sys
import os
from QCNN.models import PureQuantumNativeCNN


class QuantumNativeTrainer:
    def __init__(self, learning_rate: float = 0.002, use_bce: bool = True):
        self.learning_rate = learning_rate
        self.use_bce = use_bce
        self.quantum_optimizer = qml.AdamOptimizer(stepsize=learning_rate)

    def save_params(self, params: dict, filepath: str):
        numpy_params = {k: np.array(v) for k, v in params.items()}
        np.savez(filepath, **numpy_params)

    def load_params(self, filepath: str):
        data = np.load(filepath)
        return {k: pnp.array(data[k], requires_grad=True) for k in data.files}

    def _bce_loss(self, logits_or_expvals, labels_pm1):
        z = pnp.clip(logits_or_expvals, -1.0, 1.0)
        p = (1.0 + z) * 0.5
        y01 = (pnp.array(labels_pm1) + 1.0) * 0.5
        eps = 1e-7
        pos_w = getattr(self, '_pos_weight', 1.0)
        neg_w = getattr(self, '_neg_weight', 1.0)
        loss_pos = y01 * pnp.log(p + eps) * pos_w
        loss_neg = (1.0 - y01) * pnp.log(1.0 - p + eps) * neg_w
        return -pnp.mean(loss_pos + loss_neg)

    def train_pure_quantum_cnn(self, model: PureQuantumNativeCNN,
                               X_train: np.ndarray, y_train: np.ndarray,
                               X_test: np.ndarray, y_test: np.ndarray,
                               log_filepath='quantum_training_log.txt',
                               summary_filepath='training_summary.txt',
                               validate_data: bool = True,
                               save_weights: bool = True) -> PureQuantumNativeCNN:

        if validate_data:
            self._validate_dataset(X_train, y_train, model)
            self._validate_dataset(X_test, y_test, model)

        print(f"DEBUG: Qtrainer enters train_pure_quantum_cnn. Encoding: {model.config.encoding_type}")
        if model.config.encoding_type == 'patch' and model.quanv_layer is not None:
            import hashlib
            cache_dir = os.path.join('Results', 'Cache')
            os.makedirs(cache_dir, exist_ok=True)
            hash_input = (
                f"{X_train.shape[1:]}_"
                f"{model.quanv_layer.patch_size}_{model.quanv_layer.stride}_"
                f"{model.quanv_layer.n_filters}"
            )
            data_signature = np.ascontiguousarray(X_train[:100]).tobytes() + \
                             np.ascontiguousarray(X_test[:100]).tobytes()
            data_hash = hashlib.md5(data_signature).hexdigest()[:12]
            cache_key = hashlib.md5((hash_input + data_hash).encode()).hexdigest()[:16]
            cache_path = os.path.join(cache_dir, f"quanv_cache_{cache_key}.npz")

            if os.path.exists(cache_path):
                print(f"\n[CACHE HIT] Loading pre-calculated Quanvolutional Features from '{cache_path}'...")
                cached = np.load(cache_path)
                X_train = cached['X_train']
                X_test = cached['X_test']
                print(f"  Features loaded. Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
            else:
                n_train = len(X_train)
                total_samples = n_train + len(X_test)
                print(f"\n[CACHE MISS] Pre-calculating Quanvolutional Features for {total_samples} total samples...")
                X_combined = np.concatenate([X_train, X_test], axis=0)
                X_processed_all = model.quanv_layer.process_batch(X_combined)
                X_train = X_processed_all[:n_train]
                X_test = X_processed_all[n_train:]
                np.savez(cache_path, X_train=X_train, X_test=X_test)
                print(f"  Pre-calculation complete. Reduced shape: {X_train.shape[1:]}")
                print(f"  Cached to '{cache_path}' for future runs.")
            model.config.encoding_type = 'amplitude'

        model._quantum_preprocessed_train = X_train
        model._quantum_preprocessed_test = X_test

        pos_count = np.sum(y_train == 1)
        neg_count = np.sum(y_train == -1)
        total = len(y_train)
        if pos_count > 0 and neg_count > 0:
            self._pos_weight = total / (2.0 * pos_count)
            self._neg_weight = total / (2.0 * neg_count)
            print(f"  Class Weights: +1 (Target): {self._pos_weight:.2f} | -1 (Rest): {self._neg_weight:.2f}")
        else:
            self._pos_weight = 1.0
            self._neg_weight = 1.0

        print("Starting Training")
        print("=" * 50)

        best_accuracy = 0
        best_quantum_params = {k: v.copy() for k, v in model.quantum_params.items()}
        params_flat = model._flatten_params(model.quantum_params)
        ema_params_flat = params_flat.copy()
        ema_decay = getattr(model.config, 'ema_decay', 0.99)
        n_epochs = model.config.n_epochs
        patience_counter = 0
        patience = getattr(model.config, 'early_stopping_patience', 3)
        lr_factor = getattr(model.config, 'lr_plateau_factor', 0.5)
        current_lr = model.config.learning_rate

        for epoch in range(n_epochs):
            epoch_start = time.time()
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            epoch_quantum_loss = 0.0
            n_quantum_batches = 0

            for i in range(0, len(X_train), model.config.batch_size):
                batch_end = min(i + model.config.batch_size, len(X_train))
                X_quantum_batch = X_shuffled[i:batch_end]
                y_quantum_batch = y_shuffled[i:batch_end]

                def quantum_cost(params):
                    model.quantum_params = model._unflatten_params(params)
                    X_processed = model._preprocess_input(X_quantum_batch)
                    if model.fast_backprop:
                        # Single broadcasted pass over the whole batch (default.qubit + backprop).
                        preds = model.quantum_circuit(pnp.array(X_processed), params)
                    else:
                        preds = pnp.array([
                            model.quantum_circuit(pnp.array(X_processed[j]), params)
                            for j in range(len(X_processed))
                        ])
                    preds = pnp.atleast_1d(preds)
                    if self.use_bce:
                        loss = self._bce_loss(preds, y_quantum_batch)
                    else:
                        y_arr = pnp.array(y_quantum_batch)
                        loss = pnp.mean((preds - y_arr) ** 2)
                    return loss

                grad, loss_val = self.quantum_optimizer.compute_grad(quantum_cost, (params_flat,), {})
                grad_norm = pnp.linalg.norm(grad[0])
                if grad_norm > 5.0:
                    grad = (grad[0] * (5.0 / grad_norm),)
                params_flat = self.quantum_optimizer.apply_grad(grad, (params_flat,))[0]
                ema_params_flat = (ema_decay * ema_params_flat) + ((1.0 - ema_decay) * params_flat)
                epoch_quantum_loss += float(loss_val)
                n_quantum_batches += 1

                X_processed = model._preprocess_input(X_quantum_batch)
                if i == 0:
                    probe = X_processed[:min(5, len(X_processed))]  # only first 5 for speed
                    if model.fast_backprop:
                        quantum_outputs = pnp.atleast_1d(
                            np.asarray(model.quantum_circuit(pnp.array(probe), params_flat)))
                    else:
                        quantum_outputs = pnp.array([
                            float(model.quantum_circuit(pnp.array(probe[j]), params_flat))
                            for j in range(len(probe))
                        ])
                    print(
                        f"Epoch {epoch+1} Batch {i//model.config.batch_size+1} | "
                        f"Loss: {float(loss_val):.6f} | "
                        f"Stats: min: {quantum_outputs.min():.3f}, max: {quantum_outputs.max():.3f}, mean: {quantum_outputs.mean():.3f}"
                    )
                else:
                    print(f"Epoch {epoch+1} Batch {i//model.config.batch_size+1} | Loss: {float(loss_val):.6f}")

            avg_loss = epoch_quantum_loss / max(1, n_quantum_batches)
            model.quantum_params = model._unflatten_params(ema_params_flat)

            train_preds = model.quantum_predict_batch(X_train[:200])
            train_accuracy = np.mean(train_preds == y_train[:200])
            test_preds = model.quantum_predict_batch(X_test)
            test_accuracy = np.mean(test_preds == y_test)

            tp = np.sum((test_preds == 1) & (y_test == 1))
            tn = np.sum((test_preds == -1) & (y_test == -1))
            fp = np.sum((test_preds == 1) & (y_test == -1))
            fn = np.sum((test_preds == -1) & (y_test == 1))

            model.training_history['loss'].append(float(avg_loss))
            model.training_history['accuracy'].append(float(test_accuracy))
            elapsed = time.time() - epoch_start
            model.training_history['epoch_times'].append(float(elapsed))

            if test_accuracy > best_accuracy:
                print(f"  [STABILITY] Test accuracy improved: {best_accuracy:.2%} -> {test_accuracy:.2%} (via EMA). Saving best params.")
                best_accuracy = test_accuracy
                best_quantum_params = {k: v.copy() for k, v in model.quantum_params.items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if current_lr > 1e-4:
                        current_lr *= lr_factor
                        self.quantum_optimizer.stepsize = current_lr
                        print(f"  [STABILITY] Plateau detected. Reducing Learning Rate to {current_lr:.6f}")
                        patience_counter = 0
                    else:
                        print(f"  [STABILITY] Early Stopping triggered after {epoch+1} epochs.")
                        break

            model.quantum_params = model._unflatten_params(params_flat)
            progress_percent = ((epoch + 1) / n_epochs) * 100
            estimated_total = elapsed / ((epoch + 1) / n_epochs)
            remaining = estimated_total - elapsed
            epoch_summary_text = (
                f"--- Epoch {epoch+1}/{n_epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"Train Acc: {train_accuracy:.1%} | "
                f"Test Acc: {test_accuracy:.1%} | "
                f"CM: [TP:{tp}, TN:{tn}, FP:{fp}, FN:{fn}] | "
                f"Progress: {progress_percent:.1f}% | "
                f"ETA: {remaining:.1f}s ---"
            )
            print(epoch_summary_text)
            if summary_filepath:
                try:
                    with open(summary_filepath, 'a', encoding='utf-8') as sf:
                        sf.write(epoch_summary_text + "\n")
                except Exception as e:
                    print(f"Warning: could not write summary to {summary_filepath}: {e}")

        model.quantum_params = best_quantum_params
        final_msg = f"\nFinal Best Test Accuracy: {best_accuracy:.2%}\n"
        print(final_msg)
        if summary_filepath:
            with open(summary_filepath, 'a', encoding='utf-8') as sf:
                sf.write(final_msg)

        # save_weights=False for parallel ablation runs: they don't consume the saved
        # weights (metrics come from the in-memory model) and share this single fixed path,
        # so concurrent writers would corrupt it.
        if save_weights:
            weights_dir = os.path.join('Results', 'Weights')
            os.makedirs(weights_dir, exist_ok=True)
            weights_path = os.path.join(weights_dir, 'quantum_model_params.npz')
            self.save_params(model.quantum_params, weights_path)
            print(f"Saved trained quantum model parameters to '{weights_path}'")
        print(f"\nBest Quantum Test Accuracy: {best_accuracy:.3f}")
        return model

    def _compute_quantum_accuracy(self, model: PureQuantumNativeCNN,
                                  X: np.ndarray, y: np.ndarray) -> float:
        quantum_predictions = model.quantum_predict_batch(X)
        return np.mean(quantum_predictions == y)

    def _validate_dataset(self, X: np.ndarray, y: np.ndarray, model: 'PureQuantumNativeCNN'):
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Sample count mismatch: X has {X.shape[0]}, y has {y.shape[0]}")
        encoding = model.config.encoding_type
        n_qubits = model.config.n_qubits
        if encoding == 'patch':
            if X.ndim != 3:
                raise ValueError(f"Patch encoding expects 3D array (n_samples, h, w), got {X.shape}")
            if X.shape[1] != model.config.image_size or X.shape[2] != model.config.image_size:
                print(f"Warning: Image size {X.shape[1:]} doesn't match config {model.config.image_size}")
        elif encoding == 'amplitude':
            expected = 2 ** n_qubits
            if X.shape[1] > expected:
                raise ValueError(f"Amplitude encoding expects at most {expected} features for {n_qubits} qubits, got {X.shape[1]}")
        else:
            if X.shape[1] != n_qubits:
                raise ValueError(
                    f"Feature count {X.shape[1]} doesn't match n_qubits {n_qubits}."
                )
        unique_labels = np.unique(y)
        if not np.array_equal(unique_labels, np.array([-1, 1])):
            raise ValueError(
                f"Labels must be {{-1, +1}} for binary classification. Got {unique_labels}."
            )