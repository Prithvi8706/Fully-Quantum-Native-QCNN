#!/usr/bin/env python3
# Main execution script for Pure Quantum Native QCNN

import sys
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, precision_score, recall_score, f1_score
from QCNN.utils import splits as split_service
from QCNN.utils.run_artifacts import TestEvaluationGuard
from QCNN.utils.seeding import seed_everything
import cProfile
import pstats
import numpy as np, random
import time

# setting randomness seeds for reproducibility
np.random.seed(42)
random.seed(42)

class Logger(object):
    """Duplicates stdout/stderr to a file."""
    def __init__(self, filename="training_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        # This flush method is needed for python 3 compatibility.
        # This handles the flush command by doing nothing.
        self.terminal.flush()
        self.log.flush()

    def __del__(self):
        if hasattr(self, 'log'):
            self.log.close()

try:
    import pennylane as qml
    qml.set_seed(42)
except:
    pass

# Add the QCNN package directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import all modules
from QCNN.config.Qconfig import QuantumNativeConfig
from QCNN.models.QCNNModel import PureQuantumNativeCNN
from QCNN.training.Qtrainer import QuantumNativeTrainer
from QCNN.utils.dataset_generator import generate_quantum_binary_dataset
from QCNN.utils.dataset_loader import load_dataset
from QCNN.utils.metadata_logger import save_metadata
from QCNN.utils.metrics import (
    predict_raw_outputs,
    compute_classification_metrics,
    save_metrics_json,
)


def main(train_sample_size=None, use_bce=True, dataset_path=None, dataset_type='synthetic', 
         encoding='feature_map', image_size=None, log_file="training_log.txt", summary_log_file="training_summary.txt",
         learning_rate=None, classes=None, epochs=None, batch_size=None, seed=42):
    """Main execution function
    
    Args:
        train_sample_size (int or None): Number of training samples to use.
        use_bce (bool): If True, trainer optimizes BCE; else MSE.
        dataset_path (str or None): Path to custom dataset file
        dataset_type (str): Type of dataset ('synthetic', 'npz', 'csv', 'mnist')
        encoding (str): Encoding type ('auto', 'feature_map', 'amplitude', 'patch')
        image_size (int or None): Width/height of square image
        log_file (str): Path to output log file
        summary_log_file (str): Path to summary output log file
        classes (tuple or None): Override for config.classes (e.g. (0, 1) for MNIST).
                                 If None, the value from Qconfig is used.
    """
    # Initialize logger to capture all output to file
    sys.stdout = Logger(log_file)
    sys.stderr = sys.stdout

    print("Entered main() function.")

    # Step 1: Configuration
    print("\nStep 1: Initializing Quantum Configuration...")
    # Use factory to auto-configure based on image size and desired encoding
    if image_size is None:
        image_size = 4 if dataset_type == 'synthetic' else 16 # Default 16 for MNIST to enable amplitude encoding
    
    config = QuantumNativeConfig.from_image_size(image_size, encoding)
    config.seed = seed
    seed_everything(seed)
    if learning_rate is not None:
        config.learning_rate = learning_rate
    
    print(f" Configuration: {config.n_qubits} qubits, {config.n_conv_layers} quantum layers")
    print(f" Encoding: {config.encoding_type}")
    print(f" Image Size: {config.image_size}x{config.image_size}")
    
    if epochs is not None:
        config.n_epochs = epochs
    if batch_size is not None:
        config.batch_size = batch_size
        
    print(f" Training: {config.n_epochs} epochs, lr={config.learning_rate}, batch_size={config.batch_size}")

    # Resolve classes: CLI arg overrides Qconfig value
    if classes is not None:
        config.classes = classes
    print(f" Binary classification: classes {config.classes} mapped to +1 and -1")

    # save metadata after config is created
    save_metadata("Results/metadata.json", config)

    # Step 2: Load dataset (synthetic or custom)
    os.makedirs("Results", exist_ok=True)
    
    if dataset_type == 'synthetic':
        # Use synthetic dataset (backward compatibility)
        clean_dataset_path = os.path.join("Results", "quantum_dataset_clean_seed42.npz")
        force_regenerate = False
        
        if force_regenerate or not os.path.exists(clean_dataset_path):
            print("\nGenerating Quantum Binary Dataset (seed=42)...")
            np.random.seed(42)
            random.seed(42)
            
            X_quantum, y_quantum = generate_quantum_binary_dataset(
                n_samples=400,
                image_size=config.image_size
            )
            np.savez(clean_dataset_path, X=X_quantum, y=y_quantum)
            print(f"Saved dataset to '{clean_dataset_path}'")
        else:
            print(f"\nLoading synthetic dataset from '{clean_dataset_path}'...")
            data = np.load(clean_dataset_path)
            X_quantum, y_quantum = data['X'], data['y']
    
    else:
        # Load custom dataset
        print(f"\nStep 2: Loading Data ({dataset_type})...")
        X_quantum, y_quantum = load_dataset(
            source=dataset_path,
            dataset_type=dataset_type,
            n_qubits=config.n_qubits,
            image_size=config.image_size,
            normalization=config.preprocessing_mode,
            encoding_type=config.encoding_type,
            classes=config.classes,
        )
    
    # NEW: Limit total dataset size if requested, BEFORE splitting and expensive preprocessing.
    # This ensuring both train and test sets are small enough for quantum simulation.
    if train_sample_size is not None and len(X_quantum) > train_sample_size:
        # Step 2.1: Limit total dataset size with Stratified Sampling for balance
        total_needed = int(train_sample_size / 0.7)
        if total_needed < len(X_quantum):
            print(f"  Limiting total dataset to {total_needed} samples with stratified sampling...")
            
            # Find indices for each class
            idx_pos = np.where(y_quantum == 1)[0]
            idx_neg = np.where(y_quantum == -1)[0]
            
            # Calculate samples per class (targeted 50/50)
            n_pos_needed = total_needed // 2
            n_neg_needed = total_needed - n_pos_needed
            
            # Ensure we have enough samples of each
            n_pos_actual = min(len(idx_pos), n_pos_needed)
            n_neg_actual = min(len(idx_neg), n_neg_needed)
            
            # Draw random samples from each class
            sampled_pos = np.random.choice(idx_pos, n_pos_actual, replace=False)
            sampled_neg = np.random.choice(idx_neg, n_neg_actual, replace=False)
            
            # Combine and shuffle
            indices = np.concatenate([sampled_pos, sampled_neg])
            np.random.shuffle(indices)
            
            X_quantum = X_quantum[indices]
            y_quantum = y_quantum[indices]
            print(f"  Sampled dataset: {n_pos_actual} pos, {n_neg_actual} neg")

    # Ensure all data (including synthetic) is preprocessed for the model if not already
    from QCNN.utils.data_preprocessing import preprocess_for_quantum
    if dataset_type == 'synthetic':
        print(f"  Preprocessing synthetic dataset for {config.encoding_type}...")
        X_quantum, y_quantum = preprocess_for_quantum(
            X_quantum, y_quantum, 
            n_qubits=config.n_qubits, 
            image_size=config.image_size,
            normalization=config.preprocessing_mode,
            encoding_type=config.encoding_type
            # synthetic data already has {-1,+1} labels; no classes needed
        )
    
    print("Dataset ready.")
    # Auto-downsample only when not using amplitude encoding,
    # where we want to preserve as much image information as
    # possible for the quantum state.
    if (
        X_quantum.ndim == 3
        and X_quantum.shape[1] > 16
        and config.encoding_type not in ('amplitude', 'patch')
    ):
        print(f"  Detected high resolution ({X_quantum.shape[1:]}). Downsampling for quantum efficiency...")
        # Simple 2x2 mean pooling downsampling
        h, w = X_quantum.shape[1], X_quantum.shape[2]
        X_quantum = X_quantum[:, :h//2*2, :w//2*2].reshape(-1, h//2, 2, w//2, 2).mean(axis=(2, 4))
        config.image_size = h // 2
        print(f"  New resolution: {X_quantum.shape[1:]}")
    print(f"Shape: {X_quantum.shape}")
    print(f"Class distribution: {dict(zip(*np.unique(y_quantum, return_counts=True)))}")

    # Split data through a recorded manifest: 60/15/25, stratified, seeded.
    # Validation drives model selection; test is touched exactly once, below.
    manifest = split_service.make_split_manifest(
        y_quantum,
        seed=config.seed,
        dataset_id=f"{dataset_type}_{config.classes[0]}v{config.classes[1]}_n{len(y_quantum)}",
        class_mapping={str(config.classes[0]): 1, str(config.classes[1]): -1},
    )
    manifest_path = os.path.join(
        "Results", "manifests", f"{manifest['dataset_id']}_seed{config.seed}.json")
    split_service.save_manifest(manifest, manifest_path)
    print(f" Split manifest: {manifest_path} (id {manifest['id'][:12]})")

    X_train, y_train, X_val, y_val, X_test, y_test = split_service.apply_manifest(
        manifest, X_quantum, y_quantum)

    # Limit training samples if train_sample_size parameter set
    if train_sample_size is not None and train_sample_size < len(X_train):
        X_train = X_train[:train_sample_size]
        y_train = y_train[:train_sample_size]

    print(f" Training samples used: {len(X_train)}")
    print(f" Validation samples: {len(X_val)}")
    print(f" Test samples: {len(X_test)}")
    print(f" Class distribution (training): {dict(zip(*np.unique(y_train, return_counts=True)))}")

    # Step 3: Model initialization
    print("\n Step 3: Initializing Pure Quantum CNN...")
    quantum_model = PureQuantumNativeCNN(config)

    total_params = sum(np.prod(p.shape) for p in quantum_model.quantum_params.values())
    print(f" Quantum parameters: {total_params}")
    print(f" Hilbert space: 2^{config.n_qubits} = {2 ** config.n_qubits} dimensions")

    # Step 4: Training
    print("\n Step 4: Training Pure Quantum CNN...")
    trainer = QuantumNativeTrainer(learning_rate=config.learning_rate, use_bce=use_bce)
    
    start_time = time.time()
    trained_model = trainer.train_pure_quantum_cnn(
        quantum_model, X_train, y_train, X_val, y_val,
        log_filepath=log_file, summary_filepath=summary_log_file,
        weights_path=os.path.join("Results", "Weights", f"run_seed{config.seed}.npz"),
    )
    training_time = time.time() - start_time
    print(f" Total Training Time: {training_time:.2f} seconds")

    # Step 5: Evaluation
    print("\n Step 5: Evaluating Quantum Model...")

    # The single permitted test evaluation for this run (UPGRADE_PLAN.md 0.3).
    # The old _quantum_preprocessed_test fallback died with the leakage: the
    # trainer never sees test data now, so there is no cached representation.
    guard = TestEvaluationGuard()
    y_eval = y_test
    raw_outputs = guard.evaluate(
        predict_raw_outputs, trained_model, X_test, already_preprocessed=False)
    metrics = compute_classification_metrics(y_eval, raw_outputs)

    accuracy = metrics['accuracy']
    predictions = metrics['predictions']
    prob_scores = metrics['prob_scores']
    tp, tn, fp, fn = metrics['tp'], metrics['tn'], metrics['fp'], metrics['fn']
    precision, recall, f1 = metrics['precision'], metrics['recall'], metrics['f1']
    prediction_bias = metrics['prediction_bias']
    prediction_variance = metrics['prediction_variance']
    y_test_bin = np.where(y_eval == 1, 1, 0)

    print(f" Final Quantum Accuracy: {accuracy:.1%}")
    print(f"\n Quantum Confusion Matrix:")
    print(f"   True Positives: {tp}")
    print(f"   True Negatives: {tn}")
    print(f"\n Evaluation Metrics:")
    print(f"   Sensitivity (Recall): {recall:.3f}")
    print(f"   Precision: {precision:.3f}")
    print(f"   F1 Score: {f1:.3f}")
    print(f"   ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"   PR-AUC: {metrics['pr_auc']:.3f}")
    print(f"   Prediction Bias: {prediction_bias:.4f}")
    print(f"   Prediction Variance: {prediction_variance:.4f}")

    # Persist the metric suite as JSON for downstream aggregation / reporting.
    try:
        save_metrics_json(metrics, os.path.join("Results", "metrics.json"))
    except Exception as _me:
        print(f"  Warning: could not save metrics.json: {_me}")

    # Plot training history and save graphs
    try:
        from QCNN.utils.plotstyle import apply_paper_style
        apply_paper_style()
        graphs_dir = os.path.join('Results', 'Graphs')
        
        # Save confusion matrix
        cm_dir = os.path.join(graphs_dir, 'Confusion_Matrix')
        os.makedirs(cm_dir, exist_ok=True)
        cm_path = os.path.join(cm_dir, 'quantum_confusion_matrix.png')
        plt.figure(figsize=(6, 5))
        cm = np.array([[tn, fp], [fn, tp]])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Pred -1', 'Pred +1'],
                    yticklabels=['True -1', 'True +1'])
        plt.title('Quantum Model Confusion Matrix')
        plt.tight_layout()
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n Confusion matrix saved as '{cm_path}'")
        
        # Save ROC Curve
        fpr, tpr, _ = roc_curve(y_test_bin, prob_scores)
        roc_auc = auc(fpr, tpr)
        
        roc_dir = os.path.join(graphs_dir, 'ROC_Curve')
        os.makedirs(roc_dir, exist_ok=True)
        roc_path = os.path.join(roc_dir, 'quantum_roc_curve.png')
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(roc_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f" ROC curve saved as '{roc_path}'")
        
        # Save Precision-Recall Curve
        precision_vals, recall_vals, _ = precision_recall_curve(y_test_bin, prob_scores)
        pr_auc = auc(recall_vals, precision_vals)
        pr_path = os.path.join(roc_dir.replace('ROC_Curve', 'Evaluation_Metrics'), 'quantum_pr_curve.png')
        plt.figure(figsize=(6, 5))
        plt.plot(recall_vals, precision_vals, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
        plt.xlabel('Recall (Sensitivity)')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(pr_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f" Precision-Recall curve saved as '{pr_path}'")
        
        # Save Precision-Recall metrics bar chart
        metrics_dir = os.path.join(graphs_dir, 'Evaluation_Metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_path = os.path.join(metrics_dir, 'quantum_evaluation_metrics.png')
        plt.figure(figsize=(8, 5))
        metric_names = ['Sensitivity', 'Precision', 'F1 Score', 'Pred Bias', 'Pred Var']
        metric_vals = [recall, precision, f1, prediction_bias, prediction_variance]
        bars = plt.bar(metric_names, metric_vals, color=['royalblue', 'royalblue', 'royalblue', 'coral', 'mediumseagreen'])
        for bar in bars:
            yval = bar.get_height()
            offset = 0.02 if yval >= 0 else -0.05
            va = 'bottom' if yval >= 0 else 'top'
            plt.text(bar.get_x() + bar.get_width()/2, yval + offset, f'{yval:.3f}', ha='center', va=va)
        
        min_val = min(0, min(metric_vals))
        max_val = max(1.1, max(metric_vals))
        plt.ylim([min_val - 0.1, max_val + 0.1])
        plt.title('Model Performance Metrics')
        plt.ylabel('Score')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(metrics_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f" Evaluation metrics chart saved as '{metrics_path}'")
        
        # Per-epoch computation time breakdown
        time_dir = os.path.join(graphs_dir, 'Computation_Time')
        os.makedirs(time_dir, exist_ok=True)
        time_path = os.path.join(time_dir, 'quantum_computation_time.png')
        
        plt.figure(figsize=(10, 6))
        epoch_times = trained_model.training_history.get('epoch_times', [])
        if epoch_times:
            epochs = list(range(1, len(epoch_times) + 1))
            bars = plt.bar(epochs, epoch_times, color='dodgerblue')
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, height + (max(epoch_times)*0.01), 
                         f'{height:.1f}s', ha='center', va='bottom', fontsize=8)
            
            plt.xlabel('Epoch')
            plt.xticks(epochs)
            plt.title(f'Quantum Computation Time per Epoch (Total: {training_time:.1f}s)')
        else:
            plt.bar(['Total Training Time'], [training_time], color='dodgerblue')
            plt.text(0, training_time + (training_time*0.02), f'{training_time:.1f}s', ha='center', va='bottom')
            plt.title('Quantum Computation Time')
            
        plt.ylabel('Seconds')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(time_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f" Computation time chart saved as '{time_path}'")
        
        train_res_dir = os.path.join(graphs_dir, 'Training_Results')
        os.makedirs(train_res_dir, exist_ok=True)
        graph_path = os.path.join(train_res_dir, 'quantum_training_results.png')

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(trained_model.training_history['loss'])
        plt.title('Pure Quantum Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Quantum Loss')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(trained_model.training_history['accuracy'])
        plt.title('Pure Quantum Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Quantum Accuracy')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        print(f"\n Training plots saved as '{graph_path}'")

        try:
            from noise_sim import run_noise_simulation
            print("\n Running noise robustness simulation (this may take a few minutes)...")
            run_noise_simulation(
                weights_path="Results/Weights/quantum_model_params.npz",
                output_path="Results/Graphs/fig6_noise_robustness_real.png",
                image_size=config.image_size,
                classes=tuple(config.classes)
            )
        except Exception as noise_e:
            print(f"\n  Warning: Noise simulation failed: {noise_e}")

    except Exception as e:
        print(f"\n  Plotting failed: {e}")

    print("\n" + "-" * 60)
    print("EXECUTION OF QCNN COMPLETE!")
    print("-" * 60)
    print(f" Achieved {accuracy:.1%} accuracy with 100% quantum operations")
    return trained_model, accuracy


if __name__ == "__main__":
    print("Starting main execution script...")
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train Quantum CNN with custom or synthetic datasets')
    parser.add_argument('--dataset', type=str, default='images',
                       choices=['auto', 'synthetic', 'npz', 'csv', 'mnist', 'images', 'idx'],
                       help='Type of dataset to use')
    parser.add_argument('--path', type=str, default=None,
                       help='Path to custom dataset file or directory')
    parser.add_argument('--samples', type=int, default=None,
                       help='Number of training samples to use')
    parser.add_argument('--use-mse', action='store_true',
                       help='Use Mean Squared Error loss instead of BCE')
    parser.add_argument('--encoding', type=str, default='auto',
                       choices=['auto', 'feature_map', 'amplitude', 'patch'],
                       help='Encoding strategy to use')
    parser.add_argument('--image-size', type=int, default=None,
                       help='Width/height of square input images')
    parser.add_argument('--classes', type=int, nargs=2, default=None,
                       help='Two classes for binary classification. '
                            'Overrides classes in Qconfig.py. e.g --classes 0 1')
    parser.add_argument('--log-file', type=str, default='training_log.txt',
                       help='File to save training logs (default: training_log.txt)')
    parser.add_argument('--summary-log', type=str, default='training_summary.txt',
                       help='File to save epoch summaries (default: training_summary.txt)')
    parser.add_argument('--no-profile', action='store_true',
                       help='Disable cProfile performance profiling')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate for quantum optimization')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size for training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()
    
    try:
        if not args.no_profile:
            profiler = cProfile.Profile()
            profiler.enable()
        
        # Run main with arguments
        model, acc = main(
            train_sample_size=args.samples,
            use_bce=not args.use_mse,
            dataset_path=args.path,
            dataset_type=args.dataset,
            encoding=args.encoding,
            image_size=args.image_size,
            log_file=args.log_file,
            summary_log_file=args.summary_log,
            learning_rate=args.learning_rate,
            classes=tuple(args.classes) if args.classes else None,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        
        if not args.no_profile:
            profiler.disable()
            print(f"\nExecution completed successfully! Accuracy: {acc:.1%}")
            stats = pstats.Stats(profiler).sort_stats('cumulative')
            stats.print_stats(20)
        else:
            print(f"\nExecution completed successfully! Accuracy: {acc:.1%}")
    
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()