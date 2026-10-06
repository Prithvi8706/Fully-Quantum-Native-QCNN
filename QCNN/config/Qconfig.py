import math


class QuantumNativeConfig:
    """
    Configuration for the fully quantum-native QCNN.

    Binary classification mode
    --------------------------
    classification only.  Use ``classes`` to choose which two classes are
    treated as the target categories (+1 and -1). For MNIST this can be any
    pair of digits (e.g., (0, 1)).
    """
    
    def __init__(self):
        # Data parameters
        self.image_size = 16  # 16x16 = 256 features
        self.n_qubits = 8  # log2(256) = 8 qubits for amplitude encoding
        
        # Dataset parameters
        self.n_classes = 2  # Binary classification (one-vs-rest)
        self.n_features = self.image_size ** 2  # Total input features
        self.preprocessing_mode = 'minmax'  # 'minmax', 'standard', 'robust'

        # Strict binary classification classes
        # Set classes to the tuple of class labels you want the model to classify.
        # For MNIST: (0, 1)  |  For synthetic data: ignored (labels already {-1,+1})
        self.classes = (0, 1)
        self.binary_mode = 'strict_binary'
        
        # Encoding strategy: 'feature_map', 'amplitude', 'patch'
        #   feature_map  – 1 qubit per feature (original, image_size² qubits)
        #   amplitude    – log₂(features) qubits via amplitude embedding
        #   patch        – quanvolutional preprocessing then QCNN on reduced map
        self.encoding_type = 'amplitude'  # Captures full image details efficiently
        
        # Pure quantum architecture parameters
        self.n_conv_layers = 4
        self.kernel_size = 2  # 2x2 quantum kernels

        # ------------------------------------------------------------------
        # Ablation toggles (suggestion #4). Defaults reproduce the paper's
        # proposed architecture; alternatives let each component be switched
        # off / swapped to measure its contribution.
        # ------------------------------------------------------------------
        # Convolution kernel rotation set per qubit:
        #   'su2' – RX, RY, RZ (proposed, 3 params/qubit)
        #   'ry'  – RY only (1 param/qubit, tests SU(2) expressivity claim)
        self.kernel_rotations = 'su2'
        # Intra-window entanglement pattern:
        #   'full'         – 4 edges + both diagonals (current code, 6 CNOTs)
        #   'one_diagonal' – 4 edges + one diagonal (5 CNOTs, paper-faithful)
        #   'none'         – no entanglement (tests entanglement design claim)
        self.conv_entanglement = 'full'
        # Pooling mechanism:
        #   'unitary'     – coherence-preserving CRY/CRZ pooling (proposed)
        #   'none'        – drop qubits with no info-transfer unitary
        #   'measurement' – mid-circuit measurement pooling (breaks coherence,
        #                   models prior measurement-based QCNNs)
        self.pooling_mode = 'unitary'
        
        # Quanvolutional layer parameters (used when encoding_type == 'patch')
        self.patch_size = 4        # Patch width/height for quanvolutional filter
        self.patch_stride = 4      # Stride between patches (== patch_size → non-overlapping)
        self.n_quanv_filters = 4   # Number of random quantum filters per patch
        self.quanv_qubits = 16     # Qubits used per patch circuit (patch_size²)
        
        # Quantum training parameters
        self.learning_rate = 0.02
        self.n_epochs = 50
        self.batch_size = 32
        self.early_stopping_patience = 3  # Stop if no improvement after N epochs
        self.lr_plateau_factor = 0.5      # Multiply LR by this factor on plateau
        self.ema_decay = 0.99             # Decay for Exponential Moving Average
        
        # Random seed for reproducibility / multi-run experiments (suggestion #5).
        # Threaded into numpy, random, PennyLane, the data split, and parameter init.
        self.seed = 42

        # Quantum device
        self.device = 'lightning.qubit'
        self.shots = None  # Exact quantum simulation
    
    def configure_for_image(self, image_size: int, encoding: str = 'auto'):
        """
        Auto-configure qubit count and encoding based on image size.
        
        Args:
            image_size: Width/height of square image
            encoding: 'auto', 'feature_map', 'amplitude', or 'patch'
        """
        self.image_size = image_size
        self.n_features = image_size ** 2
        
        if encoding == 'auto':
            if image_size <= 4:
                encoding = 'feature_map'
            elif image_size <= 16:
                encoding = 'amplitude'
            else:
                encoding = 'patch'
        
        self.encoding_type = encoding
        
        if encoding == 'feature_map':
            # 1 qubit per pixel
            self.n_qubits = self.n_features
            if self.n_qubits > 25:
                # Automatically downsample if possible
                target_size = 5  # 5x5 = 25 features limit
                print(f"Warning: Cannot simulate {self.n_features} qubits via 'feature_map' encoding.")
                print(f"Auto-downsampling target resolution to {target_size}x{target_size} (25 qubits max).")
                self.image_size = target_size
                self.n_features = target_size ** 2
                self.n_qubits = self.n_features
        
        elif encoding == 'amplitude':
            # log₂(features) qubits – pad features to nearest power of 2
            n_features_padded = 2 ** math.ceil(math.log2(max(self.n_features, 2)))
            self.n_qubits = math.ceil(math.log2(n_features_padded))
            self.n_features = n_features_padded
            # Do NOT overwrite image_size; it's needed for initial data loading/preprocessing
        
        elif encoding == 'patch':
            # Quanvolutional: QCNN operates on the reduced feature map
            self.quanv_qubits = self.patch_size ** 2
            out_size = (image_size - self.patch_size) // self.patch_stride + 1
            reduced_features = out_size * out_size * self.n_quanv_filters
            # QCNN runs on the reduced map using amplitude encoding
            n_padded = 2 ** math.ceil(math.log2(max(reduced_features, 2)))
            self.n_qubits = math.ceil(math.log2(n_padded))
            self.n_features = reduced_features
            # Do NOT overwrite image_size; it's needed for initial data loading/preprocessing
        
        # Adjust conv layers based on qubit count
        if self.n_qubits < 8:
            self.n_conv_layers = min(self.n_conv_layers, 3)
        if self.n_qubits < 4:
            self.n_conv_layers = min(self.n_conv_layers, 2)
        
        return self
    
    @classmethod
    def from_image_size(cls, image_size: int, encoding: str = 'auto'):
        """
        Factory: create a config auto-tuned for a given image size.
        
        Args:
            image_size: Width/height of square images
            encoding: 'auto', 'feature_map', 'amplitude', or 'patch'
        
        Returns:
            Configured QuantumNativeConfig instance
        """
        config = cls()
        config.configure_for_image(image_size, encoding)
        return config



if __name__ == "__main__":
    config = QuantumNativeConfig()
    print(f"Configuration: {config.n_qubits} qubits, {config.n_conv_layers} quantum layers")
