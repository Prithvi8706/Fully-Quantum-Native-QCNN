import numpy as np
from typing import Tuple, Optional


def normalize_to_quantum_range(data: np.ndarray, 
                               target_range: Tuple[float, float] = (0, 1),
                               method: str = 'minmax') -> np.ndarray:
    """
    Normalize data to quantum-compatible range for angle encoding.
    
    Args:
        data: Input data array of any shape
        target_range: Target range for normalization (default: [0, 2π])
        method: Normalization method ('minmax', 'standard', 'robust')
    
    Returns:
        Normalized data in target range
    """
    data = np.array(data, dtype=np.float64)
    
    if method == 'minmax':
        # Min-max normalization to [0, 1], then scale to target range
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max - data_min < 1e-10:
            # Constant data, return middle of range
            normalized = np.full_like(data, (target_range[0] + target_range[1]) / 2)
        else:
            normalized = (data - data_min) / (data_max - data_min)
            normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]
    
    elif method == 'standard':
        # Standardize to mean=0, std=1, then clip and scale
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-10:
            normalized = np.full_like(data, (target_range[0] + target_range[1]) / 2)
        else:
            standardized = (data - mean) / std
            # Clip to [-3, 3] sigma range
            clipped = np.clip(standardized, -3, 3)
            # Scale from [-3, 3] to target range
            normalized = (clipped + 3) / 6  # to [0, 1]
            normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]
    
    elif method == 'robust':
        # Robust scaling using median and IQR
        median = np.median(data)
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        if iqr < 1e-10:
            normalized = np.full_like(data, (target_range[0] + target_range[1]) / 2)
        else:
            scaled = (data - median) / iqr
            clipped = np.clip(scaled, -3, 3)
            normalized = (clipped + 3) / 6
            normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized


def resize_images(images: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize images to match quantum circuit qubit requirements.
    
    Args:
        images: Array of images with shape (n_samples, height, width) or 
                (n_samples, height, width, channels)
        target_size: Target (height, width) for resizing
    
    Returns:
        Resized images with shape (n_samples, target_height, target_width)
    """
    try:
        from scipy.ndimage import zoom
    except ImportError:
        raise ImportError("scipy is required for image resizing. Install with: pip install scipy")
    
    if images.ndim == 3:
        # Grayscale images
        n_samples, h, w = images.shape
        has_channels = False
    elif images.ndim == 4:
        # Color images - convert to grayscale
        n_samples, h, w, c = images.shape
        has_channels = True
        # Simple grayscale conversion
        images = np.mean(images, axis=-1)
    else:
        raise ValueError(f"Expected 3D or 4D image array, got shape {images.shape}")
    
    target_h, target_w = target_size
    zoom_factors = (1, target_h / h, target_w / w)
    
    resized = zoom(images, zoom_factors, order=1)  # Bilinear interpolation
    return resized


def flatten_and_pad(data: np.ndarray, target_features: int) -> np.ndarray:
    """
    Flatten data and pad/truncate to match target feature count.
    
    Args:
        data: Input data array (n_samples, ...)
        target_features: Target number of features per sample
    
    Returns:
        Flattened and padded/truncated array (n_samples, target_features)
    """
    n_samples = data.shape[0]
    
    # Flatten all dimensions except first
    flattened = data.reshape(n_samples, -1)
    current_features = flattened.shape[1]
    
    if current_features == target_features:
        return flattened
    elif current_features < target_features:
        # Pad with zeros
        padding = np.zeros((n_samples, target_features - current_features))
        return np.concatenate([flattened, padding], axis=1)
    else:
        # Truncate
        return flattened[:, :target_features]


def encode_labels(labels: np.ndarray, 
                  n_classes: Optional[int] = None,
                  encoding: str = 'binary') -> np.ndarray:
    """
    Encode labels for quantum classification.
    
    Args:
        labels: Input labels (any format)
        n_classes: Number of classes (auto-detected if None)
        encoding: 'binary' for {-1, +1}, 'onehot' for one-hot encoding
    
    Returns:
        Encoded labels
    """
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    
    if n_classes is None:
        n_classes = len(unique_labels)
    
    if encoding == 'binary':
        if n_classes == 2:
            # Binary classification: map to {-1, +1}
            label_map = {unique_labels[0]: -1, unique_labels[1]: 1}
            return np.array([label_map[label] for label in labels])
        else:
            raise ValueError(f"Binary encoding requires 2 classes, got {n_classes}")
    
    elif encoding == 'onehot':
        # One-hot encoding for multi-class
        onehot = np.zeros((len(labels), n_classes))
        for i, label in enumerate(labels):
            class_idx = np.where(unique_labels == label)[0][0]
            onehot[i, class_idx] = 1
        return onehot
    
    else:
        raise ValueError(f"Unknown encoding: {encoding}")



def preprocess_for_quantum(X: np.ndarray, 
                           y: np.ndarray,
                           n_qubits: int,
                           image_size: Optional[int] = None,
                           normalization: str = 'minmax',
                           encoding_type: str = 'feature_map') -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete preprocessing pipeline for quantum circuit compatibility.
    
    Args:
        X: Input features (any shape)
        y: Labels
        n_qubits: Number of qubits in quantum circuit
        image_size: If provided, objective image size for resizing/reshaping
        normalization: Normalization method
        encoding_type: 'feature_map', 'amplitude', or 'patch'
    
    Returns:
        Preprocessed (X, y) ready for quantum encoding/processing
    """
    # 1. Handle image structure and feature count
    if encoding_type == 'patch':
        # Patch encoding needs to preserve 2D structure for the quanv layer
        if image_size is not None:
            if X.ndim > 2:
                # Resize if needed
                X = resize_images(X, (image_size, image_size))
            else:
                # Reshape flattened features back to 2D
                n_samples = X.shape[0]
                expected = image_size * image_size
                if X.shape[1] != expected:
                    X = flatten_and_pad(X, expected)
                X = X.reshape(n_samples, image_size, image_size)
        # We don't flatten for 'patch' mode here, QCNNModel will do it after quanv
    else:
        # Standard flattening/resizing for feature_map and amplitude
        if image_size is not None:
            expected_features = image_size * image_size
            if X.ndim > 2:
                X = resize_images(X, (image_size, image_size))
                X = X.reshape(X.shape[0], -1)
            else:
                X = flatten_and_pad(X, expected_features)
        else:
            if X.ndim > 2:
                X = X.reshape(X.shape[0], -1)
            
            # For amplitude encoding, we might need a specific power-of-2 feature count
            if encoding_type == 'amplitude':
                target_features = 2 ** n_qubits
            else:
                target_features = n_qubits
            
            X = flatten_and_pad(X, target_features)

    # 2. Normalize to quantum range [0, 2π]
    # (Even for amplitude encoding, some range scaling is helpful before state prep)
    X_normalized = normalize_to_quantum_range(X, method=normalization)
    
    # 3. Encode labels to {-1, +1}
    # Strict binary encoder (y must already have exactly 2 unique values)
    y_encoded = encode_labels(y, encoding='binary')
    
    return X_normalized, y_encoded
