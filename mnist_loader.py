"""
mnist_loader (IDX version)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Loads MNIST from raw IDX files:
- train-images-idx3-ubyte
- train-labels-idx1-ubyte
- t10k-images-idx3-ubyte
- t10k-labels-idx1-ubyte
(or the dotted-name equivalents)
"""

import numpy as np
import struct
import os

# ---------- IDX parsing ----------

def _load_idx_images(path):
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Bad magic number in {path}: expected 2051, got {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        images = data.reshape(n, rows * cols).astype(np.float32) / 255.0
    return images

def _load_idx_labels(path):
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Bad magic number in {path}: expected 2049, got {magic}")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def _pick_existing(*candidates):
    """Return first existing filepath from candidates."""
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of these paths exist: {candidates}")

# ---------- Public API (matches Nielsen style) ----------

def load_data(data_dir="."):
    """
    Return (training_data, validation_data, test_data).

    Each is a tuple: (images, labels)
    - images: ndarray shape (N, 784), float32 in [0,1]
    - labels: ndarray shape (N,), uint8 0..9

    Note: This raw IDX distribution doesn't include a separate validation set.
    We'll create one from the training set (default: last 10,000 examples).
    """
    # Support both dash and dot file naming.
    train_images_path = _pick_existing(
        os.path.join(data_dir, "train-images-idx3-ubyte"),
        os.path.join(data_dir, "train-images.idx3-ubyte"),
    )
    train_labels_path = _pick_existing(
        os.path.join(data_dir, "train-labels-idx1-ubyte"),
        os.path.join(data_dir, "train-labels.idx1-ubyte"),
    )
    test_images_path = _pick_existing(
        os.path.join(data_dir, "t10k-images-idx3-ubyte"),
        os.path.join(data_dir, "t10k-images.idx3-ubyte"),
    )
    test_labels_path = _pick_existing(
        os.path.join(data_dir, "t10k-labels-idx1-ubyte"),
        os.path.join(data_dir, "t10k-labels.idx1-ubyte"),
    )

    train_images = _load_idx_images(train_images_path)
    train_labels = _load_idx_labels(train_labels_path)
    test_images = _load_idx_images(test_images_path)
    test_labels = _load_idx_labels(test_labels_path)

    # Create a validation set from training data (Nielsen uses 50k train + 10k val)
    n_total = train_images.shape[0]
    n_val = 10000
    if n_total <= n_val:
        raise ValueError("Training set too small to split validation set.")

    # Keep first 50k as train, last 10k as validation (simple deterministic split)
    tr_images, va_images = train_images[:-n_val], train_images[-n_val:]
    tr_labels, va_labels = train_labels[:-n_val], train_labels[-n_val:]

    training_data = (tr_images, tr_labels)
    validation_data = (va_images, va_labels)
    test_data = (test_images, test_labels)

    return (training_data, validation_data, test_data)

def load_data_wrapper(data_dir="."):
    """
    Return (training_data, validation_data, test_data) as lists of (x, y):
    - training_data: y is one-hot (10,1)
    - validation_data/test_data: y is int label
    """
    tr_d, va_d, te_d = load_data(data_dir)

    training_inputs = [x.reshape(784, 1) for x in tr_d[0]]
    training_results = [vectorized_result(int(y)) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))

    validation_inputs = [x.reshape(784, 1) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, [int(y) for y in va_d[1]]))

    test_inputs = [x.reshape(784, 1) for x in te_d[0]]
    test_data = list(zip(test_inputs, [int(y) for y in te_d[1]]))

    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1), dtype=np.float32)
    e[j] = 1.0
    return e
