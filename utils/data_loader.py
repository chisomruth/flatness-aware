import numpy as np
import jax.numpy as jnp
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple

def load_cifar100_subset(n_samples: int = 5000):
    print("Loading CIFAR-100 dataset...")

    cifar100 = fetch_openml(data_id=159, as_frame=False, parser="liac-arff")
    X, y = cifar100.data, cifar100.target  

    y = LabelEncoder().fit_transform(y)

    indices = np.random.choice(len(X), n_samples, replace=False)
    X, y = X[indices], y[indices]

    X = X.astype(np.float32) / 255.0

    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"Number of classes: {len(np.unique(y))}")

    return jnp.array(X_train), jnp.array(y_train), jnp.array(X_test), jnp.array(y_test)
