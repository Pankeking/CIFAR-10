import os

import numpy as np

_CACHE_DIR = "norm_cache"
os.makedirs(_CACHE_DIR, exist_ok=True)

def get_or_compute_norm_stats(key: str, x: np.ndarray, axes=(0, 2, 3)):
    path = os.path.join(_CACHE_DIR, f"{key}_mean_std.npz")
    if os.path.exists(path):
        data = np.load(path)
        return data["mean"], data["std"]
    mean = np.mean(x, axis=axes, keepdims=True)
    std = np.std(x, axis=axes, keepdims=True) + 1e-8
    np.savez(path, mean=mean, std=std)
    return mean, std
