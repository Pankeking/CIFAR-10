import pickle
import os
import numpy as np
from utils.losses import Losses

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def print_metrics(logits: np.ndarray, labels: np.ndarray, losses: Losses):
    loss = losses.loss_fn(logits, labels)
    pred_classes = np.argmax(logits, axis=1)          # shape (64,)
    true_classes = np.argmax(labels, axis=1)
    class_diff = np.abs(pred_classes - true_classes)        # (64,)
    accuracy = np.mean(class_diff == 0)
    print(f"Accuracy: {accuracy}")
    print(f"Loss: {loss}")
    print("Class diff:", class_diff)
    print("First sample logits:", logits[0])

def load_cifar_batch(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        batch = pickle.load(f, encoding="latin1")
    data = batch["data"]          # shape (10000, 3072), uint8
    labels = np.array(batch["labels"], dtype=np.int64)  # shape (10000,)
    data = data.astype(np.float32) / 255.0
    return data, labels

def load_cifar10(root: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for i in range(1, 6):
        path = os.path.join(root, f"data_batch_{i}")
        x, y = load_cifar_batch(path)
        xs.append(x)
        ys.append(y)
    x_train = np.concatenate(xs, axis=0)   # (50000, 3072)
    y_train = np.concatenate(ys, axis=0)   # (50000,)

    x_test, y_test = load_cifar_batch(os.path.join(root, "test_batch"))
    return x_train, y_train, x_test, y_test
