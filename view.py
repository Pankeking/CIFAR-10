import numpy as np
import matplotlib.pyplot as plt

from model import Model
from utils.data_loader import load_dataset
from utils.math import softmax

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def run_view(model_filename: str, start_index: int = 0) -> None:
    model = Model()
    model.load(model_filename)

    # Load test data
    _, _, x_test, y_test = load_dataset("cifar10")

    # Normalize with train stats
    mean = np.mean(model.input_data_shape, axis=0, keepdims=True)
    std = np.std(model.input_data_shape, axis=0, keepdims=True) + 1e-8
    x_test_norm = (x_test - mean) / std

    idx = start_index

    while True:
        if idx < 0:
            idx = 0
        if idx >= x_test_norm.shape[0]:
            print("Reached end of test set.")
            break

        x_single = x_test_norm[idx:idx+1]
        logits = model.predict_logits(x_single)[0]
        probs = softmax(logits[None, :])[0]

        pred_class = int(np.argmax(probs))
        true_class = int(y_test[idx])

        img = x_test[idx].reshape(3, 32, 32).transpose(1, 2, 0)

        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Index {idx}")

        plt.show(block=False)

        print(f"\nIndex: {idx}")
        print(f"True:      {CIFAR10_CLASSES[true_class]}")
        print(f"Predicted: {CIFAR10_CLASSES[pred_class]}")
        print("Top-3 classes:")
        top3 = np.argsort(probs)[-3:][::-1]
        for k in top3:
            print(f"  {CIFAR10_CLASSES[k]:>10}: {probs[k]:.3f}")

        cmd = input("\n[Enter]=next, p=prev, q=quit > ").strip().lower()
        plt.close()
        if cmd == "q":
            break
        elif cmd == "p":
            idx -= 1
        else:
            idx += 1
