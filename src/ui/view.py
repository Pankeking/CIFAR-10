import os
import numpy as np
import matplotlib.pyplot as plt

from core.model import Model
from data.data_loader import load_dataset
from nn.math import softmax


def _load_tiny_imagenet_class_names(datasets_dir: str = "datasets") -> list[str]:
    """
    Load Tiny ImageNet class IDs (wnids) in the canonical order from wnids.txt.
    Returns a list of 200 strings like 'n01443537', 'n01629819', ...
    """
    data_dir = os.path.join(datasets_dir, "tiny-imagenet-200")
    wnids_file = os.path.join(data_dir, "wnids.txt")
    if not os.path.exists(wnids_file):
        # fallback to generic names
        return [f"class_{i}" for i in range(200)]
    with open(wnids_file, "r") as f:
        wnids = [line.strip() for line in f.readlines()]
    return wnids


def run_view(model_filename: str, start_index: int = 0) -> None:
    # 1) Load model
    model = Model()
    model.load(model_filename)

    dataset_name = model.dataset_name
    x_train, y_train, x_test, y_test = load_dataset(dataset_name)

    num_classes = int(y_train.max() + 1)

    # 2) Class names
    if dataset_name == "cifar10":
        class_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]
        assert num_classes == 10
    elif dataset_name == "tiny_imagenet":
        # wnids.txt defines the class order; labels 0..199 map to wnids list index
        class_names = _load_tiny_imagenet_class_names()
        # if counts mismatch, fall back to generic
        if len(class_names) != num_classes:
            class_names = [f"class_{i}" for i in range(num_classes)]
    else:
        class_names = [f"class_{i}" for i in range(num_classes)]

    # 3) Normalize test set with model's training stats
    if hasattr(model, "norm_mean") and model.norm_mean is not None:
        mean = model.norm_mean
        std = model.norm_std
    else:
        # fallback (should not happen if create_model set norm_mean/std)
        mean = np.mean(model.input_data_shape, axis=(0, 2, 3), keepdims=True)
        std = np.std(model.input_data_shape, axis=(0, 2, 3), keepdims=True) + 1e-8

    x_test_norm = (x_test - mean) / std

    idx = start_index

    while True:
        if idx < 0:
            idx = 0
        if idx >= x_test_norm.shape[0]:
            print("Reached end of test set.")
            break

        # 4) Run model on single sample
        x_single = x_test_norm[idx:idx+1]           # (1, C, H, W)
        logits = model.predict_logits(x_single)[0]  # (num_classes,)
        probs = softmax(logits[None, :])[0]

        pred_class = int(np.argmax(probs))
        true_class = int(y_test[idx])

        # 5) Prepare image for display (NCHW -> HWC)
        img = x_test[idx].transpose(1, 2, 0)        # (H, W, C)

        plt.imshow(img)
        plt.axis("off")
        title_true = class_names[true_class] if true_class < len(class_names) else str(true_class)
        title_pred = class_names[pred_class] if pred_class < len(class_names) else str(pred_class)
        plt.title(f"{dataset_name} | idx {idx} | true: {title_true} | pred: {title_pred}")
        plt.show(block=False)

        # 6) Console info
        print(f"\nIndex: {idx}")
        print(f"True:      {title_true}")
        print(f"Predicted: {title_pred}")
        print("Top-3 classes:")
        top3 = np.argsort(probs)[-3:][::-1]
        for k in top3:
            name = class_names[k] if k < len(class_names) else str(k)
            print(f"  {name:>15}: {probs[k]:.3f}")

        cmd = input("\n[Enter]=next, p=prev, q=quit > ").strip().lower()
        plt.close()
        if cmd == "q":
            break
        elif cmd == "p":
            idx -= 1
        else:
            idx += 1
