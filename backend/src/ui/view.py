import os

import matplotlib.pyplot as plt
import numpy as np

from data.data_loader import load_dataset
from nn.math import softmax


def _load_tiny_imagenet_class_ids(datasets_dir: str = "datasets") -> list[str]:
    data_dir = os.path.join(datasets_dir, "tiny-imagenet-200")
    wnids_file = os.path.join(data_dir, "wnids.txt")
    if not os.path.exists(wnids_file):
        return [f"class_{i}" for i in range(200)]
    with open(wnids_file) as f:
        wnids = [line.strip() for line in f.readlines()]
    return wnids


def _load_wnid_to_words(datasets_dir: str = "datasets") -> dict[str, str]:
    data_dir = os.path.join(datasets_dir, "tiny-imagenet-200")
    words_file = os.path.join(data_dir, "words.txt")
    mapping: dict[str, str] = {}
    if not os.path.exists(words_file):
        return mapping
    with open(words_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                wnid = parts[0]
                label = parts[1]
                mapping[wnid] = label
    return mapping


def run_view(model_filename: str, model_cls, start_index: int = 0) -> None:
    """
    Generic viewer for CIFAR-10 and Tiny ImageNet.
    model_cls: NumpyModel, TorchModel, HCModel.
    It must implement .load(), .predict_logits(), and expose .dataset_name.
    For normalization:
      - If model.norm_mean/std exist, they are used.
      - Otherwise, for numpy/hc we fall back to mean/std of input_data_shape.
      - For torch without norm stats, we show raw x_test.
    """
    # 1) Load model
    model = model_cls()
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
        wnids = _load_tiny_imagenet_class_ids()
        wnid_to_words = _load_wnid_to_words()
        class_names = [wnid_to_words.get(wnid, wnid) for wnid in wnids]
        if len(class_names) != num_classes:
            class_names = [f"class_{i}" for i in range(num_classes)]
    else:
        class_names = [f"class_{i}" for i in range(num_classes)]

    # 3) Normalize test set with model's training stats when available
    if hasattr(model, "norm_mean") and model.norm_mean is not None:
        mean = model.norm_mean
        std = model.norm_std
        x_test_norm = (x_test - mean) / std
    else:
        # Fallback only for numpy/hc models that still have input_data_shape
        if hasattr(model, "input_data_shape") and model.input_data_shape is not None:
            mean = np.mean(model.input_data_shape, axis=(0, 2, 3), keepdims=True)
            std = np.std(model.input_data_shape, axis=(0, 2, 3), keepdims=True) + 1e-8
            x_test_norm = (x_test - mean) / std
        else:
            # Torch models without stored stats: no normalization
            x_test_norm = x_test

    idx = start_index

    while True:
        if idx < 0:
            idx = 0
        if idx >= x_test_norm.shape[0]:
            print("Reached end of test set.")
            break

        # 4) Run model on single sample
        x_single = x_test_norm[idx:idx + 1]          # (1, C, H, W)
        logits = model.predict_logits(x_single)[0]   # (num_classes,)
        probs = softmax(logits[None, :])[0]

        pred_class = int(np.argmax(probs))
        true_class = int(y_test[idx])

        # 5) Prepare image for display (NCHW -> HWC), show original pixels
        img = x_test[idx].transpose(1, 2, 0)

        plt.figure(figsize=(6, 4))
        plt.imshow(img, interpolation="nearest")
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
