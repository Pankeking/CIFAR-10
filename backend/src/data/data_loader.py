import os
import pickle

import numpy as np
from PIL import Image


def load_dataset(dataset_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if dataset_name == "cifar10":
        return _load_cifar10()
    elif dataset_name == "tiny_imagenet":
        return _load_tiny_imagenet()
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")


def _load_cifar_batch(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        batch = pickle.load(f, encoding="latin1")
    data = batch["data"]          # shape (10000, 3072), uint8
    labels = np.array(batch["labels"], dtype=np.int64)  # shape (10000,)
    data = data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    return data, labels


def _load_cifar10() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset_name = "cifar-10-batches-py"
    datasets_dir = "datasets"
    data_dir = os.path.join(datasets_dir, dataset_name)
    xs = []
    ys = []
    for i in range(1, 6):
        path = os.path.join(data_dir, f"data_batch_{i}")
        x, y = _load_cifar_batch(path)
        xs.append(x)
        ys.append(y)
    x_train = np.concatenate(xs, axis=0)   # (50000, 3072)
    y_train = np.concatenate(ys, axis=0)   # (50000,)

    x_test, y_test = _load_cifar_batch(os.path.join(data_dir, "test_batch"))

    return x_train, y_train, x_test, y_test

def _load_tiny_imagenet() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    datasets_dir = "datasets"
    data_dir = os.path.join(datasets_dir, "tiny-imagenet-200")
    val_dir = os.path.join(data_dir, "val")

    _test_dir = os.path.join(data_dir, "test")
    # read class ids
    wnids_file = os.path.join(data_dir, "wnids.txt")
    with open(wnids_file) as f:
        wnids = [line.strip() for line in f.readlines()]
    wnid_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}

    # load train images
    train_images = []
    train_labels = []
    train_dir = os.path.join(data_dir, "train")
    for wnid in wnids:
        class_dir = os.path.join(train_dir, wnid, "images")
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith((".jpeg", ".jpg", ".png")):
                continue
            image_path = os.path.join(class_dir, fname)
            image = Image.open(image_path).convert("RGB")
            arr = np.asarray(image, dtype=np.float32) / 255.0
            arr = arr.transpose(2, 0, 1)
            train_images.append(arr)      # (3, 64, 64)
            train_labels.append(wnid_to_idx[wnid])

    x_train = np.stack(train_images, axis=0)
    y_train = np.array(train_labels, dtype=np.int64)

    # load val images + labels
    val_dir = os.path.join(data_dir, "val")
    val_img_dir = os.path.join(val_dir, "images")
    ann_path = os.path.join(val_dir, "val_annotations.txt")
    fname_to_wnid = {}
    with open(ann_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            fname, wnid = parts[0], parts[1]
            fname_to_wnid[fname] = wnid

    val_images = []
    val_labels = []
    for fname, wnid in fname_to_wnid.items():
        img_path = os.path.join(val_img_dir, fname)
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)
        val_images.append(arr)
        val_labels.append(wnid_to_idx[wnid])

    x_val = np.stack(val_images, axis=0)
    y_val = np.array(val_labels, dtype=np.int64)

    return x_train, y_train, x_val, y_val