import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from PIL import Image


def get_cifar10_loaders(data_root: str,
                        batch_size: int,
                        device: torch.device,
                        num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    train_tf = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_ds = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_tf)

    use_pin = (device.type == "cuda")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin,
    )
    return train_loader, test_loader




class TinyImageNetValDataset(Dataset):
    """
    Validation dataset for Tiny ImageNet.
    Uses val/images and val_annotations.txt, with a given class_to_idx mapping.
    """

    def __init__(self, val_dir: str, class_to_idx: dict[str, int], transform=None):
        """
        val_dir: path to tiny-imagenet-200/val
        class_to_idx: mapping wnid -> class index (from train ImageFolder)
        """
        self.transform = transform
        self.samples: list[tuple[str, int]] = []

        img_dir = os.path.join(val_dir, "images")
        ann_file = os.path.join(val_dir, "val_annotations.txt")

        # val_annotations.txt format:
        # <image_name>\t<wnid>\t<x>\t<y>\t<w>\t<h>
        img_to_wnid: dict[str, str] = {}
        with open(ann_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    img_name, wnid = parts[0], parts[1]
                    img_to_wnid[img_name] = wnid

        for img_name, wnid in img_to_wnid.items():
            if wnid not in class_to_idx:
                # Skip classes not in train set (if you subset)
                continue
            label = class_to_idx[wnid]
            img_path = os.path.join(img_dir, img_name)
            if os.path.exists(img_path):
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_tiny_imagenet_loaders(data_root: str,
                              batch_size: int,
                              device: torch.device,
                              max_samples: int | None = None,
                              num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    Tiny ImageNet loaders for torch backend.

    Train:
      - datasets.ImageFolder on train/<wnid>/images
    Val:
      - TinyImageNetValDataset using val/images + val_annotations.txt
      - Uses the same class_to_idx (wnid -> idx) as train
    """
    normalize = transforms.Normalize(
        mean=[0.4802, 0.4481, 0.3975],
        std=[0.2302, 0.2265, 0.2262],
    )

    train_tf = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        normalize,
    ])
    test_tf = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        normalize,
    ])

    train_dir = os.path.join(data_root, "train")
    val_dir   = os.path.join(data_root, "val")

    # 1) Train dataset from folder structure
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    class_to_idx = train_ds.class_to_idx  # wnid -> idx

    # Optional train subset
    if max_samples is not None and max_samples < len(train_ds):
        indices = torch.randperm(len(train_ds))[:max_samples]
        train_ds = Subset(train_ds, indices)

    # 2) Val dataset from annotations (no ImageFolder here)
    val_ds = TinyImageNetValDataset(val_dir, class_to_idx, transform=test_tf)

    use_pin = (device.type == "cuda")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin,
    )
    return train_loader, val_loader
