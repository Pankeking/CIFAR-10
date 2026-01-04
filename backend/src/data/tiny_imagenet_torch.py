# data/tiny_imagenet_torch.py
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_tiny_imagenet_loaders(
    data_root: str,
    batch_size: int,
    device,
    num_workers: int = 4,
    max_samples: int | None = None,
):
    normalize = transforms.Normalize(
        mean=[0.4802, 0.4481, 0.3975],
        std=[0.2302, 0.2265, 0.2262],
    )

    train_tf = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_ds = datasets.ImageFolder(f"{data_root}/train", transform=train_tf)
    val_ds = datasets.ImageFolder(f"{data_root}/val", transform=test_tf)

    if max_samples is not None and max_samples < len(train_ds):
        indices = torch.randperm(len(train_ds))[:max_samples]
        train_ds = Subset(train_ds, indices)

    use_pin = device.type == "cuda"

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
