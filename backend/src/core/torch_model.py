# core/torch_model.py

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.res_block import ResBlock
from core.torch_train import evaluate, train_one_epoch


class TorchModel(nn.Module):
    def __init__(
        self, in_channels: int = 3, num_classes: int = 10, base_channels: int = 32
    ):
        super().__init__()
        self.dataset_name: Optional[str] = None
        self.device = torch.device("cpu")
        self.base_channels = base_channels
        self.num_classes = num_classes

        # For viewer compatibility only
        self.norm_mean: Optional[np.ndarray] = None
        self.norm_std: Optional[np.ndarray] = None

        # Build feature extractor; head will be initialized once we know H, W
        self.features = None
        self.classifier = None
        self._in_channels = in_channels

    def _build_features(self, C_in: int, C_out: int):
        return nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C_out),
            nn.ReLU(inplace=True),
            ResBlock(C_out, C_out),
            ResBlock(C_out, C_out * 2, stride=2),
            ResBlock(C_out * 2, C_out * 2, stride=2),
            ResBlock(C_out * 2, C_out * 4, stride=2),
            #
            ResBlock(C_out * 4, C_out * 4),
            ResBlock(C_out * 4, C_out * 8, stride=2),
            ResBlock(C_out * 8, C_out * 8),
            ResBlock(C_out * 8, C_out * 8, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )

    def init_head(self, input_shape):
        # input_shape: (C, H, W)
        C, H, W = input_shape

        # Build features on CPU first
        self.features = self._build_features(C, self.base_channels)

        with torch.no_grad():
            x = torch.zeros(1, C, H, W, device=self.device)
            # Make sure features are on the correct device
            self.features.to(self.device)
            x = self.features(x)
            _, C_f, H_f, W_f = x.shape
            flat_dim = C_f * H_f * W_f

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, self.base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.base_channels * 4, self.num_classes),
        )

        # Move classifier to the correct device as well
        self.classifier.to(self.device)

    def forward(self, x):
        # Lazy head init: build features/head on first batch
        if self.features is None or self.classifier is None:
            _, C, H, W = x.shape
            self.init_head((C, H, W))
        x = self.features(x)
        x = self.classifier(x)
        return x

    # -------------------- TRAIN / EVAL -------------------- #

    def train_torch(
        self,
        epochs: int,
        train_loader: DataLoader,
        optimizer,
        optimizer_config,
        device,
        metrics: bool = False,
    ):
        self.train()
        for epoch in range(epochs):
            if (
                hasattr(optimizer_config, "start_epoch_decay")
                and epoch >= optimizer_config.start_epoch_decay
            ):
                decay_epochs = epoch - optimizer_config.start_epoch_decay + 1
                lr_multiplier = optimizer_config.decay_rate**decay_epochs
                for param_group in optimizer.param_groups:
                    param_group["lr"] = optimizer_config.learning_rate * lr_multiplier

            train_loss, train_acc = train_one_epoch(
                self, train_loader, optimizer, device
            )
            if metrics:
                print(f"Epoch {epoch}: Loss={train_loss:.4f}, Acc={train_acc:.4f}")

    def evaluate_torch(self, loader: DataLoader, device) -> float:
        self.to(device)
        test_loss, test_acc = evaluate(self, loader, device)
        print(f"Accuracy: {test_acc:.4f}")
        return test_acc

    # -------------------- VIEWER COMPATIBILITY -------------------- #

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            x_tensor = torch.from_numpy(x).float().to(self.device)
            # If norm_mean/std were saved, apply them
            if self.norm_mean is not None and self.norm_std is not None:
                mean = torch.from_numpy(self.norm_mean).float().to(self.device)
                std = torch.from_numpy(self.norm_std).float().to(self.device)
                x_tensor = (x_tensor - mean) / std
            logits = self(x_tensor)
            return logits.cpu().numpy()

    def predict_classes(self, x: np.ndarray) -> np.ndarray:
        logits = self.predict_logits(x)
        return np.argmax(logits, axis=1)

    # -------------------- SAVE / LOAD -------------------- #

    def save(self, filepath: str):
        model_directory = "models"
        os.makedirs(model_directory, exist_ok=True)

        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "dataset_name": self.dataset_name,
                "base_channels": self.base_channels,
                "num_classes": self.num_classes,
                "norm_mean": self.norm_mean,
                "norm_std": self.norm_std,
            },
            os.path.join(model_directory, filepath),
        )
        print(f"Torch model saved to {os.path.join(model_directory, filepath)}")

    def load(self, filepath: str):
        model_directory = "models"
        full_path = os.path.join(model_directory, filepath)

        ckpt = torch.load(full_path, map_location="cpu", weights_only=False)

        # Restore basic metadata
        self.dataset_name = ckpt.get("dataset_name", None)
        self.base_channels = ckpt.get("base_channels", 32)
        self.num_classes = ckpt.get("num_classes", 10)
        self.norm_mean = ckpt.get("norm_mean", None)
        self.norm_std = ckpt.get("norm_std", None)

        if self.dataset_name == "tiny_imagenet":
            H = W = 64
        else:
            H = W = 32

        self.to(torch.device("cpu"))
        dummy = torch.zeros(1, 3, H, W)
        _ = self(dummy)

        missing, unexpected = self.load_state_dict(
            ckpt["model_state_dict"], strict=False
        )

        if missing:
            print(f"Missing keys when loading: {missing}")
        if unexpected:
            print(f"Unexpected keys when loading: {unexpected}")

        print(f"Torch model loaded from {full_path} with dataset {self.dataset_name}")

    def to(self, device):
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        super().to(self.device)
        return self
