import os
import pickle

import numpy as np

from core.layers import (
    Conv2DLayer,
    FlattenLayer,
    LinearLayer,
    MaxPool2DLayer,
    ReLULayer,
)
from data.data_loader import load_dataset
from nn.losses import Loss, LossMode
from nn.optimizers import Optimizer, OptimizerMode
from utils.helpers import print_metrics


class NumpyModel:
    def __init__(self,
        loss: Loss = None,
        optimizer: Optimizer = None,
    ):
        if loss is None:
            loss = Loss()
        if optimizer is None:
            optimizer = Optimizer()
        self.input_data_shape: np.ndarray | None = None
        self.output_data_shape: np.ndarray | None = None
        self.loss = loss
        self.optimizer = optimizer
        self.dataset_name = "cifar10"


    def create_model(self, number_samples: int, dataset_name: str, C_out: int = 32) -> None:
        self.dataset_name = dataset_name
        print(f"Creating model with {self.dataset_name} dataset and {C_out} output channels")
        x_train, y_train, _, _ = load_dataset(self.dataset_name)

        x = x_train[:number_samples]          # (N, 3072)
        y_labels = y_train[:number_samples]   # (N,)

        # Z-score normalize per-channel over all pixels
        mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
        std  = np.std(x, axis=(0, 2, 3), keepdims=True) + 1e-8
        x = (x - mean) / std

        N, C, H, W = x.shape
        num_classes = int(y_train.max() + 1)   # assumes labels 0..C-1

        y_onehot = np.zeros((number_samples, num_classes), dtype=np.float32)
        y_onehot[np.arange(number_samples), y_labels] = 1.0

        layers = [
            Conv2DLayer(in_channels=C, out_channels=C_out, kernel_size=3, stride=1, padding=1),
            ReLULayer(),
            Conv2DLayer(in_channels=C_out, out_channels=C_out*2, kernel_size=3, stride=1, padding=1),
            ReLULayer(),
            MaxPool2DLayer(kernel_size=2, stride=2),
            Conv2DLayer(in_channels=C_out*2, out_channels=C_out*2, kernel_size=3, stride=1, padding=1),
            ReLULayer(),
            Conv2DLayer(in_channels=C_out*2, out_channels=C_out*4, kernel_size=3, stride=1, padding=1),
            ReLULayer(),
            MaxPool2DLayer(kernel_size=2, stride=2),
            Conv2DLayer(in_channels=C_out*4, out_channels=C_out*4, kernel_size=3, stride=1, padding=1),
            ReLULayer(),
            Conv2DLayer(in_channels=C_out*4, out_channels=C_out*4, kernel_size=3, stride=1, padding=1),
            ReLULayer(),
            MaxPool2DLayer(kernel_size=2, stride=2),
            FlattenLayer(),
            LinearLayer(in_channels=C_out*4*4*4*4, out_channels=C_out*4),
            ReLULayer(),
            LinearLayer(in_channels=C_out*4, out_channels=num_classes),
        ]
        self.layers = layers

        self.input_data_shape = x
        self.output_data_shape = y_onehot
        self.norm_mean = mean
        self.norm_std = std


    def save(self, filepath: str) -> None:
        model_directory = "models"
        with open(os.path.join(model_directory, filepath), "wb") as f:
            pickle.dump({
                "layers": self.layers,
                "loss_mode": self.loss.loss_mode,
                "learning_rate": self.optimizer.learning_rate,
                "optimizer": self.optimizer.optimizer_mode,
                "dataset_name": self.dataset_name,
                "norm_mean": self.norm_mean,
                "norm_std": self.norm_std,
            }, f)
        print(f"Model saved to {os.path.join(model_directory, filepath)}")


    def load(self, filepath: str) -> None:
        model_directory = "models"
        with open(os.path.join(model_directory, filepath), "rb") as f:
            data = pickle.load(f)
        self.layers = data["layers"]
        self.loss = Loss(loss_mode=data.get("loss_mode", LossMode.CROSS_ENTROPY))
        self.optimizer = Optimizer(optimizer_mode=data.get("optimizer", OptimizerMode.SGD))
        self.optimizer.learning_rate = data.get("learning_rate", 1e-2)
        self.dataset_name = data.get("dataset_name", "cifar10")
        self.norm_mean = data.get("norm_mean", None)
        self.norm_std = data.get("norm_std", None)
        print(f"Model loaded from {os.path.join(model_directory, filepath)} with dataset {self.dataset_name}")
            



    def train(self, epochs: int, batch_size: int, metrics: bool = False):
        self._train(epochs, batch_size, metrics)
        logits = self.predict_logits(self.input_data_shape)
        print_metrics(logits, self.output_data_shape, self.loss)
    
    
    def _train(self, epochs: int, batch_size: int = 256, metrics: bool = False):
        n = self.input_data_shape.shape[0]
        for epoch in range(epochs):
            # learning rate decay
            self.optimizer.step_epoch()

            # shuffle data
            indices = np.random.permutation(n)
            for i in range(0, n, batch_size):
                batch_idx = indices[i:i+batch_size]
                x_batch = self.input_data_shape[batch_idx]
                y_batch = self.output_data_shape[batch_idx]

                # forward pass
                out = x_batch
                for layer in self.layers:
                    out = layer.forward(out)
            
                dout = self.loss.gradient_fn(out, y_batch)

                # backward pass
                for layer in reversed(self.layers):
                    dout = layer.backward(dout)

                self.optimizer.step(self.layers)

            if metrics:
                logits = self.predict_logits(self.input_data_shape)
                print(f"Epoch {epoch}:")
                print_metrics(logits, self.output_data_shape, self.loss)



    def evaluate(self, batch_size: int = 512) -> float:
        _, _, x_test, y_test = load_dataset(self.dataset_name)

        # Use same normalization stats as training subset
        x_test = (x_test - self.norm_mean) / self.norm_std

        # Forward pass in batches to avoid RAM spikes
        n_test = x_test.shape[0]
        logits_list = []
        for i in range(0, n_test, batch_size):
            x_batch = x_test[i:i+batch_size]
            logits_batch = self.predict_logits(x_batch)
            logits_list.append(logits_batch)
        logits = np.vstack(logits_list)
        
        preds = np.argmax(logits, axis=1)
        acc = np.mean(preds == y_test)
        print(f"TEST Accuracy: {acc:.4f}")
        return acc


    def evaluate_on_train(self, batch_size: int = 512) -> float:
        x = self.input_data_shape
        # same forward as in evaluate, but on train data
        n = x.shape[0]
        logits_list = []
        for i in range(0, n, batch_size):
            x_batch = x[i:i+batch_size]
            logits_batch = self.predict_logits(x_batch)
            logits_list.append(logits_batch)
        logits = np.vstack(logits_list)
        preds = np.argmax(logits, axis=1)
        y_labels = np.argmax(self.output_data_shape, axis=1)
        acc = np.mean(preds == y_labels)
        print(f"TRAIN Accuracy (evaluate_on_train): {acc:.4f}")
        return acc
    
    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out


    def predict_classes(self, x: np.ndarray) -> np.ndarray:
        logits = self.predict_logits(x)
        return np.argmax(logits, axis=1)
