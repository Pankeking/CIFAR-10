import numpy as np
from enum import Enum
from utils.math import cross_entropy_grad, cross_entropy_loss, mean_squared_error_grad, mean_squared_error_loss

class LossMode(Enum):
    MEAN_SQUARED_ERROR = "mse"
    CROSS_ENTROPY = "cross_entropy"

class Loss:
    def __init__(self, loss_mode: LossMode = LossMode.CROSS_ENTROPY):
        self.loss_mode = loss_mode
        pass

    def gradient_fn(self, logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
        if self.loss_mode == LossMode.MEAN_SQUARED_ERROR:
            return mean_squared_error_grad(logits, labels)
        elif self.loss_mode == LossMode.CROSS_ENTROPY:
            return cross_entropy_grad(logits, labels)
        else:
            raise ValueError(f"Invalid loss mode: {self.loss_mode}")

    def loss_fn(self, logits: np.ndarray, labels: np.ndarray) -> float:
        if self.loss_mode == LossMode.MEAN_SQUARED_ERROR:
            return mean_squared_error_loss(logits, labels)
        elif self.loss_mode == LossMode.CROSS_ENTROPY:
            return cross_entropy_loss(logits, labels)
        else:
            raise ValueError(f"Invalid loss mode: {self.loss_mode}")
