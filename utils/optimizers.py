import numpy as np
from enum import Enum

class OptimizerMode(Enum):
    ADAM = "adam"
    SGD = "sgd"

class Optimizer:
    def __init__(
        self,
        optimizer_mode: OptimizerMode = OptimizerMode.SGD,
        learning_rate: float = 1e-2,
        weight_decay: float = 1e-3,
        start_epoch_decay: int = 30,
        decay_rate: float = 0.99
    ):
        self.optimizer_mode = optimizer_mode
        self.learning_rate = learning_rate
        self.base_learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.start_epoch_decay = start_epoch_decay
        self.decay_rate = decay_rate

        self.epoch: int = 0

        self.m: list[np.ndarray] | None = None
        self.v: list[np.ndarray] | None = None
        self.t: int = 0

    def step_epoch(self) -> None:
        """Update internal learning rate based on epoch and schedule."""
        if self.epoch < self.start_epoch_decay:
            self.learning_rate = self.base_learning_rate
        else:
            k = self.epoch - self.start_epoch_decay
            self.learning_rate = self.base_learning_rate * (self.decay_rate ** k)
        self.epoch += 1

    def update_weights(self, weights: list[np.ndarray], gradients: list[np.ndarray]) -> list[np.ndarray]:
        if self.optimizer_mode == OptimizerMode.SGD:
            return self.update_weights_sgd(weights, gradients)
        elif self.optimizer_mode == OptimizerMode.ADAM:
            return self.update_weights_adam(weights, gradients)
        else:
            raise ValueError(f"Invalid optimizer mode: {self.optimizer_mode}")

    def update_weights_sgd(self, weights: list[np.ndarray], gradients: list[np.ndarray]) -> list[np.ndarray]:
        updated_weights = []
        for weight, gradient in zip(weights, gradients):
            updated_weight = weight - (self.learning_rate * gradient)
            updated_weights.append(updated_weight)
        return updated_weights
    
    def update_weights_adam(self, weights: list[np.ndarray], gradients: list[np.ndarray]) -> list[np.ndarray]:
        pass


