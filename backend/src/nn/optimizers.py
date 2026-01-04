from enum import Enum

import numpy as np

from core.layers import Layer


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
        decay_rate: float = 0.99,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        self.optimizer_mode = optimizer_mode
        self.learning_rate = learning_rate
        self.base_learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.start_epoch_decay = start_epoch_decay
        self.decay_rate = decay_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
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
            self.learning_rate = self.base_learning_rate * (self.decay_rate**k)
        self.epoch += 1

    def step(self, layers: list[Layer]) -> None:
        weights = []
        gradients = []
        for layer in layers:
            weights.extend(layer.params)
            gradients.extend(layer.grads)

        if self.optimizer_mode == OptimizerMode.SGD:
            self.step_sgd(weights, gradients)
        elif self.optimizer_mode == OptimizerMode.ADAM:
            self.step_adam(weights, gradients)
        else:
            raise ValueError(f"Invalid optimizer mode: {self.optimizer_mode}")

    def step_sgd(
        self, weights: list[np.ndarray], gradients: list[np.ndarray]
    ) -> list[np.ndarray]:
        for weight, gradient in zip(weights, gradients):
            g_reg = gradient + self.weight_decay * weight
            weight -= self.learning_rate * g_reg

    def step_adam(
        self, weights: list[np.ndarray], gradients: list[np.ndarray]
    ) -> list[np.ndarray]:
        # Initialize moment vectors on first call
        if self.m is None or self.v is None:
            self.m = [np.zeros_like(w) for w in weights]
            self.v = [np.zeros_like(w) for w in weights]
            self.t = 0

        self.t += 1

        for i, (w, g) in enumerate(zip(weights, gradients)):
            g_reg = g + self.weight_decay * w

            m = self.m[i]
            v = self.v[i]

            # Update biased first moment estimate
            m = self.beta1 * m + (1.0 - self.beta1) * g_reg
            # Update biased second raw moment estimate
            v = self.beta2 * v + (1.0 - self.beta2) * (g_reg**2)

            # Bias-corrected first and second moment estimates
            m_hat = m / (1.0 - self.beta1**self.t)
            v_hat = v / (1.0 - self.beta2**self.t)

            # Parameter update
            w -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

            # Store updated state and weight
            self.m[i] = m
            self.v[i] = v
