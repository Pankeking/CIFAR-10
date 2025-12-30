import numpy as np


class Layer:
    def __init__(self):
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def params(self) -> list[np.ndarray]:
        return []

    @property
    def grads(self) -> list[np.ndarray]:
        return []


class LinearLayer(Layer):
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        self.weights = np.random.randn(in_features, out_features)
        self.bias = np.random.randn(out_features)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weights + self.bias

    def backward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weights.T


class Conv2DLayer(Layer):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.bias = np.random.randn(out_channels)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weights + self.bias