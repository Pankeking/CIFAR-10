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
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        limit = np.sqrt(6.0 / (in_features + out_features))  # glorot-like

        self.weights = np.random.uniform(-limit, limit, size=(in_features, out_features))
        self.bias = np.zeros(out_features, dtype=np.float32)

        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)

        self.cache = None


    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache = x
        return x @ self.weights + self.bias

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x = self.cache
        N = x.shape[0]
        self.grad_weights = x.T @ dout / N
        self.grad_bias = np.sum(dout, axis=0) / N
        dx = dout @ self.weights.T
        return dx

    @property
    def params(self) -> list[np.ndarray]:
        return [self.weights, self.bias]

    @property
    def grads(self) -> list[np.ndarray]:
        return [self.grad_weights, self.grad_bias]


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