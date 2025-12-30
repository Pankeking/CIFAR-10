import numpy as np

from nn.math import glorot_uniform, relu, relu_derivative

class Layer:
    def __init__(self):
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        dout: gradient w.r.t. output
        returns: gradient w.r.t. input
        """
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

        self.weights = glorot_uniform(in_features, out_features)
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


class ReLULayer(Layer):
    def __init__(self):
        super().__init__()
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache = x
        return relu(x)
        
    def backward(self, dout: np.ndarray) -> np.ndarray:
        x = self.cache
        return dout * relu_derivative(x)

    @property
    def params(self) -> list[np.ndarray]:
        return []

    @property
    def grads(self) -> list[np.ndarray]:
        return []


class Conv2DLayer(Layer):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weights = glorot_uniform(fan_in=in_channels * kernel_size * kernel_size, fan_out=out_channels)
        self.bias = np.zeros(out_channels, dtype=np.float32)

        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)

        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache = (x.shape, self.stride, self.padding)
        N, C, H, W = x.shape
        H_out = (H - 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W - 2 * self.padding - self.kernel_size) // self.stride + 1
        out = np.zeros((N, self.out_channels, H_out, W_out), dtype=np.float32)
        self.cache = (x, out.shape)
        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x, out_shape = self.cache
        return np.zeros_like(x)

    @property
    def params(self) -> list[np.ndarray]:
        return [self.weights, self.bias]

    @property
    def grads(self) -> list[np.ndarray]:
        return [self.grad_weights, self.grad_bias]
