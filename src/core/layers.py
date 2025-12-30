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
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weights = glorot_uniform(in_channels, out_channels, size=(out_channels, in_channels))
        self.bias = np.zeros(out_channels, dtype=np.float32)

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

        fan_in = in_channels * kernel_size * kernel_size
        fan_out = out_channels * kernel_size * kernel_size
        self.weights = glorot_uniform(fan_in, fan_out, size=(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = np.zeros(out_channels, dtype=np.float32)

        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)

        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        N, C_in, H, W = x.shape
        K = self.kernel_size
        S = self.stride
        P = self.padding

        if P > 0:
            x_padded = np.pad(x, ((0,0),(0,0),(P,P),(P,P)), mode="constant")
        else:
            x_padded = x

        

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x, out_shape = self.cache
        return np.zeros_like(x)

    @property
    def params(self) -> list[np.ndarray]:
        return [self.weights, self.bias]

    @property
    def grads(self) -> list[np.ndarray]:
        return [self.grad_weights, self.grad_bias]

    def _pad(self, x: np.ndarray) -> np.ndarray:
        if self.padding == 0:
            return x
        N, C, H, W = x.shape
        padded = np.zeros((N, C, H + 2*self.padding, W + 2*self.padding), dtype=np.float32)
        padded[:, :, self.padding:-self.padding, self.padding:-self.padding] = x
        return padded
