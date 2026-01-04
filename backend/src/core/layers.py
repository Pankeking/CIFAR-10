import numpy as np

from nn.math import col2im_nchw, glorot_uniform, im2col_nchw, relu, relu_derivative


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

        self.weights = glorot_uniform(
            in_channels, out_channels, size=(in_channels, out_channels)
        )
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
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        fan_in = in_channels * kernel_size * kernel_size
        fan_out = out_channels * kernel_size * kernel_size
        self.weights = glorot_uniform(
            fan_in, fan_out, size=(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = np.zeros(out_channels, dtype=np.float32)

        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)

        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        X_col, H_out, W_out = im2col_nchw(
            x, self.kernel_size, self.stride, self.padding
        )

        W_col = self.weights.reshape(self.out_channels, -1).T

        # GEMM
        out_col = X_col @ W_col
        out_col += self.bias  # broadcast over rows

        # reshape back to NCHW
        out = out_col.reshape(x.shape[0], H_out, W_out, self.out_channels)
        out = out.transpose(0, 3, 1, 2)

        self.cache = (X_col, x.shape, H_out, W_out)

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        X_col, x_padded_shape, H_out, W_out = self.cache
        N, C_out, H_out, W_out = dout.shape
        C_in = self.in_channels
        K = self.kernel_size

        dout_col = dout.transpose(0, 2, 3, 1).reshape(-1, C_out)

        W_col = self.weights.reshape(self.out_channels, -1).T

        dW_col = X_col.T @ dout_col
        self.grad_weights = dW_col.T.reshape(C_out, C_in, K, K)

        self.grad_bias = dout_col.sum(axis=0)

        dX_col = dout_col @ W_col.T

        dx = col2im_nchw(dX_col, x_padded_shape, K, self.stride, self.padding)
        return dx

    @property
    def params(self) -> list[np.ndarray]:
        return [self.weights, self.bias]

    @property
    def grads(self) -> list[np.ndarray]:
        return [self.grad_weights, self.grad_bias]


class MaxPool2DLayer(Layer):
    def __init__(self, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        N, C, H, W = x.shape
        K = self.kernel_size
        S = self.stride

        H_out = (H - K) // S + 1
        W_out = (W - K) // S + 1

        out = np.zeros((N, C, H_out, W_out), dtype=x.dtype)
        self.cache = (x, H_out, W_out)

        for i in range(H_out):
            h_start = i * S
            h_end = h_start + K
            for j in range(W_out):
                w_start = j * S
                w_end = w_start + K
                patch = x[:, :, h_start:h_end, w_start:w_end]  # (N, C, K, K)
                out[:, :, i, j] = patch.max(axis=(2, 3))

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x, H_out, W_out = self.cache
        N, C, H, W = x.shape
        K = self.kernel_size
        S = self.stride

        dx = np.zeros_like(x)

        for i in range(H_out):
            h_start = i * S
            h_end = h_start + K
            for j in range(W_out):
                w_start = j * S
                w_end = w_start + K

                patch = x[:, :, h_start:h_end, w_start:w_end]  # (N, C, K, K)
                max_vals = patch.max(axis=(2, 3), keepdims=True)  # (N, C, 1, 1)
                mask = patch == max_vals  # (N, C, K, K)

                dx[:, :, h_start:h_end, w_start:w_end] += (
                    mask * dout[:, :, i, j][:, :, None, None]
                )

        return dx

    @property
    def params(self):
        return []

    @property
    def grads(self):
        return []


class AvgPool2DLayer(Layer):
    def __init__(self, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        N, C, H, W = x.shape
        K = self.kernel_size
        S = self.stride

        H_out = (H - K) // S + 1
        W_out = (W - K) // S + 1

        out = np.zeros((N, C, H_out, W_out), dtype=x.dtype)
        self.cache_shape = (N, C, H, W, H_out, W_out)

        for i in range(H_out):
            h_start = i * S
            h_end = h_start + K
            for j in range(W_out):
                w_start = j * S
                w_end = w_start + K
                patch = x[:, :, h_start:h_end, w_start:w_end]  # (N, C, K, K)
                out[:, :, i, j] = patch.mean(axis=(2, 3))

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        dout: (N, C, H_out, W_out)
        returns: dx: (N, C, H, W)
        """
        N, C, H, W, H_out, W_out = self.cache_shape
        K = self.kernel_size
        S = self.stride

        dx = np.zeros((N, C, H, W), dtype=dout.dtype)
        scale = 1.0 / (K * K)

        for i in range(H_out):
            h_start = i * S
            h_end = h_start + K
            for j in range(W_out):
                w_start = j * S
                w_end = w_start + K

                dx[:, :, h_start:h_end, w_start:w_end] += (
                    dout[:, :, i, j][:, :, None, None] * scale
                )

        return dx

    @property
    def params(self):
        return []

    @property
    def grads(self):
        return []


class FlattenLayer(Layer):
    def __init__(self):
        super().__init__()
        self.orig_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.orig_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout.reshape(self.orig_shape)

    @property
    def params(self):
        return []

    @property
    def grads(self):
        return []
