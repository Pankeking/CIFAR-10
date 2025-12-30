import numpy as np

def glorot_uniform(fan_in: int, fan_out: int, size: tuple[int, ...]) -> np.ndarray:
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=size, dtype=np.float32)

def sigmoid(x: np.ndarray) -> np.ndarray:
    x =np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1, 0)

def mean_squared_error_loss(logits: np.ndarray, labels: np.ndarray) -> float:
    difference = logits - labels
    return np.mean(np.square(difference)) # scalar

def mean_squared_error_grad(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return 2 * (logits - labels)

def softmax(logits: np.ndarray) -> np.ndarray:
    logits_shift = logits - np.max(logits, axis=1, keepdims=True)
    exp_scores = np.exp(logits_shift)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs

def cross_entropy_loss(logits: np.ndarray, labels: np.ndarray) -> float:
    probs = softmax(logits)
    eps = 1e-12
    log_probs = np.log(probs + eps)
    loss = -np.sum(labels * log_probs) / labels.shape[0]
    return loss

def cross_entropy_grad(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    probs = softmax(logits)
    return (probs - labels) / labels.shape[0]


def im2col_nchw(x: np.ndarray, kernel_size: int, stride: int, padding: int) -> tuple[np.ndarray, int, int]:
    """
    x: (N, C, H, W)
    returns:
      X_col: (N * H_out * W_out, C * K * K)
      H_out, W_out
    """
    N, C, H, W = x.shape
    K = kernel_size
    S = stride
    P = padding

    # pad input
    if P > 0:
        x_padded = np.pad(x, ((0, 0), (0, 0), (P, P), (P, P)), mode="constant")
    else:
        x_padded = x

    H_p, W_p = x_padded.shape[2], x_padded.shape[3]
    H_out = (H_p - K) // S + 1
    W_out = (W_p - K) // S + 1

    # each row in cols is one (C*K*K) patch
    cols = np.zeros((N, C, K, K, H_out, W_out), dtype=x.dtype)

    for i in range(K):
        i_max = i + S * H_out
        for j in range(K):
            j_max = j + S * W_out
            # x_padded: (N, C, H_p, W_p)
            # slice: every S in spatial dims
            cols[:, :, i, j, :, :] = x_padded[:, :, i:i_max:S, j:j_max:S]

    # rearrange to (N*H_out*W_out, C*K*K)
    cols = cols.reshape(N, C * K * K, H_out * W_out)
    cols = cols.transpose(0, 2, 1).reshape(N * H_out * W_out, C * K * K)
    return cols, H_out, W_out
