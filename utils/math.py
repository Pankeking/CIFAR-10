import numpy as np

def glorot_uniform(in_dim: int, out_dim: int) -> np.ndarray:
    limit = np.sqrt(6.0 / (in_dim + out_dim))
    return np.random.uniform(-limit, limit, size=(in_dim, out_dim))

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