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

def mean_squared_error(y_prediction: np.ndarray, y_actual: np.ndarray) -> float:
    difference = y_prediction - y_actual # (64, 10) - (64, 10) = (64, 10)
    return np.mean(np.square(difference)) # scalar

def softmax(logits: np.ndarray) -> np.ndarray:
    logits_shift = logits - np.max(logits, axis=1, keepdims=True)
    exp_scores = np.exp(logits_shift)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs

def cross_entropy_loss(y_prediction: np.ndarray, y_actual: np.ndarray) -> float:
    probs = softmax(y_prediction)
    eps = 1e-12
    log_probs = np.log(probs + eps)
    loss = -np.sum(y_actual * log_probs) / y_prediction.shape[0]
    return loss
