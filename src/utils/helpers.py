import numpy as np
from nn.losses import Loss


def print_metrics(logits: np.ndarray, labels: np.ndarray, loss: Loss, topk: int = 3):
    loss_value = loss.loss_fn(logits, labels)
    pred = np.argmax(logits, axis=1)
    true = np.argmax(labels, axis=1)

    # top-1 accuracy
    acc = np.mean(pred == true)

    # top-k accuracy
    topk_idx = np.argsort(logits, axis=1)[:, -topk:]  # last k = largest
    topk_hits = np.any(topk_idx == true[:, None], axis=1)
    topk_acc = np.mean(topk_hits)

    print(f"Accuracy: {acc:.4f}")
    print(f"Top-{topk} accuracy: {topk_acc:.4f}")
    print(f"Loss: {loss_value:.6f}")
    print("First sample logits:", logits[0])
    print("First sample true / pred:", int(true[0]), int(pred[0]))
