import numpy as np
from nn.losses import Loss

def print_metrics(labels: np.ndarray, loss: Loss):
    loss_value = loss.loss_fn(logits, labels)
    pred_classes = np.argmax(logits, axis=1)          # shape (64,)
    true_classes = np.argmax(labels, axis=1)
    class_diff = np.abs(pred_classes - true_classes)        # (64,)
    accuracy = np.mean(class_diff == 0)
    print(f"Accuracy: {accuracy}")
    print(f"Loss: {loss_value}")
    print("Class diff:", class_diff)
    print("First sample logits:", logits[0])
