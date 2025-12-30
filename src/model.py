import pickle
import os
import numpy as np
from utils.helpers import print_metrics
from utils.data_loader import load_dataset
from mathlib.mathlib import glorot_uniform, sigmoid, relu, relu_derivative
from mathlib.losses import Loss, LossMode
from mathlib.optimizers import Optimizer, OptimizerMode

class Model:
    def __init__(self,
        weight_decay: float = 1e-3,
        loss: Loss = Loss(),
        activation_function: str = "relu",
        optimizer: Optimizer = Optimizer(),
    ):
        self.input_data_shape: np.ndarray | None = None
        self.output_data_shape: np.ndarray | None = None
        self.weights: list[np.ndarray] | None = None
        self.y_prediction: np.ndarray | None = None
        self.weight_decay = weight_decay
        self.loss = loss
        self.optimizer = optimizer
        self.dataset_name = "cifar10"

        if activation_function == "relu":
            self.activation_function = relu
        elif activation_function == "sigmoid":
            self.activation_function = sigmoid
        else:
            raise ValueError(f"Invalid activation function: {activation_function}")


    def create_model(self, number_samples: int, hidden_layer_size: int = 512, dataset_name: str = "cifar10") -> None:
        self.dataset_name = dataset_name
        print(f"Creating model with {self.dataset_name} dataset")
        x_train, y_train, _, _ = load_dataset(self.dataset_name)

        x = x_train[:number_samples]          # (N, 3072)
        y_labels = y_train[:number_samples]   # (N,)

        x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)  # Z-score normalize

        dimension_input = x.shape[1]
        num_classes = int(y_train.max() + 1)   # assumes labels 0..C-1
        dimension_output = num_classes

        y_onehot = np.zeros((number_samples, dimension_output), dtype=np.float32)
        y_onehot[np.arange(number_samples), y_labels] = 1.0

        weights = [
            glorot_uniform(dimension_input, hidden_layer_size),
            glorot_uniform(hidden_layer_size, hidden_layer_size // 2),
            glorot_uniform(hidden_layer_size // 2, hidden_layer_size // 4),
            glorot_uniform(hidden_layer_size // 4, dimension_output),
        ]

        self.input_data_shape = x
        self.output_data_shape = y_onehot
        self.weights = weights


    def save(self, filepath: str) -> None:
        model_directory = "models"
        with open(os.path.join(model_directory, filepath), 'wb') as f:
            pickle.dump({
                'weights': self.weights,
                'input_data_shape': self.input_data_shape,
                'output_data_shape': self.output_data_shape,
                'loss_mode': self.loss.loss_mode,
                'activation_function_name': (
                    'relu' if self.activation_function is relu
                    else 'sigmoid' if self.activation_function is sigmoid
                    else None
                ),
                'learning_rate': self.optimizer.learning_rate,
                'weight_decay': self.weight_decay,
                'optimizer': self.optimizer.optimizer_mode,
                'dataset_name': self.dataset_name,
            }, f)
        print(f"Model saved to {os.path.join(model_directory, filepath)}")


    def load(self, filepath: str) -> None:
        model_directory = "models"
        with open(os.path.join(model_directory, filepath), 'rb') as f:
            data = pickle.load(f)
        self.weights = data['weights']
        self.input_data_shape = data['input_data_shape']
        self.output_data_shape = data['output_data_shape']
        self.weight_decay = data.get('weight_decay', 1e-3)
        self.loss = Loss(loss_mode=data.get('loss_mode', LossMode.CROSS_ENTROPY))
        self.optimizer = Optimizer(optimizer_mode=data.get('optimizer', OptimizerMode.SGD))
        self.optimizer.learning_rate = data.get('learning_rate', 1e-2)
        self.dataset_name = data.get('dataset_name', 'cifar10')
        activation_function_name = data.get('activation_function_name', 'relu')
        if activation_function_name == "relu":
            self.activation_function = relu
        elif activation_function_name == "sigmoid":
            self.activation_function = sigmoid
        else:
            raise ValueError(f"Invalid activation function: {activation_function_name}")
        print(f"Model loaded from {os.path.join(model_directory, filepath)} with dataset {self.dataset_name}")
            
        self.y_prediction = None


    def set_activation_function(self, strategy: str) -> None:
        if strategy == "sigmoid":
            self.activation_function = sigmoid
        elif strategy == "relu":
            self.activation_function = relu
        else:
            raise ValueError(f"Invalid activation function: {strategy}")


    def train(self, epochs: int, batch_size: int, metrics: bool = False):
        self._train(epochs, batch_size, metrics)
        self.y_prediction = self._predict_full()
        print_metrics(
            self.y_prediction,
            self.output_data_shape,
            self.loss
        )
    
    
    def _train(self, epochs: int, batch_size: int = 256, metrics: bool = False):
        n = self.input_data_shape.shape[0]
        for epoch in range(epochs):
            # learning rate decay
            self.optimizer.step_epoch()

            # shuffle data
            indices = np.random.permutation(n)
            for i in range(0, n, batch_size):
                batch_idx = indices[i:i+batch_size]
                x_batch = self.input_data_shape[batch_idx]
                y_batch = self.output_data_shape[batch_idx]

                # forward pass
                hidden_activations = []
                for layer_idx, weight in enumerate(self.weights):
                    if layer_idx == 0:
                        layer_input = x_batch
                    else:
                        layer_input = hidden_activations[layer_idx - 1] # output of previous layer
                    
                    linear_output = layer_input.dot(weight)

                    if layer_idx < len(self.weights) - 1:
                        output = self.activation_function(linear_output)
                    else:
                        output = linear_output
                    hidden_activations.append(output)
                
                # backward pass
                gradients = []
                logits = hidden_activations[-1]
                gradient_next = self.loss.gradient_fn(logits, y_batch)
                for idx in reversed(range(len(self.weights))):
                    if idx == 0:
                        layer_input = x_batch
                    else:
                        layer_input = hidden_activations[idx - 1]

                    gradient_weight = layer_input.T.dot(gradient_next) + self.weight_decay * self.weights[idx]

                    if idx > 0:
                        gradient_hidden = gradient_next.dot(self.weights[idx].T)
                        relu_grad = relu_derivative(hidden_activations[idx - 1])
                        gradient_next = gradient_hidden * relu_grad
                    
                    gradients.insert(0, gradient_weight)

                self.weights = self.optimizer.update_weights(self.weights, gradients)
            
                self.y_prediction = hidden_activations[-1]
            if metrics:
                self.y_prediction = self._predict_full()
                print(f"Epoch {epoch}:")
                print_metrics(
                    self.y_prediction,
                    self.output_data_shape,
                    self.loss
                )


    def _predict_full(self) -> np.ndarray:
        """Forward pass on full dataset for final metrics."""
        return self.predict_logits(self.input_data_shape)


    def evaluate(self, batch_size: int = 512) -> float:
        _, _, x_test, y_test = load_dataset(self.dataset_name)

        # Use same normalization stats as training subset
        mean = np.mean(self.input_data_shape, axis=0, keepdims=True)
        std = np.std(self.input_data_shape, axis=0, keepdims=True) + 1e-8
        x_test = (x_test - mean) / std

        # Forward pass in batches to avoid RAM spikes
        n_test = x_test.shape[0]
        logits_list = []
        for i in range(0, n_test, batch_size):
            x_batch = x_test[i:i+batch_size]
            logits_batch = self.predict_logits(x_batch)
            logits_list.append(logits_batch)
        logits = np.vstack(logits_list)
        
        preds = np.argmax(logits, axis=1)
        acc = np.mean(preds == y_test)
        print(f"TEST Accuracy: {acc:.4f}")
        return acc


    def evaluate_on_train(self, batch_size: int = 512) -> float:
        x = self.input_data_shape
        # same forward as in evaluate, but on train data
        n = x.shape[0]
        logits_list = []
        for i in range(0, n, batch_size):
            x_batch = x[i:i+batch_size]
            logits_batch = self.predict_logits(x_batch)
            logits_list.append(logits_batch)
        logits = np.vstack(logits_list)
        preds = np.argmax(logits, axis=1)
        y_labels = np.argmax(self.output_data_shape, axis=1)
        acc = np.mean(preds == y_labels)
        print(f"TRAIN Accuracy (evaluate_on_train): {acc:.4f}")
        return acc
    
    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        hidden = x
        for idx, weight in enumerate(self.weights):
            hidden = hidden @ weight
            if idx < len(self.weights) - 1:
                hidden = self.activation_function(hidden)
        return hidden


    def predict_classes(self, x: np.ndarray) -> np.ndarray:
        logits = self.predict_logits(x)
        return np.argmax(logits, axis=1)


    def get_normalized_test(self) -> tuple[np.ndarray, np.ndarray]:
        """Return normalized CIFAR-10 test data using train stats."""
        _, _, x_test, y_test = load_dataset(self.dataset_name)
        mean = np.mean(self.input_data_shape, axis=0, keepdims=True)
        std = np.std(self.input_data_shape, axis=0, keepdims=True) + 1e-8
        x_test_norm = (x_test - mean) / std
        return x_test_norm, y_test
