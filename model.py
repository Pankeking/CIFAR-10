import pickle
import os
import numpy as np
from utils.helpers import print_metrics, load_cifar10
from utils.math import glorot_uniform, sigmoid, relu, relu_derivative, mean_squared_error, cross_entropy_loss, softmax

class Model:
    def __init__(self, learning_rate: float = 1e-2, weight_decay: float = 1e-3, loss_mode: str = "cross_entropy", activation_function: str = "relu"):
        self.input_data_shape: np.ndarray | None = None
        self.output_data_shape: np.ndarray | None = None
        self.weights: list[np.ndarray] | None = None
        self.y_prediction: np.ndarray | None = None
        
        if activation_function == "relu":
            self.activation_function = relu
        elif activation_function == "sigmoid":
            self.activation_function = sigmoid
        else:
            raise ValueError(f"Invalid activation function: {activation_function}")

        self.learning_rate: float | None = learning_rate
        self.weight_decay: float | None = weight_decay
        self.loss_mode: str | None = loss_mode

    def update_weights(self, weights: list[np.ndarray]) -> None:
        self.weights = weights

    def set_learning(self, learning_rate: float, weight_decay: float) -> None:
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
  
    def create_model(self, strategy: str, number_samples: int) -> None:
        print(f"Creating model with strategy: {strategy}")
        if strategy == "hardcode":
            self.build_hardcode_network()
        elif strategy == "cifar10":
            self.build_cifar10_network(number_samples)
        else:
            raise ValueError(f"Invalid strategy: {strategy}")

    def loss_function(self, y_prediction: np.ndarray = None, y_actual: np.ndarray = None) -> float:
        y_prediction = y_prediction if y_prediction is not None else self.y_prediction
        y_actual = y_actual if y_actual is not None else self.output_data_shape

        if self.loss_mode == "mse": # mean squared error
            return mean_squared_error(y_prediction, y_actual)
        elif self.loss_mode == "cross_entropy":
            return cross_entropy_loss(y_prediction, y_actual)
        else:
            raise ValueError(f"Invalid loss mode: {self.loss_mode}")

    def build_cifar10_network(self, number_samples: int) -> None:
        x_train, y_train, x_test, y_test = load_cifar10("cifar-10-batches-py")

        x = x_train[:number_samples]          # (N, 3072)
        y_labels = y_train[:number_samples]   # (N,)

        x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)  # Z-score normalize

        dimension_input = 32 * 32 * 3
        hidden_layer_size = 1024
        dimension_output = 10

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

    def build_hardcode_network(self) -> None:
        print("Hardcoding layer sizes...")
        number_samples = 10000
        dimension_input = 32 * 32 * 3
        hidden_layer_size = 1024
        dimension_output = 10
        layer_sizes = [dimension_input, hidden_layer_size, hidden_layer_size // 8, dimension_output]
        
        weights = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            weights.append(glorot_uniform(in_dim, out_dim))
        
        input_dim = layer_sizes[0]
        output_dim = layer_sizes[-1]

        self.input_data_shape = np.random.randn(number_samples, input_dim)
        self.output_data_shape = np.random.randn(number_samples, output_dim)
        self.weights = weights

    def save(self, filepath: str) -> None:
        model_directory = os.path.dirname("models")
        with open(os.path.join(model_directory, filepath), 'wb') as f:
            pickle.dump({
                'weights': self.weights,
                'input_data_shape': self.input_data_shape,
                'output_data_shape': self.output_data_shape,
                'loss_mode': self.loss_mode,
                'activation_function_name': (
                    'relu' if self.activation_function is relu
                    else 'sigmoid' if self.activation_function is sigmoid
                    else None
                ),
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay
            }, f)
        print(f"Model saved to {os.path.join(model_directory, filepath)}")

    def save_weights_only(self, filepath: str) -> None:
        model_directory = os.path.dirname("models")
        np.savez_compressed(os.path.join(model_directory, filepath), weights=self.weights)
        print(f"Weights saved to {os.path.join(model_directory, filepath)}")

    def load(self, filepath: str, rebuild_data: bool = False) -> None:
        model_directory = os.path.dirname("models")
        with open(os.path.join(model_directory, filepath), 'rb') as f:
            data = pickle.load(f)
        self.weights = data['weights']
        self.input_data_shape = data['input_data_shape']
        self.output_data_shape = data['output_data_shape']
        self.learning_rate = data.get('learning_rate', 1e-2)
        self.weight_decay = data.get('weight_decay', 1e-3)
        self.loss_mode = data.get('loss_mode', "cross_entropy")
        activation_function_name = data.get('activation_function_name', 'relu')
        if activation_function_name == "relu":
            self.activation_function = relu
        elif activation_function_name == "sigmoid":
            self.activation_function = sigmoid
        else:
            raise ValueError(f"Invalid activation function: {activation_function_name}")
        print(f"Model loaded from {os.path.join(model_directory, filepath)}")
            
        if rebuild_data:  # Rebuild if data shapes missing
            self.create_model("cifar10")
        
        self.y_prediction = None  # Reset prediction cache

    

    def set_activation_function(self, strategy: str) -> None:
        if strategy == "sigmoid":
            self.activation_function = sigmoid
        elif strategy == "relu":
            self.activation_function = relu
        else:
            raise ValueError(f"Invalid activation function: {strategy}")

    def train(self, strategy: str, epochs: int, batch_size: int, metrics: bool = False):
        print(f"Training model with strategy: {strategy}")
        if strategy == "hardcode":
            self.train_hardcode(epochs, metrics)
        elif strategy == "layered":
            self.train_layered(epochs, batch_size, metrics)
        else:
            raise ValueError(f"Invalid strategy: {strategy}")
        self.y_prediction = self._predict_full()
        print_metrics(self.y_prediction, self.output_data_shape, self.loss_function())
        return 0
    
    
    def train_layered(self, epochs: int, batch_size: int = 256, metrics: bool = False):
        n = self.input_data_shape.shape[0]
        for epoch in range(epochs):
            start_epoch_decay = 30
            if epoch < start_epoch_decay:
                lr = self.learning_rate
            else:
                lr = self.learning_rate * (0.99 ** (epoch - start_epoch_decay))
            indices = np.random.permutation(n)
            for i in range(0, n, batch_size):
                batch_idx = indices[i:i+batch_size]
                x_batch = self.input_data_shape[batch_idx]
                y_batch = self.output_data_shape[batch_idx]

                # forward pass
                # numpy.dot inverts (transposes) the order of the matrices row/column wise
                # and then multiplies the matrices
                # x.dot(y) is equivalent to x^T * y
                hidden_activations = []
                for layer_idx, weight in enumerate(self.weights):
                    if layer_idx == 0:
                        layer_input = x_batch
                    else:
                        layer_input = hidden_activations[layer_idx - 1] # output of previous layer
                    
                    linear_output = layer_input.dot(weight)

                    # linear_output = (linear_output - np.mean(linear_output, axis=0,keepdims=True)) / (np.std(linear_output, axis=0, keepdims=True) + 1e-8)

                    if layer_idx < len(self.weights) - 1:
                        output = self.activation_function(linear_output)
                    else:
                        output = linear_output

                    hidden_activations.append(output)
                
                # backward pass
                new_weights = []
                logits = hidden_activations[-1]
                if self.loss_mode == "mse":
                    gradient_next = 2 * (logits - y_batch) # (64, 10)
                elif self.loss_mode == "cross_entropy":
                    probs = softmax(logits) # (64, 10)
                    gradient_next = (probs - y_batch) / y_batch.shape[0]
                else:
                    raise ValueError("Unknown loss_mode")

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
                    new_weights.insert(0, self.weights[idx] - (lr * gradient_weight))
                    

                self.update_weights(new_weights)

            
                self.y_prediction = hidden_activations[-1]
            if metrics:
                self.y_prediction = self._predict_full()
                print(f"Epoch {epoch}:")
                print_metrics(self.y_prediction, self.output_data_shape, self.loss_function())

        return 0


    def train_hardcode(self, epochs: int, metrics: bool = False):
        for epoch in range(epochs):
            hidden_linear = self.input_data_shape.dot(self.weights[0]) # (64, 1000) * (1000, 100) = (64, 100)
            hidden_activation = self.activation_function(hidden_linear) # hidden layer activation (64, 100)

            hidden_linear_2 = hidden_activation.dot(self.weights[1])
            hidden_activation_2 = self.activation_function(hidden_linear_2)

            logits = hidden_activation_2.dot(self.weights[2]) # (64, 100) * (100, 10) = (64, 10)
            # y_prediction = sigmoid(logits) # output layer activation (64, 10)
            # we can apply softmax to the logits to get the probability distribution
            self.y_prediction = logits

            new_weights = []

            gradient_y_prediction = 2 * (self.y_prediction - self.output_data_shape) # (64, 10) - (64, 10) = (64, 10)

            gradient_weights_3 = hidden_activation_2.T.dot(gradient_y_prediction) # (100, 64) * (64, 10) = (100, 10)
            gradient_hidden_activation_2 = gradient_y_prediction.dot(self.weights[2].T)
            gradient_hidden_linear_2 = gradient_hidden_activation_2 * hidden_activation_2 * (1 - hidden_activation_2)
            gradient_weights_2 = hidden_activation.T.dot(gradient_hidden_linear_2)
            gradient_hidden_activation = gradient_hidden_linear_2.dot(self.weights[1].T)
            gradient_hidden_linear = gradient_hidden_activation * hidden_activation * (1 - hidden_activation)
            gradient_weights_1 = self.input_data_shape.T.dot(gradient_hidden_linear)

            new_weights.append(self.weights[0] - (self.learning_rate * gradient_weights_1))
            new_weights.append(self.weights[1] - (self.learning_rate * gradient_weights_2))
            new_weights.append(self.weights[2] - (self.learning_rate * gradient_weights_3))

            self.update_weights(new_weights)

            if metrics and epoch % 100 == 0:
                print(f"Epoch {epoch}:")
                print_metrics(self.y_prediction, self.output_data_shape, self.loss_function(self.y_prediction, self.output_data_shape))

        return 0

    def _predict_full(self) -> np.ndarray:
        """Forward pass on full dataset for final metrics."""
        hidden_activations = []
        for idx, weight in enumerate(self.weights):
            if idx == 0:
                layer_input = self.input_data_shape
            else:
                layer_input = hidden_activations[idx - 1]
            
            linear_output = layer_input.dot(weight)
            if idx < len(self.weights) - 1:
                output = self.activation_function(linear_output)
            else:
                output = linear_output
            hidden_activations.append(output)
        return hidden_activations[-1]

    def evaluate(self, batch_size: int = 512) -> float:
        _, _, x_test, y_test = load_cifar10("cifar-10-batches-py")
        number_samples = self.input_data_shape.shape[0]

        # Use same normalization stats as training subset
        mean = np.mean(self.input_data_shape, axis=0, keepdims=True)
        std = np.std(self.input_data_shape, axis=0, keepdims=True) + 1e-8
        x_test = (x_test - mean) / std

        # Forward pass in batches to avoid RAM spikes
        n_test = x_test.shape[0]
        logits_list = []
        for i in range(0, n_test, batch_size):
            x_batch = x_test[i:i+batch_size]
            hidden = x_batch
            for idx, weight in enumerate(self.weights):
                hidden = hidden @ weight
                if idx < len(self.weights) - 1:
                    hidden = self.activation_function(hidden)
            logits_list.append(hidden)
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
            hidden = x_batch
            for idx, weight in enumerate(self.weights):
                hidden = hidden @ weight
                if idx < len(self.weights) - 1:
                    hidden = self.activation_function(hidden)
            logits_list.append(hidden)
        logits = np.vstack(logits_list)
        preds = np.argmax(logits, axis=1)
        y_labels = np.argmax(self.output_data_shape, axis=1)
        acc = np.mean(preds == y_labels)
        print(f"TRAIN Accuracy (evaluate_on_train): {acc:.4f}")
        return acc