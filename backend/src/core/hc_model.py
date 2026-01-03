from typing import override

import numpy as np

from core.numpy_model import NumpyModel
from nn.losses import Loss
from nn.math import glorot_uniform, relu
from nn.optimizers import Optimizer
from utils.helpers import print_metrics


class HCModel(NumpyModel):
    def __init__(self,
        learning_rate: float = 1e-2,
        weight_decay: float = 1e-3,
        loss: Loss = None,
        optimizer: Optimizer = None,
    ):
        if loss is None:
            loss = Loss()
        if optimizer is None:
            optimizer = Optimizer()
        super().__init__(learning_rate, weight_decay, loss, optimizer)
        self.activation_function = relu

    @override
    def create_model(self, number_samples: int, hidden_layer_size: int = 512) -> None:
        print("Creating model with hardcoded network")
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


    @override
    def _train(self, epochs: int, metrics: bool = False):
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

            gradient_y_prediction = 2 * (self.y_prediction - self.output_data_shape)

            gradient_weights_3 = hidden_activation_2.T.dot(gradient_y_prediction)
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
                print_metrics(
                    self.y_prediction,
                    self.output_data_shape,
                    self.loss
                )
