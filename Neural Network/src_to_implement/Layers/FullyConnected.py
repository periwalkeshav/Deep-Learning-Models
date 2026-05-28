import numpy as np
from .Base import BaseLayer


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(0, 1, (input_size + 1, output_size))
        self.input_tensor = None
        self._gradient_weights = None
        self._optimizer = None
        
    def forward(self, input_tensor):
        batch_size = input_tensor.shape[0]
        bias_column = np.ones((batch_size, 1))
        self.input_tensor = np.concatenate([input_tensor, bias_column], axis=1)
        return np.dot(self.input_tensor, self.weights)
    
    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def backward(self, error_tensor):
        self._gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        weights_no_bias = self.weights[:-1, :]
        error = np.dot(error_tensor, weights_no_bias.T)
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        return error

    @property
    def gradient_weights(self):
        return self._gradient_weights
    
    