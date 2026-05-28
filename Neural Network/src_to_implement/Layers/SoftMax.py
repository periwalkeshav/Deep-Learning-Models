import numpy as np
from .Base import BaseLayer


class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_tensor = None
        self.output_tensor = None
        self.error_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        row_max = np.max(input_tensor, axis=1, keepdims=True)
        exponentials = np.exp(input_tensor - row_max)
        row_sums = np.sum(exponentials, axis=1, keepdims=True)
        self.output_tensor = exponentials / row_sums
        return self.output_tensor

    def backward(self, error_tensor):
        #y.(error - sum(error*y)) 
        innerTerm = np.sum(error_tensor * self.output_tensor, axis=1, keepdims=True)
        self.error_tensor = self.output_tensor * (error_tensor - innerTerm)
        return self.error_tensor