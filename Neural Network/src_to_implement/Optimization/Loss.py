import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor = None
        self.label_tensor = None
        self.loss = None
        self.error_tensor = None
        self.eps = np.finfo(float).eps

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        self.label_tensor = label_tensor
        #-sum(y*log(p))
        self.loss = -np.sum(self.label_tensor * np.log(self.prediction_tensor + self.eps))
        return self.loss

    def backward(self, label_tensor):
       # -y/p
        self.label_tensor = label_tensor
        self.error_tensor = -self.label_tensor / (self.prediction_tensor + self.eps)
        return self.error_tensor