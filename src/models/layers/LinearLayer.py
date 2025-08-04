import numpy as np
import torch
import torch.nn as nn

class LinearLayer:
    def __init__(self, in_features, out_features):
        self.weights = nn.Parameter(torch.randn(out_features, in_features) * np.sqrt(2. / in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, input):
        return input @ self.weights.T + self.bias

    def update_parameters(self, learning_rate):
        if self.weights.grad is not None:
            with torch.no_grad():
                self.weights -= learning_rate * self.weights.grad
                self.weights.grad.zero_()

        if self.bias.grad is not None:
            with torch.no_grad():
                self.bias -= learning_rate * self.bias.grad
                self.bias.grad.zero_()

    def parameters(self):
        return [self.weights, self.bias]