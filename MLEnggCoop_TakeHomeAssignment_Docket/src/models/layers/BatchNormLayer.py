import numpy as np
import torch
import torch.nn as nn

class BatchNormLayer:
    def __init__(self, num_features, epsilon=1e-5, momentum=0.1):
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.epsilon = epsilon
        self.momentum = momentum
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)
        #self.train_mode = True  # Initialize in train_mode mode

    def forward(self, input, training=True):
        if training:
            # Compute mean and variance along the batch dimension
            batch_mean = input.mean(dim=(0, 2, 3), keepdim=True)
            batch_var = input.var(dim=(0, 2, 3), keepdim=True, unbiased=False)

            # Normalize
            normalized_input = (input - batch_mean) / np.sqrt(batch_var + self.epsilon)

            # Update running stats
            self.running_mean = self.momentum * batch_mean + (1 - self.momentum) * self.running_mean
            self.running_var = self.momentum * batch_var + (1 - self.momentum) * self.running_var
        else:
            # Use running stats for inference
            normalized_input = (input - self.running_mean) / torch.sqrt(self.running_var + self.epsilon)

        # Scale and shift
        output = self.gamma * normalized_input + self.beta
        return output

    def update_parameters(self, learning_rate):
        if self.gamma.grad is not None:
            with torch.no_grad():
                self.gamma -= learning_rate * self.gamma.grad
                self.gamma.grad.zero_()
        # Update beta
        if self.beta.grad is not None:
            with torch.no_grad():
                self.beta -= learning_rate * self.beta.grad
                self.beta.grad.zero_()

    def parameters(self):
        return [self.gamma, self.beta]

    # def train(self):
    #     """Set the layer to train_mode mode."""
    #     self.train_mode = True
    #
    # def eval(self):
    #     """Set the layer to evaluation mode."""
    #     self.train_mode = False