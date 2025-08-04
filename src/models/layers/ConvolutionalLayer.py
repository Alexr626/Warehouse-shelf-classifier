import numpy as np
import torch
import torch.nn as nn
import torch.autograd.functional as auto_f

class ConvolutionalLayer:

    def __init__(self,
                 input: np.ndarray,
                 stride: int = 1,
                 padding: int = 1,
                 kernel_size: int = 5,
                 out_channels: int = 3,
                 in_channels: int = 3):

        self.input = input
        self.weights = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.in_channels = in_channels

    def pad(self, input):
        return np.pad(input, ((self.padding, self.padding),
                              (self.padding, self.padding),
                              (0, 0)), mode='constant')

    def forward(self, input):
        if self.padding > 0:
            input = self.pad(input)

        output_height = (input.shape[0] - self.kernel_size) // self.stride + 1
        output_width = (input.shape[1] - self.kernel_size) // self.stride + 1
        # Output should be of these dimensions (output_height, output_width, 3)
        output = np.zeros((output_height, output_width, self.out_channels))


        for out_channel in range(self.out_channels):
            conv_sum = np.zeros((output_height, output_width))

            for in_channel in range(self.in_channels):
                weight = self.weights[out_channel, in_channel]
                channel_input = input[..., in_channel]

                # Stride tricks for efficiency
                strided_view = np.lib.stride_tricks.as_strided(
                    channel_input,
                    shape=(output_height, output_width, self.kernel_size, self.kernel_size),
                    strides=(
                        channel_input.strides[0] * self.stride,
                        channel_input.strides[1] * self.stride,
                        channel_input.strides[0],
                        channel_input.strides[1],
                    ),
                )

                # Element-wise multiplication and summing
                conv_sum += np.einsum('ijkl,kl->ij', strided_view, weight)

            output[..., out_channel] = conv_sum + self.bias[out_channel].item()

        return output

    def backward(self):
        pass

    def update_parameters(self, learning_rate: float):
        # Check that they exist before updating
        if self.weights.grad is not None:
            with torch.no_grad():
                self.weights -= learning_rate * self.weights.grad
                self.bias -= learning_rate * self.bias.grad
                self.weights.grad.zero_()
                self.bias.grad.zero_()

    def parameters(self):
        return [self.weights, self.bias]