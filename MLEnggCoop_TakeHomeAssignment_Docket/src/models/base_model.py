from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy.signal import convolve
import wandb
import os
import constants
import torch
import torch.nn as nn


class BaseModel(ABC):

    def __init__(self, type):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError


class InputLayer:
    inputs: np.ndarray
    weights: np.ndarray = None
    delta_weights: np.ndarray = None

    def __init__(self, inputs: np.ndarray):
        pass

class ConvolutionalLayer:
    input: np.ndarray
    stride: int = 1
    padding: int = 0
    kernel_size: int = 5

    def __init__(self,
                 input: np.ndarray,
                 stride: int,
                 padding: int,
                 kernel_size: int,
                 out_channels: int,
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

        # Output should be of these dimensions
        output = np.zeros((output_height, output_width, self.out_channels))

        for out_channel in range(self.out_channels):

            conv_sum = np.zeros((output_height, output_width))
            for in_channel in range(self.in_channels):

                # Extract weight kernel and slice input for this channel
                weight = self.weights[out_channel, in_channel].detach().numpy()
                channel_input = input[..., in_channel]

                # Convolution
                for i in range(output_height):
                    for j in range(output_width):

                        # Extract input region
                        input_region = channel_input[
                                        i * self.stride : i * self.stride + self.kernel_size,
                                        j * self.stride : j * self.stride + self.kernel_size]

                        # element-wise product and sum
                        conv_sum[i, j] += np.sum(input_region * weight)

            output[..., out_channel] = conv_sum + self.bias[out_channel].item()

        return output


