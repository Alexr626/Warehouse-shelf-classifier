import numpy as np
import torch
import torch.nn as nn

class MaxPoolingLayer:
    def __init__(self, kernel_size=5, stride=1):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input):
        output_height = (input.shape[0] - self.kernel_size) // self.stride + 1
        output_width = (input.shape[1] - self.kernel_size) // self.stride + 1
        pooled_output = np.zeros((output_height, output_width, input.shape[2]))

        for i in range(output_height):
            for j in range(output_width):
                region = input[
                         i * self.stride : i * self.stride + self.kernel_size,
                         j * self.stride : j * self.stride + self.kernel_size,
                         ]
                pooled_output[i, j] = np.max(region, axis=(0, 1))

        return pooled_output