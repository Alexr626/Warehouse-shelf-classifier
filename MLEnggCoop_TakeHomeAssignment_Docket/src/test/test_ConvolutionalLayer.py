import unittest
import numpy as np
import torch

from src.models.layers.ConvolutionalLayer import ConvolutionalLayer
import os
import constants
import torch.nn as nn
class TestLayers(unittest.TestCase):

    def get_test_input_array(self, array_name):
        array_path = os.path.join(constants.ROOT_DIR, "training_data", "preprocessed_arrays", array_name)
        return np.load(array_path)

    def run_padding_test(self, C, input):
        output = C.pad(input)
        self.assertEqual(output.shape, (512 + 2*C.padding, 512 + 2*C.padding, 3))
    def run_convolve_test(self, C, input):
        output = C.forward(input)

        input_tensor = torch.tensor(input, dtype=torch.float32).permute(2,0,1).unsqueeze(0)
        kernel_tensor = C.weights.clone()
        bias_tensor = C.bias.clone()

        comparison_output = nn.functional.conv2d(input=input_tensor, weight=kernel_tensor, bias=bias_tensor, stride=C.stride, padding=C.padding)
        comparison_output = comparison_output.squeeze(0).permute(1, 2, 0).numpy()

        print(comparison_output[0:2,0:2,1])
        print(output[0:2,0:2,1])

        self.assertEqual(output.shape, comparison_output.shape)
        self.assertTrue(np.allclose(output, comparison_output, atol=1e-2))

    def test_convolutional_layer(self):
        image_array = self.get_test_input_array("0000_aug_0_normalization_blurring.npy")
        C = ConvolutionalLayer(input=image_array,
                               stride=1,
                               padding=2,
                               kernel_size=5,
                               out_channels=3,
                               in_channels=3)
        i, j = 100, 100
        image_array_chunk = image_array[i:i+5, j:j+5, :]

        self.run_padding_test(C, image_array)
        self.run_convolve_test(C, image_array)

if __name__ == "__main__":
    unittest.main()