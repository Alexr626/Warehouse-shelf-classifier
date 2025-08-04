from src.models.base_model import BaseModel
from src.models.layers.ConvolutionalLayer import ConvolutionalLayer
from src.models.layers.BatchNormLayer import BatchNormLayer
from src.models.layers.ReLULayer import ReLULayer
from src.models.layers.LinearLayer import LinearLayer
from src.models.layers.MaxPoolingLayer import MaxPoolingLayer
from src.models.layers.ResidualConnection import ResidualConnection
import torch.nn as nn
class ResNetBlock(BaseModel):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__("ResNetBlock")

        # Define the individual layers for the ResNet block
        self.conv1 = ConvolutionalLayer(in_channels, out_channels, kernel_size, stride, padding)
        self.batch_norm1 = BatchNormLayer(out_channels)
        self.relu1 = ReLULayer()

        self.conv2 = ConvolutionalLayer(out_channels, out_channels, kernel_size, stride, padding)
        self.batch_norm2 = BatchNormLayer(out_channels)

        # Define the residual connection
        self.residual = ResidualConnection(self.conv1, self.conv2)
        self.train_mode = True


    def forward(self, x, training=True):
        residual = x

        # Pass through the first convolutional layer, batch normalization, and ReLU
        out = self.conv1.forward(x)
        out = self.batch_norm1.forward(out, training=training)
        out = self.relu1.forward(out)

        # Pass through the second convolutional layer and batch normalization
        out = self.conv2.forward(out)
        out = self.batch_norm2.forward(out, training=training)

        # Apply the residual connection
        out = self.residual.forward(out, residual)  # Assuming the residual layer combines `x` and `out`

        return out


    def update_parameters(self, learning_rate):
        # Call the update_parameters method for each layer with learnable parameters
        self.conv1.update_parameters(learning_rate)
        self.conv2.update_parameters(learning_rate)
        self.batch_norm1.update_parameters(learning_rate)
        self.batch_norm2.update_parameters(learning_rate)


    def parameters(self):
        params = []
        for layer in [self.conv1, self.batch_norm1, self.conv2, self.batch_norm2]:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())
        return params

    def to(self, device):
        # Recursively move all parameters to the specified device
        for layer in [self.conv1, self.batch_norm1, self.relu1, self.conv2, self.batch_norm2]:
            if hasattr(layer, 'parameters'):
                for param in layer.parameters():
                    param.data = param.data.to(device)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False