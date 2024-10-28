class ResidualConnection:
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

    def forward(self, input):
        # Compute the output of two layers
        output = self.layer1.forward(input)
        output = self.layer2.forward(output)

        # Add the original input (skip connection)
        return input + output