class InputLayer:
    def __init__(self, target_shape):
        self.target_shape = target_shape

    def forward(self, input):
        # Reshape or normalize the input as necessary
        return input.reshape(self.target_shape)