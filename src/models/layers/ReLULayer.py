import numpy as np
import torch
import torch.nn as nn
import torch.autograd.functional as auto_f

class ReLULayer:
    def __init__(self):
        pass
    def forward(self, input):
        return input * (input > 0)
