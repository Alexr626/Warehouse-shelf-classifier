import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from src.models.base_model import BaseModel

class ResNetWarehouseClassifier(nn.Module, BaseModel):
    def __init__(self, num_classes=2):
        super(ResNetWarehouseClassifier, self).__init__()
        # Load pre-trained ResNet-18 model
        weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=weights)
        # Modify the final fully connected layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)