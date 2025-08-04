import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from src.models.base_model import BaseModel

class EfficientNetWarehouseClassifier(nn.Module, BaseModel):
    def __init__(self, num_classes=2):
        super(EfficientNetWarehouseClassifier, self).__init__()
        # Load pre-trained EfficientNet-B0 model
        weights = EfficientNet_B0_Weights.DEFAULT
        self.model = efficientnet_b0(weights=weights)
        # Modify the classifier
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_ftrs, num_classes),
        )

    def forward(self, x):
        return self.model(x)