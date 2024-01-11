import torch.nn as nn

from torchvision.models import resnet101

class ResNet101(nn.Module):
    def __init__(self, weights='pretrained', num_classes=10):
        super().__init__()
        backbone = resnet101(weights=weights)
        backbone.conv1.stride = (1, 1)
        backbone.maxpool.stride = (1, 1)
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.head = nn.Linear(backbone.fc.in_features, num_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = x.mean([-1, -2])
        x = self.head(x)
        return x