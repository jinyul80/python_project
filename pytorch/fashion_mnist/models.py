import torch
from torch import nn
from torchvision import models


class MyCNNModel(nn.Module):
    def __init__(self):

        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )

        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(7 * 7 * 64, 256)
        self.fc2 = nn.Linear(256, 10)

        self.dropout25 = nn.Dropout(p=0.25)
        self.dropout50 = nn.Dropout(p=0.5)

    def forward(self, data):

        data = self.conv1(data)
        data = torch.relu(data)
        data = self.pooling(data)
        data = self.dropout25(data)

        data = self.conv2(data)
        data = torch.relu(data)
        data = self.pooling(data)
        data = self.dropout25(data)

        data = data.view(-1, 7 * 7 * 64)

        data = self.fc1(data)
        data = torch.relu(data)
        data = self.dropout50(data)

        logits = self.fc2(data)

        return logits


class MyTransferLearningModel(torch.nn.Module):

    def __init__(self, feature_extractor, num_classes=10):
        # feature_extractor = True: Feature Extractor,  False: Fine Tuning

        super().__init__()

        # pretrained_model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        pretrained_model = models.efficientnet_b2(
            weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1
        )
        print(pretrained_model)

        if feature_extractor:
            for param in pretrained_model.parameters():
                param.require_grad = False

        # vision transformer 에서의 classifier 부분은 heads 로 지정
        pretrained_model.heads = torch.nn.Sequential(
            torch.nn.Linear(pretrained_model.heads[0].in_features, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, num_classes),
        )

        self.model = pretrained_model

    def forward(self, data):

        logits = self.model(data)

        return logits


class MyEffNetModel(torch.nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        pretrained_model = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        )
        print(pretrained_model)

        pretrained_model.classifier[1] = torch.nn.Linear(
            in_features=pretrained_model.classifier[1].in_features,
            out_features=num_classes,
        )

        self.model = pretrained_model

    def forward(self, data):

        logits = self.model(data)

        return logits
