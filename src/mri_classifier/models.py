import torch
from torch import nn
from torchvision import models

class BaselineCNN(nn.Module):
    """3-block CNN baseline without regularization (expects 128x128 RGB input)."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class RegularizedCNN(nn.Module):
    """3-block CNN with spatial Dropout2d and fully-connected Dropout."""

    def __init__(
        self,
        num_classes: int = 4,
        dropout2d: float = 0.25,
        dropout_fc: float = 0.5,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout2d = nn.Dropout2d(dropout2d)
        self.dropout = nn.Dropout(dropout_fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout2d(self.pool(self.relu(self.conv1(x))))
        x = self.dropout2d(self.pool(self.relu(self.conv2(x))))
        x = self.dropout2d(self.pool(self.relu(self.conv3(x))))
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)


def build_model(exp_cfg: dict, num_classes: int = 4) -> nn.Module:
    model_name = exp_cfg["model"]
    if model_name == "baseline_cnn":
        return BaselineCNN(num_classes=num_classes)
    if model_name == "regularized_cnn":
        return RegularizedCNN(
            num_classes=num_classes,
            dropout2d=exp_cfg.get("dropout2d", 0.25),
            dropout_fc=exp_cfg.get("dropout_fc", 0.5),
        )
    if model_name == "resnet18":
        return build_resnet18(
            num_classes=num_classes,
            dropout_fc=exp_cfg.get("dropout_fc", 0.3),
            pretrained=exp_cfg.get("pretrained", True),
        )
    raise ValueError(f"Unknown model architecture: {model_name}")


def build_resnet18(
    num_classes: int = 4,
    dropout_fc: float = 0.3,
    pretrained: bool = True,
) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout_fc),
        nn.Linear(in_features, num_classes),
    )
    return model