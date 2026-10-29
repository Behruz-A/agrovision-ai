"""Small convolutional network trained from scratch."""
import torch
from torch import nn


class AgroVisionCNN(nn.Module):
    def __init__(self, num_classes: int = 3, first_stride: int = 1) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = 3
        for width in (16, 32, 64, 128):
            layers.extend([nn.Conv2d(channels, width, 3, padding=1, bias=False,
                                     stride=first_stride if channels == 3 else 1),
                           nn.BatchNorm2d(width), nn.ReLU(inplace=True), nn.MaxPool2d(2)])
            channels = width
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((2, 2)))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.4),
                                        nn.Linear(128 * 4, 128), nn.ReLU(),
                                        nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
