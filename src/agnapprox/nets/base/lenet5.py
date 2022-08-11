"""
Class definition for vanilla LeNet5 implementation
"""
import torch.nn as nn


class LeNet5(nn.Module):
    """
    Defintion of vanilla LeNet5 architecture torch.nn.Module
    """

    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.linear1 = nn.Linear(400, 120, bias=False)
        self.batchnorm1 = nn.BatchNorm1d(120)
        self.act1 = nn.ReLU()
        self.linear2 = nn.Linear(120, 84, bias=False)
        self.batchnorm2 = nn.BatchNorm1d(84)
        self.act2 = nn.ReLU()
        self.linear3 = nn.Linear(84, num_classes)

    def forward(self, features):
        out = self.conv1(features)
        out = self.conv2(out)
        out = out.reshape(out.size(0), -1)
        out = self.linear1(out)
        out = self.batchnorm1(out)
        out = self.act1(out)
        out = self.linear2(out)
        out = self.batchnorm2(out)
        out = self.act2(out)
        out = self.linear3(out)
        return out
