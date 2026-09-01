"""A small ResNet-style classifier for the end-to-end check.

`examples/tiny-vit.py` in this repo is written for the static parser and does
not forward-pass (it calls nn.MultiheadAttention with one argument), so the
tracer needs a model that actually runs. This one has a residual merge and a
shared ReLU, the two things static parsing gets wrong.
"""
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity() if stride == 1 and cin == cout else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.relu(y + self.shortcut(x))


class MiniResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.layer1 = BasicBlock(32, 32)
        self.layer2 = BasicBlock(32, 64, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.layer2(self.layer1(self.stem(x)))
        return self.fc(torch.flatten(self.pool(x), 1))
