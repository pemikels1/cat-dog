import torch
from torch import nn


class CatDog(nn.Module):
    def __init__(self):
        super(CatDog, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 7, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 7, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 5, 1, 1)
        self.conv4 = nn.Conv2d(64, 128, 5, 1, 1)
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv6 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv7 = nn.Conv2d(256, 512, 3, 1, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.25)
        self.maxpool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(10240, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.maxpool(x)
        x = self.conv7(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
