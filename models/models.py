from torch import nn
import torch.nn.functional as F

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        return x

class FullyConnectedNetwork(nn.Module):
    def __init__(self,
                 input_dim,
                 num_classes,
                 hidden_dim=100):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class CifarNet(nn.Module):
    def __init__(self,
                 input_dim,
                 num_classes,
                 hidden_channels=64):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=hidden_channels,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.bn1 = nn.BatchNorm2d(hidden_channels)

        self.conv2 = nn.Conv2d(in_channels=hidden_channels,
                               out_channels=hidden_channels,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.bn2 = nn.BatchNorm2d(hidden_channels)

        self.flatten = nn.Flatten()

        self.mlp = nn.Sequential(
            nn.Linear((input_dim // 4)**2 * hidden_channels, 384, bias=False),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(384, 192, bias=False),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(192, num_classes, bias=False)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.flatten(x)
        x = self.mlp(x)
        return x