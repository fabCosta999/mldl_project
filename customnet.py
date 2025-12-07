import torch
from torch import nn

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Define layers of the neural network
        self.convLayers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x112x112
            nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x56x56
        )
        self.flatten = nn.Flatten()

        self.fullLayers = nn.Sequential(
            nn.Linear(16 * 56 * 56, 256),
            nn.ReLU(),
            nn.Linear(256, 200) # 200 is the number of classes in TinyImageNet
        )

    def forward(self, x):
        # Define forward pass
        x = self.fullLayers(self.flatten(self.convLayers(x)))
        return x