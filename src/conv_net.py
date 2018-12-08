import torch
from torch import nn


t = torch.rand(5)
print(t)

_ordering = [
    nn.Conv2d(1, 12, kernel_size=7, stride=2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(12, 24, kernel_size=5, stride=2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(24, 36, kernel_size=3, stride=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Linear(6 * 6 * 36, 6 * 6 * 36),
    nn.Linear(6 * 6 * 36, 6 * 6 * 36),
    nn.Linear(6 * 6 * 36, 1)
]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(12, 24, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(24, 36, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.lin1 = nn.Linear(6 * 6 * 36, 6 * 6 * 36)
        self.lin2 = nn.Linear(6 * 6 * 36, 6 * 6 * 36)
        self.o = nn.Linear(6 * 6 * 36, 1)

    def forward(self, inputs):
        inputs = self.conv1(inputs)
        inputs = self.conv2(inputs)
        inputs = self.conv3(inputs)
        inputs = inputs.view(inputs.size(0), -1)
        inputs = self.lin1(inputs)
        inputs = self.lin2(inputs)
        output = self.o(inputs)
        return output


cnn = CNN()
print(cnn)
