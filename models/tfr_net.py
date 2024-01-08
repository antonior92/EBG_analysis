import torch.nn as nn
import torch


class TFRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 8, kernel_size=(3, 3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)

        # self.conv2 = nn.Conv2d(8, 8, kernel_size=(5, 5), stride=1, padding=1)
        # self.act2 = nn.ReLU()
        # self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = nn.Conv2d(8, 4, kernel_size=(5, 5), stride=1, padding=1)
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.flat = nn.Flatten()

        self.fc3 = nn.Linear(35052, 8154)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(8154, 512)
        self.act4 = nn.ReLU()
        self.drop4 = nn.Dropout(0.5)

        self.fc5 = nn.Linear(512, 1)

    def forward(self, x):
        # input 3x32x32, output 32x32x32
        x = self.act1(self.conv1(x))
        x = self.drop1(x)
        # input 32x32x32, output 32x32x32
        # x = self.act2(self.conv2(x))
        # input 32x32x32, output 32x16x16
        # x = self.pool2(x)

        x = self.act3(self.conv3(x))
        x = self.pool3(x)

        # input 32x16x16, output 8192
        x = self.flat(x)
        # input 8192, output 512
        x = self.act3(self.fc3(x))
        x = self.drop3(x)

        x = self.act4(self.fc4(x))
        x = self.drop4(x)
        # input 512, output 10
        x = self.fc5(x)
        return x

