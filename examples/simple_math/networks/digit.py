import numpy as np
import torch
import torch.nn as nn

class DigitCNN(nn.Module):
    def __init__(self) -> None:
        super(DigitCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.fc1 = nn.Linear(64 * 7 * 7, 10) # 10 classes for digits

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.pool1(out)
        out = self.relu2(self.conv2(out))
        out = self.pool2(out)
        out = out.view(-1)
        logits = self.fc1(out)

        return logits


    
