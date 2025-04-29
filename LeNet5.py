import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 1 input channel (grayscale), 6 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)  # Çıktı: [batch, 6, 28, 28] → [batch, 6, 24, 24]
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)  # [batch, 6, 24, 24] → [batch, 6, 12, 12]
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # [batch, 6, 12, 12] → [batch, 16, 8, 8] → pool → [batch, 16, 4, 4]
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10 sınıf için çıkış

    def forward(self, x):
        x = F.relu(self.conv1(x))     # Conv1 + ReLU
        x = self.pool(x)              # Pool1
        x = F.relu(self.conv2(x))     # Conv2 + ReLU
        x = self.pool(x)              # Pool2
        x = x.view(-1, 16 * 4 * 4)     # Flatten
        x = F.relu(self.fc1(x))       # FC1
        x = F.relu(self.fc2(x))       # FC2
        x = self.fc3(x)               # FC3 (çıktı)
        return x
