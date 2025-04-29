import torch.nn as nn
import torch.nn.functional as F

class ImprovedLeNet5(nn.Module):
    def __init__(self):
        super(ImprovedLeNet5, self).__init__()
        
        # 1. Evrişim katmanı
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(6)  # Batch Normalization
        
        # 2. Evrişim katmanı
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)
        
        # 3. Tam bağlantılı katman
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.dropout1 = nn.Dropout(0.3)  # Dropout ekledik
        
        # 4. Tam bağlantılı katman
        self.fc2 = nn.Linear(120, 84)
        self.dropout2 = nn.Dropout(0.3)
        
        # 5. Çıkış katmanı
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))   # Conv1 + BN + ReLU
        x = F.max_pool2d(x, 2)                # MaxPool
        
        x = F.relu(self.bn2(self.conv2(x)))   # Conv2 + BN + ReLU
        x = F.max_pool2d(x, 2)                # MaxPool

        x = x.view(-1, 16 * 5 * 5)            # Flatten
        x = F.relu(self.fc1(x))              
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))              
        x = self.dropout2(x)
        x = self.fc3(x)                       # No activation here (CrossEntropyLoss includes softmax)
        return x
