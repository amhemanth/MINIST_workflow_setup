import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        
        # Third convolutional layer
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 3 * 3, 32)
        self.fc2 = nn.Linear(32, 10)
        
        # Dropout
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)

    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Third block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Flatten
        x = x.view(-1, 16 * 3 * 3)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 