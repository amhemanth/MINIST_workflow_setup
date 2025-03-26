import torch
import torch.nn as nn

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        
        # First convolutional layer with fewer filters
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        
        # Second convolutional layer with fewer filters
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        
        # Fully connected layers with fewer neurons
        self.fc1 = nn.Linear(8 * 7 * 7, 40)
        self.dropout = nn.Dropout(0.4)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(40, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = x.view(-1, 8 * 7 * 7)
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 