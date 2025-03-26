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
        
        # Initialize weights for better GPU performance
        self._initialize_weights()

    def forward(self, x):
        # Use contiguous tensors for better GPU memory access
        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)  # inplace operation saves memory
        x = F.max_pool2d(x, 2)
        
        # Second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)  # inplace operation saves memory
        x = F.max_pool2d(x, 2)
        
        # Third block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x, inplace=True)  # inplace operation saves memory
        x = F.max_pool2d(x, 2)
        
        # Flatten - use contiguous for better memory layout
        x = x.view(-1, 16 * 3 * 3).contiguous()
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x, inplace=True)  # inplace operation saves memory
        x = self.dropout1(x)
        x = self.fc2(x)
        
        return x
    
    def _initialize_weights(self):
        # Initialize weights for better convergence on GPU
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def to_gpu(self):
        """Optimized method to move model to GPU with the right settings"""
        self.cuda()
        # Set these flags for better performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        return self

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 