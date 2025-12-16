import torch
import torch.nn as nn
import torch.nn.functional as F

class AsaltoNet(nn.Module):
    def __init__(self):
        super(AsaltoNet, self).__init__()
        # Input: 3 channels (Empty, Rebel, Officer) x 7 x 7
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 1) # Output value of current state (Value)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(-1, 64 * 7 * 7) # Flatten
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)) # Output range [-1, 1]