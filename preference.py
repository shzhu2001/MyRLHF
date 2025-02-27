import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class SimplePreferenceModel(nn.Module):
    def __init__(self):
        super(SimplePreferenceModel, self).__init__()
        self.fc = nn.Linear(1,1)
    
    def forward(self, reward):
        return self.fc(reward)
        
        



