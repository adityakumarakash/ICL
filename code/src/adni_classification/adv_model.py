import torch

from torch import nn
from torch.autograd import Variable

from src.adni_classification.utils import Swish

class Adv(nn.Module):
    # New adversarial model for baseline which uses ReLU activation
    # The output would be logits
    def __init__(self, name='Adv', input_dim=64, output_dim=10, hidden_dim=64, hidden_layers=0, dropout=0.):
        super(Adv, self).__init__()
        self.name = name
        layers = []
        prev_dim = input_dim
        self.output_dim = output_dim
        for i in range(0, hidden_layers + 1):
            if dropout > 0.:
                layers.append(nn.Dropout(p=dropout))
            if i == 0:
                prev_dim = input_dim
            else:
                prev_dim = hidden_dim
            
            if i == hidden_layers:
                layers.append(nn.Linear(prev_dim, output_dim))
            else:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))  # This is different from the previous adv
                layers.append(Swish())
        self.adv = nn.Sequential(*layers)
        
    def forward(self, x):
        output = self.adv(x)
        if self.output_dim == 1:
            output = output.squeeze()
        return output