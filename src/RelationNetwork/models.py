import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

class RegularEncoder(nn.Module):
    def __init__(self, n_layer=4, batch_norm = [1, 1, 1, 1], mp = [1, 1, 0, 0]):
        super(RegularEncoder, self).__init__()

        layers = []
        in_c = 3
        for i in range(n_layer):
            layers.append(nn.Conv2d(in_c, 64,3))
            in_c = 64
            if batch_norm[i]:
                layers.append(nn.BatchNorm2d(64))
            layers.append(nn.ReLU())
            if mp[i]:
                layers.append(nn.MaxPool2d(2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class RelationNet(nn.Module):
    def __init__(self, cnn_layers=2, batch_norm = [1, 1], mp=[1,1], hidden_dim = 512):
        super(RelationNet, self).__init__()

        layers = []
        in_c = 64*2
        for i in range(cnn_layers):
            layers.append(nn.Conv2d(in_c, 64,3))
            in_c = 64
            if batch_norm[i]:
                layers.append(nn.BatchNorm2d(64))
            layers.append(nn.ReLU())
            if mp[i]:
                layers.append(nn.MaxPool2d(2))
        self.cnns = nn.Sequential(*layers)

        self.fc1 = nn.Linear(1600, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.cnns(x)
        h = h.view(h.shape[0], -1)
        h = F.relu(self.fc1(h))
        h = torch.sigmoid(self.fc2(h))

        return h

            