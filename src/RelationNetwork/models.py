import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

import transformers

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        print(self.i, x.shape)
        return x

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

class inception(nn.Module):
    
    def __init__(self, inp, n1, n3r, n3, n5r, n5, mxp):
        super(inception, self).__init__()
        
        layers = []
        layers += [nn.Conv2d(inp, n1, 1)]
        layers += [nn.ReLU(True)]
        self.one = nn.Sequential(*layers)
        
        layers = []
        layers += [nn.Conv2d(inp, n3r, 1)]
        layers += [nn.ReLU(True)]
        layers += [nn.Conv2d(n3r, n3, 3, padding=1)]
        layers += [nn.ReLU(True)]
        self.three = nn.Sequential(*layers)
        
        layers = []
        layers += [nn.Conv2d(inp, n5r, 1)]
        layers += [nn.ReLU(True)]
        layers += [nn.Conv2d(n5r, n5, 3, padding=1)]
        layers += [nn.ReLU(True)]
        layers += [nn.Conv2d(n5, n5, 3, padding=1)]
        layers += [nn.ReLU(True)]
        self.five = nn.Sequential(*layers)
        
        layers = []
        layers += [nn.MaxPool2d(3, 1, 1)]
        layers += [nn.Conv2d(inp, mxp, 1)]
        layers += [nn.ReLU(True)]
        
        self.maxp = nn.Sequential(*layers)
        
    def forward(self, x):
        h1 = self.one(x)
        h2 = self.three(x)
        h3 = self.five(x)
        h4 = self.maxp(x)
        
        h = torch.cat([h1, h2, h3, h4], 1)
        
        return h


class InceptionEncoder(nn.Module):
    def __init__(self, cnn_layer=1, inception_layer=3, lcnn_layer=2):
        super(InceptionEncoder, self).__init__()

        layers = []
        in_c = 3
        for i in range(cnn_layer):
            layers.append(nn.Conv2d(in_c, 64,3))
            in_c = 64
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
        self.base_layers = nn.Sequential(*layers)

        layers = []
        mp = 4
        for i in range(inception_layer):
            layers.append(inception(in_c, 32, 32, 64, 32, 64, mp))
            in_c = 32+64+64+mp
            mp=mp*2
        self.inception_layers = nn.Sequential(*layers)

        layers = []
        out_c = in_c//2
        for i in range(lcnn_layer):
            layers.append(nn.Conv2d(in_c, out_c, 3))
            in_c = in_c//2
            if i==0:
                out_c = 64
                layers.append(nn.MaxPool2d(2))
        self.final_layers = nn.Sequential(*layers)

    def forward(self, x):
        h = self.base_layers(x)
        h = self.inception_layers(h)
        h = self.final_layers(h)
        return h

class ResNet(nn.Module):
    def __init__(self, in_c):
        super(ResNet, self).__init__()
        self.l1 = nn.Conv2d(in_c, in_c, 3, 1, 1)
        self.l2 = nn.Conv2d(in_c, in_c, 3, 1, 1)

        self.b1 = nn.BatchNorm2d(in_c)
        self.b2 = nn.BatchNorm2d(in_c)

    def forward(self, x):
        h = F.relu(self.b1(x))
        h = self.l1(h)

        h = F.relu(self.b2(h))
        h = self.l2(h)

        return h+x

class ResEncoder(nn.Module):
    def __init__(self, cnn_layer=2, res_layer=3, lcnn_layer=2):
        super(ResEncoder, self).__init__()

        layers = []
        in_c = 3
        for i in range(cnn_layer):
            layers.append(nn.Conv2d(in_c, 64,3))
            in_c = 64
            if i==0:
                layers.append(nn.ReLU())
                layers.append(nn.MaxPool2d(2))
        self.base_layers = nn.Sequential(*layers)

        layers = []
        for i in range(res_layer):
            layers.append(ResNet(in_c))
        self.res_layers = nn.Sequential(*layers)

        layers = []
        for i in range(lcnn_layer):
            layers.append(nn.Conv2d(in_c, in_c, 3))
            if i==0:
                layers.append(nn.MaxPool2d(2))
        self.final_layers = nn.Sequential(*layers)

    def forward(self, x):
        h = self.base_layers(x)
        h = self.res_layers(h)
        h = self.final_layers(h)
        return h

class TransferRelationNetwork(nn.Module):
    def __init__(self):
        super(TransferRelationNetwork, self).__init__()

        self.fc1 = nn.Linear(1024*2, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = torch.sigmoid(self.fc3(h))
        return h


class TextEncoder(nn.Module):
    def __init__(self, bert_path):
        super(TextEncoder, self).__init__()
        self.bert_path = bert_path
        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased')

    def forward(self, ids, mask,  token_type_ids):
        o1, o2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        
        apool = torch.mean(o1, 1)
        mpool, _ = torch.max(o1, 1)
        cat = torch.cat((apool, mpool), 1)

        return cat

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

class TextRelationNetwork(nn.Module):
    def __init__(self):
        super(TextRelationNetwork, self).__init__()

        self.fc1 = nn.Linear(768*2*2, 768)
        self.fc2 = nn.Linear(768, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = torch.sigmoid(self.fc3(h))
        return h