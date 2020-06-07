import numpy as np
import sys
from PIL import Image

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim
import itertools

import torchvision
from torchvision import transforms, datasets, models
from torch import Tensor

import math
import matplotlib.pyplot as plt 
import scipy
import time

import dataloader as dl
import models as md

from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter('runs/RelationNet_exp1')



if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


## Few Config
k_shot = 20
n_way = 5
learning_rate = 0.0001


#### Define networks
EmbeddingNetwork = md.RegularEncoder().to(device)
RelationNetwork = md.RelationNet().to(device)

#### Get training, validation, and testing classes
tr, val, te = dl.get_labels("./data/102flowers/imagelabels.mat")

#### Define dataloaders
TrainData = dl.Few_shot_dataset("./data/102flowers/jpg/", "./data/102flowers/imagelabels.mat", n_way, k_shot, tr)
ValData = dl.Few_shot_dataset("./data/102flowers/jpg/", "./data/102flowers/imagelabels.mat", n_way, k_shot, val)
TestData = dl.Few_shot_dataset("./data/102flowers/jpg/", "./data/102flowers/imagelabels.mat", n_way, k_shot, te)

TrainDataLoader = DataLoader(TrainData, batch_size=1, shuffle=True, num_workers=0)  # num_workers=0 for windows OS
ValDataLoader = DataLoader(TrainData, batch_size=1, shuffle=True, num_workers=0)  # num_workers=0 for windows OS
TestDataLoader = DataLoader(TrainData, batch_size=1, shuffle=True, num_workers=0)  # num_workers=0 for windows OS


#### Define Loss function
criterion = nn.MSELoss()

#### Define optimizers
optimizer = torch.optim.Adam(itertools.chain(EmbeddingNetwork.parameters(), RelationNetwork.parameters()), lr=learning_rate, betas=(0.5, 0.999))


def training(data_loader, n_epoch):
    EmbeddingNetwork.train()
    RelationNetwork.train()

    running_loss = 0

    for en, (sx, qx, sy, qy) in enumerate(data_loader):

        ## Remove additional dimension
        ## (k_shot*n_way)*3*128*128
        sx = Variable(sx.squeeze(0)).to(device)
        sy = Variable(sy.squeeze(0)).to(device)
        qx = Variable(qx.squeeze(0)).to(device)
        qy = Variable(qy.squeeze(0)).to(device)

        ## extract embeddings
        sx_f = EmbeddingNetwork(sx)
        qx_f = EmbeddingNetwork(qx)

        ## Concatenating support and quey set
        sx_f = torch.sum(sx_f.view(n_way, k_shot, 64, sx_f.shape[-2], sx_f.shape[-1]), 1).squeeze(1).unsqueeze(0).repeat(qx.shape[0], 1, 1, 1, 1)
        qx_f = qx_f.unsqueeze(1).repeat(1, n_way, 1, 1, 1)
        pairs = torch.cat((sx_f, qx_f), 2).view(-1, 64*2, sx_f.shape[-2], sx_f.shape[-1])

        ## Get relation scores
        scores = RelationNetwork(pairs).view(qx.shape[0], n_way, 1).squeeze(2)


        optimizer.zero_grad()
        loss = criterion(scores, qy)
        loss.backward()
        optimizer.step()

        running_loss+=loss.cpu().item()


        print ("[Epoch: %d] [Iter: %d/%d] [loss: %f]" % (n_epoch, en, len(TrainDataLoader), loss.cpu().data.numpy()))

def validation(data_loader, n_epoch):
    EmbeddingNetwork.eval()
    RelationNetwork.eval()

    current = 0
    total = 0

    for en, (sx, qx, sy, qy) in enumerate(data_loader):

        ## Remove additional dimension
        ## (k_shot*n_way)*3*128*128
        sx = Variable(sx.squeeze(0)).to(device)
        sy = Variable(sy.squeeze(0)).to(device)
        qx = Variable(qx.squeeze(0)).to(device)
        qy = Variable(qy.squeeze(0)).to(device)

        ## extract embeddings
        sx_f = EmbeddingNetwork(sx)
        qx_f = EmbeddingNetwork(qx)

        ## Concatenating support and quey set
        sx_f = torch.sum(sx_f.view(n_way, k_shot, 64, sx_f.shape[-2], sx_f.shape[-1]), 1).squeeze(1).unsqueeze(0).repeat(qx.shape[0], 1, 1, 1, 1)
        qx_f = qx_f.unsqueeze(1).repeat(1, n_way, 1, 1, 1)
        pairs = torch.cat((sx_f, qx_f), 2).view(-1, 64*2, sx_f.shape[-2], sx_f.shape[-1])

        ## Get relation scores
        scores = RelationNetwork(pairs).view(qx.shape[0], n_way, 1).squeeze(2)

        _, true_labels = torch.max(qy.data, 1)
        _, pred_labels = torch.max(scores.data,1)

        correct += np.sum([1 if pred_labels[j]==true_labels[j] else 0 for j in range(qy.shape[0])])
        total += qy.shape[0]

    return (correct*100)/total


isTrain = True
save_epoch = 5
checkpoints_path = "./src/ReletionNetwork/checkpoints"


if isTrain:
    epochs = 100
    for  i in range(epochs):
        training(TrainDataLoader, i+1)

        val_acc = validation(ValDataLoader i+1)

        if (i+1)%save_epoch==0:
            torch.save(EmbeddingNetwork, join(checkpoints_path, "emb_net.pth"))
            torch.save(RelationNetwork, join(checkpoints_path, "relation_net.pth"))
