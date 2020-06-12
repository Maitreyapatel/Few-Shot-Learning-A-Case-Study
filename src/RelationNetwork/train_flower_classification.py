import numpy as np
import sys
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from PIL import Image

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim
from pytorch_metric_learning import miners, losses
import itertools

import torchvision
from torchvision import transforms, datasets, models
from torch import Tensor

import math
import matplotlib.pyplot as plt 
import scipy
import time
import pickle
import argparse

import dataloader as dl
import models as md

from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description="Learning to learn: RelationNetwork Parameters")
parser.add_argument("-n","--n_way",type = int, default = 5)
parser.add_argument("-k","--k_shot",type = int, default = 5)
parser.add_argument("-lr","--learning_rate",type = float, default = 0.000001)
parser.add_argument("-save","--saving_rate",type = int, default = 5)
parser.add_argument("-ckp","--checkpoint_path",type = str)
parser.add_argument("-ep","--epochs",type = int, default = 150)
parser.add_argument("-train","--do_training",type = bool, default = True)
parser.add_argument("-test","--test_class_path",type = str, default = "")
parser.add_argument("-tsf", "--transfer_learning", type=bool, default=False)

args = parser.parse_args()


### Training functions
def training(data_loader, n_epoch):
    EmbeddingNetwork.train()
    RelationNetwork.train()

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
        if args.transfer_learning == False:
            sx_f = torch.sum(sx_f.view(n_way, k_shot, 64, sx_f.shape[-2], sx_f.shape[-1]), 1).squeeze(1).unsqueeze(0).repeat(qx.shape[0], 1, 1, 1, 1)
            qx_f = qx_f.unsqueeze(1).repeat(1, n_way, 1, 1, 1)
            pairs = torch.cat((sx_f, qx_f), 2).view(-1, 64*2, sx_f.shape[-2], sx_f.shape[-1])
        else:
            sx_f = torch.sum(sx_f.view(n_way, k_shot, sx_f.shape[-1]), 1).squeeze(1).unsqueeze(0).repeat(qx.shape[0], 1, 1)
            qx_f = qx_f.unsqueeze(1).repeat(1, n_way, 1)
            pairs = torch.cat((sx_f, qx_f), 2).view(-1, sx_f.shape[-1]*2)

        ## Get relation scores
        scores = RelationNetwork(pairs).view(qx.shape[0], n_way, 1).squeeze(2)

        _, sy_label = torch.max(qy.data, 1)

        optimizer.zero_grad()
        loss = criterion(scores, qy)# + triplet(sx_f, sy_label)*0.01
        loss.backward()
        optimizer.step()

        running_loss = 0
        running_loss+=loss.cpu().item()


        writer.add_scalar('Training Loss (MSE)',
                        running_loss,
                        (n_epoch-1)*len(TrainDataLoader) + (en+1))


        print ("[Epoch: %d] [Iter: %d/%d] [loss: %f]" % (n_epoch, en, len(TrainDataLoader), loss.cpu().data.numpy()))


### Validation Function
def validation(data_loader, n_epoch):
    EmbeddingNetwork.eval()
    RelationNetwork.eval()

    correct = 0
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
        if args.transfer_learning == False:
            sx_f = torch.sum(sx_f.view(n_way, k_shot, 64, sx_f.shape[-2], sx_f.shape[-1]), 1).squeeze(1).unsqueeze(0).repeat(qx.shape[0], 1, 1, 1, 1)
            qx_f = qx_f.unsqueeze(1).repeat(1, n_way, 1, 1, 1)
            pairs = torch.cat((sx_f, qx_f), 2).view(-1, 64*2, sx_f.shape[-2], sx_f.shape[-1])
        else:
            sx_f = torch.sum(sx_f.view(n_way, k_shot, sx_f.shape[-1]), 1).squeeze(1).unsqueeze(0).repeat(qx.shape[0], 1, 1)
            qx_f = qx_f.unsqueeze(1).repeat(1, n_way, 1)
            pairs = torch.cat((sx_f, qx_f), 2).view(-1, sx_f.shape[-1]*2)

        ## Get relation scores
        scores = RelationNetwork(pairs).view(qx.shape[0], n_way, 1).squeeze(2)

        _, true_labels = torch.max(qy.data, 1)
        _, pred_labels = torch.max(scores.data,1)

        correct += np.sum([1 if pred_labels[j]==true_labels[j] else 0 for j in range(qy.shape[0])])
        total += qy.shape[0]

    return (correct*100)/total





## Few Config
checkpoints_path = args.checkpoint_path
isTrain = args.do_training
save_epoch = args.saving_rate
k_shot = args.k_shot
n_way = args.n_way
learning_rate = args.learning_rate
epochs = args.epochs

if isdir(checkpoints_path)==False:
    makedirs(checkpoints_path)


writer = SummaryWriter('runs/RelationNet_{}_way_{}_shot_TransferLearning'.format(n_way, k_shot))

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

#### Define networks

if args.transfer_learning == False:
    EmbeddingNetwork = md.ResEncoder().to(device)
    RelationNetwork = md.RelationNet().to(device)
else:
    EmbeddingNetwork = models.densenet161(pretrained=True)

    for param in EmbeddingNetwork.parameters():
        param.requires_grad = False   
        
    EmbeddingNetwork.classifier = torch.nn.Sequential(
                torch.nn.Linear(2208, 1024))

    EmbeddingNetwork = EmbeddingNetwork.to(device)
    RelationNetwork = md.TransferRelationNetwork().to(device)

#### Get training, validation, and testing classes
tr, val, te = dl.get_labels("./data/102flowers/imagelabels.mat")
with open(join(checkpoints_path, 'traning_class.pkl'), 'wb') as f1:
    pickle.dump(tr, f1)
with open(join(checkpoints_path, 'validation_class.pkl'), 'wb') as f2:
    pickle.dump(val, f2)
with open(join(checkpoints_path, 'validation_class.pkl'), 'wb') as f3:
    pickle.dump(te, f3)

#### Define dataloaders
TrainData = dl.Few_shot_dataset("./data/102flowers/jpg/", "./data/102flowers/imagelabels.mat", n_way, k_shot, tr)
ValData = dl.Few_shot_dataset("./data/102flowers/jpg/", "./data/102flowers/imagelabels.mat", n_way, k_shot, val)
TestData = dl.Few_shot_dataset("./data/102flowers/jpg/", "./data/102flowers/imagelabels.mat", n_way, k_shot, te)

TrainDataLoader = DataLoader(TrainData, batch_size=1, shuffle=True, num_workers=0)  # num_workers=0 for windows OS
ValDataLoader = DataLoader(ValData, batch_size=1, shuffle=True, num_workers=0)  # num_workers=0 for windows OS
TestDataLoader = DataLoader(TestData, batch_size=1, shuffle=True, num_workers=0)  # num_workers=0 for windows OS


#### Define Loss function
criterion = nn.MSELoss()
triplet = losses.TripletMarginLoss(margin=0.01)

#### Define optimizers
optimizer = torch.optim.Adam(itertools.chain(EmbeddingNetwork.parameters(), RelationNetwork.parameters()), lr=learning_rate, betas=(0.5, 0.999))




if isTrain:
    for  i in range(epochs):
        training(TrainDataLoader, i+1)

        val_acc = validation(ValDataLoader, i+1)

        writer.add_scalar('Validation Accuracy',
                        val_acc,
                        (i+1))

        if (i+1)%save_epoch==0:
            torch.save(EmbeddingNetwork, join(checkpoints_path, "emb_net_{}.pth".format(i+1)))
            torch.save(RelationNetwork, join(checkpoints_path, "relation_net_{}.pth".format(i+1)))
        