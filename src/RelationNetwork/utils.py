import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from thop import profile

import dataloader as dl
import models as md

import argparse


parser = argparse.ArgumentParser(description="Learning to learn: RelationNetwork Parameters")
parser.add_argument("-tsf", "--transfer_learning", type=bool, default=False)

def get_params(n_way, k_shot):
    args = parser.parse_args()
    if args.transfer_learning == False:
        EmbeddingNetwork = md.InceptionEncoder()
        RelationNetwork = md.RelationNet()
    else:
        EmbeddingNetwork = models.densenet161(pretrained=True)

        for param in EmbeddingNetwork.parameters():
            param.requires_grad = False   
            
        classifier = torch.nn.Sequential(
                    torch.nn.Linear(2208, 1024))
        EmbeddingNetwork.classifier = torch.nn.Sequential(
                    torch.nn.Linear(2208, 1024))

        RelationNetwork = md.TransferRelationNetwork()

    tr, _, _ = dl.get_labels("./data/102flowers/imagelabels.mat")
    TrainData = dl.Few_shot_dataset("./data/102flowers/jpg/", "./data/102flowers/imagelabels.mat", n_way, k_shot, tr)
    TrainDataLoader = DataLoader(TrainData, batch_size=1, shuffle=True, num_workers=0)  # num_workers=0 for windows OS 

    for sx,qx,sy,qy in TrainDataLoader:
        sx = Variable(sx.squeeze(0))
        sy = Variable(sy.squeeze(0))
        qx = Variable(qx.squeeze(0))
        qy = Variable(qy.squeeze(0))

        ## extract embeddings
        if args.transfer_learning == False:
            macs1, params1 = profile(EmbeddingNetwork, inputs=(sx, ))
        else:
            macs1, params1 = profile(classifier, inputs=(torch.randn((n_way*k_shot, 2208)), ))
        # print(sx.shape)
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
        macs2, params2 = profile(RelationNetwork, inputs=(pairs, ))

        print("macs:{}\tparams:{}".format(macs1+macs2, params1+params2))

        break


if __name__=="__main__":
    get_params(5, 5)