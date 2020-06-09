import numpy as np
import sys
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from scipy.io import loadmat
from PIL import Image
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

import transformers
import tokenizers


def get_labels(mat_file, tarin_prob=0.5, val_prob=0.2, test_prob=0.3):
    labels = np.unique(loadmat(mat_file)['labels'][0])
    train_class = []
    test_class = []
    val_class = []

    for i in labels:
        r = np.random.rand()
        if r<=tarin_prob:
            train_class.append(i)
        elif r<=tarin_prob+val_prob:
            val_class.append(i)
        else:
            test_class.append(i)

    return train_class, val_class, test_class

class Few_shot_dataset(Dataset):
    def __init__(self, image_path, mat_file, n_way, k_shot, target_class):
        self.image_path = image_path
        self.mat_file = mat_file

        self.images = listdir(self.image_path)
        self.labels = loadmat(mat_file)['labels'][0]


        self.label2img = {}

        for i, j in enumerate(self.images):

            if self.labels[i] in target_class:
                if self.labels[i] in self.label2img:
                    self.label2img[self.labels[i]].append(join(self.image_path, j))
                else:
                    self.label2img[self.labels[i]] = [join(self.image_path, j)]
        
        self.support_set_x = []
        self.query_set_x = []

        self.support_set_y = []
        self.query_set_y = []

        for i in range(100):
            cls = np.random.choice(target_class, n_way, False)
            sup_tmp_x = []
            q_tmp_x = []

            sup_tmp_y = []
            q_tmp_y = []

            for k,j in enumerate(cls):
                imgs = np.random.choice(len(self.label2img[j]), 2*k_shot, False)
                sup_tmp_x.append(np.array(self.label2img[j])[np.array(imgs[:k_shot])].tolist())
                q_tmp_x.append(np.array(self.label2img[j])[np.array(imgs[k_shot:])].tolist())

                ch = torch.zeros(n_way)
                ch[k] = 1
                ch = ch.unsqueeze(0).repeat(k_shot, 1)

                sup_tmp_y.append(ch.numpy())
                q_tmp_y.append(ch.numpy())
            
            self.support_set_x.append(sup_tmp_x)
            self.query_set_x.append(q_tmp_x)
            
            self.support_set_y.append(sup_tmp_y)
            self.query_set_y.append(q_tmp_y)


        self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                transforms.Resize((128, 128)),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                            ])

    def __getitem__(self, index):
        sx = []
        sy = []
        qx = []
        qy = []

        for i in range(len(self.support_set_x[index])):
            for j in range(len(self.support_set_x[index][i])):
                sx.append(self.transform(self.support_set_x[index][i][j]).numpy())
                sy.append(self.support_set_y[index][i][j])

        for i in range(len(self.query_set_x[index])):
            for j in range(len(self.query_set_x[index][i])):
                qx.append(self.transform(self.query_set_x[index][i][j]).numpy())
                qy.append(self.query_set_y[index][i][j])

        sx = torch.from_numpy(np.array(sx))
        sy = torch.from_numpy(np.array(sy))
        qx = torch.from_numpy(np.array(qx))
        qy = torch.from_numpy(np.array(qy))

        return sx, qx, sy, qy

    def __len__(self):
        return 100

def get_text_labels(data_file, tarin_prob=0.5, val_prob=0.2, test_prob=0.3):
    labels = np.unique(pd.read_json(data_file, lines=True)['category'].to_numpy())
    train_class = []
    test_class = []
    val_class = []

    for i in labels:
        r = np.random.rand()
        if r<=tarin_prob:
            train_class.append(i)
        elif r<=tarin_prob+val_prob:
            val_class.append(i)
        else:
            test_class.append(i)

    return train_class, val_class, test_class

class Few_shot_text_dataset(Dataset):
    def __init__(self, data_file, n_way, k_shot, target_class, tokenizer):
        self.tokenizer = tokenizer

        self.data_file = data_file
        self.data = pd.read_json(data_file, lines=True)[['category', 'headline']]
        self.label2text = {}

        for i in range(len(self.data)):

            if self.data.category[i] in target_class:
                if self.data.category[i] in self.label2text:
                    self.label2text[self.data.category[i]].append(self.data.headline[i])
                else:
                    self.label2text[self.data.category[i]] = [self.data.headline[i]]
        
        self.support_set_x = []
        self.query_set_x = []

        self.support_set_y = []
        self.query_set_y = []

        for i in range(100):
            cls = np.random.choice(target_class, n_way, False)
            sup_tmp_x = []
            q_tmp_x = []

            sup_tmp_y = []
            q_tmp_y = []

            for k,j in enumerate(cls):
                txts = np.random.choice(len(self.label2text[j]), 2*k_shot, False)
                sup_tmp_x.append(np.array(self.label2text[j])[np.array(txts[:k_shot])].tolist())
                q_tmp_x.append(np.array(self.label2text[j])[np.array(txts[k_shot:])].tolist())

                ch = torch.zeros(n_way)
                ch[k] = 1
                ch = ch.unsqueeze(0).repeat(k_shot, 1)

                sup_tmp_y.append(ch.numpy())
                q_tmp_y.append(ch.numpy())
            
            self.support_set_x.append(sup_tmp_x)
            self.query_set_x.append(q_tmp_x)
            
            self.support_set_y.append(sup_tmp_y)
            self.query_set_y.append(q_tmp_y)


    def process_data(self, text, max_len):
        text = " " + " ".join(str(text).split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=max_len,
        )
        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]
        
        padding_length = max_len - len(ids)
        
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        }


    def __getitem__(self, index):
        sx = {
            'ids': [],
            'mask': [],
            'token_type_ids': []
        }
        sy = []
        qx = {
            'ids': [],
            'mask': [],
            'token_type_ids': []
        }
        qy = []

        for i in range(len(self.support_set_x[index])):
            for j in range(len(self.support_set_x[index][i])):
                tmp = self.process_data(self.support_set_x[index][i][j], 64)
                sx['ids'].append(tmp['ids'].numpy())
                sx['mask'].append(tmp['mask'].numpy())
                sx['token_type_ids'].append(tmp['token_type_ids'].numpy())

                sy.append(self.support_set_y[index][i][j])

        for i in range(len(self.query_set_x[index])):
            for j in range(len(self.query_set_x[index][i])):
                tmp = self.process_data(self.query_set_x[index][i][j], 64)
                qx['ids'].append(tmp['ids'].numpy())
                qx['mask'].append(tmp['mask'].numpy())
                qx['token_type_ids'].append(tmp['token_type_ids'].numpy())

                qy.append(self.query_set_y[index][i][j])

        # sx = torch.from_numpy(np.array(sx))
        sy = torch.from_numpy(np.array(sy))
        # qx = torch.from_numpy(np.array(qx))
        qy = torch.from_numpy(np.array(qy))


        sx = {
            'ids': torch.tensor(sx['ids'], dtype=torch.long),
            'mask': torch.tensor(sx['mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(sx['token_type_ids'], dtype=torch.long),
        }
        qx = {
            'ids': torch.tensor(qx['ids'], dtype=torch.long),
            'mask': torch.tensor(qx['mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(qx['token_type_ids'], dtype=torch.long),
        }
        return sx, qx, sy, qy

    def __len__(self):
        return 100
