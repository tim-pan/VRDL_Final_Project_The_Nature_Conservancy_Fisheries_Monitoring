import os
import json
from os.path import join

import numpy as np
import scipy
from scipy import io
import scipy.misc
from PIL import Image
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import imageio

import torch
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url, list_dir, check_integrity, extract_archive, verify_str_arg

class fish(Dataset):
    def __init__(self, root, is_train=True, transform=None):
        self.root = root
        self.is_train = is_train
        self.transform = transform
        self.classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
        
        self.dict_str2num = {}
        for cla, ind in zip(self.classes, range(8)):
            self.dict_str2num[cla] = ind
            
        self.dict_num2str = {}
        for ind, cla in zip(range(8), self.classes):
            self.dict_num2str[ind] = cla
        
        self.img_list = []
        self.img_name_list = []
        self.img_label_list = []

        if self.is_train:
            for cla in self.classes:
                for file_name in os.listdir(os.path.join(self.root, cla)):
                    # train_img = imageio.imread(os.path.join(self.root, cla, file_name))
                    # self.img_list.append(train_img)
                    self.img_name_list.append(file_name)
                    self.img_label_list.append(cla)

        if not self.is_train:
            for cla in self.classes:
                for file_name in os.listdir(os.path.join(self.root, cla)):
                    # test_img = imageio.imread(os.path.join(self.root, cla, file_name))
                    # self.img_list.append(test_img)
                    self.img_name_list.append(file_name)
                    self.img_label_list.append(cla)


    def __getitem__(self, index):
        if self.is_train:
            img_name, target = self.img_name_list[index], self.img_label_list[index]
            label = self.dict_str2num[target]
            
            img = Image.open(os.path.join(self.root, target, img_name))
            
            # if len(img.shape) == 2:
            #     img = np.stack([img] * 3, 2)
            # img = Image.fromarray(img, mode='RGB')
            if self.transform is not None:
                img = self.transform(img)
            return img, label

        else:
            img_name, target = self.img_name_list[index], self.img_label_list[index]
            label = self.dict_str2num[target]
            
            img = Image.open(os.path.join(self.root, target, img_name))
            
            # if len(img.shape) == 2:
            #     img = np.stack([img] * 3, 2)
            # img = Image.fromarray(img, mode='RGB')
            if self.transform is not None:
                img = self.transform(img)

            return img, label

    def __len__(self):
        return len(self.img_name_list)
                
class fish_test(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
        
        self.dict_num2str = {}
        for ind, cla in zip(range(8), self.classes):
            self.dict_num2str[ind] = cla
        
        # self.img_list = []
        self.img_name_list = []

        self.test_stg1_name = os.listdir(os.path.join(self.root, 'test_stg1'))
        self.test_stg1_name = sorted(self.test_stg1_name)
        self.test_stg2_name = os.listdir(os.path.join(self.root, 'test_stg2'))
        self.test_stg2_name = sorted(self.test_stg2_name)

        self.img_name_list = self.test_stg1_name + self.test_stg2_name
        
    def __getitem__(self, index):

        img_name = self.img_name_list[index]
        
        if index<=999:
            # img = imageio.imread(os.path.join(self.root, 'test_stg1', img_name))
            img = Image.open(os.path.join(self.root, 'test_stg1', img_name))
            # img = Image.fromarray(img, mode='RGB')
        else:
            # img = imageio.imread(os.path.join(self.root, 'test_stg2', img_name))
            img = Image.open(os.path.join(self.root, 'test_stg2', img_name))
            img_name = 'test_stg2/' + img_name
            # img = Image.fromarray(img, mode='RGB')
            
        if self.transform is not None:
            img = self.transform(img)

        return img, img_name

    def __len__(self):
        return len(self.img_name_list)
                
