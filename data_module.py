# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 16:56:59 2021

@author: tamiryuv
"""
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

####
# GLOBAL PARAMS
###
CWD = os.getcwd()
VALID_IMAGES_PATH = os.path.join(CWD,r'toothbrush\train\good')
INVALID_IMAGES_PATH = os.path.join(CWD,r'toothbrush\test\defective')


class VALID_Image_DataSet(Dataset):
    def __init__(self,images_path = VALID_IMAGES_PATH, augmantation = None):
        super().__init__()
        global CWD, VALID_IMAGES_PATH
        self.images_list = os.listdir(VALID_IMAGES_PATH)
                                    
        self.augmantation = augmantation

    def __len__(self):
        return len(self.images_list)
    
    def __getitem__(self,idx):
        image = Image.open(os.path.join(VALID_IMAGES_PATH,self.images_list[idx]))
        if self.augmantation is not None:
            image = self.augmantation(image)
            
        return image
    
    
class INVALID_Image_DataSet(Dataset):
    def __init__(self,images_path = INVALID_IMAGES_PATH, augmantation = None):
        super().__init__()
        global CWD, INVALID_IMAGES_PATH
        self.images_list = os.listdir(INVALID_IMAGES_PATH)
                                    
        self.augmantation = augmantation

    def __len__(self):
        return len(self.images_list)
    
    def __getitem__(self,idx):
        image = Image.open(os.path.join(INVALID_IMAGES_PATH,self.images_list[idx]))
        if self.augmantation is not None:
            image = self.augmantation(image)
            
        return image