# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 16:47:22 2021

@author: tamiryuv
"""
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from VQ_VAE_MODEL import model
import torch.nn.functional as F
from data_module import VALID_Image_DataSet, INVALID_Image_DataSet
from torch.utils.data import DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt
from SimplePixelCNN import PixelCNN
from scipy.stats import norm
from utils import *


def save_model(model,filename,model_type = 'VQ VAE'):
    print(f'saving {model_type} model as {filename}')
    checkpoint = {
        "state_dict": model.state_dict(),
        #"optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    



if __name__ == '__main__':
    print('Start Training')
    vq_model = train_vq_vae(train_loader, model)
    pixel_cnn_model = train_AR_prior(vq_model, train_loader)
    save_model(vq_model, 'vq_vae.pth')
    save_model(pixel_cnn_model,'pixelCNN.pth', model_type = 'Auto-Regressive')
    
    
    
    