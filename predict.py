# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 18:20:24 2021

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


def load_model(model, saved_model_path):
    print('=> Loading model')
    checkpoint = torch.load(saved_model_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


if __name__ == '__main__':
    print('Start Predicting')
    cnn = PixelCNN()
    vq_model = load_model(model, 'vq_vae.pth')
    pixelcnn_model = load_model(cnn, 'pixelCNN.pth')
    direct_recons,resampled_recons,anomaly_scores = prediction(vq_model, pixelcnn_model, test_loader)