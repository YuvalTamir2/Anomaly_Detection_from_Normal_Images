# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 12:12:28 2021

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
import time


### HP : 
IMAGE_SIZE = 256
LEARNING_RATE = 4e-3
EPOCHES = 100
### load data : 
    
transforms = T.Compose([T.Resize((IMAGE_SIZE,IMAGE_SIZE)), T.ToTensor()])
### load train data
dataset = VALID_Image_DataSet(augmantation = transforms)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
tl = DataLoader(dataset, batch_size=1, shuffle=True)
### load test data
test_dataset = INVALID_Image_DataSet(augmantation = transforms)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


    
### pre-train model :     
def train_vq_vae(train_loader, vq_model):

    vq_model.train()
    vq_optimizer = torch.optim.Adam(params = vq_model.parameters(), lr = LEARNING_RATE)
    since = time.time()
    for epoch in range(EPOCHES):
        train_res_recon_error = []
        train_res_perplexity = []
        print(f'Epoch {epoch} on the way')
        for idx,data in enumerate(train_loader):
            #data = data / 255.
            vq_optimizer.zero_grad()
            vq_loss, data_recon, perplexity = vq_model(data)
            recon_error = F.binary_cross_entropy(data_recon, data)
            loss = recon_error + vq_loss
            loss.backward()
        
            vq_optimizer.step()
            
            train_res_recon_error.append(recon_error.item())
            train_res_perplexity.append(perplexity.item())
        
           
        print('%d iterations' % (epoch+1))
        print('recon_error: %.3f' % np.mean(train_res_recon_error))
        print('perplexity: %.3f' % np.mean(train_res_perplexity))
        print()
    
    print(f'ended run {EPOCHES} epoches in {round(time.time() - since, 1)} seconds\n')
    return vq_model
#vq_model = train_vq_vae(train_loader, model)
### train auto-regressive prior : 
    

def train_AR_prior(vq_model,train_loader):
    print('Training prior AR model')
    cnn = PixelCNN()
    cnn.train()
    ar_optimizer = torch.optim.Adam(cnn.parameters(), lr=4e-3)
    c = 1
    model.eval()
    train_loss = []
    train_con = []
    test_loss = []
    test_con = []
    for i in range(100):
        for data in train_loader:
            b = data.shape[0]
            z = model._encoder(data)
            z = model._pre_vq_conv(z)
            loss,quantized,pre,encoding = vq_model._vq_vae(z)
            X = torch.argmax(encoding, dim = 1).reshape(b,c,64,64).float()
            ar_optimizer.zero_grad()
            loss = F.cross_entropy(input=cnn(X), target=torch.squeeze(X).long().reshape(-1,64,64))
            loss.backward()
            ar_optimizer.step()
            
        print (f'Epoch {i}, Loss: {loss}' )
    return cnn
#pixel_cnn_model = train_AR_prior(vq_model, train_loader)
#### Find Anomalys ### :
    
def prediction(vq_model, pixelcnn_model, data_loader, some_arbitrary_thresh = 0.01):
    vq_model.train(False)
    pixelcnn_model.train(False)
    #num chanllens for latent : 
    c = 1
    starting_point=(0, 0)
    anomaly_scores = []
    direct_recons = []
    resampled_recons = []
    for anomaly_image in test_loader:
        b = anomaly_image.shape[0]
        z = vq_model._encoder(anomaly_image)
        z = vq_model._pre_vq_conv(z)
        loss,quantized,pre,encoding = vq_model._vq_vae(z)
        resmaple_shape = quantized.shape
        X_orig = torch.argmax(encoding, dim = 1).reshape(b,c,64,64).float()
        X_new = torch.clone(X_orig)
        some_arbitrary_thresh = 0.005 
        for i in range(64):  # codebook size (rows)
            for j in range(64): # codebook size (cols)
                if i < starting_point[0] or (i == starting_point[0] and j < starting_point[1]):
                    continue
                out = pixelcnn_model(X_orig)
                probs = F.softmax(out[:, :, i, j]).data
                mle = torch.argmax(probs, 1).float()
                observed_likelihood = probs[:,X_orig[:,:,i,j].long().item()]
                if observed_likelihood >= some_arbitrary_thresh:
                    continue
                else:
                    print(i,j)
                    X_new[:, :, i, j] = torch.argmax(probs, 1).long()
        encodings = torch.zeros(X_new.reshape(-1,1).shape[0], 512)
        encodings.scatter_(1, X_new.reshape(-1,1).long(), 1)
        re_sample_quantized = torch.matmul(encodings, model._vq_vae._embedding.weight).view(resmaple_shape).permute(0,3,1,2)
        direct_recon = model._decoder(quantized)
        resample_recon = model._decoder(re_sample_quantized) 
        diff = torch.abs(direct_recon - resample_recon)
        
        # to return 
        
        anomaly_scores.append(diff.sum().item())
        direct_recons.append(direct_recon.data)
        resampled_recons.append(resample_recon.data)
        
        #break
    return direct_recons,resampled_recons,anomaly_scores
    
    
    

## plot the findings :
    
def plot_anomalys(direct_recons,resampled_recons,anomaly_scores,loader,thresh = 10):
    for idx, image in enumerate(loader):
        anomaly_score = anomaly_scores[idx]
        if anomaly_score <= thresh:
            continue
        else:
            print(f'Plotting Anomalys.\nthe threshold is {thresh}. consider tweaking for more accurate results.')
            plt.imshow(image[0].detach().numpy().transpose(1,2,0))
            plt.pause(0.001)
            plt.imshow(direct_recons[idx][0].detach().numpy().transpose(1,2,0))
            plt.pause(0.001)
            plt.imshow(resampled_recons[idx][0].detach().numpy().transpose(1,2,0))
            plt.pause(0.001)
            diff = torch.abs(direct_recons[idx][0] - resampled_recons[idx][0])
            plt.imshow(diff[0].detach().numpy().transpose(1,2,0))



