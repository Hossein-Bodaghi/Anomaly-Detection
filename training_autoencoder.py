#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 21:33:07 2020
Convolutional autoencoder
@author: hossein
"""
import scipy.io as sio
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import transforms


path = '/home/hossein/Downloads/hazelnut/train/good/'
images = glob.glob(path + '*.png')

train = []
for img in images:
    image = cv2.imread(img)
    train.append(image)

class Autoencoder(nn.Module):
    
    def __init__(self):
        super(Autoencoder,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(32, 32, kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(32, 32, kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(32, 64, kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(64, 64, kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(64, 128, kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(128, 64, kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(64, 32, kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(32, 100, kernel_size=8,stride=1,padding=0),
            nn.LeakyReLU(0.2,inplace=True)
            )        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(100, 32, kernel_size=8,stride=1,padding=0),
            nn.LeakyReLU(0.2,inplace=True),
            nn.ConvTranspose2d(32, 64, kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.ConvTranspose2d(64, 128, kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4,stride=2,padding=1), 
            nn.LeakyReLU(0.2,inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4,stride=2,padding=1),
            nn.Sigmoid()
            )
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

transform = transforms.Compose([transforms.ToPILImage(),
            transforms.ToTensor()])

class ADdata(Dataset):
    
    def __init__(self,train,transforms=None):
        self.data = train
        self.transform = transforms
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        x = self.data[idx]
        if self.transform:
            x = self.transform(x)
        else: pass
        return (x,x)
    
train_data = ADdata(train,transforms=transform)
trainloader = DataLoader(train_data , batch_size=32 , shuffle=True)
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder = Autoencoder().to(device)        

criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(),lr=0.00001)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)        
    
def train(net, trainloader):
    for epoch in range(5):
        running_loss = 0
        for data in trainloader:
            inputs , labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = autoencoder(inputs)
            loss = criterion(outputs , labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step(epoch)
        print('[Epoch %d] loss: %.3f' %
                      (epoch + 1, running_loss/len(trainloader)))  
    print('Training is Done')

train(autoencoder, trainloader)
