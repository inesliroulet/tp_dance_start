
import numpy as np
import cv2
import os
import pickle
import sys
import math

import matplotlib.pyplot as plt

from torchvision.io import read_image
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton
from GenVanillaNN import * 



# Based on "Improved Techniques for Training GANs" (2016) by Salimans et al.
class BatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, batch_size, num_kernels=100, kernel_dim=5):
        super(BatchDiscrimination, self).__init__()
        self.batch_size = batch_size
        self.num_kernels = num_kernels
        self.kernel_dim = kernel_dim
        
        # Initialize the tensor for the learned matrix
        self.T = nn.Parameter(torch.randn(in_features, num_kernels * kernel_dim))
        
    def forward(self, x):
        # Multiply input by the learned matrix T
        M = x @ self.T  # Shape: [batch_size, num_kernels * kernel_dim]
        M = M.view(-1, self.num_kernels, self.kernel_dim)  # Shape: [batch_size, num_kernels, kernel_dim]

        # Compute L1 distance between all pairs within the batch
        M1 = M.unsqueeze(0)  # Shape: [1, batch_size, num_kernels, kernel_dim]
        M2 = M.unsqueeze(1)  # Shape: [batch_size, 1, num_kernels, kernel_dim]
        L1_distance = torch.abs(M1 - M2).sum(3)  # Sum across kernel_dim, shape: [batch_size, batch_size, num_kernels]

        # Apply exponential function to distance and sum across batch dimension
        exp_distances = torch.exp(-L1_distance)  # Shape: [batch_size, batch_size, num_kernels]
        output = exp_distances.sum(1)  # Shape: [batch_size, num_kernels]
        
        return torch.cat([x, output], dim=1)  # Concatenate original features with batch discrimination features


class Discriminator(nn.Module):
    def __init__(self, ngpu=0, batch_size=32):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.batch_size = batch_size
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # [batch_size, 64, 32, 32]
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # [batch_size, 128, 16, 16]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # [batch_size, 256, 8, 8]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # [batch_size, 512, 4, 4]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),  # [batch_size, 1024, 2, 2]
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
        )
        
        # Define the fully connected layer after the convolutions
        self.fc = nn.Linear(1024 * 2 * 2, 512)
        
        # Add the Batch Discrimination layer
        self.batch_discrimination = BatchDiscrimination(512, out_features=100, batch_size=batch_size)
        
        # Output layer (real/fake classification)
        self.out = nn.Sequential(
            nn.Linear(512 + 100, 1),  # Output with batch discrimination features added
            nn.Sigmoid()
        )

    def forward(self, input):
        input = self.model(input)
        input = input.view(input.size(0), -1)  # Flatten for fully connected layer
        input = self.fc(input)
        input = self.batch_discrimination(input)
        return self.out(input).view(-1, 1).squeeze(1)
        # return self.model(input)
    



class GenGAN():
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
    """
    def __init__(self, videoSke, loadFromFile=False, optSkeOrImage=2):
        self.optSkeOrImage = optSkeOrImage
        self.netG = GenNNSkeToImage(optSkeOrImage=self.optSkeOrImage)
        self.netD = Discriminator()
        self.real_label = 1.
        self.fake_label = 0.
        self.filename = 'data/Dance/DanceGenGAN.pth'
        tgt_transform = transforms.Compose(
                            [transforms.Resize((64, 64)),
                            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            transforms.CenterCrop(64),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform, optSkeOrImage=self.optSkeOrImage)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=32, shuffle=True)
        if loadFromFile and os.path.isfile(self.filename):
            print("GenGAN: Load=", self.filename, "   Current Working Directory=", os.getcwd())
            self.netG = torch.load(self.filename)


    def train(self, n_epochs=20):
        criterion = nn.BCELoss()
        optimizerD = optim.Adam(self.netD.parameters(), lr=0.00005, betas=(0.5, 0.999))
        optimizerG = optim.Adam(self.netG.parameters(), lr=0.0001, betas=(0.5, 0.999))

        for epoch in range(n_epochs):
            for skeletons, real_images in self.dataloader:
                
                # === Train Discriminator ===
                self.netD.zero_grad()
                
                # Label smoothing: use slightly smaller real labels to encourage stability
                real_labels = torch.full((real_images.size(0),), 0.95, dtype=torch.float32)
                fake_labels = torch.full((real_images.size(0),), 0.05, dtype=torch.float32)

                # Forward real images through discriminator
                output = self.netD(real_images)
                lossD_real = criterion(output, real_labels)
                lossD_real.backward()

                # Generate fake images from skeletons
                if self.optSkeOrImage==1:
                    skeletons = skeletons.view(skeletons.size(0), -1)
                    
                fake_images = self.netG(skeletons)
                
                # Forward fake images through discriminator
                output = self.netD(fake_images.detach())
                lossD_fake = criterion(output, fake_labels)
                lossD_fake.backward()
                optimizerD.step()

                # === Train Generator ===
                self.netG.zero_grad()
                label_g = torch.full((real_images.size(0),), 1, dtype=torch.float32)  # Generator wants to fool D to think images are real
                output = self.netD(fake_images)
                lossG = criterion(output, label_g)
                lossG.backward()
                optimizerG.step()

            print(f'Epoch [{epoch+1}/{n_epochs}] | LossD: {(lossD_real + lossD_fake).item():.4f} | LossG: {lossG.item():.4f}')


    def generate(self, ske):           # TP-TODO
        """ generator of image from skeleton """
        if self.optSkeOrImage==1:
            ske_t = self.dataset.preprocessSkeleton(ske)
            ske_t = ske_t.view(1, -1)  # Reshape for batch dimension (1, 26)
        if self.optSkeOrImage==2:
            ske_image = self.dataset.skeToImageTransform(ske)
            ske_image = np.array(ske_image, dtype=np.float32)
            ske_t = transforms.ToTensor()(ske_image).unsqueeze(0) # Convert to tensor and normalize
        
        normalized_output = self.netG(ske_t)
        res = self.dataset.tensor2image(normalized_output[0])
        return res




if __name__ == '__main__':
    force = False
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "tp/dance/data/taichi1.mp4"
    print("GenGAN: Current Working Directory=", os.getcwd())
    print("GenGAN: Filename=", filename)

    targetVideoSke = VideoSkeleton(filename)

    #if False:
    if True:    # train or load
        # Train
        gen = GenGAN(targetVideoSke, False)
        gen.train(4) #5) #200)
    else:
        gen = GenGAN(targetVideoSke, loadFromFile=True)    # load from file        


    for i in range(targetVideoSke.skeCount()):
        image = gen.generate(targetVideoSke.ske[i])
        #image = image*255
        nouvelle_taille = (256, 256) 
        image = cv2.resize(image, nouvelle_taille)
        cv2.imshow('Image', image)
        key = cv2.waitKey(-1)

