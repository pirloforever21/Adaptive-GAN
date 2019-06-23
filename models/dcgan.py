#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:28:23 2019

@author: yangzhenhuan

Define the Deep Convolutional Generative Adversarial Network class

nc - Number of channels in the training images. For color images this is 3
nz - Size of z latent vector (i.e. size of generator input)
nfg - Size of feature maps in generator
nfd - Size of feature maps in discriminator
"""

import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, nc=3, nz=100, nfg=64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, nfg * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(nfg * 8),
            nn.ReLU(True),
            # state size. (nfg*8) x 4 x 4
            nn.ConvTranspose2d(nfg * 8, nfg * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nfg * 4),
            nn.ReLU(True),
            # state size. (nfg*4) x 8 x 8
            nn.ConvTranspose2d( nfg * 4, nfg * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nfg * 2),
            nn.ReLU(True),
            # state size. (nfg*2) x 16 x 16
            nn.ConvTranspose2d( nfg * 2, nfg, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nfg),
            nn.ReLU(True),
            # state size. (nfg) x 32 x 32
            nn.ConvTranspose2d( nfg, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, nc=3, nfd=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, nfd, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nfd) x 32 x 32
            nn.Conv2d(nfd, nfd * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nfd * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nfd*2) x 16 x 16
            nn.Conv2d(nfd * 2, nfd * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nfd * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nfd*4) x 8 x 8
            nn.Conv2d(nfd * 4, nfd * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nfd * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nfd*8) x 4 x 4
            nn.Conv2d(nfd * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)