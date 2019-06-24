#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
Created on Thu Jun 13 15:45:13 2019

@author: yangzhenhuan

Training Generative Adversarial Network with loss and first order algorithm 
'''

import os
import csv
import argparse

import models
import lib
import utils

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

# Parse arguments for command line
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--v', action='store_true', help='if True verbose training')
parser.add_argument('--ds', choices=('cifar10','mnist'), default='cifar10', help='dataset')
parser.add_argument('--po', type=float, default=0.001, help='portion of data to train')
parser.add_argument('--arc', choices=('dcgan'), default='dcgan', help='architecture of GAN')
parser.add_argument('--loss', choices=('gan', 'lsgan'), default='gan', help='loss function of GAN')
parser.add_argument('--alg', choices=('Adam', 'AdaGrad', 'ExtraSGD'), default='Adam', help='optimization algorithm')
parser.add_argument('--ne', type=int, default=2, help='number of epochs of training')
parser.add_argument('--bs', type=int, default=20, help='size of the batches')
parser.add_argument('--sd', type=int, default=1234, help='random seed')
parser.add_argument('--lrg', type=float, default=0.0002, help='learning rate of generator')
parser.add_argument('--lrd', type=float, default=0.0002, help='learning rate of discriminator')
parser.add_argument('--ema', type=float, default=0.9999, help='exponential moving average')
parser.add_argument('--b1', type=float, default=0.5, help='Adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='Adam: decay of first order momentum of gradient')
parser.add_argument('--ka', type=float, default=1000.0, help='AccSGD: long step')
parser.add_argument('--xi', type=float, default=10.0, help='AccSGD: Statistical advantage parameter between 1.0 to sqrt(ka)')
parser.add_argument('--dv', type=int, default=1, help='gpu: device number')
parser.add_argument('--nz', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img', type=int, default=64, help='size of each image dimension')
parser.add_argument('--nc', type=int, default=3, help='number of image channels')
parser.add_argument('--nfd', type=int, default=64, help='number of filters of discriminator')
parser.add_argument('--nfg', type=int, default=64, help='number of filters of generator')
parser.add_argument('--ins', action='store_true', help='if True compute inception score')
parser.add_argument('--fid', action='store_true', help='if True compute fid score')
parser.add_argument('--save', action='store_true', help='if True save state every epoch')
args = parser.parse_args(['--v'])

# Define Hyper-parameter from 
VERBOSE = args.v
DATA = args.ds
PORTION = args.po
ARCHITECTURE = args.arc
LOSS = args.loss
ALG = args.alg
NUM_EPOCH = args.ne
BATCH_SIZE = args.bs
SEED = args.sd
torch.manual_seed(SEED)
LEARNING_RATE_G = args.lrg
LEARNING_RATE_D = args.lrd
BETA_EMA = args.ema
BETA_1 = args.b1
BETA_2 = args.b2
KAPPA = args.ka
XI = args.xi
DEVICE = args.dv
NUM_LATENT = args.nz
IMAGE_SIZE = args.img
NUM_CHANNELS = args.nc
NUM_FILTER_G = args.nfg
NUM_FILTER_D = args.nfd
INCEPTION_SCORE_FLAG = args.ins
FID_SCORE_FLAG = args.fid
SAVE_FLAG = args.save

# Define output location of one specific run
OUTPUT_PATH = 'result'
OUTPUT_PATH = os.path.join(OUTPUT_PATH,'%s'%ARCHITECTURE, '%s'%LOSS, '%s'%ALG, 'lrg=%.5e_lrd=%.5e'%(LEARNING_RATE_G,LEARNING_RATE_D))
if not os.path.exists(os.path.join(OUTPUT_PATH, 'checkpoints')):
    os.makedirs(os.path.join(OUTPUT_PATH, 'checkpoints'))
loss_f = open(os.path.join(OUTPUT_PATH, 'loss.csv'), 'ab')
loss_writter = csv.writer(loss_f)
    
if INCEPTION_SCORE_FLAG or FID_SCORE_FLAG:
    NUM_SAMPLES = 1000
    fixed_noise = torch.randn(NUM_SAMPLES, NUM_LATENT, 1, 1, device=DEVICE)
    
    if INCEPTION_SCORE_FLAG:
        inception_f = open(os.path.join(OUTPUT_PATH, 'inception_score.csv'), 'ab')
        inception_writter = csv.writer(inception_f)
    if FID_SCORE_FLAG:
        fid_f = open(os.path.join(OUTPUT_PATH, 'fid_score.csv'), 'ab')
        fid_writter = csv.writer(fid_f)

print('Initializing...')
# Load and transform data
transform=transforms.Compose([transforms.Resize(IMAGE_SIZE),transforms.CenterCrop(IMAGE_SIZE),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
if DATA == 'cifar10':
    trainset = dset.CIFAR10(root='./data', train=True, transform=transform, download=True)
    testset = dset.CIFAR10(root='./data', train=False, transform=transform, download=True)    
elif DATA == 'mnist':
    trainset = dset.MNIST(root='./data', train=True, transform=transform, download=True)
    testset = dset.MNIST(root='./data', train=False, transform=transform, download=True)
    NUM_CHANNELS = 1
indices = list(range(int(len(trainset)*PORTION)))
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(indices), num_workers=0)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, num_workers=0)

# Decide which device we want to run on
device = torch.device('cuda:%i'%DEVICE if torch.cuda.is_available() else 'cpu')
print(device)

# Define and initialize models
if ARCHITECTURE == 'dcgan':
    netG = models.dcgan.Generator(nc=NUM_CHANNELS, nfg=NUM_FILTER_G).to(device)
    netD = models.dcgan.Discriminator(nc=NUM_CHANNELS, nfd=NUM_FILTER_D).to(device)
netG.apply(utils.weights_init)
netD.apply(utils.weights_init)
print(netG)
print(netD)

# Define optimizer 
if ALG == 'Adam':
    import torch.optim as optim
    optG = optim.Adam(netG.parameters(), lr=LEARNING_RATE_G, betas=(BETA_1, BETA_2))
    optD = optim.Adam(netD.parameters(), lr=LEARNING_RATE_D, betas=(BETA_1, BETA_2))
elif ALG == 'AccSGD':
    import optim
    optG = optim.AccSGD(netG.parameters(), lr=LEARNING_RATE_G, kappa = KAPPA, xi = XI)
    optD = optim.AccSGD(netD.parameters(), lr=LEARNING_RATE_D, kappa = KAPPA, xi = XI)
elif ALG == 'Adagrad':
    import torch.optim as optim
    optG = optim.Adagrad(netG.parameters(), lr=LEARNING_RATE_G)
    optD = optim.Adagrad(netD.parameters(), lr=LEARNING_RATE_D)
elif ALG == 'ExtraSGD':
    import optim
    optG = optim.ExtraSGD(netG.parameters(), lr=LEARNING_RATE_G)
    optD = optim.ExtraSGD(netD.parameters(), lr=LEARNING_RATE_D)

# Initialize Loss function
if LOSS == 'gan':
    criterion = nn.BCELoss()
elif LOSS == 'lsgan':
    criterion = nn.MSELoss()

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Lists to keep track of progress
netG_param_avg = []
netG_param_ema = []
for param in netG.parameters():
    netG_param_avg.append(param.data.clone())
    netG_param_ema.append(param.data.clone())

iters = 0

print('Starting Training Loop...')
# For each epoch
for epoch in range(NUM_EPOCH):
    # For each batch in the dataloader
    for i, data in enumerate(trainloader, 0): # data contains x and y

        ############################
        # (1) Update D network: 
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()
        
        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, NUM_LATENT, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1) # Has to detach to avoid backpropagate!!!!!!
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        if ALG == 'ExtraSGD':
            if (i+1)%2:
                optD.extrapolation()
            else:
                optD.step()
        else:
            optD.step()
        
        ############################
        # (2) Update G network: 
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1) # view -1 to reshape output as a 1-dim tensor
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        if ALG == 'ExtraSGD':
            if (i+1)%2:
                optG.extrapolation()
            else:
                optG.step()
        else:
            optG.step()
        
        iters += 1
        
        for j, param in enumerate(netG.parameters()):
                netG_param_avg[j] = netG_param_avg[j]*iters/(iters+1.) + param.data.clone()/(iters+1.)
                netG_param_ema[j] = netG_param_ema[j]*BETA_EMA+ param.data.clone()*(1-BETA_EMA)
        
        # Output training stats
        if i % 10 == 0: 
            if VERBOSE:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' 
                      % (epoch, NUM_EPOCH, i, len(trainloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            loss_writter.writerow((errD.item(), errG.item()))
            loss_f.flush()
            if INCEPTION_SCORE_FLAG or FID_SCORE_FLAG:
                fake_imgs = netG(fixed_noise).detach().cpu()
                if INCEPTION_SCORE_FLAG:
                    inception_writter.writerow(lib.get_inception_score(fake_imgs))
                    inception_f.flush()
                if FID_SCORE_FLAG:
                    fid_writter.writerow(lib.get_fid_score(fake_imgs))
                    fid_f.flush()
                
    if SAVE_FLAG:    
        #https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-models-in-pytorch
        torch.save({'args': vars(args), 'state_gen': netG.state_dict(), 'param_avg': netG_param_avg, 'param_ema': netG_param_ema}, os.path.join(OUTPUT_PATH, 'checkpoints/gen-%i.state'%epoch))
        torch.save({'args': vars(args), 'state_dis': netD.state_dict()}, os.path.join(OUTPUT_PATH, 'checkpoints/dis-%i.state'%epoch))
