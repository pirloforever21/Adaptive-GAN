#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 15:45:13 2019

@author: yangzhenhuan

Training Generative Adversarial Network with loss and first order algorithm 
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
import argparse
import pickle

import model
import utils

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

# Parse arguments for command line
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--out", action="store_true", help="if True store output to result")
parser.add_argument("--dis", action="store_true", help="if True display training")
parser.add_argument("--ds", default="cifar10", help="dataset")
parser.add_argument("--po", type=float, default=0.1, help="portion of data to train")
parser.add_argument("--arc", default="dcgan", help="architecture of GAN")
parser.add_argument("--loss", default="gan", help="loss function of GAN")
parser.add_argument("--alg", default="Adam", help="optimization algorithm")
parser.add_argument("--ne", type=int, default=100, help="number of epochs of training")
parser.add_argument("--bs", type=int, default=64, help="size of the batches")
parser.add_argument("--sd", type=int, default=1234, help="random seed")
parser.add_argument("--lrg", type=float, default=0.0002, help="learning rate of generator")
parser.add_argument("--lrd", type=float, default=0.0002, help="learning rate of discriminator")
parser.add_argument("--b1", type=float, default=0.5, help="Adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="Adam: decay of first order momentum of gradient")
parser.add_argument("--ka", type=float, default=1000.0, help="AccSGD: long step")
parser.add_argument("--xi", type=float, default=10.0, help="AccSGD: Statistical advantage parameter between 1.0 to sqrt(ka)")
parser.add_argument("--dv", type=int, default=1, help="gpu: device number")
parser.add_argument("--ncpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--ngpu", type=int, default=8, help="number of gpu threads to use during batch generation")
parser.add_argument("--nz", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img", type=int, default=64, help="size of each image dimension")
parser.add_argument("--nc", type=int, default=3, help="number of image channels")
parser.add_argument("--nfd", type=int, default=64, help="number of filters of discriminator")
parser.add_argument("--nfg", type=int, default=64, help="number of filters of generator")
args = parser.parse_args()

# Define Hyper-parameter from 
OUTPUT = args.out
DISPLAY = args.dis
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
BETA_1 = args.b1
BETA_2 = args.b2
KAPPA = args.ka
XI = args.xi
DEVICE = args.dv
NUM_CPU = args.ncpu
NUM_GPU = args.ngpu
NUM_LATENT = args.nz
IMAGE_SIZE = args.img
NUM_CHANNELS = args.nc
NUM_FILTER_G = args.nfg
NUM_FILTER_D = args.nfd

# Define output location of one specific run
if OUTPUT:
    OUTPUT_PATH = "result"
    OUTPUT_PATH = os.path.join(OUTPUT_PATH,"%s"%ARCHITECTURE, "%s"%LOSS, "%s"%ALG, "lrg=%.5e_lrd=%.5e"%(LEARNING_RATE_G,LEARNING_RATE_D))
    if not os.path.exists(os.path.join(OUTPUT_PATH, 'checkpoints')):
        os.makedirs(os.path.join(OUTPUT_PATH, 'checkpoints'))
    if not os.path.exists(os.path.join(OUTPUT_PATH, 'loss')):
        os.makedirs(os.path.join(OUTPUT_PATH, 'loss'))

print("Initializing...")
# Load and transform data
transform=transforms.Compose([transforms.Resize(IMAGE_SIZE),transforms.CenterCrop(IMAGE_SIZE),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
if DATA == "cifar10":
    trainset = dset.CIFAR10(root='./data', train=True, transform=transform, download=True)
    testset = dset.CIFAR10(root='./data', train=False, transform=transform, download=True)    
elif DATA == "mnist":
    trainset = dset.MNIST(root='./data', train=True, transform=transform, download=True)
    testset = dset.MNIST(root='./data', train=False, transform=transform, download=True)
indices = list(range(int(len(trainset)*PORTION)))
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(indices), num_workers=0)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, num_workers=0)

# Decide which device we want to run on
device = torch.device("cuda:%i"%DEVICE if (torch.cuda.is_available() and NUM_GPU> 0) else "cpu")
print(device)

# Define and initialize model
if ARCHITECTURE == "dcgan":
    netG = model.dcgan.Generator(NUM_GPU,nc=NUM_CHANNELS, nfg=NUM_FILTER_G).to(device)
    netD = model.dcgan.Discriminator(NUM_GPU,nc=NUM_CHANNELS, nfd=NUM_FILTER_D).to(device)
    netG.apply(utils.weights_init)
    netD.apply(utils.weights_init)
    print(netG)
    print(netD)

# Define optimizer 
if ALG == "Adam":
    import torch.optim as optim
    optG = optim.Adam(netG.parameters(), lr=LEARNING_RATE_G, betas=(BETA_1, BETA_2))
    optD = optim.Adam(netD.parameters(), lr=LEARNING_RATE_D, betas=(BETA_1, BETA_2))
elif ALG == "AccSGD":
    import optim
    optG = optim.AccSGD(netG.parameters(), lr=LEARNING_RATE_G, kappa = KAPPA, xi = XI)
    optD = optim.AccSGD(netD.parameters(), lr=LEARNING_RATE_D, kappa = KAPPA, xi = XI)
elif ALG == "Adagrad":
    import torch.optim as optim
    optG = optim.Adagrad(netG.parameters(), lr=LEARNING_RATE_G)
    optD = optim.Adagrad(netD.parameters(), lr=LEARNING_RATE_D)
elif ALG == "ExtraSGD":
    import optim
    optG = optim.ExtraSGD(netG.parameters(), lr=LEARNING_RATE_G)
    optD = optim.ExtraSGD(netD.parameters(), lr=LEARNING_RATE_D)

# Initialize Loss function
if LOSS == "gan":
    criterion = nn.BCELoss()
elif LOSS == "lsgan":
    criterion = nn.MSELoss()

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Lists to keep track of progress
img_list = []
lossG = []
lossD = []
iters = 0

print("Starting Training Loop...")
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
        if ALG == "ExtraSGD":
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
        if ALG == "ExtraSGD":
            if (i+1)%2:
                optG.extrapolation()
            else:
                optG.step()
        else:
            optG.step()
        
        # Output training stats
        if (i % 10 == 0) and DISPLAY:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, NUM_EPOCH, i, len(trainloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        lossG.append(errG.item())
        lossD.append(errD.item())
        
        iters += 1
    
    if OUTPUT:    
        #https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch
        torch.save({'args': vars(args), 'state_gen': netG.state_dict()}, os.path.join(OUTPUT_PATH, "checkpoints/gen-%i.state"%epoch))
        torch.save({'args': vars(args), 'state_dis': netD.state_dict()}, os.path.join(OUTPUT_PATH, "checkpoints/dis-%i.state"%epoch))

if OUTPUT:    
    pickle.dump(lossG,open(os.path.join(OUTPUT_PATH, "loss/gen.p"),"wb"))
    pickle.dump(lossD,open(os.path.join(OUTPUT_PATH, "loss/dis.p"),"wb"))
