import torch
import time
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import numpy as np
import argparse
import os
import json
import csv

import models
import utils
import lib

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', choices=('resnet', 'dcgan'), default='dcgan')
parser.add_argument('-alg', '--algorithm', choices=('SGD','ExtraSGD','OMD','Adam','ExtraAdam','OptimisticAdam','UMP'), default='Adam')
parser.add_argument('--cuda', action='store_true')
parser.add_argument('-bs' ,'--batch-size', default=64, type=int)
parser.add_argument('--num-iter', default=500000, type=int)
parser.add_argument('-lrd', '--learning-rate-dis', default=2e-4, type=float)
parser.add_argument('-lrg', '--learning-rate-gen', default=2e-5, type=float)
parser.add_argument('-b1' ,'--beta1', default=0.5, type=float)
parser.add_argument('-b2' ,'--beta2', default=0.9, type=float)
parser.add_argument('-ema', default=0.9999, type=float)
parser.add_argument('-nz' ,'--num-latent', default=128, type=int)
parser.add_argument('-nfd' ,'--num-filters-dis', default=128, type=int)
parser.add_argument('-nfg' ,'--num-filters-gen', default=128, type=int)
parser.add_argument('-gp', '--gradient-penalty', default=10, type=float)
parser.add_argument('-m', '--mode', choices=('gan','ns-gan', 'wgan'), default='wgan')
parser.add_argument('-c', '--clip', default=0.01, type=float)
parser.add_argument('-d', '--distribution', choices=('normal', 'uniform'), default='normal')
parser.add_argument('--batchnorm-dis', action='store_true')
parser.add_argument('--seed', default=1234, type=int)
parser.add_argument('--inception-score', action='store_true')
parser.add_argument('--fid-score', action='store_true')
parser.add_argument('--default', action='store_true')
parser.add_argument('--save', action='store_true')
args = parser.parse_args()

CUDA = args.cuda
ALGORITHM = args.algorithm
MODEL = args.model
MODE = args.mode
GRADIENT_PENALTY = args.gradient_penalty
INCEPTION_SCORE_FLAG = args.inception_score
FID_SCORE_FLAG = args.fid_score
DEFAULT = args.default
SAVE_FLAG = args.save

if DEFAULT:
    try:
        if GRADIENT_PENALTY:
            config = "config/default_%s_%s-gp_%s.json"%(MODEL, MODE, ALGORITHM)
        else:
            config = "config/default_%s_%s_%s.json"%(MODEL, MODE, ALGORITHM)
    except:
        raise ValueError("Not default config available under the current setting")

    with open(config) as f:
        data = json.load(f)
    args = argparse.Namespace(**data) # Will flush the original namespace

BATCH_SIZE = args.batch_size
N_ITER = args.num_iter
LEARNING_RATE_G = args.learning_rate_gen # It is really important to set different learning rates for the discriminator and generator
LEARNING_RATE_D = args.learning_rate_dis
BETA_1 = args.beta1
BETA_2 = args.beta2
BETA_EMA = args.ema
N_LATENT = args.num_latent
N_FILTERS_G = args.num_filters_gen
N_FILTERS_D = args.num_filters_dis
CLIP = args.clip
DISTRIBUTION = args.distribution
BATCH_NORM_G = True
BATCH_NORM_D = args.batchnorm_dis
N_SAMPLES = 50000
N_CHANNEL = 3
START_EPOCH = 0
EVAL_FREQ = 10000
SEED = args.seed
torch.manual_seed(SEED)
np.random.seed(SEED)
n_gen_update = 0
n_dis_update = 0
total_time = 0

OUTPUT_PATH = 'results'
if GRADIENT_PENALTY:
    OUTPUT_PATH = os.path.join(OUTPUT_PATH, '%s/%s-gp'%(MODEL, MODE), '%s/lrd=%.1e_lrg=%.1e/s%i/%i'%(ALGORITHM, LEARNING_RATE_D, LEARNING_RATE_G, SEED, int(time.time())))
else:
    OUTPUT_PATH = os.path.join(OUTPUT_PATH, '%s/%s'%(MODEL, MODE), '%s/lrd=%.1e_lrg=%.1e/s%i/%i'%(ALGORITHM, LEARNING_RATE_D, LEARNING_RATE_G, SEED, int(time.time())))

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, num_workers=0)

print 'Init....'
if not os.path.exists(os.path.join(OUTPUT_PATH, 'checkpoints')):
    os.makedirs(os.path.join(OUTPUT_PATH, 'checkpoints'))
if not os.path.exists(os.path.join(OUTPUT_PATH, 'gen')):
    os.makedirs(os.path.join(OUTPUT_PATH, 'gen'))

if INCEPTION_SCORE_FLAG or FID_SCORE_FLAG:
    if INCEPTION_SCORE_FLAG:
        inception_f = open(os.path.join(OUTPUT_PATH, 'inception_score.csv'), 'ab')
        inception_writter = csv.writer(inception_f)
    if FID_SCORE_FLAG:
        fid_f = open(os.path.join(OUTPUT_PATH, 'fid_score.csv'), 'ab')
        fid_writter = csv.writer(fid_f)

if MODEL == "resnet":
    gen = models.ResNet32Generator(N_LATENT, N_CHANNEL, N_FILTERS_G, BATCH_NORM_G)
    dis = models.ResNet32Discriminator(N_CHANNEL, 1, N_FILTERS_D, BATCH_NORM_D)
elif MODEL == "dcgan":
    gen = models.DCGAN32Generator(N_LATENT, N_CHANNEL, N_FILTERS_G, batchnorm=BATCH_NORM_G)
    dis = models.DCGAN32Discriminator(N_CHANNEL, 1, N_FILTERS_D, batchnorm=BATCH_NORM_D)

if CUDA:
    gen = gen.cuda(0)
    dis = dis.cuda(0)

gen.apply(lambda x: utils.weight_init(x, mode='normal'))
dis.apply(lambda x: utils.weight_init(x, mode='normal'))

if ALGORITHM == 'Adam':
    import torch.optim as optim
    dis_optimizer = optim.Adam(dis.parameters(), lr=LEARNING_RATE_D, betas=(BETA_1, BETA_2))
    gen_optimizer = optim.Adam(gen.parameters(), lr=LEARNING_RATE_G, betas=(BETA_1, BETA_2))
elif ALGORITHM == 'ExtraAdam':
    import optim
    dis_optimizer = optim.ExtraAdam(dis.parameters(), lr=LEARNING_RATE_D, betas=(BETA_1, BETA_2))
    gen_optimizer = optim.ExtraAdam(gen.parameters(), lr=LEARNING_RATE_G, betas=(BETA_1, BETA_2))
elif ALGORITHM == 'OptimisticAdam':
    import optim
    dis_optimizer = optim.OptimisticAdam(dis.parameters(), lr=LEARNING_RATE_D, betas=(BETA_1, BETA_2))
    gen_optimizer = optim.OptimisticAdam(gen.parameters(), lr=LEARNING_RATE_G, betas=(BETA_1, BETA_2))
elif ALGORITHM =='SGD':
    import torch.optim as optim
    dis_optimizer = optim.SGD(dis.parameters(), lr=LEARNING_RATE_D)
    gen_optimizer = optim.SGD(gen.parameters(), lr=LEARNING_RATE_G)
elif ALGORITHM == 'ExtraSGD':
    import optim
    dis_optimizer = optim.ExtraSGD(dis.parameters(), lr=LEARNING_RATE_D)
    gen_optimizer = optim.ExtraSGD(gen.parameters(), lr=LEARNING_RATE_G)
elif ALGORITHM == 'OMD':
    import optim
    dis_optimizer = optim.OMD(dis.parameters(), lr=LEARNING_RATE_D)
    gen_optimizer = optim.OMD(gen.parameters(), lr=LEARNING_RATE_G)
elif ALGORITHM == 'UMP':
    import optim
    dis_optimizer = optim.UMP(dis.parameters())
    gen_optimizer = optim.UMP(gen.parameters())

with open(os.path.join(OUTPUT_PATH, 'config.json'), 'wb') as f:
    json.dump(vars(args), f)

dataiter = iter(testloader)
examples, labels = dataiter.next()
torchvision.utils.save_image(utils.unormalize(examples), os.path.join(OUTPUT_PATH, 'examples.png'), 10)

z_examples = utils.sample(DISTRIBUTION, (100, N_LATENT))
if CUDA:
    z_examples = z_examples.cuda(0)

gen_param_avg = []
gen_param_ema = []
for param in gen.parameters():
    gen_param_avg.append(param.data.clone())
    gen_param_ema.append(param.data.clone())

f = open(os.path.join(OUTPUT_PATH, 'results.csv'), 'ab')
f_writter = csv.writer(f)

print 'Training...'
n_iteration_t = 0
gen_inception_score = 0
while n_gen_update < N_ITER:   
    t = time.time()
    avg_loss_G = 0
    avg_loss_D = 0
    avg_penalty = 0
    num_samples = 0
    penalty = Variable(torch.Tensor([0.]))
    if CUDA:
        penalty = penalty.cuda(0)
    for i, data in enumerate(trainloader):
        _t = time.time()
        x_true, _ = data
        x_true = Variable(x_true)

        z = Variable(utils.sample(DISTRIBUTION, (len(x_true), N_LATENT)))
        if CUDA:
            x_true = x_true.cuda(0)
            z = z.cuda(0)

        x_gen = gen(z)
        p_true, p_gen = dis(x_true), dis(x_gen)

        gen_loss = utils.compute_gan_loss(p_true, p_gen, mode=MODE)
        dis_loss = - gen_loss.clone()
        if GRADIENT_PENALTY:
            penalty = dis.get_penalty(x_true.data, x_gen.data)
            dis_loss += GRADIENT_PENALTY*penalty

        for p in gen.parameters():
            p.requires_grad = False
            
        dis_optimizer.zero_grad()
        # https://github.com/pytorch/examples/issues/116
        dis_loss.backward(retain_graph=True)
        
        if ALGORITHM == 'ExtraAdam' or ALGORITHM == 'ExtraSGD':
            if (n_iteration_t+1)%2 != 0:
                dis_optimizer.extrapolation()
            else:
                dis_optimizer.step()
                n_dis_update += 1
        elif ALGORITHM == 'UMP':
            if (n_iteration_t+1)%2 != 0:
                dis_optimizer.extrapolation()
                n_dis_update += 1
            else:
                dis_optimizer.step()
        else:
            dis_optimizer.step()
            n_dis_update += 1
            
        if MODE =='wgan' and not GRADIENT_PENALTY:
            for p in dis.parameters():
                p.data.clamp_(-CLIP, CLIP)
                
        for p in gen.parameters():
            p.requires_grad = True
        
        for p in dis.parameters():
            p.requires_grad = False
            
        gen_optimizer.zero_grad()
        gen_loss.backward()
        if ALGORITHM == 'ExtraAdam' or ALGORITHM == 'ExtraSGD':
            if (n_iteration_t+1)%2 != 0:
                gen_optimizer.extrapolation()
            else:
                gen_optimizer.step()
                n_gen_update += 1
                for j, param in enumerate(gen.parameters()):
                    gen_param_avg[j] = gen_param_avg[j]*n_gen_update/(n_gen_update+1.) + param.data.clone()/(n_gen_update+1.)
                    gen_param_ema[j] = gen_param_ema[j]*BETA_EMA+ param.data.clone()*(1-BETA_EMA)
        elif ALGORITHM == 'UMP':
            if (n_iteration_t+1)%2 != 0:
                gen_optimizer.extrapolation()
                n_gen_update += 1
                for j, param in enumerate(gen.parameters()):
                    gen_param_avg[j] = gen_param_avg[j]*n_gen_update/(n_gen_update+1.) + param.data.clone()/(n_gen_update+1.)
                    gen_param_ema[j] = gen_param_ema[j]*BETA_EMA+ param.data.clone()*(1-BETA_EMA)
            else:
                gen_optimizer.step()
        else:
            gen_optimizer.step()
            n_gen_update += 1
            for j, param in enumerate(gen.parameters()):
                gen_param_avg[j] = gen_param_avg[j]*n_gen_update/(n_gen_update+1.) + param.data.clone()/(n_gen_update+1.)
                gen_param_ema[j] = gen_param_ema[j]*BETA_EMA+ param.data.clone()*(1-BETA_EMA)
                
        for p in dis.parameters():
            p.requires_grad = True

        total_time += time.time() - _t

        avg_loss_D += dis_loss.item()*len(x_true)
        avg_loss_G += gen_loss.item()*len(x_true)
        avg_penalty += penalty.item()*len(x_true)
        num_samples += len(x_true)

        if n_gen_update%EVAL_FREQ == 0:
            if INCEPTION_SCORE_FLAG or FID_SCORE_FLAG:
                fake_imgs = gen(z_examples).detach()#.cpu()
            if INCEPTION_SCORE_FLAG:
                inception_score = lib.get_inception_score(fake_imgs,cuda=CUDA)
                inception_writter.writerow(inception_score)
                inception_f.flush()
            if FID_SCORE_FLAG:
                fid_score = lib.get_fid_score(fake_imgs,cuda=CUDA)
                fid_writter.writerow([fid_score])
                fid_f.flush()
            
            if SAVE_FLAG:    
                #https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-models-in-pytorch
                torch.save({'args': vars(args), 'n_gen_update': n_gen_update, 'total_time': total_time, 'state_gen': gen.state_dict(), 'gen_param_avg': gen_param_avg, 'gen_param_ema': gen_param_ema}, os.path.join(OUTPUT_PATH, "checkpoints/%i.state"%n_gen_update))
                torch.save({'args': vars(args), 'state_dis': dis.state_dict()}, os.path.join(OUTPUT_PATH, 'checkpoints/dis-%i.state'%n_dis_update))

        n_iteration_t += 1
        
    avg_loss_G /= num_samples
    avg_loss_D /= num_samples
    avg_penalty /= num_samples

    print('Iter: %i, Loss Generator: %.4f, Loss Discriminator: %.4f, Penalty: %.2e, Time: %.4f'%(n_gen_update, avg_loss_G, avg_loss_D, avg_penalty, time.time() - t))

    f_writter.writerow((n_gen_update, avg_loss_G, avg_loss_D, avg_penalty, time.time() - t))
    f.flush()

    x_gen = gen(z_examples)
    x = utils.unormalize(x_gen)
    torchvision.utils.save_image(x.data, os.path.join(OUTPUT_PATH, 'gen/%i.png' % n_gen_update), 10)
