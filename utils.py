#  MIT License

# Copyright (c) Facebook, Inc. and its affiliates.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# written by Hugo Berard (berard.hugo@gmail.com) while at Facebook.

import torch
import torch.nn as nn

def clip_weights(params, clip=0.01):
    for p in params:
        p.clamp_(-clip, clip)

def unormalize(x):
    return x/2. + 0.5

def sample(name, size):
    if name == 'normal':
        return torch.zeros(size).normal_()
    elif name == 'uniform':
        return torch.zeros(size).uniform_()
    else:
        raise ValueError()

def weight_init(m, mode='normal'):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        if mode == 'normal':
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            nn.init.constant_(m.bias.data, 0.)
        elif mode == 'kaimingu':
            nn.init.kaiming_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.)
        elif mode == 'orthogonal':
            nn.init.orthogonal_(m.weight.data, 0.8)

def compute_gan_loss(p_true, p_gen, mode='gan', gen_flag=False):
    if mode == 'ns-gan' and gen_flag:
        loss = (p_true.clamp(max=0) - torch.log(1+torch.exp(-p_true.abs()))).mean() - (p_gen.clamp(max=0) - torch.log(1+torch.exp(-p_gen.abs()))).mean()
    elif mode == 'gan' or mode == 'gan++':
        loss = (p_true.clamp(max=0) - torch.log(1+torch.exp(-p_true.abs()))).mean() - (p_gen.clamp(min=0) + torch.log(1+torch.exp(-p_gen.abs()))).mean()
    elif mode == 'wgan':
        loss = p_true.mean() - p_gen.mean()
    else:
        raise NotImplementedError()

    return loss
