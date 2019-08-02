#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:20:02 2019

@author: neyozhyang

Layer-wise adaptive rate scaling for SGD in PyTorch
"""

import math

import torch
from torch.optim.optimizer import Optimizer, required

class LARS(Optimizer):
    r"""Implements LARS algorithm.
    It has been proposed again in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-6)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0.1)
    .. _Large Batch Optimization for Deep Learning: Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962
    """

    def __init__(self, params, lr=required, momemtum=0.99,
                 weight_decay=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value".format(momentum))
    
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        
        super(LARS, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
 
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)

                exp_avg = state['exp_avg']

                state['step'] += 1
                
                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first moment running average coefficient
                # m_t
                exp_avg.mul_(group['momentum']).add_(1 - group['momentum'], grad)

                # Paper v3 does not use debiasing.
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']
                # Apply bias to lr to avoid broadcast.
                step_size = group['lr'] # * math.sqrt(bias_correction2) / bias_correction1

                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)

                exp_avg_norm = exp_avg.pow(2).sum().sqrt()
                
                trust_ratio = weight_norm / exp_avg_norm

                p.data.add_(-step_size * trust_ratio, exp_avg)

        return loss
