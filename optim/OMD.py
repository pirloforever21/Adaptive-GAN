#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:07:29 2019

@author: yangzhenhuan
"""

import torch
from torch.optim import Optimizer

required = object()

class OMD(Optimizer):
    def __init__(self, params, lr=required):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)
        super(OMD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(OMD, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['previous_update'] = torch.zeros_like(d_p)

                p.data.add_(-2*group['lr'], d_p).add_(group['lr']*state['previous_update'])

                state['previous_update'] = d_p

        return loss