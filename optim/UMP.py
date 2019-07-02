#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:05:00 2019

@author: neyozhyang
"""

import torch
from torch.optim import Optimizer


class UMP(Optimizer):
    """Implement Universal Mirror Prox Algorithm by Francis Bach and Kfir Y. Levy

    It has been proposed in `A Universal Algorithm for Variational Inequalities Adaptive to Smoothness and Noise`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lr_decay (float, optional): learning rate decay (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    .. _Adaptive Subgradient Methods for Online Learning and Stochastic
        Optimization: http://jmlr.org/papers/v12/duchi11a.html
    """

    def __init__(self, params, lr=1e-2, lr_decay=0, weight_decay=0, initial_accumulator_value=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= lr_decay:
            raise ValueError("Invalid lr_decay value: {}".format(lr_decay))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= initial_accumulator_value:
            raise ValueError("Invalid initial_accumulator_value value: {}".format(initial_accumulator_value))

        defaults = dict(lr=lr, lr_decay=lr_decay, weight_decay=weight_decay,
                        initial_accumulator_value=initial_accumulator_value)
        super(UMP, self).__init__(params, defaults)
        self.params_copy = []
        self.grads_copy = []
        
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['sum'] = torch.full_like(p.data, initial_accumulator_value) # this returns a tensor

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['sum'].share_memory_()
        
    def extrapolation(self):
        """Performs the extrapolation step and save a copy of the current parameters for the update step.
        """
        # Check if a copy of the parameters was already made.
        is_params_empty = len(self.params_copy) == 0
        is_grads_empty = len(self.grads_copy) == 0
        
        for group in self.param_groups:
            for p in group['params']:
                
                if is_params_empty:
                    # Save the current parameters for the update step. 
                    # Several extrapolation step can be made before each update but only the parameters before the first extrapolation step are saved.
                    self.params_copy.append(p.data.clone())
                    
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                
                if is_grads_empty:
                    self.grads_copy.append(p.grad.data.clone())
                
                state = self.state[p]
                state['step'] += 1
                
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)
                
                clr = group['lr'] / (1 + (state['step'] - 1) * group['lr_decay'])
                
                std = state['sum'].sqrt().add_(1e-10)
                
                # Update the current parameters
                p.data.addcmul_(-clr, grad, std)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if len(self.params_copy) == 0 or len(self.grads_copy) == 0:
            raise RuntimeError('Need to call extrapolation before calling step.')

        loss = None
        if closure is not None:
            loss = closure()
        
        i = -1
        for group in self.param_groups:
            for p in group['params']:
                
                i += 1
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                state = self.state[p] # state of outside p
                
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)
                
                clr = group['lr'] / (1 + (state['step'] - 1) * group['lr_decay'])
                
                std = state['sum'].sqrt().add_(1e-10)
                
                # Update the parameters saved during the extrapolation step
                p.data = self.params_copy[i].addcmul_(-clr, grad, std)
                
                state['sum'].addcmul_(1 / 5, self.grads_copy[i] - grad, self.grads_copy[i] - grad)
                state['sum'].addcmul_(1 / 5, self.grads_copy[i], self.grads_copy[i])
                
                
        # Free the old parameters
        self.params_copy = []
        self.grads_copy = []
        return loss
