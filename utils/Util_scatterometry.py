#!/usr/bin/env python

import numpy as np
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# returns (x,y) pairs in a dataloader fashion, where y is given by the forward model
def get_epoch_data_loader(batch_size, forward_model,a, b,lambd_bd):
    x = torch.tensor(inverse_cdf_prior(np.random.uniform(size=(8*batch_size,3)),lambd_bd),dtype=torch.float,device=device)
    y = forward_model(x)
    y += torch.randn_like(y) * b + torch.randn_like(y)*a*y
    def epoch_data_loader():
        for i in range(0, 8*batch_size, batch_size):
            yield x[i:i+batch_size].clone(), y[i:i+batch_size].clone()

    return epoch_data_loader


# returns (negative) log_posterior evaluation for the scatterometry model
# likelihood is determined by the error model
# uniform prior is approximated via boundary loss for a.e. differentiability
def get_log_posterior(samples, forward_model, a, b, ys,lambd_bd):
    relu=torch.nn.ReLU()
    forward_samps=forward_model(samples)
    prefactor = ((a*forward_samps)**2+b**2)
    p = .5*torch.sum(torch.log(prefactor), dim = 1)
    p2 =  0.5*torch.sum((ys-forward_samps)**2/prefactor, dim = 1)
    p3 = lambd_bd*torch.sum(relu(samples-1)+relu(-1-samples), dim = 1)
    return p+p2+p3


# returns samples from the boundary loss approximation prior
# lambd_bd controlling the strength of boundary loss
def inverse_cdf_prior(x,lambd_bd):
    x*=(2*lambd_bd+2)/lambd_bd
    y=np.zeros_like(x)
    left=x<1/lambd_bd
    y[left]=np.log(x[left]*lambd_bd)-1
    middle=np.logical_and(x>=1/lambd_bd,x < 2+1/lambd_bd)
    y[middle]=x[middle]-1/lambd_bd-1
    right=x>=2+1/lambd_bd
    y[right]=-np.log(((2+2/lambd_bd)-x[right])*lambd_bd)+1
    return y
