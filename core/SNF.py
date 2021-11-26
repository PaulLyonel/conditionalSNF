#!/usr/bin/env python

import numpy as np
import torch
from tqdm import tqdm
from torch import nn
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet, ConditionNode
from FrEIA.modules import GLOWCouplingBlock



device = 'cuda' if torch.cuda.is_available() else 'cpu'

# creates conditional stochastic normalizing flow model
#
# num_layers: number of layers needed
# sub_net_size: number of hidden neurons in the fc subnetworks
# log_posterior: method that evalutes the log posterior
# metr_steps_per_block: how many steps of the MH algorithm are done in every stoch MCMC layer
# dimension_condition: dimension of y
# dimension: dimension of x
# noise_std: standard deviation of gaussian noise in MH steps
# num_inn_layers: how many INN layers per determinstic block
# lang_steps: number of Langevin steps
# lang_steps_prop: number of proposal langevin steps (in MALA layers)
# step_size: step size in Langevin steps
# langevin_prop: use langevin as proposal (use MALA layer)
#
# returns an snf object

def create_snf(num_layers, sub_net_size,log_posterior,metr_steps_per_block=3,dimension_condition=5,dimension=5,noise_std=0.4,num_inn_layers=1,
                 lang_steps = 0,lang_steps_prop=1, step_size = 5e-3, langevin_prop = False):
    snf=SNF()
    for k in range(num_layers):
        lambd = (k+1)/(num_layers)
        snf.add_layer(deterministic_layer(num_inn_layers,sub_net_size,dimension_condition=dimension_condition,dimension=dimension))
        if metr_steps_per_block>0:
            if lang_steps>0:
                snf.add_layer(Langevin_layer(log_posterior,lambd,lang_steps,step_size))
            if langevin_prop:
                snf.add_layer(MALA_layer(log_posterior,lambd,metr_steps_per_block,lang_steps_prop,step_size))
            else:
                snf.add_layer(MCMC_layer(log_posterior,lambd,noise_std,metr_steps_per_block))

    return snf
# defines a fully connected subnetwork with input size c_in, output size c_out and hidden sizes sub_net_size
def subnet_fc(c_in, c_out,sub_net_size):
    return nn.Sequential(nn.Linear(c_in, sub_net_size), nn.ReLU(),
                         nn.Linear(sub_net_size, sub_net_size), nn.ReLU(),
                         nn.Linear(sub_net_size,  c_out))


# SNF class inheriting from nn.Module
class SNF(nn.Module):
    
    # initialize SNF
    def __init__(self,layers=[]):
        super(SNF, self).__init__()
        self.layers=layers
        self.param_counter=0
        
    # adds layer and registers parameters
    def add_layer(self,layer):
        self.layers.append(layer)
        for param in layer.parameters():
            self.register_parameter(name='parameter'+str(self.param_counter),param=param)
            self.param_counter+=1
            
    # performs a forward step of SNF with input zs and conditions ys
    # returns final vector and logdet term
    def forward(self,zs,ys):
        logdet = torch.zeros(len(zs), device = device)
        for k in range(len(self.layers)):
            out = self.layers[k].forward(zs, ys)
            zs = out[0]
            logdet += out[1]
        return (zs,logdet)
    
    # performs a forward step and returns the path (also the interpolating samples)
    def forward_all(self,zs,ys):
        outs = []
        outs.append(zs)
        for k in range(len(self.layers)):
            out = self.layers[k].forward(zs, ys)
            outs.append(out[0])
            zs = out[0]
        return (outs)

    # performs a backward step of the SNF with input zs and conditions ys
    # returns output and logdet term
    def backward(self,zs, ys):
        logdet = torch.zeros(len(zs), device = device)
        for k in range(len(self.layers)-1,-1, -1):
            out = self.layers[k].backward(zs, ys)
            zs = out[0]
            logdet += out[1]
        return (zs,logdet)

# class implementing deterministic layer based on Freia package
# inherits from nn.Module
# num_inn_layers: number of INN layers
# sub_net_size: number of hidden neurons in subnetworks
# dimension_condition: dimension of y
# dimension: dimension of x-space
#
# returns layer with forward function

class deterministic_layer(nn.Module):
    # initializes this layer
    def __init__(self,num_inn_layers,sub_net_size,dimension_condition=5,dimension=5):
        super(deterministic_layer, self).__init__()
        k=1
        nodes = [InputNode(dimension, name=F'input_{k}')]
        cond = ConditionNode(dimension_condition, name=F'condition_{k}')
        for i in range(num_inn_layers):
            nodes.append(Node(nodes[-1],
                              GLOWCouplingBlock,
                              {'subnet_constructor':lambda c_in,c_out: subnet_fc(c_in,c_out,sub_net_size), 'clamp':1.4},
                              conditions = cond,
                              name=F'coupling_{k}'+str(i)))

        nodes.append(OutputNode(nodes[-1], name=F'output_{k}'))
        self.model=ReversibleGraphNet(nodes + [cond], verbose=False).to(device)
        for param in self.model.parameters():
            self.register_parameter(name='parameter',param=param)

    # defines the forward propagation of input xs, with condition ys
    # returns output and logdet
    def forward(self,xs,ys):
        return self.model(xs,c=ys)
    
    # defines the backward propagation of input xs, with condition ys
    # returns output and logdet
    def backward(self,xs,ys):
        return self.model(xs,c=ys,rev=True)
    
# implements MCMC layer
#
# log_posterior: function which returns the log_posterior
# lambd: controls the interpolation schedule
# noise_std: the standard deviation of gaussian noise in the proposal
# metr_steps_per_block: number of metropolis hastings steps per block
class MCMC_layer(nn.Module):
    def __init__(self,log_posterior,lambd,noise_std,metr_steps_per_block):
        super(MCMC_layer,self).__init__()
        self.noise_std=noise_std
        self.metr_steps_per_block=metr_steps_per_block
        self.get_log_posterior=log_posterior
        self.lambd=lambd
    def forward(self,xs,ys):
        zs,e=anneal_to_energy(xs, get_interpolated_energy_fun(ys, self.lambd,self.get_log_posterior),self.metr_steps_per_block,noise_std=self.noise_std)
        e=e.view(len(e))
        return zs,e
    def backward(self,xs,ys):
        return self.forward(xs,ys)

# implements MALA layer (i.e. MCMC with langevin proposal)
#
# log_posterior: function which returns the log_posterior
# lambd: controls the interpolation schedule
# noise_std: the standard deviation of gaussian noise in the proposal
# metr_steps_per_block: number of metropolis hastings steps per block
# lang_steps: number of langevin proposal steps
# step_size: step size of langevin dynamics
class MALA_layer(nn.Module):
    def __init__(self,log_posterior,lambd,metr_steps_per_block,lang_steps,stepsize):
        super(MALA_layer,self).__init__()
        self.metr_steps_per_block=metr_steps_per_block
        self.get_log_posterior=log_posterior
        self.lambd=lambd
        self.lang_steps=lang_steps
        self.stepsize=stepsize
    def forward(self,xs,ys):
        zs,e=anneal_to_energy(xs, get_interpolated_energy_fun(ys, self.lambd,self.get_log_posterior),
                              self.metr_steps_per_block,langevin_prop=True,lang_steps=self.lang_steps,stepsize=self.stepsize/self.lambd)
        e=e.view(len(e))
        return zs,e
    def backward(self,xs,ys):
        return self.forward(xs,ys)

#implements a Langevin layer
#
# log_posterior: function which returns the log_posterior
# lambd: controls the interpolation schedule
# lang_steps: number of langevin proposal steps
# step_size: step size of langevin dynamics
class Langevin_layer(nn.Module):
    def __init__(self,log_posterior,lambd,lang_steps,stepsize):
        super(Langevin_layer,self).__init__()
        self.get_log_posterior=log_posterior
        self.lambd=lambd
        self.lang_steps=lang_steps
        self.stepsize=stepsize
    def forward(self,xs,ys):
        zs,log_det,_,_=langevin_step(xs,self.stepsize,get_interpolated_energy_fun(ys, self.lambd,self.get_log_posterior),self.lang_steps)
        return zs,log_det
    def backward(self,xs,ys):
        return self.forward(xs,ys)

# returns the interpolated energy given the condition, the lambda and the log_posterior
# i.e. this realizes the intermediate densities to which we anneal within mcmc and langevin
def get_interpolated_energy_fun(ys,lambd,get_log_posterior):
    if lambd==0.:
        def energy(x):
            return 0.5*torch.sum(x**2, dim = 1)
        return energy
    if lambd==1.:
        def energy(x):
            return get_log_posterior(x, ys)
        return energy
    def energy(x):
        return lambd*(get_log_posterior(x, ys)).view(len(x))+(1-lambd)*0.5*torch.sum(x**2, dim = 1)
    return energy

# calculates and returns the grad of the energy ( i.e. log posterior) as needed in the langevin steps
def energy_grad(x, energy):
    x = x.requires_grad_(True)
    e = energy(x)
    return torch.autograd.grad(e.sum(), x,retain_graph=True)[0],e

# anneals to energy
#
# x_curr: current starting x
# energy: energy to which we anneal to
# noise_std: standard deviation of Gaussian noise proposal
# langevin_prop: use langevin proposal
# lang_steps: number of langevin steps
# step_size: step size within Langevin
#
# returns the annealed points and "logdet"

def anneal_to_energy(x_curr, energy,metr_steps_per_block,noise_std=0.1, langevin_prop=False, lang_steps=None, stepsize=None):
    e0 = energy(x_curr)

    for i in range(metr_steps_per_block):
        if langevin_prop == True:
            x_prop, log_det,e_curr,e_prop = langevin_step(x_curr, stepsize, energy, lang_steps)
            e_diff = torch.exp(-e_prop+e_curr+log_det)
        else:
            noise = noise_std * torch.randn_like(x_curr, device = device)
            x_prop = x_curr + noise
            e_prop=energy(x_prop)
            e_curr=energy(x_curr)

            e_diff = torch.exp(-e_prop+e_curr)
     
        r = torch.rand_like(e_diff, device = device)
        acc = (r < e_diff).float().view(len(x_prop),1)
        rej = 1. - acc
        rej = rej.view(len(rej),1)
        x_curr = rej * x_curr + acc * x_prop
    if langevin_prop == True:
        e = rej * e_curr.view(len(e_curr),1) + acc * e_prop.view(len(e_prop),1)
    else:
        e = rej * e_curr.view(len(e_curr),1) + acc * e_prop.view(len(e_prop),1)

    return (x_curr, e.view(len(e))-e0.view(len(e)))

# realizes the langevin steps
#
# x: current points
# step_size: step_size for langevin
# energy: energy to anneal to
# lang_steps: number of langevin steps
#
# returns a tuple of annealed points, "logdet", former and final gradients of energy

def langevin_step(x, stepsize, energy, lang_steps):
        log_det = torch.zeros((x.shape[0], 1), device = device)
        beta=1.
        for i in range(lang_steps):
            eta = torch.randn_like(x, device = device)
            grad_x,e_x=energy_grad(x,energy)
            if i==0:
                energy_x=e_x
            y = x - stepsize * grad_x + np.sqrt(2*stepsize/beta) * eta
            grad_y,energy_y=energy_grad(y, energy)

            eta_ = (x - y + stepsize* grad_y) / np.sqrt(2*stepsize/beta)
            log_det += 0.5 * (eta**2 - eta_**2).sum(axis=1, keepdims=True)
            x = y
        return (x, log_det.view(len(x)),energy_x,energy_y)

# realizes an "epoch" of training of SNF
#
# optimizer: the optimizer for the snf
# snf: an object from the snf class
# epoch_data_loader: data loader containing (x,y) training data
# forward_model: pretrained forward_model (needed for forward KL loss)
# a, b: error model parameters
# get_prior_log_likelihood: method giving log prior
# convex_comb_factor: weighting between forward and backward KL

def train_SNF_epoch(optimizer, snf, epoch_data_loader,forward_model,a,b,get_prior_log_likelihood):
    mean_loss = 0
    for k, (x, y) in enumerate(epoch_data_loader()):
        cur_batch_size = len(x)
        loss = 0
        out = snf.backward(x, y)
        invs = out[0]
        jac_inv = out[1]

        l5 = 0.5* torch.sum(invs**2, dim=1) - jac_inv
        loss += torch.sum(l5) / cur_batch_size 
        mean_loss = mean_loss * k / (k + 1) + loss.data.item() / (k + 1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return mean_loss




