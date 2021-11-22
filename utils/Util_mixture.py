#!/usr/bin/env python

import numpy as np
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# draw num_samples samples from the distributions given by the mixture_params
# returns those samples
def draw_mixture_dist(mixture_params, num_samples):
    n = len(mixture_params)
    sigmas=torch.stack([torch.sqrt(sigma) for w,mu,sigma in mixture_params])
    probs=np.array([w for w, mu, sigma in mixture_params])
    zs = np.random.choice(n, (num_samples,), p=probs/probs.sum())
    mus = torch.stack([mu for w, mu, sigma in mixture_params])[zs]
    sigmas_samples = sigmas[zs]
    multinomial_samples = torch.randn(num_samples, mus.shape[1], device=device)
    if len(sigmas_samples.shape)==1:
        sigmas_samples=sigmas_samples.unsqueeze(-1)
    out_samples = mus + multinomial_samples*sigmas_samples
    return out_samples


# gets mean and covariance of the Gaussian posterior with linear forward model
# mean, sigma are the parameters of the prior distribution
def get_single_gaussian_posterior(mean, sigma, forward_mat, b_sq, y):
    ATA = forward_mat**2/b_sq
    cov_gauss = 1/(ATA+1/sigma)

    mean_gauss = cov_gauss*forward_mat*y/b_sq+cov_gauss*mean/sigma
    return mean_gauss, cov_gauss

# returns the mixture parameters of the posterior given the mixture parameters of the
# prior, the forward model and the likelihood (for a specific y)
def get_mixture_posterior(x_gauss_mixture_params, forward_mat, b_sq, y):
    out_mixtures = []
    nenner = 0
    ws=torch.zeros(len(x_gauss_mixture_params),device=device)
    mus_new=[]
    sigmas_new=[]
    log_zaehler=torch.zeros(len(x_gauss_mixture_params),device=device)
    for k,(w, mu, sigma) in enumerate(x_gauss_mixture_params):
        mu_new, sigma_new = get_single_gaussian_posterior(mu, sigma, forward_mat, b_sq, y)
        mus_new.append(mu_new)
        sigmas_new.append(sigma_new)
        ws[k]=w
        log_zaehler[k]=torch.log(torch.tensor(w,device=device,dtype=torch.float))+(0.5*torch.sum(mu_new**2/sigma_new)-0.5*torch.sum(mu**2)/sigma)
    const=torch.max(log_zaehler)
    log_nenner=torch.log(torch.sum(torch.exp(log_zaehler-const)))+const
    for k in range(len(x_gauss_mixture_params)):
        out_mixtures.append((torch.exp(log_zaehler[k]-log_nenner).detach().cpu().numpy(),mus_new[k],sigmas_new[k]))
    return out_mixtures

# creates forward map
# scale controls how illposed the problem is

def create_forward_model(scale,dimension):
    s = torch.ones(dimension, device = device)
    for i in range(dimension):
        s[i] = scale/(i+1)
    return s

# evaluates forward_map
def forward_pass(x, forward_map):
    return x*forward_map


# gets the log of the prior of some samples given its mixture parameters

def get_prior_log_likelihood(samples, mixture_params):
    exponents = torch.zeros((samples.shape[0], len(mixture_params)), device=device)
    dimension=samples.shape[1]
    for k, (w, mu, sigma) in enumerate(mixture_params):
        log_gauss_prefactor = (-dimension / 2)* (np.log(2 * np.pi)   + torch.log(sigma))
        tmp = -0.5 * torch.sum((samples - mu[None, :])**2, dim=1)/sigma
        exponents[:, k] = tmp + np.log(w) + log_gauss_prefactor

    max_exponent = torch.max(exponents, dim=1)[0].detach()
    exponents_=exponents-max_exponent.unsqueeze(-1)
    exp_sum=torch.log(torch.sum(torch.exp(exponents_),dim=1))+max_exponent
    return exp_sum
     
#returns the (negative) log posterior given a y, the mixture params of the prior, the likelihood model b and y
def get_log_posterior(samples, forward_map, mixture_params, b, y):
    p = -get_prior_log_likelihood(samples, mixture_params)
    p2 = 0.5 * torch.sum((y-forward_pass(samples, forward_map))**2 * (1/b**2), dim=1)
    return (p+p2).view(len(samples))
    


# creates a data loader returning (x,y) pairs of the joint distribution
def get_epoch_data_loader(mixture_params, num_samples_per_epoch, batch_size, forward_map, b):
    x = draw_mixture_dist(mixture_params, num_samples_per_epoch)
    y = forward_pass(x, forward_map)
    y += torch.randn_like(y) * b
    def epoch_data_loader():
        for i in range(0, num_samples_per_epoch, batch_size):
            yield x[i:i+batch_size].clone(), y[i:i+batch_size].clone()

    return epoch_data_loader







