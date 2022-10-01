# This code belongs to the paper
#
# P. Hagemann, J. Hertrich, G. Steidl (2022).
# Stochastic Normalizing Flows for Inverse Problems: a Markov Chains Viewpoint
# SIAM/ASA Journal on Uncertainty Quantification, vol. 10 (3), pp. 1162-1190.
#
# Please cite the paper, if you use this code.
# This script reproduces the numerical example from Section 7.2 of the paper.
#
from torch.optim import Adam
import torch
import numpy as np
from tqdm import tqdm
import scipy
import time
import os

from utils.Util_scatterometry import *
from utils.histogram_plot import make_image
from core.SNF import *
from core.INN import *

# define parameters
num_epochs_SNF = 40
batch_size = 1600
DIMENSION = 3

num_epochs_INN = 5000

# mcmc parameters for "discovering" the ground truth
NOISE_STD_MCMC = 0.5
METR_STEPS = 1000
lambd_bd = 1000
a = 0.2
b = 0.01
relu = torch.nn.ReLU()
retrain = False
# regularization parameter for KL calculation
reg=1e-10



# load forward model
forward_model = nn.Sequential(nn.Linear(3, 256), nn.ReLU(),
                              nn.Linear(256, 256), nn.ReLU(),
                              nn.Linear(256, 256), nn.ReLU(),
                              nn.Linear(256,  23)).to(device)

forward_model.load_state_dict(torch.load('models_scatterometry/forward_model_new.pt'))
for param in forward_model.parameters():
    param.requires_grad=False

def train_and_eval(a, b, testing_ys, forward_model):
    # define networks
    log_posterior=lambda samples, ys:get_log_posterior(samples,forward_model,a,b,ys,lambd_bd)
    snf = create_snf(4,64,log_posterior,metr_steps_per_block=10,dimension=3,dimension_condition=23,noise_std=0.4)
    INN = create_INN(4,64,dimension=3,dimension_condition=23)

    if retrain:
        # training
        optimizer = Adam(snf.parameters(), lr = 1e-3)

        prog_bar = tqdm(total=num_epochs_SNF)
        for i in range(num_epochs_SNF):
            data_loader=get_epoch_data_loader(batch_size,forward_model,a,b,lambd_bd)
            loss = train_SNF_epoch(optimizer, snf, data_loader,forward_model, a, b,None)
            prog_bar.set_description('SNF loss:{:.3f}'.format(loss))
            prog_bar.update()
        prog_bar.close()

        optimizer_inn = Adam(INN.parameters(), lr = 1e-3)


        prog_bar = tqdm(total=num_epochs_INN)
        for i in range(num_epochs_INN):
            data_loader=get_epoch_data_loader(batch_size,forward_model,a,b,lambd_bd)
            loss = train_inn_epoch(optimizer_inn, INN, data_loader)
            prog_bar.set_description('determ INN loss:{:.3f}'.format(loss))
            prog_bar.update()
        prog_bar.close()
        if not os.path.isdir('models_scatterometry'):
            os.mkdir('models_scatterometry')

        torch.save(snf.state_dict(),'models_scatterometry/snf.pt')
        torch.save(INN.state_dict(),'models_scatterometry/inn.pt')

    else:
        snf.load_state_dict(torch.load('models_scatterometry/snf.pt'))
        INN.load_state_dict(torch.load('models_scatterometry/inn.pt'))

    # parameters for testing
    testing_x_per_y = int(16000*(1.5)**3)

    testing_num_y = len(testing_ys)
    weights1, weights2 = np.ones((testing_x_per_y,)) / testing_x_per_y, np.ones((testing_x_per_y,)) / testing_x_per_y

    kl1_sum = 0.
    kl2_sum = 0.
    kl1_vals=[]
    kl2_vals=[]
    nbins=75
    repeats=10
    tic=time.time()
    for i, y in enumerate(testing_ys):
        # testing
        hist_mcmc_sum=np.zeros((nbins,nbins,nbins))
        hist_snf_sum=np.zeros((nbins,nbins,nbins))
        hist_inn_sum=np.zeros((nbins,nbins,nbins))
        for asdf in range(repeats):
            # run methods for posterior samplings
            inflated_ys = y[None, :].repeat(testing_x_per_y, 1)
            mcmc_energy=lambda x:get_log_posterior(x,forward_model,a,b,inflated_ys,lambd_bd)
            true_posterior_samples = anneal_to_energy(torch.rand(testing_x_per_y,3, device = device)*2-1,mcmc_energy,METR_STEPS,noise_std=NOISE_STD_MCMC )[0].detach().cpu().numpy()
            samples1 = snf.forward(torch.randn(testing_x_per_y, DIMENSION, device=device), inflated_ys)[0].detach().cpu().numpy()
            samples2 = INN(torch.randn(testing_x_per_y, DIMENSION, device=device), c = inflated_ys)[0].detach().cpu().numpy()
            # generate histograms
            hist_mcmc,_ = np.histogramdd(true_posterior_samples, bins = (nbins,nbins,nbins), range = ((-1,1),(-1,1),(-1,1)))
            hist_snf,_ = np.histogramdd(samples1, bins = (nbins,nbins,nbins), range = ((-1,1),(-1,1),(-1,1)))
            hist_inn,_ = np.histogramdd(samples2, bins = (nbins,nbins,nbins), range = ((-1,1),(-1,1),(-1,1)))

            hist_mcmc_sum+=hist_mcmc
            hist_snf_sum+=hist_snf
            hist_inn_sum+=hist_inn
        # save histograms
        #make_image(true_posterior_samples,samples1, 'SNF_img'+str(i),'Images_comp')
        #make_image(true_posterior_samples,samples2, 'INN_img'+str(i),'Images_comp')

        hist_mcmc = hist_mcmc_sum/hist_mcmc_sum.sum()
        hist_snf = hist_snf_sum/hist_snf_sum.sum()
        hist_inn = hist_inn_sum/hist_inn_sum.sum()
        hist_mcmc+=reg
        hist_snf+=reg
        hist_inn+=reg
        hist_mcmc/=hist_mcmc.sum()
        hist_snf/=hist_snf.sum()
        hist_inn/=hist_inn.sum()
        
        # evaluate KL divergence
        kl1=np.sum(scipy.special.rel_entr(hist_mcmc,hist_snf))
        kl2=np.sum(scipy.special.rel_entr(hist_mcmc,hist_inn))
        kl1_sum += kl1
        kl2_sum += kl2
        kl1_vals.append(kl1)
        kl2_vals.append(kl2)
        toc=time.time()-tic
        print('Iteration: {} of {}, Time: {:.3f}, Time left (estimated): {:.3f}'.format(i+1,testing_num_y,toc,toc/(i+1)*(testing_num_y-i-1)))
        print('KL_SNF: {:.3f}, KL_INN {:.3f}'.format(kl1,kl2))
        print('KL_SNF mean: {:.3f} +- {:.3f}, KL_INN mean: {:.3f} +- {:.3f}'.format(np.mean(kl1_vals),np.std(kl1_vals),np.mean(kl2_vals),np.std(kl2_vals)))
    kl1_vals=np.array(kl1_vals)
    kl2_vals=np.array(kl2_vals)
    kl1_var=np.sum((kl1_vals-kl1_sum/testing_num_y)**2)/testing_num_y
    kl2_var=np.sum((kl2_vals-kl2_sum/testing_num_y)**2)/testing_num_y
    print('KL1:', kl1_sum / testing_num_y,'+-',kl1_var)
    print('KL2:', kl2_sum / testing_num_y,'+-',kl2_var)


    return (kl1_sum / testing_num_y, kl2_sum/testing_num_y)

# define seed and call methods
np.random.seed(0)
torch.manual_seed(0)
# creating ys for evaluating the model
testing_num_y = 100
testing_xs = torch.rand(testing_num_y, 3, device = device)*2-1
testing_ys = forward_model(testing_xs) + b * torch.randn_like(forward_model(testing_xs)) + forward_model(testing_xs)*a*torch.randn_like(forward_model(testing_xs))
# return the KL distances
kl1,kl2 = train_and_eval(a, b, testing_ys, forward_model)

print('FINAL KL1:',kl1)
print('FINAL KL2:',kl2)










