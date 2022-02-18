# This code belongs to the paper
#
# P. Hagemann, J. Hertrich, G. Steidl (2021).
# Stochastic Normalizing Flows for Inverse Problems: a Markov Chains Viewpoint
# Arxiv preprint arXiv:2109.11375
#
# Please cite the paper, if you use this code.
# This script reproduces the numerical example from Section 7.1 of the paper.
#
from torch.optim import Adam
import ot
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy
import os
import pickle
import time

from utils.Util_mixture import *
from utils.histogram_plot import make_image
from core.SNF import *
from core.INN import *

batch_size = 1024
num_samples_per_epoch = 1024

num_epochs_SNF = 20000
num_epochs_INN = 20000

DIMENSION=100

# train from scratch or just use pretrained model
retrain=True

# trains and evaluates both the INN and SNF and returns the Wasserstein distance on the mixture example
# parameters are the mixture params (parameters of the mixture model in the prior), b (likelihood parameter)
# a set of testing_ys and the forward model (forward_map)
#
# prints and returns the Wasserstein distance of SNF/INN
def train_and_eval(mixture_params, b, testing_ys, forward_map,training_run):

    forward_model=lambda x: forward_pass(x, forward_map)
    log_posterior=lambda samples,y:get_log_posterior(samples,forward_map,mixture_params,b,y)
    snf = create_snf(2,128,log_posterior,metr_steps_per_block=3,dimension=DIMENSION,dimension_condition=DIMENSION,noise_std=0.05,num_inn_layers=4, lang_steps = 0, step_size = 1e-4, langevin_prop = True)
    INN = create_INN(8,128,dimension=DIMENSION,dimension_condition=DIMENSION)
    if retrain:
        optimizer = Adam(snf.parameters(), lr = 1e-4)

        prog_bar = tqdm(total=num_epochs_SNF)
        for i in range(num_epochs_SNF):
            data_loader = get_epoch_data_loader(mixture_params, num_samples_per_epoch, batch_size, forward_map, b)
            loss = train_SNF_epoch(optimizer, snf, data_loader, forward_model,0.,b,lambda samples: get_prior_log_likelihood(samples, mixture_params))
            prog_bar.set_description('loss: {:.4f}, b: {}, n_mix: {}'.format(loss, b, len(mixture_params)))
            prog_bar.update()
        prog_bar.close()

        optimizer_inn = Adam(INN.parameters(), lr = 1e-4)

        prog_bar = tqdm(total=num_epochs_INN)

        for i in range(num_epochs_INN):
            data_loader = get_epoch_data_loader(mixture_params, num_samples_per_epoch, batch_size, forward_map, b)
            loss = train_inn_epoch(optimizer_inn, INN, data_loader)
            prog_bar.set_description('loss: {:.4f}, b: {}, n_mix: {}'.format(loss, b, len(mixture_params)))
            prog_bar.update()
        prog_bar.close()
        if not os.path.isdir('models_mixture'):
            os.mkdir('models_mixture')

        torch.save(snf.state_dict(),'models_mixture/snf_'+str(training_run)+'.pt')
        torch.save(INN.state_dict(),'models_mixture/inn_'+str(training_run)+'.pt')
    else:
        snf.load_state_dict(torch.load('models_mixture/snf_'+str(training_run)+'.pt'))
        INN.load_state_dict(torch.load('models_mixture/inn_'+str(training_run)+'.pt'))
    
    testing_x_per_y = 5000
    testing_x_per_y_more = 10000

    testing_num_y = len(testing_ys)
    weights1, weights2 = np.ones((testing_x_per_y,)) / testing_x_per_y, np.ones((testing_x_per_y,)) / testing_x_per_y

    weights1 = weights1.astype(np.float64)
    weights2 = weights2.astype(np.float64)
    weights1_large, weights2_large = np.ones((testing_x_per_y_more,)) / testing_x_per_y_more, np.ones((testing_x_per_y_more,)) / testing_x_per_y_more

    weights1_large = weights1_large.astype(np.float64)
    weights2_large = weights2_large.astype(np.float64)
    w1=[]
    w2=[]

    
    w1_large=[]
    w2_large=[]
    tic=time.time()
    for i, y in enumerate(testing_ys):
        true_posterior_params = get_mixture_posterior(mixture_params, forward_map, b**2, y)
        true_posterior_samples = draw_mixture_dist(true_posterior_params, testing_x_per_y).cpu().numpy()
        inflated_ys = y[None, :].repeat(testing_x_per_y, 1)
        inp_samps=torch.randn(testing_x_per_y, DIMENSION, device=device)
        samples1 = snf.forward(inp_samps, inflated_ys)[0].detach().cpu().numpy()
        samples_INN = INN(inp_samps, c = inflated_ys)[0].detach().cpu().numpy()
        
        if ((i <10) and (training_run ==0)):
            make_image(true_posterior_samples, samples1, 'SNF_mixtures{}_b{}_{}.png'.format(len(mixture_params), b,i),directory='Images',inds=[0,49,99])
            make_image(true_posterior_samples, samples_INN, 'INN_mixtures{}_b{}_{}.png'.format(len(mixture_params), b,i),directory='Images',inds=[0,49,99])

        M1 =ot.dist(samples1, true_posterior_samples, metric='euclidean')
        M2 =ot.dist(samples_INN, true_posterior_samples, metric='euclidean')
        
        w1.append(ot.emd2(weights1, weights2, M1, numItermax=1000000))
        w2.append(ot.emd2(weights1, weights2, M2, numItermax=1000000))
        true_posterior_samples = draw_mixture_dist(true_posterior_params, testing_x_per_y_more).cpu().numpy()
        
        inflated_ys = y[None, :].repeat(testing_x_per_y_more, 1)
        inp_samps=torch.randn(testing_x_per_y_more, DIMENSION, device=device)
        samples1 = snf.forward(inp_samps, inflated_ys)[0].detach().cpu().numpy()
        samples_INN = INN(inp_samps, c = inflated_ys)[0].detach().cpu().numpy()
        M1 =ot.dist(samples1, true_posterior_samples, metric='euclidean')
        M2 =ot.dist(samples_INN, true_posterior_samples, metric='euclidean')
        
        w1_large.append(ot.emd2(weights1_large, weights2_large, M1, numItermax=1000000))
        w2_large.append(ot.emd2(weights1_large, weights2_large, M2, numItermax=1000000))
        toc=time.time()-tic
    w1_mean=np.mean(w1)
    w2_mean=np.mean(w2)
    w1_std=np.std(w1)
    w2_std=np.std(w2)

    w1_mean_large=np.mean(w1_large)
    w2_mean_large=np.mean(w2_large)
    print('W SNF:', w1_mean,'+-',w1_std)
    print('W INN:', w2_mean,'+-',w2_std)
    
    print('W SNF large:',  w1_mean_large)
    print('W INN large:', w2_mean_large)

    return w1_mean,w2_mean,w1_mean_large,w2_mean_large



#numbers of testing_ys
testing_num_y = 100
# likelihood parameter
b = 0.1
# forward_model
forward_map = create_forward_model(scale = 0.1,dimension=DIMENSION)
# number of mixtures
n_mixtures=12
np.random.seed(0)
torch.manual_seed(0)
results_array = np.zeros((5,4))
for r in range(5):

    mixture_params=[]
    # create mixture params (weights, means, covariances)
    for i in range(n_mixtures):
        mixture_params.append((1./n_mixtures,torch.tensor(np.random.uniform(size=DIMENSION)*2-1, device = device,dtype=torch.float),torch.tensor(0.0001,device=device,dtype=torch.float)))
    with open('models_mixture/models_mixture' +str(r), 'wb') as fp:
        pickle.dump(mixture_params, fp)
    # draws testing_ys
    testing_xs = draw_mixture_dist(mixture_params, testing_num_y)
    testing_ys = forward_pass(testing_xs, forward_map) + b * torch.randn(testing_num_y, DIMENSION, device=device)

    results = train_and_eval(mixture_params,b,testing_ys,forward_map,training_run = r)
    results_array[r,0] = results[0]
    results_array[r,1] = results[1]
    results_array[r,2] = results[2]
    results_array[r,3] = results[3]

print('MEANS OF SNF and INN')
print(np.mean(results_array[:,0]))
print(np.mean(results_array[:,1]))
print('STD OF SNF and INN')

print(np.std(results_array[:,0]))
print(np.std(results_array[:,1]))


print('MEANS OF SNF and INN LARGE')
print(np.mean(results_array[:,2]))
print(np.mean(results_array[:,3]))
print('STD OF SNF and INN LARGE')

print(np.std(results_array[:,2]))
print(np.std(results_array[:,3]))




