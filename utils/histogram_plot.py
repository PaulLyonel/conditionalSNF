#!/usr/bin/env python

import os
import matplotlib.pyplot as plt
import numpy as np


# makes images of 1d and 2d distributions of "true" samples and predicted (either by INN or SNF) samples
# saves the matplotlib plot in the directory
# in case of many dimensions you might wanna specify some indices of dimensions to restrict this to (inds)

def make_image(true_samples, pred_samples, img_name,directory='Images',inds=None):
    cmap = plt.cm.tab20
    range_param = 1.2
    if inds is None:
        no_params = min(5, true_samples.shape[1])
        inds=range(no_params)
    else:
        no_params=len(inds)
    fig, axes = plt.subplots(figsize=[12,12], nrows=no_params, ncols=no_params, gridspec_kw={'wspace':0., 'hspace':0.});

    for j, ij in enumerate(inds):
        for k, ik in enumerate(inds):
            axes[j,k].get_xaxis().set_ticks([])
            axes[j,k].get_yaxis().set_ticks([])
            # if k == 0: axes[j,k].set_ylabel(j)
            # if j == len(params)-1: axes[j,k].set_xlabel(k);
            if j == k:
                axes[j,k].hist(pred_samples[:,ij], bins=50, color=cmap(0), alpha=0.3, range=(-range_param,range_param))
                axes[j,k].hist(pred_samples[:,ij], bins=50, color=cmap(0), histtype="step", range=(-range_param,range_param))

                axes[j,k].hist(true_samples[:,ij], bins=50, color=cmap(2), alpha=0.3, range=(-range_param,range_param))
                axes[j,k].hist(true_samples[:,ij], bins=50, color=cmap(2), histtype="step", range=(-range_param,range_param))
            else:
                val, x, y = np.histogram2d(pred_samples[:,ij], pred_samples[:,ik], bins=25, range = [[-range_param, range_param], [-range_param, range_param]])
                axes[j,k].contour(val, 8, extent=[x[0], x[-1], y[0], y[-1]], alpha=0.5, colors=[cmap(0)])

                val, x, y = np.histogram2d(true_samples[:,ij], true_samples[:,ik], bins=25, range = [[-range_param, range_param], [-range_param, range_param]])
                axes[j,k].contour(val, 8, extent=[x[0], x[-1], y[0], y[-1]], alpha=0.5, colors=[cmap(2)])
    if not os.path.isdir(directory):
        os.mkdir(directory)
    plt.savefig('./'+directory+'/'+img_name, bbox_inches='tight',pad_inches = 0.05)
    plt.close()
