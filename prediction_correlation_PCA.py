import sys
import pycochleagram.cochleagram as cgram
from pycochleagram import utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import sys
import os
from os import listdir

from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

from sound_dataset import SoundDataset_kfold
from AE_architectures import AE_RNN
from dense_nn import NN

from load_pd_data.py import *

cell_ids = np.arange(816) 

path = "/path to elements saved after running poisson_NN_regression_model.py"
nn_out = listdir(path)

dictionaries = {}

for ids in cell_ids:
    loss_i = None
    r2 = None
    r2_training = None
    NN_net = None
    reg_lambda = None

    for element in nn_out:
        full_path = "/".join([path, element])
        if int(element.split('_')[1]) == ids:
            if str(element.split('_')[2]) == 'ls':
                with open(full_path) as f:
                    loss_i = json.loads(json.load(f))
            elif str(element.split('_')[2]) == 'r2.npy':
                r2 = np.load(full_path)
            elif str(element.split('_')[2]) == 'r2' and str(element.split('_')[3]) == 'training.npy':
                r2_training = np.load(full_path)
            elif str(element.split('_')[2]).startswith('NN'):
                NN_net = NN(ni=128,no=1,nh_list=[], nonlinearity=None, nonlinearity_out=None) #ni=128,no=1,nh_list=[10,5] for NN
                
                NN_net.load_state_dict(torch.load(full_path, map_location='cpu'))
            elif str(element.split('_')[2]) == 'reg':
                reg_lambda = np.load(full_path)
                

    dict_name = f"dict_{ids}"
    dictionaries[dict_name] = {'ls_NN': loss_i, 'model': NN_net, 'r2': r2, 'r2_training': r2_training, 'reg': reg_lambda}

r2_training_values = [dictionary['r2_training'] for dictionary in dictionaries.values()]
r2_values = [dictionary['r2'] for dictionary in dictionaries.values()]
reg_lambda_values = [dictionary['reg'] for dictionary in dictionaries.values()]
losses = [dictionary['ls_NN'] for dictionary in dictionaries.values()]
NNs = [dictionary['model'] for dictionary in dictionaries.values()]

#Hidden Unit Activity
full_sound_input_normalized = np.load("/nfs/gatsbystor/arosinski/msc_project/full_coch_np/full_pennington_david.npy")/1000
full_sound_input_normalized_reshaped = full_sound_input_normalized[:,:-25]
full_sound_input_tensor = torch.tensor(full_sound_input_normalized, requires_grad=False) 
full_sound_input_tensor = full_sound_input_tensor[None,None,:,:]

batch_sz=1
hidden_size=128
time_lag=30 
burn_in=30 

#Get time-varying PCA model activity 
pc_pred = np.load("/path to .npy time-varying PC predictions")

ls_pred_corr =[]

x_val = pc_pred.T
x_val = torch.tensor(x_val[:1296,:]).float()   

for i, NN_poisson in enumerate(NNs):
    
    print(f'Neuron: {i}')
    spikes_val = get_spikes_val("/path to raster_cells_val.npy", cell_id=i).detach().numpy()
    spikes_val_single_pres = get_spikes_val_single_pres("/path to raster_cells_val.npy", cell_id=i)

    spikes_val = spikes_val[29:,:]
    spikes_val_single_pres = spikes_val_single_pres[:,29:]

    y_pred_val = torch.exp(NN_poisson(x_val).float())                                             
    y_pred_val = y_pred_val.detach().numpy()

    correlation = np.corrcoef(spikes_val[:,0], y_pred_val[:,0])[0, 1]

    #corr for single trial presentations
    ls_corr_single_pres = []
    for j in range(spikes_val_single_pres.shape[0]):
        ls_corr_single_pres.append(np.corrcoef(spikes_val_single_pres[j,:], y_pred_val[:,0])[0,1])
    corr_single_pres_avg = np.nanmean(ls_corr_single_pres)

    #GET TTRC
    correlation_coefficients = np.corrcoef(spikes_val_single_pres)
    correlation_coefficients = np.ma.masked_array(correlation_coefficients, np.eye(correlation_coefficients.shape[0], dtype=bool))      #exclude diag values i.e., correlation of 1
    ttrc = np.nanmean(correlation_coefficients) 

    if ttrc > 0.025:
        ls_pred_corr.append(corr_single_pres_avg/np.sqrt(ttrc))
    else:
        ls_pred_corr.append(correlation)                                                                                               #dealing with unstable approximation for very small TTRC (Pennington & David)


np.save("/save path for pred corr pca", np.array(ls_pred_corr))