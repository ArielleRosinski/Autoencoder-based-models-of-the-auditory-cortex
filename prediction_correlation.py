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

from forest_sound_dataset import SoundDataset,SoundDataset_kfold
from AE_architectures_6_2_2023 import AE_RNN_burn_in, AE_RNN_burn_in2,AE_RNN_burn_in3
from AE_architectures_7_4_2023 import AE_RNN_test , AE_LSTM_test
from AE_architectures_conv_13_6_2023 import AE_Conv_Decoder
from dense_nn_5_7_2023 import NN

#Get output from regression models
Poisson = True
cell_ids = np.arange(816) 
path = "/nfs/gatsbystor/arosinski/msc_project/pennington_david/NN_poisson_untrained_nn"
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
                if Poisson: 
                    NN_net = NN(ni=128,no=1,nh_list=[10,5], nonlinearity='relu', nonlinearity_out=None) 
                else:
                    NN_net = NN(ni=128, no=1, nh_list=[10,5])
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

#Get Pennington & David (2023) sound input
full_sound_input_normalized = np.load("/nfs/gatsbystor/arosinski/msc_project/full_coch_np/full_pennington_david.npy")/1000
full_sound_input_normalized_reshaped = full_sound_input_normalized[:,:-25]
full_sound_input_tensor = torch.tensor(full_sound_input_normalized, requires_grad=False) 
full_sound_input_tensor = full_sound_input_tensor[None,None,:,:]

batch_sz=1
hidden_size=128
time_lag=30 
burn_in=30 

#Get hidden Unit Activity
device = torch.device('cpu')
model = AE_RNN_test(time_lag=time_lag, burn_in=burn_in, hidden_size=hidden_size).float()
model_path="/nfs/gatsbystor/arosinski/msc_project/models/model_pre_training_128_hu.pt"
model.load_state_dict(torch.load(model_path, map_location=device))
x=(model.encoder(full_sound_input_tensor.float(),initialization=None)[0]).detach().numpy()  
x_comp = x[:,:-25,:]

#Get A1 responses (average every two bins to get 20 ms temporal resolution)
def get_spikes_val(file, cell_id=None):
    spikes_val = np.load(file)
    spikes_val = spikes_val[:,cell_id,np.newaxis,:]
    spikes_val = np.mean(spikes_val, axis=0).T
    spikes_val = spikes_val.reshape(-1, 20)
    spikes_val = torch.tensor(np.sum(spikes_val, axis=1))
    spikes_val = spikes_val[:,None]
    return spikes_val

def get_spikes_val_single_pres(file, cell_id):
    spikes_val = np.load(file)                                   
    spikes_val = spikes_val[:,cell_id,np.newaxis,:]                
    spikes_val = spikes_val.reshape(spikes_val.shape[0], -1, 20)
    spikes_val = np.sum(spikes_val, axis=2)                       
    return spikes_val

#Compute prediction correlation
ls_pred_corr =[]
x_val = x_comp[:,:1325,:].squeeze()                
x_val = torch.tensor(x_val)
for i, NN_poisson in enumerate(NNs):
    print(f'Neuron: {i}')
    spikes_val = get_spikes_val("/nfs/nhome/live/arosinski/msc_project_files/pennington_david_files/raster_cells_val.npy", cell_id=i).detach().numpy()
    spikes_val_single_pres = get_spikes_val_single_pres("/nfs/nhome/live/arosinski/msc_project_files/pennington_david_files/raster_cells_val.npy", cell_id=i)    
    y_pred_val = torch.exp(NN_poisson(x_val).float())                                             
    y_pred_val = y_pred_val.detach().numpy()    

    #Correlations between single presentations
    ls_corr_single_pres = []
    for j in range(spikes_val_single_pres.shape[0]):
        ls_corr_single_pres.append(np.corrcoef(spikes_val_single_pres[j,:], y_pred_val[:,0])[0,1])
    corr_single_pres_avg = np.nanmean(ls_corr_single_pres)   
    
    #TTRC
    correlation_coefficients = np.corrcoef(spikes_val_single_pres)
    correlation_coefficients = np.ma.masked_array(correlation_coefficients, np.eye(correlation_coefficients.shape[0], dtype=bool))    
    ttrc = np.nanmean(correlation_coefficients)    

    ls_pred_corr.append(corr_single_pres_avg/np.sqrt(ttrc))


np.save("/nfs/gatsbystor/arosinski/msc_project/pennington_david/pred_corr/pred_corr_poisson_untrained_nn", np.array(ls_pred_corr))