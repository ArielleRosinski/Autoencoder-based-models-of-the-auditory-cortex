import sys
import pycochleagram.cochleagram as cgram
from pycochleagram import utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import sys
import os
import time
from os import listdir

from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

from AE_architectures import AE_RNN
from dense_nn import NN


model_path= str(sys.argv[1]) 
resp = np.load(str(sys.argv[2]))   
resp_val = np.load(str(sys.argv[3]))  
cell_id = int(sys.argv[7])
folder_name = str(sys.argv[8])

ls_lambda = str(sys.argv[9])
ls_lambda = ls_lambda.split(',')
ls_lambda = [float(element) for element in ls_lambda]

method = str(sys.argv[10])

#Get save path
parent_dir="/nfs/gatsbystor/arosinski/msc_project/pennington_david"
save_path = os.path.join(parent_dir, folder_name) 

if not os.path.exists(save_path):
    try:
        os.makedirs(save_path)
    except FileExistsError:
        time.sleep(0.5)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

#Get Pennington & David (2023) sound inputs
full_sound_input_normalized = np.load("/nfs/gatsbystor/arosinski/msc_project/full_coch_np/full_pennington_david.npy")/1000
full_sound_input_normalized_reshaped = full_sound_input_normalized[:,:-25]
full_sound_input_tensor = torch.tensor(full_sound_input_normalized, requires_grad=False) 
full_sound_input_tensor = full_sound_input_tensor[None,None,:,:]

batch_sz=1
hidden_size=128
time_lag=30 
burn_in=30 

#Get A1 responses (estimation and validation data)
def get_spikes_est(file, full_raster_file = True, cell_id=None):
    spikes_est = file.T
    if full_raster_file is False:
        spikes_est = spikes_est.reshape(-1, 20)
        spikes_est = torch.tensor(np.sum(spikes_est, axis=1))
        spikes_est = spikes_est[:,None]
    else: 
        spikes_est = spikes_est[:,cell_id,np.newaxis]
        spikes_est = spikes_est.reshape(-1, 20)
        spikes_est = torch.tensor(np.sum(spikes_est, axis=1))
        spikes_est = spikes_est[:,None]
    return spikes_est

def get_spikes_val(file, full_raster_file = True, cell_id=None):
    spikes_val = file

    spikes_val = spikes_val[:,cell_id,np.newaxis,:]
    spikes_val = np.mean(spikes_val, axis=0).T
    spikes_val = spikes_val.reshape(-1, 20)
    spikes_val = torch.tensor(np.sum(spikes_val, axis=1))
    spikes_val = spikes_val[:,None]
    return spikes_val


#Get hidden unit activity
device = torch.device('cpu')
model = AE_RNN(time_lag=time_lag, burn_in=burn_in, hidden_size=hidden_size).float()
model.load_state_dict(torch.load(model_path, map_location=device))
x=(model.encoder(full_sound_input_tensor.float(),initialization=None)[0]).detach().numpy()    #x_shape = (1, 44475, 128)
x_comp = x[:,:-25,:]

r2 = float('-inf') 
r2_training=float('-inf')
y_pred, ls_NN, NN_poisson = None, None, None


x_input = x_comp[:,1350:,:].squeeze() 
x_input = torch.tensor(x_input)


spikes_est = get_spikes_est(resp, full_raster_file = True, cell_id=cell_id)
for param in ls_lambda: 
    ls_NN_temp = []

    num_epochs=2000
    lr=1e-3                                                       
    
    #Poisson NN
    if method == 'nn':
        NN_poisson_temp=NN(ni=128,no=1,nh_list=[10,5], nonlinearity='relu', nonlinearity_out=None).float()  #CHANGE NH LIST
        print('method is', method)

    #Poisson LN
    elif method == '1layer':
        NN_poisson_temp=NN(ni=128,no=1,nh_list=[], nonlinearity=None, nonlinearity_out=None).float()    
        print('method is', method)
    
    optimizer=optim.Adam(NN_poisson_temp.parameters(),lr=lr)

    for i in range(num_epochs):
      total_loss=0

      y_pred_temp = NN_poisson_temp(x_input).float()                                             
      y_pred_temp = torch.exp(y_pred_temp)                         
      y = spikes_est.float()                           
      
      reg_lambda_temp = float(param)
      reg = reg_lambda_temp * sum(w.pow(2.0).sum() for w in NN_poisson_temp.parameters())
      loss = torch.mean(y_pred_temp - y *torch.log(y_pred_temp)) + reg
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      ls_NN_temp.append(loss.item()) 

    r2_training_temp = r2_score(y.detach().numpy(), y_pred_temp.detach().numpy())
    
    spikes_val = get_spikes_val(resp_val, full_raster_file = True, cell_id=cell_id)

    x_val = x_comp[:,:1325,:].squeeze()                
    x_val = torch.tensor(x_val)

    y_pred_val = torch.exp(NN_poisson_temp(x_val).float())                                             
    y_pred_val = y_pred_val.detach() 

    r2_temp = r2_score(spikes_val, y_pred_val)   

    if r2_temp > r2:
        r2 = r2_temp
        y_pred, ls_NN, r2_training, r2, NN_poisson, reg_lambda = y_pred_temp, ls_NN_temp ,r2_training_temp, r2_temp, NN_poisson_temp, reg_lambda_temp


###########################################################################################################################################################

#Save regression models
ls_path ="/".join([save_path, f"cell_{cell_id}_ls_NN.json"])
save_ls_NN = open(ls_path, "w")
json.dump(json.dumps(ls_NN), save_ls_NN)
save_ls_NN.close()

model_path ="/".join([save_path, f"cell_{cell_id}_NN_poisson.pt"]) 
torch.save(NN_poisson.state_dict(), model_path)

r2_training_path ="/".join([save_path, f"cell_{cell_id}_r2_training"])
np.save(r2_training_path, r2_training) 

r2_path ="/".join([save_path, f"cell_{cell_id}_r2"])
np.save(r2_path, r2)

reg_lambda_path ="/".join([save_path, f"cell_{cell_id}_reg_lambda"])
np.save(reg_lambda_path, reg_lambda)