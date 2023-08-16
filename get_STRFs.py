import pycochleagram.cochleagram as cgram
from pycochleagram import utils

import json
import sys
import os 
from os import listdir

import matplotlib.pyplot as plt
import numpy as np
import math

from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

from forest_sound_dataset import SoundDataset,SoundDataset_kfold
from AE_architectures_7_4_2023 import AE_RNN_test
from AE_architectures_6_2_2023 import AE_RNN_burn_in
from AE_architectures_conv_13_6_2023 import AE_Conv_Decoder

""" Calculating STRFs using ridge regression with L2 weight regularization. 30 dynamic random chord stimuli, each lasting for 200000 ms (i.e., 10000 chords), were used for STRF estimation in MSc project  """

time_lag=30
burn_in=30
device = torch.device('cpu')

model_path = str(sys.argv[1])
model_name = str(sys.argv[2])
get_DRC_strf = str(sys.argv[3])
get_DRS_strf = str(sys.argv[4])
hidden_size= int(sys.argv[5])

print ("Argument List:", str(sys.argv))


model = AE_Conv_Decoder(hidden_size=hidden_size, time_lag=time_lag, burn_in=burn_in).float()
model.load_state_dict(torch.load(model_path, map_location=device))

parent_dir="/nfs/gatsbystor/arosinski/msc_project/"
save_path = os.path.join(parent_dir, model_name) 

if not os.path.exists(save_path):
   os.makedirs(save_path)

dt=30 #time lag 



#################################################################################################################################
#GET STRFs with Ridge regression
def get_STRFs(dt, final_x,final_stimulus,save_name,save_path,save=True):
    regr = RidgeCV(alphas=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
    alpha_list=[]

    final_x=final_x[dt:,:]

    strf_matrix=np.zeros((final_stimulus.shape[0], dt, final_x.shape[1])) 
    for i in range(final_x.shape[1]): 
        for t in range(dt):
            regr.fit((final_stimulus[:,dt-t:final_stimulus.shape[1]-t]).T, final_x[:,i]) 
            strf_matrix[:,t,i]=regr.coef_
            alpha_list.append(regr.alpha_)
            print(f'done with lag {t}')
        print(f'done with neuron {i}')
    
    if save==True:
        strf_matrix_path="/".join([save_path, save_name])
        np.save(strf_matrix_path, strf_matrix)

#################################################################################################################################
""" STRFs estimated using DRCs (dynamic random chords)"""
if get_DRC_strf=="True": 
    DRC_data=SoundDataset('/nfs/gatsbystor/arosinski/msc_project/csv_data/DRC_test_28_2_2023_clust.csv', uint8=False) 

    batch_sz=30
    DRC_loader = DataLoader(DRC_data,batch_size=batch_sz,shuffle=False)

    #For 10 000-bins length batch elements 
    ls_x=[]
    ls_stimulus=[]
    for batch in (DRC_loader):
        image=batch[0]

        x=(model.encoder(image.float(),initialization=None)[0]).detach().numpy()  

        stimulus=(image.squeeze()).detach().numpy()        
    
        ls_x.append(np.vstack(x))
        ls_stimulus.append(np.hstack(stimulus))
    final_x=np.vstack(ls_x)                   
    final_stimulus=np.hstack(ls_stimulus)      

    get_STRFs(dt, final_x,final_stimulus,"strf_DRC_matrix",save_path)

    
#####################################################################################################################################################################
""" STRFs estimated using DRSs (dynamic ripple stimuli)"""
if get_DRS_strf=="True": 
    DRS_data=SoundDataset("/nfs/gatsbystor/arosinski/msc_project/csv_data/DRS_cluster_4_11_5_2023.csv", uint8=False) 

    batch_sz=30
    DRS_loader = DataLoader(DRS_data,batch_size=batch_sz,shuffle=False)

    ls_x=[]
    ls_stimulus=[]
    T=1
    for epoch in range(T):
        for batch in (DRS_loader):
            image=batch[0]

            x=(model.encoder(image.float(),initialization=None)[0]).detach().numpy()    
            x = x[:,burn_in:,:]                                                  

            stimulus=(image.squeeze()).detach().numpy()                                   
            stimulus=stimulus[:,:,burn_in:]                                                 
            
            ls_x.append(np.vstack(x))
            ls_stimulus.append(np.hstack(stimulus))
    final_x=np.vstack(ls_x)                                                             
    final_stimulus=np.hstack(ls_stimulus)                                              

    get_STRFs(dt, final_x,final_stimulus,"strf_DRS_matrix",save_path)

