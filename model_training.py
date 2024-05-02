
import sys
import pycochleagram.cochleagram as cgram
from pycochleagram import utils
import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import os
from os import listdir

from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

from sound_dataset import SoundDataset, SoundDataset_kfold
from AE_architectures import AE_RNN


#Get files (i.e., cochleagrams)
path="/nfs/gatsbystor/arosinski/msc_project/full_coch_np"
sound_files=listdir(path)

file_names=[]
files=[]
for file in sound_files:
    if file.endswith(".npy") and file.startswith("full"): 
        file_names.append(file)
        full_path=[path,file]
        full_path="/".join(full_path)
        files.append(full_path)
print(f"length of files list is {len(files)}")

#Normalize the data and get 2-seconds random crops from the training set 
def get_dataset(files, kfold, fold_id, file_names, training=True, norm=True):
    ls_coch=[]
    crop_size=(81,100)
    random_crop=transforms.Compose([transforms.RandomCrop(crop_size),])
   
    for i,file in enumerate(files): 
        file=np.load(file)

        for fold, (train_ids, test_ids) in enumerate(kfold.split(file.T)):
            if fold==fold_id:
                
                if norm==True:
                    if file_names[i].endswith("coch.npy"):
                        file=file/2500
                    elif file_names[i] == "full_pennington_david.npy":
                        file=file/1000
                    else: 
                        file=file/5000
                
                if training:
                    indices=train_ids
                else:
                    indices=test_ids

                #Ensure that no 2-seconds window combines time windows from different parts of the recordings:
                split_id=None                           
                for i in range(len(indices)-1): 
                    if indices[i+1] != (indices[i]+1):
                        split_id=i+1
                    
                if split_id is None:
                    cropped_coch=torch.tensor(file.T[indices].T)
                    
                    for i in range(int(cropped_coch.shape[1]/20)): 
                        cochleogram=random_crop(cropped_coch)
                        ls_coch.append(cochleogram.numpy())

                else:
                    cropped_coch1 = torch.tensor(file.T[indices][:split_id].T)
                    cropped_coch2 = torch.tensor(file.T[indices][split_id:].T)

                    for i in range(int(cropped_coch1.shape[1]/20)): 
                        cochleogram=random_crop(cropped_coch1)
                        ls_coch.append(cochleogram.numpy())      

                    for i in range(int(cropped_coch2.shape[1]/20)):  
                        cochleogram=random_crop(cropped_coch2)
                        ls_coch.append(cochleogram.numpy())
    return ls_coch

#L1/L2 on hidden unit activity:
def get_reg(regularizer,reg_lambda,model,image,initialization):
    if regularizer == 'l1':
        dimensionality = torch.numel(model.encoder(image,initialization)[0])      
        reg = ( torch.norm(model.encoder(image,initialization)[0], 1) )/dimensionality
        reg = reg_lambda * reg    
    elif regularizer == 'l2':
        reg = reg_lambda * (torch.norm(model.encoder(image,initialization)[0], 2)**2)     
    else:
        reg = 0
    return reg

#Exponential decay in the MSE loss function
def get_exp_decay(SE,device):
    weight = (np.logspace(-2, 0, num=SE.shape[2]))[::-1]        
    weight = torch.tensor(np.copy(weight[None, None, :, None])).to(device)
    weighted_SE = SE * weight
    return weighted_SE

#Check if GPU is available
print(torch.cuda.is_available())
if torch.cuda.is_available():
  device=torch.device("cuda")
else:
  device=torch.device("cpu")
print(device)

batch_sz=32
burn_in=30
T = 500

k_folds=5
kfold = KFold(n_splits=k_folds, shuffle=False) 

#Get training parameters 
regularizer = str(sys.argv[1])
reg_lambda = float(sys.argv[2])
fold_id =int(sys.argv[3])
hidden_size = int(sys.argv[4]) 
exp_decay = str(sys.argv[5])
folder_name = str(sys.argv[6])
time_lag= int(sys.argv[7])


#Get save path
parent_dir="/nfs/gatsbystor/arosinski/msc_project/runs"
save_path = os.path.join(parent_dir, folder_name) 
if not os.path.exists(save_path):
    os.makedirs(save_path)

#Training
model = AE_RNN(time_lag=time_lag, hidden_size=hidden_size).float() 
model=model.to(device)

loss_function=nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 

train_loss_epoch = []
train_loss_iter = []
test_loss_iter_with_reg = []
test_loss_iter_no_reg = []

for epoch in range(T):
    train_temp_loss_epoch=0
            
    train_data=SoundDataset_kfold(get_dataset(files, kfold, fold_id, file_names, norm=True), transform = transforms.Compose([transforms.ToTensor(),]),uint8=False ) 
    train_loader = DataLoader(train_data,batch_size=batch_sz,shuffle=True) 

    #TRAINING
    for batch in train_loader:
        image = batch[0].float().to(device)

        #For a time-scrambled model:
        #idx = torch.randperm(image.shape[-1]) 
        #image = image[:,:,:, idx]

        #Initialize initial hidden state of encoder RNN to zeros
        initialization=torch.zeros((1,image.shape[0],hidden_size)).to(device)
                
        reconstructed=model.forward_train(image.float(),initialization,device) 
                        
        #Adjust input dimensionality to match output shape (altered due to parallelization of reconstrcutions during training)
        x=torch.swapaxes(image.squeeze(),1,2)                  
        x_list = [x[:,time_lag+burn_in:,:]]
        for i in range(time_lag-1):
            x_t_minus_i=x[:,time_lag+burn_in-(i+1):-(i+1),:]
            x_list.append(x_t_minus_i)                                        
        img_comparison=torch.stack(x_list,axis=2) 

        #Add regularization if specified 
        reg=get_reg(regularizer,reg_lambda,model,image,initialization)

        #Compute MSE with exponential decay
        SE=((img_comparison-reconstructed)**2)
        if exp_decay == 'True':
            L = get_exp_decay(SE, device).mean() + reg
        else: 
            L = SE.mean() + reg

        #Adjust weights
        optimizer.zero_grad()
        L.backward()
        nn.utils.clip_grad_value_(model.parameters(), clip_value=1)
        optimizer.step()

        train_loss_iter.append(L.item())
        train_temp_loss_epoch+=L.item()

    train_loss_epoch.append(train_temp_loss_epoch)

    #Save models
    if epoch % (T/5)==0:
        model_path ="/".join([save_path, f"model_epoch_{epoch}_regularizer_{regularizer}_lambda_{reg_lambda}_fold_{fold_id}.pt"])
        torch.save(model.state_dict(), model_path)
   

        plt.figure()
        plt.plot(train_loss_iter)
        plt.xlabel('iterations')
        plt.ylabel('MSE')
        fig_path ="/".join([save_path, f"loss_epoch_{epoch}_regularizer_{regularizer}_lambda_{reg_lambda}_fold_{fold_id}.png"])
        plt.savefig(fig_path)
       
        plt.figure()
        plt.semilogy(train_loss_iter)
        plt.xlabel('iterations')
        plt.ylabel('MSE_log_scale')
        fig_path ="/".join([save_path, f"loss_semilog_epoch_{epoch}_regularizer_{regularizer}_lambda_{reg_lambda}_fold_{fold_id}.png"])
        plt.savefig(fig_path)

print("end of training!")


model_path ="/".join([save_path, f"model_final_regularizer_{regularizer}_lambda_{reg_lambda}_fold_{fold_id}.pt"]) 
torch.save(model.state_dict(), model_path)

plt.figure()
plt.plot(train_loss_iter)
plt.xlabel('iterations')
plt.ylabel('MSE')
fig_path ="/".join([save_path, f"loss_final_regularizer_{regularizer}_lambda_{reg_lambda}_fold_{fold_id}.png"])
plt.savefig(fig_path)

plt.figure()
plt.semilogy(train_loss_iter)
plt.xlabel('iterations')
plt.ylabel('MSE_log_scale')
fig_path ="/".join([save_path, f"loss_semilog_final_regularizer_{regularizer}_lambda_{reg_lambda}_fold_{fold_id}.png"])
plt.savefig(fig_path)