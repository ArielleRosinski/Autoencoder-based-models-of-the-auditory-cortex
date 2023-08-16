import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import pandas as pd 

""" Dataset PyTorch class to get DRCs and natural sound cochleagrams """


class SoundDataset(Dataset):
    def __init__(self, csv_file, transform=None, uint8=False): 
        self.transform = transform
        self.uint8=uint8
        self.df = pd.read_csv(csv_file)
        self.sound_name=self.df['FilePath'].to_list()
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        cochleogram= np.load(self.sound_name[idx])

        if self.uint8:
            cochleogram = cochleogram.astype('uint8')

        if self.transform:
            cochleogram=self.transform(cochleogram)

        data=[]
        data.append(np.asarray(cochleogram))

        return data
    
class SoundDataset_kfold(Dataset):
    def __init__(self, ls, transform=None, uint8=False): 
        self.transform = transform
        self.uint8=uint8
        self.ls = ls 
       
    
    def __len__(self):
        return len(self.ls)
    
    def __getitem__(self, idx):
        cochleogram= self.ls[idx] 

        if self.uint8:
            cochleogram = cochleogram.astype('uint8')

        if self.transform:
            cochleogram=self.transform(cochleogram)

        data=[]
        data.append(np.asarray(cochleogram))

        return data

