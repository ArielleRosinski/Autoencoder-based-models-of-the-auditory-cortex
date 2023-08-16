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

""" Generic fully connected NN architecture """


class NN(nn.Module):
  def __init__(self,ni,no,nh_list,nonlinearity='relu',nonlinearity_out='relu'):  
    super().__init__()

    self.nonlinearity = nonlinearity
    self.nonlinearity_out = nonlinearity_out
    self.input=ni
    self.num_hidden_layers = len(nh_list)

    self.layers=nh_list
    self.layers.insert(0, ni)
    self.layers.append(no)
     
    self.dense_layers = nn.ModuleList()
    for i in range(self.num_hidden_layers):
      self.dense_layers.append(nn.Linear(self.layers[i], self.layers[i + 1])) #e.g., 128 --> 10; 10 --> 5

    self.output_layer=nn.Linear(self.layers[-2], self.layers[-1])             #e.g., 5 --> 1

  def forward(self,x):
    x=x.view(-1,self.input)
    for layer in self.dense_layers:
      if self.nonlinearity == 'relu':
        x = torch.relu(layer(x))
      elif self.nonlinearity is None:
        x = layer(x)
    
    if self.nonlinearity_out == 'relu':
      out = torch.relu(self.output_layer(x))
    elif self.nonlinearity_out is None:
      out = self.output_layer(x)

    return out