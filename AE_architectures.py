import pycochleagram.cochleagram as cgram
from pycochleagram import utils
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

""" AE RNN is the autoencoder architecture used for MSc Project Autoencoder-based models of the auditory cortex"""
"""The input is processed at each time step by the non-RNN encoder (conv_encoder and dense_encoder). The output of the non-RNN encoder is the input to the RNN encoder, which
processes the 2-s input sequence over time. Its output is the initialization to the decoder, which contains an RNN that runs backwards in time. Its output is passed through the non-RNN decoder (dense_decoder and conv_decoder_1D) and reconstructs the last N (30) timesteps. """
class AE_RNN(torch.nn.Module):
    def __init__(self, C=1, time_lag=30, hidden_size=128, input_shape=(81, 100),burn_in=30):
        super().__init__()

        self.C=C
        self.time_lag=time_lag
        self.hidden_size=hidden_size
        self.freq_values=input_shape[0]
        self.seq_len=input_shape[1]
        self.burn_in=burn_in 

        #ARCHITECTURE#
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,1), padding='same'), 
            nn.ReLU(True),
            nn.MaxPool2d((2,1)),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,1), stride=1, padding='same'),       
            nn.ReLU(True),
            nn.MaxPool2d((2,1)),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,1), stride=1, padding='same'),       
            nn.ReLU(True),
        )

        self.dense_encoder = nn.Linear(20*32,40)          
            
        self.RNN_encoder = nn.RNN(input_size=40, hidden_size=self.hidden_size, num_layers = 1, nonlinearity='relu', batch_first=True)

        self.RNN_decoder = nn.RNNCell(1, hidden_size=self.hidden_size, nonlinearity='relu')

        self.dense_decoder = nn.Linear(self.hidden_size,32*10)

        self.conv_decoder_1D = nn.Sequential(
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=3, stride=2,padding=1, output_padding=1), 
            nn.ReLU(True),
            nn.ConvTranspose1d(in_channels=16, out_channels=8, kernel_size=3, stride=2,padding=0, output_padding=0), 
            nn.ReLU(True),
            nn.ConvTranspose1d(in_channels=8, out_channels=1, kernel_size=3, stride=2,padding=1, output_padding=0), 
        )

    def encoder(self,x,initialization):
        encoded = self.conv_encoder(x)                                                #encoded shape is [batch, channel, freq, time]                         

        encoded = torch.flatten(torch.swapaxes(encoded,1,3),start_dim=2,end_dim=3)    #encoded shape is [batch, time, channel*freq]
        
        encoded = self.dense_encoder(encoded)                                        
        
        if initialization is not None:
            out_rnn, h_n = self.RNN_encoder(encoded,initialization) 
        else:
            out_rnn, h_n = self.RNN_encoder(encoded)                    
   
        return out_rnn, h_n                                                           #out_rnn is [batch, time, 128=hidden units] and h_n is [1, batch, 128=hidden units] #burn in here
    
    def decoder_train(self, x, device): 
        out_final = torch.tensor([])                                                  #out_rnn went from [batch, time, 128] to [batch*time, 128] 
        h_i = x                              
        rnn_cell_xinput=torch.zeros(h_i.shape[0],1).to(device)                        
       
        for i in range(self.time_lag):          
            h_i = self.RNN_decoder(rnn_cell_xinput, h_i)                                            
            out_dense_decoder = self.dense_decoder(h_i)                             
                
            x=out_dense_decoder.view(-1,32,10)                                        
            decoded=self.conv_decoder_1D(x)                                          
            
            decoded=decoded.view(-1,self.seq_len-self.time_lag-self.burn_in,1,self.freq_values)                     
            
            out_final = out_final.to(device)
            out_final = torch.cat([out_final, decoded], axis=2)   
        return out_final                                                              #out_final is [batch, time, time_lag, freq]

    def forward_train(self,x,initialization,device): 
        x = self.encoder(x,initialization)[0]
        x = x[:, self.time_lag+self.burn_in:, :]     
        x = torch.reshape(x, (-1, self.hidden_size))                 
        x = self.decoder_train(x,device)
        return(x)
    

############################################################################################################################################################################################

""" Implementation with LSTM """
class AE_LSTM(torch.nn.Module):
    def __init__(self, C=1, time_lag=30, hidden_size=128, input_shape=(81, 100),burn_in=30):
        super().__init__()

        self.C=C
        self.time_lag=time_lag
        self.hidden_size=hidden_size
        self.freq_values=input_shape[0]
        self.seq_len=input_shape[1]
        self.burn_in=burn_in 

        #ARCHITECTURE#
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,1), padding='same'), #81*100
            nn.ReLU(True),
            nn.MaxPool2d((2,1)),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,1), stride=1, padding='same'),       
            nn.ReLU(True),
            nn.MaxPool2d((2,1)),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,1), stride=1, padding='same'),       
            nn.ReLU(True),
        )

        self.dense_encoder = nn.Linear(20*32,40)           
            
        self.LSTMCell_encoder = nn.LSTMCell(input_size=40, hidden_size=hidden_size)

        self.LSTM_decoder = nn.LSTMCell(1, hidden_size=self.hidden_size)

        self.dense_decoder = nn.Linear(self.hidden_size,32*10)

        self.conv_decoder_1D = nn.Sequential(
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=3, stride=2,padding=1, output_padding=1), #19
            nn.ReLU(True),
            nn.ConvTranspose1d(in_channels=16, out_channels=8, kernel_size=3, stride=2,padding=0, output_padding=0), #41
            nn.ReLU(True),
            nn.ConvTranspose1d(in_channels=8, out_channels=1, kernel_size=3, stride=2,padding=1, output_padding=0), #81
        )

        self.conv_decoder_2D = nn.ConvTranspose2d(in_channels=self.C, out_channels=1, kernel_size=(2,1), stride=(2,1),padding=0, output_padding=(1,0)) 

    def encoder(self,x,device):
        encoded = self.conv_encoder(x)                                                #encoded shape is [batch, channel, freq, time]                         

        encoded = torch.flatten(torch.swapaxes(encoded,1,3),start_dim=2,end_dim=3)    #encoded shape is [batch, time, channel*freq]
        
        encoded = self.dense_encoder(encoded)                                        
        
        hidden_state = torch.zeros(encoded.shape[0], self.hidden_size).to(device)  
        cell_state = torch.zeros(encoded.shape[0], self.hidden_size).to(device)  

        hidden_states = []
        cell_states = []
        for i in range(encoded.shape[1]):
            hidden_state, cell_state = self.LSTMCell_encoder(encoded[:,i,:], (hidden_state, cell_state))
            hidden_states.append(hidden_state)
            cell_states.append(cell_state)
        hidden_states = torch.stack(hidden_states)
        cell_states = torch.stack(cell_states)

        return hidden_states, cell_states                                             #Hidden/Cell states shape: torch.Size([100, 32, 128]) --> becomes permuted, and account for time lag + burn in

                                                       
    
    def decoder_train(self, hidden_states, cell_states, device): 
        out_final = torch.tensor([])                                                  #Hidden/Cell states went from [32, time=40 (because of burn in and lag), 128] to [batch*time=1280, 128] 
        h_i = (hidden_states, cell_states)                               
        rnn_cell_xinput=torch.zeros(hidden_states.shape[0],1).to(device)                                   
       
        for i in range(self.time_lag):          
            h_i = self.LSTM_decoder(rnn_cell_xinput, h_i)                     
            out_dense_decoder = self.dense_decoder(h_i[0])                        
                
            x=out_dense_decoder.view(-1,32,10)         
            decoded=self.conv_decoder_1D(x)                       
            
            decoded=decoded.view(-1,self.seq_len-self.time_lag-self.burn_in,1,self.freq_values)                     
            
            out_final = out_final.to(device)
            out_final = torch.cat([out_final, decoded], axis=2)   
        return out_final                                                              #out_final is [32, 40=time, 30=time_lag, freq=81]

    def forward_train(self,x,device): 
        hidden_states, cell_states = self.encoder(x,device)

        hidden_states = hidden_states.permute(1,0,2)
        cell_states = cell_states.permute(1,0,2)

        hidden_states = hidden_states[:, self.time_lag+self.burn_in:, :]   
        cell_states = cell_states[:, self.time_lag+self.burn_in:, :]      

        hidden_states = torch.reshape(hidden_states, (-1, self.hidden_size))    
        cell_states = torch.reshape(cell_states, (-1, self.hidden_size))  

        x = self.decoder_train(hidden_states,cell_states,device)
        return(x)


