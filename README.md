# Autoencoder-based-models-of-the-auditory-cortex
This repository contains the code used for MSc Project "An Autoencoder-based Model of Biological Auditory
Representation Learning".

The files it contains are:
(1) "model_training.py". Main file used to train the autoencoder model. Note that running the code 
will require changing the paths to the cochleagram data, and to the other .py files used. 
(2) "AE_architectures.py". File containing the recurrent autoencoder architecture. 
(3) "sound_dataset.py". PyTorch dataset class to obtain the training and validation cochleagram inputs.
(4) "get_STRFs.py". Code to compute the spectrotemporal receptive fields (STRFs) of the autoencoder hidden units.
(5) "poisson_NN_regression_model.py". Implementation of the Poisson LN and NN linking hypotheses to predict ferret A1 data. 
It uses the (6) "dense_nn.py" file to generate the 3-layers (NN) or 1-layer neural (LN) neural networks. 
(7) "PCA_model.py". Convolution of the cochleagram matrix from Pennington & David (2023) with the principal 
components to obtain time-varying traces that can be used to predict the ferret A1 PSTHs.

