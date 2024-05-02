# Autoencoder-based-models-of-the-auditory-cortex
This repository contains the code used for MSc Project "An Autoencoder-based Model of Biological Auditory
Representation Learning".

The files it contains include:

(1) "model_training.py": main file used to train the autoencoder model. Note that running the code will require changing the paths to the cochleagram data, and to the other .py files used. 

(2) "AE_architectures.py": file containing the recurrent autoencoder architecture. 

(3) "sound_dataset.py": PyTorch dataset class to obtain the training and validation cochleagram inputs.

(4) "get_STRFs.py": code to compute the spectrotemporal receptive fields (STRFs) of the autoencoder hidden units.

(5) "poisson_NN_regression_model.py" and "poisson_NN_regression_model_PCA.py": implementation of the Poisson LN and NN linking hypotheses to predict ferret A1 data for autoencoders and PCA-based models, respectively. 

(6) "dense_nn.py": file generating the 3-layer (NN) or 1-layer (LN) neural networks for the Poisson-based linking hypotheses. 

(7) "PCA_model.py". Convolution of the cochleagram matrix from Pennington & David (2023) with the principal components to obtain time-varying traces that can be used to predict the ferret A1 PSTHs.

(8) "prediction_correlation.py" and "prediction_correlation_PCA.py": computing prediction correlation scores from Poisson LN or NN linking hypotheses for autoencoders and PCA-based models, respectively. 

(9) "PCA_pennington_david.ipynb": get PCs from Pennington & David (2023) sounds (as a cochleagram). 
