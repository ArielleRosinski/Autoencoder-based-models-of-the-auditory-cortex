import numpy as np
from scipy.signal import convolve


#Get Pennington & David (2023) sounds
full_sound_input_normalized = np.load("/Users/ariellerosinski/My Drive/UCL/MSc/Project/Code/full_coch/full_pennington_david.npy")/1000
full_sound_input_normalized_reshaped = full_sound_input_normalized[:,:-25]

#Get principal component matrix (obtained using sklearn.decomposition.PCA) and keep only first 128
pc_matrix = np.load("/Users/ariellerosinski/My Drive/UCL/MSc/Project/Runs/pennington_david/pennington_david_full_dataset/Images/PCA/PCs_30bins.npy") 
pc_matrix_filt = pc_matrix[:128,:,:]

#Convolve PCs with the Pennington & David (2023) sound matrix to get activity over time 
ls_act = []
for i in range(pc_matrix_filt.shape[0]):
    filter = pc_matrix_filt[i,:,:]  
    result = convolve(full_sound_input_normalized_reshaped, filter, mode='valid')
    ls_act.append(result)

pc_pred = np.vstack(ls_act)
np.save("/Users/ariellerosinski/My Drive/UCL/MSc/Project/Runs/pennington_david/pennington_david_full_dataset/Images/PCA/Convolution/pc_pred", pc_pred)