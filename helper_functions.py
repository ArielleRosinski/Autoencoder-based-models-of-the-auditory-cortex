import numpy as np

def get_windows(files, kfold, fold_id, file_name, split_size=30, norm=True, training=True):
    ls_coch=[]
    ls_file=[]
    ls_file_name=[]
    for i,file in enumerate(files): 
        file=np.load(file)

        for fold, (train_ids, test_ids) in enumerate(kfold.split(file.T)):
            if fold==fold_id:
                if norm==True:
                    if file_name[i].endswith("coch.npy"):
                        file=file/2500
                    if file_name[i] == "full_pennington_david.npy":
                        file=file/1000
                    else: 
                        file=file/5000

                if training:
                    indices=train_ids
                else:
                    indices=test_ids
                
                file=file.T[indices].T
                ls_file.append(file)
    
                split_start=0
                split_end=split_start+split_size
                for j in range(file.shape[1]//split_size):
                    window=file[:,split_start:split_end]
                    split_start += split_size
                    split_end += split_size
                    ls_coch.append(window)
                         
    return ls_coch, ls_file

