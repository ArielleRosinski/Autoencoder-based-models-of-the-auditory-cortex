def get_spikes_est(file, cell_id=None):
    spikes_est = np.load(file).T    
    spikes_est = spikes_est[:,cell_id,np.newaxis]                   #(862000, 816 --> 1)
    spikes_est = spikes_est.reshape(-1, 20)                         #(43100, 20)
    spikes_est = torch.tensor(np.sum(spikes_est, axis=1))           #torch.Size([43100])
    spikes_est = spikes_est[:,None]                                 #torch.Size([43100, 1])
    return spikes_est

def get_spikes_val(file, cell_id=None):
    spikes_val = np.load(file)                                      #(20, 816, 26500)
    spikes_val = spikes_val[:,cell_id,np.newaxis,:]                 #(20, 1, 26500)           
    spikes_val = np.mean(spikes_val, axis=0).T                      #(26500, 1)         
    spikes_val = spikes_val.reshape(-1, 20)                         #(1325, 20)  
    spikes_val = torch.tensor(np.sum(spikes_val, axis=1))           #torch.Size([1325])
    spikes_val = spikes_val[:,None]                                 #torch.Size([1325, 1])
    return spikes_val

def get_spikes_val_single_pres(file, cell_id):
    spikes_val = np.load(file)                                      #(20, 816, 26500)
    spikes_val = spikes_val[:,cell_id,np.newaxis,:]                 #(20, 1, 26500)
    spikes_val = spikes_val.reshape(spikes_val.shape[0], -1, 20)
    spikes_val = np.sum(spikes_val, axis=2)                         #(20, 1325)
    return spikes_val
