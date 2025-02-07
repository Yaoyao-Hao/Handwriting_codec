from scipy.io import loadmat
import numpy as np
from sklearn import metrics
import os
from scipy import stats
import scipy.io as io

def fd_continuous_pos(arr):    #find segments of same positive value
    positive_indices = np.where(arr > 0)[0]
    segments = []
    start = positive_indices[0]
    for i in range(1, len(positive_indices)):
        if positive_indices[i] != positive_indices[i - 1] + 1:
            end = positive_indices[i - 1]
            segments.append((start, end))
            start = positive_indices[i]
    segments.append((start, positive_indices[-1]))

    return segments
def fd_continuous_neg(arr): #find segments of same negative value
    negative_indices  = np.where(arr < 0)[0]
    segments = []
    start = negative_indices [0]
    for i in range(1, len(negative_indices)):
        if negative_indices [i] != negative_indices [i - 1] + 1:
            end = negative_indices [i - 1]
            segments.append((start, end))
            start = negative_indices [i]
    segments.append((start, negative_indices [-1]))

    return segments

# bits/spike calculation for strokes of an example session

data = loadmat('data0701.mat')
bined_spk = data['bined_spk'] # neural data of the session
break_ind = data['break_ind']
trial_target = data['trial_target']
trial_mask = data['trial_mask']
# neurons with spike rate<1Hz were removed
firing_rates = bined_spk / 0.2
neuron_ind = np.where(np.mean(firing_rates,1) >= 1)[0]
bined_spk = bined_spk[neuron_ind,:]

segments = fd_continuous_pos(break_ind[0,:]) # find each stroke

encoding_results = ['Vxy.mat','Vxyz.mat','Vxy_grip.mat','Vxy_pres.mat','Vxy_emg.mat','full.mat']# encoding results of different models
bps = np.zeros((bined_spk.shape[0],6)) # neurons x models
i_file=0
for filename in encoding_results:
    data = loadmat(filename)
    lamda = data['lamda_pre']
    bps_file = np.zeros((bined_spk.shape[0],len(segments)))
    for neuron in range(bined_spk.shape[0]):
        for seg in range(len(segments)): # each stroke
            predict = lamda[neuron, segments[seg][0]:segments[seg][1] + 1] # predicted firing rates
            if (np.any(bined_spk[neuron,segments[seg][0]:
            segments[seg][1]+1]!=0))&(not np.isnan(predict).any()):
                predict_disc = np.floor(predict / 0.1) + 1 # discretization
                y_true = bined_spk[neuron,segments[seg][0]:segments[seg][1]+1] # true spike count
                mi = metrics.mutual_info_score(y_true, predict_disc) # mutual information
                bps_file[neuron,seg] = mi/len(predict_disc)/np.mean(y_true) # bits per spike
            else:
                bps_file[neuron,seg] = np.nan
    bps[:,i_file] = np.nanmean(bps_file,1) # mean bits per spike of each model
    i_file = i_file+1
