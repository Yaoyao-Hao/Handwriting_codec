import time
from scipy.io import loadmat
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import math
import scipy.io as io
from scipy.interpolate import interp1d
import torch
from sklearn import preprocessing


def find_samesegments(arr):
  segments = []
  n = len(arr)
  if n == 0:
    return segments
  start_idx = 0
  for i in range(1, n):
    if arr[i] != arr[start_idx]:
      segments.append((start_idx, i - 1))
      start_idx = i
  segments.append((start_idx, n - 1))

  return segments
def cal_spike_counts(predict):
  lambdas = predict.detach().cpu().numpy()
  spike_count = np.empty_like(lambdas)
  for i, lmbda_val in enumerate(lambdas):
    if np.isnan(lmbda_val):
      spike_count[i] = np.nan
    else:
      spike_count[i] = np.random.poisson(lmbda_val)
  spike_count = torch.tensor(spike_count, dtype=torch.float).to('cuda')

  return spike_count

def fit_lnp(variables, spikes,num_epochs,input_dim):
  dataset = TensorDataset(variables, spikes)
  batch_size = 256
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  # Use a random vector of weights to start (mean 0, sd .1)
  w = torch.normal(0, 0.1, (1,input_dim+1)).to('cuda').requires_grad_(True)
  optimizer = torch.optim.SGD([w], lr=1e-3)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
  for epoch in range(num_epochs):
    start = time.time()
    total_log_lik = 0
    for X, y in dataloader:
      rate = torch.exp(torch.sum(X * w[0,0:input_dim],dim=1) + w[0,-1]).float()
      # Compute the Poisson log likelihood
      log_lik = -(torch.unsqueeze(y, 0) @ torch.log(rate) - rate.sum())
      optimizer.zero_grad()
      log_lik.backward()
      optimizer.step()
      total_log_lik += log_lik.item()
    scheduler.step()

    avg_loss = total_log_lik / len(dataloader)
    end = time.time()
    print(f"Epoch {epoch + 1}/{num_epochs}, log_lik: {avg_loss:.4f}, time:{end - start:.4f}s")
  return w

def predict_fr_lnp(variables, input_dim, theta=None):
  yhat = torch.exp(torch.sum(variables * theta[0,0:input_dim],dim=1) + theta[0,-1])
  return yhat

# encoding of an example session
data = loadmat('data0623.mat')
bined_spk = data['bined_spk'] # neural data
break_ind = data['break_ind']
trial_mask = data['trial_mask']
trial_target = data['trial_target']
velocity_xy = data['velocity'][:2,:]  # average handwriting data
velocity_z = data['velocity'][2,:]
Fgrip = data['Fgrip']/1000*9.8 # g -> N
Fpres = data['Fpres']/1000*9.8

data = loadmat('0623_s1.mat')
emg = data['emg_data']
# neurons with spike rate<1Hz were removed
firing_rates = bined_spk / 0.2
neuron_ind = np.where(np.mean(firing_rates,1) >= 1)[0]
bined_spk = bined_spk[neuron_ind,:]

num_epochs = 70

stroke_ind = np.where(break_ind[0]>0)
velocity_z[stroke_ind[0]] = 0
# velocity normalization
v_max = np.nanmax(velocity_xy)
v_min = np.nanmin(velocity_xy)
for i in range(velocity_xy.shape[0]):
  for j in range(velocity_xy.shape[1]):
    velocity_xy[i][j] = (2*(velocity_xy[i][j]-v_min)/(v_max-v_min))-1 #[-1,1]

z_scale = max(np.abs(np.nanmin(velocity_z)),np.abs(np.nanmax(velocity_z)))
velocity_z = velocity_z/z_scale
velocity = torch.tensor(np.vstack([velocity_xy, velocity_z])).to('cuda')
bined_spk = torch.tensor(bined_spk,dtype=torch.float).to('cuda')
# grip force and pressure normalization
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
Fgrip = (min_max_scaler.fit_transform(Fgrip.T)).T
Fpres = (min_max_scaler.fit_transform(Fpres.T)).T
# emg normalization
emg_max = np.max(emg)
emg_min = np.min(emg)
for i in range(emg.shape[0]):
  for j in range(emg.shape[1]):
    emg[i][j] = (emg[i][j]-emg_min)/(emg_max-emg_min)

full_data = torch.cat((velocity,torch.tensor(Fgrip).to('cuda'),torch.tensor(Fpres).to('cuda'),torch.tensor(emg).to('cuda')),dim=0)
dims = ['Vxy','Vxyz','Vxy_grip','Vxy_pres','Vxy_emg']

for encoding_dim in dims:  # add new input dims on Vxy
  if encoding_dim=='Vxy':
    handwriting_data = full_data[:2,:]
    input_dim = 2
  elif encoding_dim=='Vxyz':
    handwriting_data = full_data[:3, :]
    input_dim = 3
  elif encoding_dim=='Vxy_grip':
    handwriting_data = full_data[[0,1,3], :]
    input_dim = 3
  elif encoding_dim=='Vxy_pres':
    handwriting_data = full_data[[0,1,4], :]
    input_dim = 3
  else:
    handwriting_data = torch.cat((full_data[:2,:], full_data[5,:]), dim=0)
    input_dim = 10

  for i_stroke in range(2):
    if i_stroke == 0:
      ind = np.where(break_ind[0] < 0) # cohesion ind
      print('cohesion')
    else:
      ind = np.where(break_ind[0] > 0)  # stroke ind
      print('stroke')

    lamda_pre = torch.zeros((bined_spk.shape[0], bined_spk.shape[1]), dtype=torch.double).to('cuda')
    spk_predict = torch.zeros((bined_spk.shape[0], bined_spk.shape[1]), dtype=torch.float).to('cuda')
    # encode each neuron in turn
    for channel_num in range(bined_spk.shape[0]):
      bined_spk_channelnum = bined_spk[channel_num, :] # each neuron

      numbers = list(range(1, 31))  # 30 characters in this session
      for count in range(10):  # 10 fold
        print(f'neuron:{channel_num + 1},fold:{count + 1}')
        test_numbers = np.array(numbers[:3])
        del numbers[:3]
        character_ind = np.where(trial_target == test_numbers)[0]
        test_indices = np.where(trial_mask[0].reshape(-1, 1) == (character_ind + 1))[0] # character index for test
        tr_target_indices = np.setdiff1d(np.arange(len(trial_mask[0])), test_indices) # character index for training

        test_indices = np.intersect1d(ind, test_indices) # stroke/cohesion index for test
        tr_target_indices = np.intersect1d(ind, tr_target_indices) # stroke/cohesion index for training

        tr_idx = []
        seg_tr = find_samesegments(break_ind[0, tr_target_indices])
        for i_seg in range(len(seg_tr)):
          seg_s = bined_spk_channelnum.T[tr_target_indices[seg_tr[i_seg][0]]:tr_target_indices[seg_tr[i_seg][1]] + 1]
          if torch.any(seg_s.ne(0)):
            tr_idx.append(tr_target_indices[seg_tr[i_seg][0]:seg_tr[i_seg][1] + 1])
        tr_idx = np.concatenate(tr_idx)

        X_test, y_test = handwriting_data.T[test_indices], bined_spk_channelnum[test_indices] # test data
        X_train, y_train = handwriting_data.T[tr_idx], bined_spk_channelnum[tr_idx]   # training data

        # Fit LNP model
        theta_lnp = fit_lnp(X_train, y_train, num_epochs, input_dim)

        predict = predict_fr_lnp(X_test,input_dim, theta_lnp)  ## spike rate
        lamda_pre[channel_num, test_indices] = predict
        # spike counts
        spike_count = cal_spike_counts(predict)
        spk_predict[channel_num, test_indices] = spike_count
      # save results
      if i_stroke == 0:
        io.savemat(f'cohesion/{encoding_dim}.mat', {'spk_predict': spk_predict.detach().cpu().numpy(),
                                                     'lamda_pre': lamda_pre.detach().cpu().numpy(),})
      else:
        io.savemat(f'stroke/{encoding_dim}.mat', {'spk_predict': spk_predict.detach().cpu().numpy(),
                                                   'lamda_pre': lamda_pre.detach().cpu().numpy()})

