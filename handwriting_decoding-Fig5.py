import numpy as np
from scipy.io import loadmat
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error as mse

def fd_continuous_pos(arr):
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
def fd_continuous_neg(arr):
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

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(0)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[-1, :, :]
        out = self.fc(out)
        return out

def norm_data(data):
    data_min = np.full((data.shape[0], 1), np.nan)
    data_max = np.full((data.shape[0], 1), np.nan)

    for i in range(data.shape[0]):
        data_min[i] = np.nanmin(data[i, :])
        data_max[i] = np.nanmax(data[i, :])
    data_norm = np.copy(data)

    for i in range(3):
        span = data_max[i] - data_min[i]
        if span == 0: span = 1e-6
        data_norm[i, :] = 2 * (data[i, :] - data_min[i]) / span - 1
    for i in range(3, 5):
        span = data_max[i] - data_min[i]
        if span == 0: span = 1e-6
        data_norm[i, :] = (data[i, :] - data_min[i]) / span

    return data_norm,data_min,data_max

def invert_norm_data(data_norm,data_min,data_max):
    data = np.copy(data_norm)

    for i in range(3):
        span = data_max[i] - data_min[i]
        if span == 0: span = 1e-6
        data[i, :] = (data_norm[i, :] + 1) / 2 * span + data_min[i]

    for i in range(3, 5):
        span = data_max[i] - data_min[i]
        if span == 0: span = 1e-6
        data[i, :] = data_norm[i, :] * span + data_min[i]

    return data

class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float))

    def forward(self, input, target):
        squared_error = (input - target)**2
        weighted_squared_error = squared_error * self.weights.cuda()
        return torch.mean(weighted_squared_error)

data = loadmat('0701_s6.mat') # data of one session for a single subject
Vxy = data['velocity_data'][0:2,:]
Vz = data['velocity_data'][2,:]
Fgrip = data['Fgrip_data']
Fpres = data['Fpres_data']
trial_target = data['trial_target'] # character index
trial_mask = data['trial_mask'] # trial index
bined_spk = data['bined_spk'] # neuron number x T
break_ind = data['break_ind'] # strokes,>0; cohesions,<0;

data = np.vstack((Vxy,Vz, Fgrip,Fpres))
prediction = np.zeros((data.shape[0],data.shape[1]))

lr = 1e-3

for character in range(1,31): # 30 characters in a session
    print(character)
    character_ind = np.where(trial_target == character)[0]
    test_indices = np.where(trial_mask[0].reshape(-1, 1) == (character_ind + 1))[0] #index for test
    train_indices = np.setdiff1d(np.arange(len(trial_mask[0])), test_indices) #index for training

    if np.isnan(data[:, test_indices]).any():
        continue
    else:
        # tr
        data_cv = np.delete(data, test_indices, 1)
        bined_spk_cv = np.delete(bined_spk, test_indices, 1)
        break_ind_cv = np.delete(break_ind, test_indices, 1)

        # normalization
        data_cv_norm, data_cv_min, data_cv_max = norm_data(data_cv)
        data_cv = data_cv_norm

        var_norm = np.nanvar(data_cv_norm, axis=1)
        weights = 1.0 / np.maximum(var_norm, 1e-8)
        criterion = WeightedMSELoss(weights=weights)

        bined_spk_cv = torch.tensor(bined_spk_cv.T, dtype=torch.float).to('cuda')
        data_cv = torch.tensor(data_cv.T, dtype=torch.float).to('cuda')

        diffs = np.diff(np.concatenate(([0], break_ind_cv[0], [0])))
        startIdx = np.where(diffs != 0)[0][:-1]
        endIdx = startIdx[1:] - 1
        endIdx = np.append(endIdx, break_ind_cv.shape[1] - 1)

        single_model = LSTM(input_dim=bined_spk_cv.shape[1], hidden_dim=32, num_layers=1, output_dim=5).to('cuda')
        optimizer = optim.SGD(single_model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

        for epoch in range(20):
            losses = 0
            single_model.train()
            for stroke in range(startIdx.shape[0]):
                inputs = bined_spk_cv[startIdx[stroke]:endIdx[stroke] + 1, :]
                targets = data_cv[startIdx[stroke]:endIdx[stroke] + 1, :]
                if not torch.isnan(targets).any():
                    optimizer.zero_grad()
                    outputs = single_model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    losses += loss
                    optimizer.step()
            scheduler.step()
            # val
            single_model.eval()
            with torch.no_grad():
                val_losses = 0
                for i_trial in character_ind:
                    trial_indices = np.where(trial_mask[0].reshape(-1, 1) == (i_trial + 1))[0]
                    spk_target = torch.tensor(bined_spk[:, trial_indices], dtype=torch.float).to('cuda')
                    data_target = data[:, trial_indices]
                    data_target_std = (data_target - data_cv_min) / (data_cv_max - data_cv_min)
                    data_target_std = torch.tensor(data_target_std, dtype=torch.float).to('cuda')

                    stroke_predict = single_model(spk_target.permute(1, 0))
                    loss = criterion(stroke_predict, data_target_std.permute(1, 0))
                    val_losses += loss

            print(f'Epoch [{epoch + 1}/{20}], Loss: {losses.item() / 87:.4f}, val_Loss: {val_losses.item() / character_ind.shape[0]:.4f},'
                f'LR: {scheduler.get_last_lr()[0]}')
            # test
            single_model.eval()
            with torch.no_grad():
                for i_trial in character_ind:
                    trial_indices = np.where(trial_mask[0].reshape(-1, 1) == (i_trial + 1))[0]
                    spk_target = torch.tensor(bined_spk[:, trial_indices], dtype=torch.float).to('cuda')

                    stroke_predict_std = single_model(spk_target.permute(1, 0))
                    stroke_predict_std = stroke_predict_std.cpu().numpy().T
                    stroke_pred_ori = invert_norm_data(stroke_predict_std, data_cv_min, data_cv_max)

                    prediction[:, trial_indices] = stroke_pred_ori
                    print('trial: ', i_trial + 1)