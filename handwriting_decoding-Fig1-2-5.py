import numpy as np
from scipy.io import loadmat
import torch.optim as optim
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error as mse

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

class stroke_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(stroke_LSTM, self).__init__()
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
class cohesion_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(cohesion_LSTM, self).__init__()
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


data = loadmat('0701_s6.mat') # data of one session for a single subject
Vxy = data['velocity_data'][0:2,:]
Vz = data['velocity_data'][2,:]
Fgrip = data['Fgrip_data']
Fpres = data['Fpres_data']
trial_target = data['trial_target'] # character index
trial_mask = data['trial_mask'] # trial index
bined_spk = data['bined_spk'] # neuron number x T
break_ind = data['break_ind'] # strokes,>0; cohesions,<0;

stroke_data = np.vstack((Vxy,Fgrip,Fpres))
cohesion_data =  np.vstack((Vxy,Vz,Fgrip))

prediction = np.zeros((stroke_data.shape[0]+1,stroke_data.shape[1]))

batch_size = 256
criterion = nn.MSELoss()
lr = 1e-3

for character in range(1,31): # 30 characters in a session
    print(character)
    character_ind = np.where(trial_target == character)[0]
    test_indices = np.where(trial_mask[0].reshape(-1, 1) == (character_ind + 1))[0] #index for test
    train_indices = np.setdiff1d(np.arange(len(trial_mask[0])), test_indices) #index for training

    if np.isnan(stroke_data[:,test_indices]).any(): # skip incorrect stroks/cohesions
        continue
    else:
        # train
        stroke_data_cv = np.delete(stroke_data, test_indices, 1)
        cohesion_data_cv = np.delete(cohesion_data, test_indices, 1)
        bined_spk_cv = np.delete(bined_spk, test_indices, 1)
        break_ind_cv = np.delete(break_ind, test_indices, 1)

        bined_spk_cv = torch.tensor(bined_spk_cv.T,dtype=torch.float).to('cuda')
        stroke_data_cv = torch.tensor(stroke_data_cv.T, dtype=torch.float).to('cuda')
        cohesion_data_cv = torch.tensor(cohesion_data_cv.T, dtype=torch.float).to('cuda')

        for i_stroke in range(2):
            if i_stroke==0: #stroke
                print('stroke model')
                # stroke-model training
                diffs = np.diff(np.concatenate(([0], break_ind_cv[0] > 0, [0])))
                startIdx = np.where(diffs == 1)[0] # start index for each training stroke
                endIdx = np.where(diffs == -1)[0] - 1 # end index for each training stroke

                stroke_model = stroke_LSTM(input_dim=bined_spk_cv.shape[1], hidden_dim=16, num_layers=1, output_dim=4).to('cuda')
                optimizer = optim.SGD(stroke_model.parameters(), lr=lr)
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.01)

                for epoch in range(20):
                    losses = 0
                    stroke_model.train()
                    for stroke in range(startIdx.shape[0]):
                        inputs = bined_spk_cv[startIdx[stroke]:endIdx[stroke]+1,:]
                        targets = stroke_data_cv[startIdx[stroke]:endIdx[stroke]+1,:]
                        if not torch.isnan(targets).any():
                            optimizer.zero_grad()
                            outputs = stroke_model(inputs)
                            loss = criterion(outputs, targets)
                            loss.backward()
                            losses += loss
                            optimizer.step()
                    scheduler.step()
                    # stroke-model evaluation
                    stroke_model.eval()
                    with torch.no_grad():
                        val_losses = 0
                        stroke_num = 0
                        for i_ind in range(character_ind.shape[0]):
                            character_idx = np.where(trial_mask[0].reshape(-1, 1) == (character_ind[i_ind] + 1))[0]
                            spk_character = torch.tensor(bined_spk[:, character_idx], dtype=torch.float).to('cuda')
                            stroke_data_target = torch.tensor(stroke_data[:, character_idx], dtype=torch.float).to('cuda')

                            stroke_seg = fd_continuous_pos(break_ind[0, character_idx])
                            stroke_num += len(stroke_seg)
                            for stroke in range(len(stroke_seg)):
                                stroke_predict = stroke_model(spk_character[:, stroke_seg[stroke][0]:stroke_seg[stroke][1] + 1].permute(1, 0))
                                loss = criterion(stroke_predict, stroke_data_target[:, stroke_seg[stroke][0]:stroke_seg[stroke][1] + 1].permute(1, 0))
                                val_losses += loss

                    print(f'Epoch [{epoch + 1}/{20}], Loss: {losses.item() / startIdx.shape[0]:.4f}, val_Loss: {val_losses.item() / stroke_num:.4f},'
                          f'LR: {scheduler.get_last_lr()[0]}')

            else:
                print('cohesion model')
                # cohesion-model training
                diffs = np.diff(np.concatenate(([0], break_ind_cv[0] < 0, [0])))
                startIdx = np.where(diffs == 1)[0] # start index for each training cohesion
                endIdx = np.where(diffs == -1)[0] - 1 # end index for each training cohesion

                cohesion_model = cohesion_LSTM(input_dim=bined_spk_cv.shape[1], hidden_dim=32, num_layers=1, output_dim=4).to('cuda')
                optimizer = optim.SGD(cohesion_model.parameters(), lr=lr)
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.01)

                for epoch in range(20):
                    losses = 0
                    cohesion_model.train()
                    for cohesion in range(startIdx.shape[0]):
                        inputs = bined_spk_cv[startIdx[cohesion]:endIdx[cohesion] + 1, :]
                        targets = cohesion_data_cv[startIdx[cohesion]:endIdx[cohesion] + 1, :]
                        if not torch.isnan(targets).any():
                            optimizer.zero_grad()
                            outputs = cohesion_model(inputs)
                            loss = criterion(outputs, targets)
                            loss.backward()
                            losses += loss
                            optimizer.step()
                    scheduler.step()
                    # cohesion-model evaluation
                    cohesion_model.eval()
                    with torch.no_grad():
                        val_losses = 0
                        cohesion_num = 0
                        for i_ind in range(character_ind.shape[0]):
                            character_idx = np.where(trial_mask[0].reshape(-1, 1) == (character_ind[i_ind] + 1))[0]
                            spk_character = torch.tensor(bined_spk[:, character_idx], dtype=torch.float).to('cuda')
                            cohesion_data_target = torch.tensor(cohesion_data[:, character_idx], dtype=torch.float).to('cuda')

                            cohesion_seg = fd_continuous_neg(break_ind[0, character_idx])
                            cohesion_num += len(cohesion_seg)
                            for cohesion in range(len(cohesion_seg)):
                                cohesion_predict = cohesion_model(spk_character[:, cohesion_seg[cohesion][0]:cohesion_seg[cohesion][1] + 1].permute(1, 0))
                                loss = criterion(cohesion_predict,cohesion_data_target[:, cohesion_seg[cohesion][0]:cohesion_seg[cohesion][1] + 1].permute(1, 0))
                                val_losses += loss

                    print(f'Epoch [{epoch + 1}/{20}], Loss: {losses.item() / startIdx.shape[0]:.4f}, val_Loss: {val_losses.item() / cohesion_num:.4f},'
                        f'LR: {scheduler.get_last_lr()[0]}')
        # test
        stroke_model.eval()
        cohesion_model.eval()
        with torch.no_grad():
            for i_ind in range(character_ind.shape[0]):
                character_idx = np.where(trial_mask[0].reshape(-1,1) == (character_ind[i_ind]+1))[0]
                spk_character = torch.tensor(bined_spk[:,character_idx],dtype=torch.float).to('cuda')
                predict_character = np.zeros((5,spk_character.shape[1]))
                # stroke
                stroke_seg = fd_continuous_pos(break_ind[0,character_idx])
                for stroke in range(len(stroke_seg)):
                    stroke_predict = stroke_model(spk_character[:,stroke_seg[stroke][0]:stroke_seg[stroke][1]+1].permute(1,0))
                    predict_character[[0,1,3,4],stroke_seg[stroke][0]:stroke_seg[stroke][1]+1] = stroke_predict.cpu().numpy().T
                # cohesion
                cohesion_seg = fd_continuous_neg(break_ind[0,character_idx])
                for cohesion in range(len(cohesion_seg)):
                    cohesion_predict = cohesion_model(spk_character[:, cohesion_seg[cohesion][0]:cohesion_seg[cohesion][1] + 1].permute(1, 0))
                    predict_character[[0,1,2,3],cohesion_seg[cohesion][0]:cohesion_seg[cohesion][1] + 1] = cohesion_predict.cpu().numpy().T

                prediction[:,character_idx] = predict_character
                print('trial: ',character_ind[i_ind]+1)

