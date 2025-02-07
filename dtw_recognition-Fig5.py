from scipy.io import loadmat
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import os
import scipy.io as io
from scipy.interpolate import interp1d
from scipy.stats import zscore

def create_database(character_num,data):
    trial_data = data['aver_data']
    trial_target = data['target_ind'][0]
    trial_data[2, np.where(trial_data[3, :] != 0)] = 0

    database = [None] * character_num
    for target in range(1, character_num+1):
        target_ind = np.where(trial_target == target)[0]
        database[target - 1] = trial_data[:,target_ind]
    return database

def dtw_match(vel,database):
    distance = np.zeros((190, 1))
    paths = []
    for i in range(len(database)):
        d, path = fastdtw(vel.T, database[i].T, dist=euclidean)
        distance[i, 0] = d
        paths.append(path)
    return distance,paths

# create character library
sessions = ['0419', '0614', '0616', '0623', '0624', '0630', '0701'] # sessions
character_num = [10, 30, 30, 30, 30, 30, 30] # characters in each session
library = []
for date, num in zip(sessions, character_num):
    data = loadmat(f'{date}_except6.mat') # average handwriting data except subject S6
    library_session = create_database(num, data)
    library.append(library_session)

library_total = library[0]
for db in library[1:]:
    library_total += db
# interpolation
interp_library = []
for array in library_total:
    interp_func = interp1d(np.linspace(0, 1, array.shape[1]), array, axis=1)
    new_array = interp_func(np.linspace(0, 1, 200))
    interp_library.append(new_array)
# z-score
interp_library = [zscore(arr, axis=1) for arr in interp_library]
interp_library[7][2,:]=0 #character library

decoding_file = ['0419_s6.mat', '0614_s6.mat', '0616_s6.mat', '0623_s6.mat',
                 '0624_s6.mat','0630_s6.mat','0701_s6.mat'] #decoding results of a subject in each session
target_file = ['data0419.mat', 'data0614.mat', 'data0616.mat',
               'data0623.mat','data0624.mat','data0630.mat',
               'data0701.mat'] # nerual data in each session

correct_rate = np.zeros((1,7))
i_file=0
rate_dtw = np.zeros((7,5))
Path_vel2d = []
Path_vel3d = []
Path_grip = []
Path_pres = []
Path_full = []
match_num_vel2d = np.zeros((640,1)) #640 trial
match_num_vel3d = np.zeros((640,1))
match_num_grip = np.zeros((640,1))
match_num_pres = np.zeros((640,1))
match_num_full = np.zeros((640,1))
trial_count = 0
for filename in decoding_file:
    count = 0
    data = loadmat(os.path.join('decoding_results', filename))
    prediction = data['prediction']

    data = loadmat(f'{target_file[i_file]}')
    trial_target = data['trial_target']
    trial_mask = data['trial_mask']
    print(target_file[i_file])
    if i_file==0:
        character_num=10
    else:
        character_num=30
    for character in range(1,character_num+1):
        if i_file == 0:
            match_index = character - 1 # labels
        else:
            match_index = 10 + (i_file - 1) * 30 + (character - 1)
        target_ind = np.where(trial_target == character)[0]
        for trial in range(target_ind.shape[0]):
            print(f'character: {character}, trial: {trial}')
            idx = np.where(trial_mask[0].reshape(-1,1) == (target_ind[trial]+1))[0]
            # decoding result interpolation (each trial)
            prediction_trial = prediction[:, idx]
            interp_func = interp1d(np.linspace(0, 1, prediction_trial.shape[1]), prediction_trial, axis=1)
            prediction_trial = interp_func(np.linspace(0, 1, 200))
            # z-score
            prediction_trial = zscore(prediction_trial, axis=1)
            if i_file==0 and character==8:
                prediction_trial[2,:]=0

            if not np.any(np.isnan(prediction_trial)):
                count = count+1
                prediction_vel2d = prediction_trial[:2,:]
                prediction_vel3d = prediction_trial[:3, :]
                prediction_grip = np.vstack((prediction_trial[:2, :], prediction_trial[3, :].reshape(1, -1)))
                prediction_pres = np.vstack((prediction_trial[:2, :], prediction_trial[4, :].reshape(1, -1)))

                # dtw_match return dtw_distance and dtw_path
                dist_vel2d,paths_vel2d = dtw_match(prediction_vel2d, [arr[[0, 1], :] for arr in interp_library]) # vel_2d for recognition
                dist_vel3d,paths_vel3d = dtw_match(prediction_vel3d, [arr[[0, 1, 2], :] for arr in interp_library]) # vel_3d for recognition
                dist_grip,paths_grip = dtw_match(prediction_grip, [arr[[0, 1, 4], :] for arr in interp_library]) # vel_3d+grip for recognition
                dist_pres,paths_pres = dtw_match(prediction_pres, [arr[[0, 1, 3], :] for arr in interp_library]) # vel_3d+pressure for recognition
                dist_full,paths_full = dtw_match(prediction_trial, interp_library) # all variables for recognition
                match_target_vel2d = np.argmin(dist_vel2d) # character corresponding to the shortest distance
                Path_vel2d.append(paths_vel2d[match_target_vel2d])
                match_target_vel3d = np.argmin(dist_vel3d)
                Path_vel3d.append(paths_vel3d[match_target_vel3d])
                match_target_vel2d_grip = np.argmin(dist_grip)
                Path_grip.append(paths_grip[match_target_vel2d_grip])
                match_target_vel2d_pres = np.argmin(dist_pres)
                Path_pres.append(paths_pres[match_target_vel2d_pres])
                match_target_full = np.argmin(dist_full)
                Path_full.append(paths_full[match_target_full])

                # check whether the recognition result is consistent with the label
                if match_target_vel2d == match_index:
                    rate_dtw[i_file, 0] += 1
                    match_num_vel2d[trial_count, 0] += 1
                if match_target_vel3d == match_index:
                    rate_dtw[i_file, 1] += 1
                    match_num_vel3d[trial_count, 0] += 1
                if match_target_vel2d_grip == match_index:
                    rate_dtw[i_file, 2] += 1
                    match_num_grip[trial_count, 0] += 1
                if match_target_vel2d_pres == match_index:
                    rate_dtw[i_file, 3] += 1
                    match_num_pres[trial_count, 0] += 1
                if match_target_full == match_index:
                    rate_dtw[i_file, 4] += 1
                    match_num_full[trial_count, 0] += 1
            print(f'trial_count: {trial_count}')
            trial_count = trial_count + 1
    rate_dtw[i_file, :] = rate_dtw[i_file, :] / count
    print(count)
    print(rate_dtw[i_file, :])
    i_file = i_file + 1

