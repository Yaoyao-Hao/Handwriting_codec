# Neural encoding and decoding model for handwriting
## Overview
This repository contains the code for the paper Wang et al., 2025 "Cortical representation of multidimensional handwriting movement and implications for neuroprostheses".

## Installation
Set up a new environment and using the provided `requirements.txt` as follows:
```
pip install -r requirements.txt
```
This will install the key Python packages to run this code

## Usage
* **encoding model(fig4d,figS6).py:**
This module uses multidimensional handwriting data to predict single-neuron firing rates during imagined handwriting. We systematically test different combinations of multidimensional handwriting data as inputs:
```Python
dims = ['Vxy','Vxyz','Vxy_grip','Vxy_pres','Vxy_emg']
```
For both strokes and pen lifts, we fit a Linear-Nonlinear Poisson (LNP) model separately with 10-fold cross-validation strategy：
```Python
theta_lnp = fit_lnp(X_train, y_train, num_epochs, input_dim) # Fit LNP model
predict = predict_fr_lnp(X_test,input_dim, theta_lnp) # Test LNP model
```
* **full model-(fig4f,figS7).py:**
This module evaluates the uniqueness of each handwriting dimension in neural encoding by removing corresponding dimension from the full model. Firstly, train a full model using all dimensions of handwriting data：
```Python
theta_lnp = fit_lnp(X_train, y_train, num_epochs) # Fit the full model
```
Then, remove certain dimension of handwriting data as the inputs to encode single-neuron firing rates:
```Python
theta_lnp_delV = theta_lnp[:, 3:] # example：remove model weights for velocity
predict_delV = torch.exp(torch.sum(X_test_delV * theta_lnp_delV[0, :-1], dim=1) + theta_lnp_delV[0, -1]) # example：predict reults after removing handwriting velocity from the input
```
* **bits_per_spike.py:**
This module is used to assess the encoding performance in bits/spike as follows:
```Python
mi = metrics.mutual_info_score(y_true, predict_disc) # mutual information
bps_file[neuron,seg] = mi/len(predict_disc)/np.mean(y_true) # bits per spike
```
* **handwriting_decoding(fig5).py:**
This module decodes each subject's multidimensional handwriting data (velocity, grip force and pressure) from neural signals using LSTM networks, with separate decoder models trained for stroke and pen lift.
```Python
stroke_model = stroke_LSTM(input_dim=bined_spk_cv.shape[1], hidden_dim=16, num_layers=1, output_dim=4).to('cuda')
cohesion_model = cohesion_LSTM(input_dim=bined_spk_cv.shape[1], hidden_dim=32, num_layers=1, output_dim=4).to('cuda')
```
* **dtw_recognition(fig5).py:**
This module recognizes the decoded results as Chinese characters in the character library using the dynamic time warping (DTW) algorithm.
```Python
dist_vel2d,paths_vel2d = dtw_match(prediction_vel2d, [arr[[0, 1], :] for arr in interp_library]) # example: using decoded velocity of each character to recognize.
d, path = fastdtw(vel.T, database[i].T, dist=euclidean) # employ the fastdtw algorithm to compute the warping path between decoded results and all character templates in the character library.
match_target_vel2d = np.argmin(dist_vel2d)  # character with the minimum distance is the recognition result.
```


