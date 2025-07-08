# Handwriting_code
## Overview
This repository contains the code for the paper "Cortical representation of multidimensional handwriting movement" (unpublished).

## Installation
Clone this repository, change into the project directory, and install with 'pip install .'. Feel free to contact us if you encounter any installation issues.

## Usage
Run 'encoding model(fig4d,figS6).py' to encode multidimensional handwriting data into neural signals via linear-nonlinear Poisson (LNP) model.

Run 'full model-(fig4f,figS7).py' to evaluate the uniqueness of each handwriting dimension in neural encoding by removing corresponding dimension from the full model.

Run 'bits_per_spike.py' to assess the encoding performance in bits/spike.

Run 'handwriting_decoding(fig5).py' to decode multidimensional handwriting data from neural signals via LSTM.

Run 'dtw_recognition(fig5).py' to recognize the decoded results as Chinese characters in the library using the dynamic time warping (DTW) algorithm.


