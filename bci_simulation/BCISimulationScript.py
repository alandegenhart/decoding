#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BCI simulation analysis script

This script trains and analyzes a recurrent neural network designed to
simulate (simplified) BCI cursor trajectories similar to those observed in the
energy landscape experiments.

@author: alandegenhart
"""

#%% --- Setup ---

# Import standard modules (some of these may be unused)
import sys
import os
import datetime
import numpy as np
import pandas as pd
import copy

# Import numerical libraries
import sklearn as skl
from sklearn import linear_model, metrics
import tensorflow as tf
from tensorflow import keras
print('Tensorflow version: {}'.format(tf.__version__))

# Set up plotting
import matplotlib as mpl
import matplotlib.pyplot as plt

# Setup paths and import custom modules
modules_dir = '/Users/alandegenhart/Documents/GitHub/python/modules/'
sys.path.append(modules_dir)
import plottools as pt

# Import any local modules (in the local directory)
import bci_simulation as bci

# Setup autoreload
%reload_ext autoreload
%autoreload 2

#%% --- Specify simulation parameters ---

n_seq = 100  # Number of sequences to generate
n_rnn = 128
n_epochs = 2000

#%% --- Generate training data ---

train_data = bci.gen_training_ex(n_seq)
fh = bci.plot_training_ex(train_data, 'Simulated training data')
fig_name = 'SimulatedData'
fh.savefig('results/BCISimulation_{}.pdf'.format(fig_name))

#%% --- Build the model, test predictions ---

# Build the model
n_inputs = train_data['X'].shape[2]
model = bci.build_model(n_inputs, n_rnn, use_regularizer=True)
model.build()
model.summary()

# Now try running some test data through to get the output
Y_predict = model.predict(train_data['X'])
test_data = copy.deepcopy(train_data)
test_data['Y'] = Y_predict

fh = bci.plot_training_ex(test_data, 'Predicted - pre-training')
fig_name = 'Predicted_PreTraining'
fh.savefig('results/BCISimulation_{}.pdf'.format(fig_name))

#%% --- Fit the model, plot training/validation error, save model weights ---

# Fit the model
history = model.fit(
    train_data['X'], train_data['Y'],
    epochs=n_epochs, validation_split=0.2, verbose=0,
    callbacks=[bci.PrintDot()])

# Plot training and validation error
fh = bci.plot_history(history)
plt.savefig('results/TrainError.pdf')

# Save model weights
save_name = './models/bci_sim_weights_units_{}_epochs_{}_trials_{}'.format(
    n_rnn, n_epochs, n_seq)
model.save(save_name)

#%% --- Load saved model (if desired) ---

save_name = './models/bci_sim_weights_units_{}_epochs_{}_trials_{}'.format(
    n_rnn, n_epochs, n_seq)
model = tf.keras.models.load_model(save_name)
model.summary()

#%% --- Plot training results ---

# Plot predictions for training data
Y_predict = model.predict(train_data['X'])
test_data = copy.deepcopy(train_data)
test_data['Y'] = Y_predict

fh = bci.plot_training_ex(test_data, 'Predicted - post-training')
fig_name = 'Predicted_PostTraining'
plt.savefig('results/BCISimulation_{}.pdf'.format(fig_name))

#%% --- Run PCA analysis ---

bci.pca_analysis(model, test_data)

# %%
""" Notes (2020.03.18):

1. The simulated network does a reasonable job of reproducing the desired
   cursor trajectories.  However, they're not perfect.  This is probably due
   to a limited amount of training data (only 100 unique trials).  The
   simulation could be updated to generate many more trials, possibly with
   randomized start and end positions.

2. The PCA analysis shows that the dimensionality of the network is only ~2D.
   This is in stark constrast to real data, which is closer to 5-10D.  It is
   interesting to consider what components might be missing here -- 
   specifically, there is not a "condition-invariant" response.  Are there
   certains types of inputs that might be required to generate such a response?
   For instance, a global "go" signal might do this.  Here, such a signal is
   not necessary, as the task timing can be completely inferred from the input
   and target.  However, this need not be the case.

3. Even though there is little variance captured by the higher dimensions of
   the hidden units, this activity still does show "asymmetry-like" activity.
   This is encouraging.  It might be worth implementing the steifel
   optimization here and explicitly looking for rotated-mapping-like
   projections.

To-do:
- Update simulation to generate more trials, with random start/end positions.
  This data can be used to train the network.  We can then use a second set of
  "canonical" trajectories to look at the behavior of the network.

- Port over stiefel optimization code from MATLAB

- Use stiefel optimization w/ simplified objective function to identify
  asymmetries in the simulatied data.  Note that the current optimization uses
  means for each condition (with the exception of the variance term), so it
  *should* be fairly straightforward to modify this to use the canonical
  trajectories used in the simulated model.
"""
