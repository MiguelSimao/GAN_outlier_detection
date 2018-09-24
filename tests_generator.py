#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests_generator.py

Script to run the GAN generator tests for the UC2018 DualMyo data set.

Author: Miguel Simão (miguel.simao@uc.pt)
"""

# imports
import numpy as np
import pickle

from sklearn import preprocessing
import matplotlib.pyplot as plt
from metrics import ganmetrics

# ENSURE REPRODUCIBILITY ######################################################
#Python/Numpy
import os
import random
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1337)
random.seed(12345)
#TF
#import tensorflow as tf
#from keras import backend as K
#session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
#                              inter_op_parallelism_threads=1)
#tf.set_random_seed(123)
#sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#K.set_session(sess)
###############################################################################


#%% Load dataset

with open('./data/dualmyo_generated_noise.pkl','rb') as file:
    data = pickle.load(file)

    
    
#%% Process dataset

# Load data
X_train  = data['X_train']
t_train  = data['t_train']
X_val    = data['X_val']
t_val    = data['t_val']
X_test   = data['X_test']
t_test   = data['t_test']
X_others = data['X_others']
t_others = data['t_others']

X_train_gen = data['X_train_gen']
t_train_gen = data['t_train_gen']
o_train_gen = np.zeros((X_train_gen.shape[0],1))
X_train = np.concatenate((X_train, X_train_gen), 0)
t_train = np.concatenate((t_train, t_train_gen), 0)

t_train_ind = np.argmax(t_train,1)
t_train_gen_ind = np.argmax(t_train_gen,1)

# Generate some test noise (by class)
scaler = preprocessing.StandardScaler().fit(X_train)
X_noise = np.zeros_like(X_train_gen)
t_noise = t_train_gen
t_noise_ind = t_noise.argmax(1)
for i in range(t_noise_ind.max()):
    x = X_noise[t_noise_ind==i]
    mean, std = x.mean(0), x.std(0)
    n = x.shape[0]
    X_noise[t_noise_ind==i] = np.random.normal(mean, std, x.shape)
    
#%% GENERATION L2-DISTANCE CALCULATION

# Baseline (train set)
similarity_base = ganmetrics.class_l2_norm(X_train, t_train_ind, X_train, t_train_ind)

# Baseline (gaussian noise)
similarity_noise = ganmetrics.class_l2_norm(X_train, t_train_ind, X_noise, t_noise_ind)

# Generated set / real set distance
similarity_gen = ganmetrics.class_l2_norm(X_train, t_train_ind, X_train_gen, t_train_gen_ind)
# Generated set distance
similarity_gen[:,2] = ganmetrics.class_l2_norm(
        X_train_gen, t_train_gen_ind, X_train_gen, t_train_gen_ind)[:,2]


# Print results (CSV)
print(',Baseline,,GAN,,GaussianNoise,')
print('Class,Mean,Std,Mean,Std,Mean,Std')
for i in range(len(similarity_base)):
    tmp = np.hstack((similarity_base[i],similarity_gen[i,1:], similarity_noise[i,1:]))
    print('%i,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f' % tuple(tmp))
    
#%% VISUALIZATION

# Sensor order
def myoind():
    myo0, myo1 = np.arange(8), np.arange(8,16)
    sensor_order = np.empty((myo0.size+myo1.size), dtype=np.int)
    sensor_order[0::2] = myo0
    sensor_order[1::2] = myo1
    return sensor_order

num_examples = 32 # to draw

# Training samples targets:
t_train_ind = np.argmax(t_train, axis=1)

# Select real examples to show:
vis_X = []
for i in range(t_train_ind.max() + 1):
    I = np.argwhere(t_train_ind==i)[:num_examples].squeeze()
    vis_X.append(X_train[I])
# "Other" class examples (all zeros)
vis_X.append(np.zeros_like(vis_X[0]))

# Generate samples:
vis_X_gen = [] #gen'ed storage
# for each class
for i in range(7):
    I = np.argwhere(t_train_gen_ind==i)[:num_examples].squeeze()
    vis_X_gen.append(X_train_gen[I])

# Scale data for representation:
scaler = preprocessing.MinMaxScaler().fit(np.concatenate(vis_X_gen+vis_X))
for i in range(7):
    vis_X[i] = scaler.transform(vis_X[i])
    vis_X_gen[i] = scaler.transform(vis_X_gen[i])
vis_X[-1] = np.zeros_like(vis_X[0])

fig, axs = plt.subplots(nrows=2, ncols=7,
                        sharex=True, sharey=True,
                        figsize=(5.7, 3))

for row_ind in range(axs.shape[0]):
    for col_ind in range(axs.shape[1]):
        plt.axes(axs[row_ind,col_ind])
        if col_ind == 0:
            if row_ind == 0:
                plt.ylabel('Real')
            else:
                plt.ylabel('Generated')
        if row_ind == 1:
            plt.xlabel('G%i' % col_ind)
            axs[row_ind,col_ind].imshow(vis_X_gen[col_ind][:,myoind()])
        else:
            axs[row_ind,col_ind].imshow(vis_X[col_ind][:,myoind()])


