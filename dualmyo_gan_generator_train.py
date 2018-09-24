#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 17:19:33 2018

@author: simao
"""

import numpy as np
import time
import pickle
import matplotlib.pyplot as plt

from dataset.dualmyo.utils import Loader, SynteticSequences
from dataset.dualmyo import dualmyofeatures
from tools import toolstimeseries as tts
from tools import toolsfeatures
from tools.postprocessing import PostProcessor

from sklearn import preprocessing

# ENSURE REPRODUCIBILITY ######################################################
import os
import random
import tensorflow as tf
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1337)
random.seed(12345)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

tf.set_random_seed(123)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
###############################################################################

from classifiers import ACGAN
from metrics import ganmetrics
import keras
from keras import Model
from keras.optimizers import Adam
import keras.layers as kls

#%% LOAD DATA

# Use the loading util class to load from the default path
DataLoader = Loader()
sample_data, sample_target = DataLoader.load()
# List to np.array
sample_data = np.concatenate( [sample.reshape((1,) + sample.shape) for sample in sample_data], axis=0 )
sample_target = np.array(sample_target)
# Set class 7 aside
ind_7 = np.argwhere(sample_target == 7)
sample_data_7 = sample_data[ind_7].squeeze()
sample_target_7 = sample_target[ind_7]
sample_data = np.delete(sample_data,ind_7,0)
sample_target = np.delete(sample_target,ind_7,0)

# Dataset split
ind_train, ind_val, ind_test = DataLoader.split(sample_target)

#%% FEATURE EXTRACTION

# Feature extraction
X_train = np.vstack([dualmyofeatures.extract_std(sample) for sample in sample_data[ind_train]])
X_val = np.vstack([dualmyofeatures.extract_std(sample) for sample in sample_data[ind_val]])
X_test = np.vstack([dualmyofeatures.extract_std(sample) for sample in sample_data[ind_test]])
X_test_7 = np.vstack([dualmyofeatures.extract_std(sample) for sample in sample_data_7])

# Feature scaling
feature_scaler = preprocessing.StandardScaler().fit(X_train)
X_train = feature_scaler.transform(X_train)
X_val = feature_scaler.transform(X_val)
X_test = feature_scaler.transform(X_test)
X_test_7 = feature_scaler.transform(X_test_7)


# Target processing
t_train = sample_target[ind_train]
t_val = sample_target[ind_val]
t_test = sample_target[ind_test]
t_test_7 = sample_target_7.squeeze()
t_train = toolsfeatures.onehotencoder(t_train, 8) # consider the 8th class to be "others"
t_val = toolsfeatures.onehotencoder(t_val, 8)
t_test = toolsfeatures.onehotencoder(t_test, 8)
t_test_7 = toolsfeatures.onehotencoder(t_test_7, 8)

# Set data backups aside
X_train_bak, X_val_bak, X_test_bak = X_train.copy(), X_val.copy(), X_test.copy()
t_train_bak, t_val_bak, t_test_bak = t_train.copy(), t_val.copy(), t_test.copy()


#%% GAN MODEL INSTANCING

# Add noise to discriminator training ?
par_noise = True
par_save = True

batch_size = 64
d_lr = 0.0004
g_lr = 0.0010
g_loss_w = [1.3, .8]
epochs = 300
runid = 0

# Noise:
#if par_noise:
#    gan = ACGAN(batch_size=64, num_classes=7, gesture_size=16, latent_dim=8, label_noise=[.9,1.0])
#else:
#    gan = ACGAN(batch_size=64, num_classes=7, gesture_size=16, latent_dim=8)
gan = ACGAN(batch_size=batch_size, num_classes=7, gesture_size=16, latent_dim=8, 
            label_noise=[.9,1.0], d_lr=.0002, g_lr=.001, g_loss_weights=[1.3, 0.8])

    
if par_save:
    gan.discriminator.save('./nets/dualmyo_untrained_discriminator.h5')

#%% ----- TRAIN ----- #
plt.ioff()
try:
    time_start = time.time()
    history = gan.fit(X_train, t_train,
                      batch_size=32,
                      epochs=epochs,
                      validation_data=(X_val, t_val),
                      plot=50,
                      runid=runid)
except KeyboardInterrupt:
    raise KeyboardInterrupt
time_elapsed = time.time() - time_start
print('Training time: %.1f seconds.' % time_elapsed)

G = np.array(history['g_loss']).squeeze()
D = np.array(history['d_loss']).squeeze()


# Default plot configurations
plt.rc('font', family='serif')
plt.rc('text', usetex=True)
plt.rc('lines', markeredgewidth=0.5, linewidth=0.5)

plt.figure(figsize=(4,3), dpi=300)
plt.plot(G)
plt.plot(D[:,0])
plt.legend(('G total loss','G-v loss','G-c loss','D loss'))
plt.ylim([0, 3])
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.savefig('gan_myo_train_losses.pdf', bbox_inches = 'tight')
plt.show()


#%% VISUALIZATION

plt.ion()

# Sensor order
def myoind():
    myo0, myo1 = np.arange(8), np.arange(8,16)
    sensor_order = np.empty((myo0.size+myo1.size), dtype=np.int)
    sensor_order[0::2] = myo0
    sensor_order[1::2] = myo1
    return sensor_order

num_examples = 32

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
for i in range(t_train_ind.max() + 2):
    # for each sample (16)
    tmp_y = np.zeros_like(vis_X[0])
    for j in range(num_examples):
        noise = np.random.normal(0, 1, size=(1, gan.latent_dim))

        t = np.zeros((1, 8))
        t[0,i] = 1
        t = toolsfeatures.onehotnoise(np.array([i]), 8, 0.4)
        tmp_y[j] = gan.generator.predict([noise,t])
    vis_X_gen.append(tmp_y)

# Scale data:
scaler = preprocessing.MinMaxScaler().fit(np.concatenate(vis_X_gen+vis_X))
for i in range(len(vis_X)):
    vis_X[i] = scaler.transform(vis_X[i])
    vis_X_gen[i] = scaler.transform(vis_X_gen[i])
vis_X[-1] = np.zeros_like(vis_X[0])

fig, axs = plt.subplots(nrows=2, ncols=8,
                        sharex=True, sharey=True,
                        figsize=(12, 5.7))

for row_ind in range(axs.shape[0]):
    for col_ind in range(axs.shape[1]):
        plt.axes(axs[row_ind,col_ind])
        plt.xticks(np.arange(0,16,3))
        if col_ind == 0:
            if row_ind == 0:
                plt.ylabel('Real Sample Index')
            else:
                plt.ylabel('Generated Sample Index')
        if row_ind == 1:
            plt.xlabel('G%i' % col_ind)
            axs[row_ind,col_ind].imshow(vis_X_gen[col_ind][:,myoind()])
        else:
            axs[row_ind,col_ind].imshow(vis_X[col_ind][:,myoind()])

#fig.suptitle('[Epochs:%i][d_lr=%g][g_lr=%g][w_loss=(%g,%g)]' % (
#                  epochs,d_lr,g_lr,g_loss_w[0],g_loss_w[1]))
fig.show()
fig.savefig('samples.pdf', bbox_inches = 'tight')
#    fig.savefig('gan%03i.%04i.png' % (0, ep))


#%% SAVE
            
if par_noise:
    filename = './nets/dualmyo_trainedGan_noise_'
else:
    filename = './nets/dualmyo_trainedGan_nonoise_'
    
if par_save:
    gan.discriminator.save(filename + 'discriminator.h5')
    gan.generator.save(filename + 'generator.h5')


#%% DATASET GENERATION
# Gestures 0-6 -> regular gestures
# Gesture 7 -> outliers (others)

# Recover original data:
X_train, X_val, X_test = X_train_bak.copy(), X_val_bak.copy(), X_test_bak.copy()
t_train, t_val, t_test = t_train_bak.copy(), t_val_bak.copy(), t_test_bak.copy()

n_outliers = 0 # number of outliers to be generated
n_per_class = 32 # number of samples per class to be generated
n_per_batch = (n_outliers + n_per_class) * 7

# Source Noise
noise = np.random.normal(0, 1, ((n_per_batch, gan.latent_dim)))
# Generate class targets vector
gened_labels = np.tile(np.arange(7), (n_per_class,))
gened_labels = np.concatenate((gened_labels, np.tile(7, n_outliers)))
gened_labels = toolsfeatures.onehotencoder(gened_labels, 8)
# Generate samples
gened_samples = gan.generator.predict([noise, gened_labels])

X_train_gen = gened_samples
t_train_gen = gened_labels
t_train_gen = gan.discriminator.predict(X_train_gen)[1]
o_train_gen = np.zeros_like(gened_labels)
o_train = np.ones_like((t_train_bak))

generated_dataset = {'X_train': X_train,
                     't_train': t_train,
                     'X_val': X_val,
                     't_val': t_val,
                     'X_test': X_test,
                     't_test': t_test,
                     'X_others': X_test_7,
                     't_others': t_test_7,
                     'X_train_gen': X_train_gen,
                     't_train_gen': t_train_gen,
                     'o_train_gen': o_train_gen
                     }

#%% GENERATION METRIC CALCULATION

# Baseline (train set)
similarity_base = ganmetrics.class_l2_norm(X_train, t_train_ind, X_train, t_train_ind)

# Generated set
t_train_gen_ind = np.argmax(t_train_gen,1)
similarity_gen = ganmetrics.class_l2_norm(X_train, t_train_ind, X_train_gen, t_train_gen_ind)
for i in range(len(similarity_base)):
    print(np.hstack((similarity_base[i],similarity_gen[i])))
    
#%% SAVE DATA

if par_noise:
    filename = './data/dualmyo_generated_noise.pkl'
else:
    filename = './data/dualmyo_generated_nonoise.pkl'

if par_save:
    with open(filename,'wb') as file:
        pickle.dump(generated_dataset, file)
    

