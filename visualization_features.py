#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualization_features.py

Script to produce visualizations of the features used for the GAN discriminator 
tests.

Author: Miguel Simão (miguel.simao@uc.pt)
"""


import numpy as np

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import scale

import matplotlib.pyplot as plt

# ENSURE REPRODUCIBILITY ######################################################
import os
import random
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1337)
random.seed(12345)
###############################################################################

import keras
from tools import toolsfeatures
from dataset.dualmyo.utils import Loader
from dataset.dualmyo import dualmyofeatures

#%% LOAD DATA

DataLoader = Loader()

sample_data, sample_target = DataLoader.load()
sample_data = np.concatenate( [sample.reshape((1,) + sample.shape) for sample in sample_data], axis=0 )
sample_target = np.array(sample_target)

# Data split
ind_train, ind_val, ind_test = DataLoader.split(sample_target)

num_classes = 8

#%% FEATURE EXTRACTION

# Feature extraction
X_train = np.vstack([dualmyofeatures.extract_std(sample) for sample in sample_data[ind_train]])
X_val = np.vstack([dualmyofeatures.extract_std(sample) for sample in sample_data[ind_val]])
X_test = np.vstack([dualmyofeatures.extract_std(sample) for sample in sample_data[ind_test]])
X_ts = np.concatenate(
        [dualmyofeatures.extract_ts(sample)[np.newaxis] for sample in sample_data],
        axis=0)

# Feature scaling
feature_scaler = preprocessing.StandardScaler().fit(X_train)
X_train = feature_scaler.transform(X_train)
X_val = feature_scaler.transform(X_val)
X_test = feature_scaler.transform(X_test)
X_master = np.concatenate((X_train, X_val, X_test), axis=0)


# Target processing
t_train = sample_target[ind_train]
t_val = sample_target[ind_val]
t_test = sample_target[ind_test]
t_master = np.concatenate((t_train, t_val, t_test), axis=0)

#%% SENSOR REORDERING
myo0, myo1 = np.arange(8), np.arange(8,16)
sensor_order = np.empty((myo0.size+myo1.size), dtype=np.int)
sensor_order[0::2] = myo0
sensor_order[1::2] = myo1
X_master = X_master[:,sensor_order]
X_ts = X_ts[:,:,sensor_order]

#%% PLOT DEFAULT SETTINGS
#X_ts = np.abs(X_ts)/128.0

# Default configurations
plt.rc('font', family='serif')
#plt.rc('xtick', labelsize='x-small')
#plt.rc('ytick', labelsize='x-small')
plt.rc('text', usetex=True)
plt.rc('legend', edgecolor=(0,0,0), fancybox=False)
plt.rc('lines', markeredgewidth=0.5, linewidth=0.5)

#%% PLOT FEATURES BY CLASS SIDE-BY-SIDE


fig,ax = plt.subplots(nrows=1,ncols=1, dpi=300, figsize=(6.4,3))

X = []
for i in range(num_classes):
    I = np.argwhere(t_master==i)[:16].squeeze()
    X.append(X_master[I])

X = np.concatenate(X,0).transpose()

X = np.concatenate((X[:,:64],X[:,-64:]), axis=0)

ax.imshow(X, aspect='auto', cmap='Greys')
plt.xticks(np.arange(15,50,16) + .5)
plt.yticks([15.5])
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.tick_params(axis='both', length=0)
plt.grid(color='k')
plt.ylabel('Channels')
#plt.suptitle('Samples')

for i in range(4):
    plt.text(8+16*i,-1,'G%i' % i, ha='center')
    plt.text(8+16*i,32,'G%i' % (i+4), ha='center', va='top')
    
#fig.savefig('aaa.pdf', bbox_inches='tight')

#%% PLOT FEATURES LATENT SPACE PCA

Xb = preprocessing.scale(X_master)
pca = PCA(n_components=2).fit(Xb)
#Xb = TSNE(perplexity=5,early_exaggeration=8).fit_transform(Xb)
Xb = pca.transform(Xb)

fig,ax = plt.subplots(nrows=1,ncols=1, figsize=(4,4), dpi=300)

labels = ['G%i' % i for i in range(8)]
for i in range(8):
    I = t_master == i    
    ax.scatter(Xb[I,0],Xb[I,1],
               edgecolors='k',
               linewidth=0.5,
               c=plt.get_cmap('Dark2')(i),
               label=labels[i],
               )
plt.xlabel('Component 1')
plt.ylabel('Component 2')
lg = plt.legend(labels, fancybox=False, loc=5, bbox_to_anchor=(1.25,0.5))
fig.savefig('aaa.pdf', bbox_inches='tight', bbox_extra_artists=(lg,))

#%% USE GENERATOR TO CREATE FEATURES FROM NOISE

Xb = preprocessing.scale(X_master)

generator = keras.models.load_model('trainedGan_generator7.h5')
t_train_gen_ind = np.tile( np.arange(num_classes), (16,))
t_train_gen = toolsfeatures.onehotnoise(t_train_gen_ind, num_classes, 0.4)
X_gen = generator.predict([np.random.normal(0, 1, (t_train_gen.shape[0], 16)), t_train_gen])

Xb_gen = pca.transform(X_gen)


for i in range(num_classes):
    I = t_train_gen_ind == i    
    ax.scatter(Xb_gen[I,0],Xb_gen[I,1],
               edgecolors='k',
               marker='x',
               linewidth=0.5,
               c=plt.get_cmap('Dark2')(i),
               label=labels[i],
               )
    
#%% USE GENERATOR TO CREATE FEATURES FROM NOISE (TSNE)

i_real = np.arange(Xb.shape[0])
i_gened = np.arange(Xb_gen.shape[0]) + i_real[-1] + 1
X2 = TSNE().fit_transform( np.concatenate((Xb, X_gen), axis=0) )
Xb = X2[i_real]

Xb_gen = X2[i_gened]

fig,ax = plt.subplots(nrows=1,ncols=1, figsize=(6,6), dpi=300)
for i in range(8):
    I = t_master == i    
    ax.scatter(Xb[I,0],Xb[I,1],
               edgecolors='k',
               linewidth=0.5,
               c=plt.get_cmap('Dark2')(i),
               label=labels[i],
               )
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(labels, fancybox=False)

for i in range(num_classes):
    I = t_train_gen_ind == i    
    ax.scatter(Xb_gen[I,0],Xb_gen[I,1],
               edgecolors='k',
               marker='x',
               linewidth=0.5,
               c=plt.get_cmap('Dark2')(i),
               label=labels[i],
               )