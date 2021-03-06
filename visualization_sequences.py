#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualization_sequences.py

Script to produce visualizations of the sequences of the UC2018 DualMyo data 
set.

Author: Miguel Simão (miguel.simao@uc.pt)
"""


import numpy as np
from dataset.dualmyo.utils import Loader, SynteticSequences
from dataset.dualmyo import dualmyofeatures
from tools import toolsfeatures
from tools import toolstimeseries as tts
from sklearn import preprocessing

import matplotlib.pyplot as plt

# ENSURE REPRODUCIBILITY ######################################################
import os
import random
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1337)
random.seed(12345)
###############################################################################

#%% LOAD DATA

DataLoader = Loader()

sample_data, sample_target = DataLoader.load()
sample_data = np.concatenate( [sample.reshape((1,) + sample.shape) for sample in sample_data], axis=0 )
sample_target = np.array(sample_target)

# Data split
ind_train, ind_val, ind_test = DataLoader.split(sample_target)

# Data generation

def generate_sequences(sample_data, sample_target):
    X, T = SynteticSequences((sample_data, sample_target)).load_sequences()
    X = np.concatenate([tts.window(seq, 100, 1) for seq in X])
    T = np.concatenate([tts.window(seq, 100, 1) for seq in T])
    T = np.array([toolsfeatures.targetmode(seq)[0] for seq in T])
    T[T==-1] = 0
    return X, T

X1,T1 = generate_sequences(sample_data[ind_train], sample_target[ind_train])
X2,T2 = generate_sequences(sample_data[ind_val], sample_target[ind_val])
X3,T3 = generate_sequences(sample_data[ind_test], sample_target[ind_test])

#%% FEATURE EXTRACTION

# Feature extraction
X_train = np.vstack([dualmyofeatures.extract_std(sample) for sample in X1])
X_val = np.vstack([dualmyofeatures.extract_std(sample) for sample in X2])
X_test = np.vstack([dualmyofeatures.extract_std(sample) for sample in X3])

# Feature scaling
feature_scaler = preprocessing.StandardScaler().fit(X_train)
X_train = feature_scaler.transform(X_train)
X_val = feature_scaler.transform(X_val)
X_test = feature_scaler.transform(X_test)

# Target processing
T_train = T1
T_val = T2
T_test = T3


#%% SENSOR REORDERING
myo0, myo1 = np.arange(8), np.arange(8,16)
sensor_order = np.empty((myo0.size+myo1.size), dtype=np.int)
sensor_order[0::2] = myo0
sensor_order[1::2] = myo1

X = X_train[:20000].transpose()
T = T_train[:20000].transpose()
T1, T2 = T[:10000], T[-10000:]

# Transitions
Td = T[1:] != T[:-1]
Td = np.concatenate((np.array([False]), Td), axis=0)
Td1, Td2 = Td[:10000], Td[-10000:]

idx1, idx2 = np.argwhere(Td1 != 0), np.argwhere(Td2 != 0)
Tind1, Tind2 = T1[idx1], T2[idx2]
idx1, idx2 = idx1[Tind1 != 0], idx2[Tind2 != 0]
Tind1, Tind2 = Tind1[Tind1 != 0], Tind2[Tind2 != 0]

# Reorder channels
X = X[sensor_order]

#%% PLOT DEFAULT SETTINGS

# Default configurations
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('text', usetex=True)
plt.rc('legend', edgecolor=(0,0,0), fancybox=False)
plt.rc('lines', markeredgewidth=0.5, linewidth=0.5)

#%% PLOT FEATURES BY CLASS SIDE-BY-SIDE

fig,ax = plt.subplots(nrows=2, ncols=1,
                      sharex=True, sharey=True,
                      dpi=300, figsize=(6.4,3))

ax[0].imshow(X[:,:10000], aspect='auto', cmap='Greys')
ax[1].imshow(X[:,-10000:], aspect='auto', cmap='Greys')

ax[1].set_xticklabels(ax[1].get_xticks() / 200)


for i in range(len(Tind1)):
    ax[0].text(idx1[i],-1.1,'G%i' % Tind1[i])
for i in range(len(Tind2)):
    ax[1].text(idx2[i],-1.1,'G%i' % Tind2[i])


fig.text(0.0, 0.5, 'Channels', va='center', rotation='vertical')
plt.xlabel('Time (s)')

plt.tight_layout()

fig.savefig('myo_synth_seqs.pdf', pad_inches=0)
