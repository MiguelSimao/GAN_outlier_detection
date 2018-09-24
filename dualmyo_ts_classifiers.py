#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 17:19:33 2018

@author: simao
"""

import numpy as np
import time
from dataset.dualmyo.utils import Loader, SynteticSequences
from dataset.dualmyo import dualmyofeatures
import toolstimeseries as tts
import toolsfeatures
from postprocessing import PostProcessor
from scipy import stats

from sklearn import preprocessing, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

# ENSURE REPRODUCIBILITY ######################################################
import os
import random
import tensorflow as tf
from keras import backend as K

def reset_random():
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(1337)
    random.seed(12345)
    
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.set_random_seed(123)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
reset_random()
###############################################################################

from keras.models import Model
import keras.layers as kls
from keras.layers import Input, Dense, GaussianNoise
from keras.layers import BatchNormalization
import keras.optimizers as optimizers
from keras.callbacks import EarlyStopping
from keras import utils

###############################################################################


#%% LOAD DATA

DataLoader = Loader()

sample_data, sample_target = DataLoader.load()
sample_data = np.concatenate( [sample.reshape((1,) + sample.shape) for sample in sample_data], axis=0 )
sample_target = np.array(sample_target)

# Data split
ind_train, ind_val, ind_test = DataLoader.split(sample_target)

#%% FEATURE EXTRACTION

# Data generation
def extractstd(sample_data, sample_target, span=200, step=5):
    X, T = SynteticSequences((sample_data, sample_target)).load_sequences()
    X = X.reshape((-1,X.shape[-1]))
    T = T.reshape((-1,))
    nsteps = (X.shape[0] - span) // step
    X2 = np.zeros((nsteps, X.shape[1]))
    T2 = np.zeros((nsteps, ))
    for i in range(nsteps):
        X2[i] = np.std( X[i*step:i*step+span], 0)
        T2[i] = stats.mode( T[i*step:i*step+span] )[0][0]
    T2[T2==-1] = 0
    
    return X2, T2

X1,T1 = extractstd(sample_data[ind_train], sample_target[ind_train], 100, step=1)
X2,T2 = extractstd(sample_data[ind_val], sample_target[ind_val], 100, step=1)
X3,T3 = extractstd(sample_data[ind_test], sample_target[ind_test], 100, step=1)

X1 = np.delete(X1, [0,1,10,11], axis=1)
X2 = np.delete(X2, [0,1,10,11], axis=1)
X3 = np.delete(X3, [0,1,10,11], axis=1)


# Feature scaling
feature_scaler = preprocessing.StandardScaler().fit(X1)
X_train = feature_scaler.transform(X1)
X_val = feature_scaler.transform(X2)
X_test = feature_scaler.transform(X3)


# Target processing
t_train = T1
t_val = T2
t_test = T3


#%% CONCATENATE TIMESTEPS

def concatenatewindow(X, T, span=200, step=5):
    nsteps = (X.shape[0] - span) // step
    X2 = np.zeros((nsteps, X.shape[1]*span))
    T2 = np.zeros((nsteps, ))
    for i in range(nsteps):
        X2[i] = X[i*step:i*step+span].reshape((-1,))
        T2[i] = stats.mode( T[i*step:i*step+span] )[0][0]
    return X2, T2

X_train, t_train = concatenatewindow(X_train, t_train, span=200, step=1)
X_val,   t_val   = concatenatewindow(X_val,   t_val,   span=200, step=1)
X_test,  t_test  = concatenatewindow(X_test,  t_test,  span=200, step=1)


#%% BATCH SAMPLES

batch_size = 256
def fill_samples(X, batch_size):
    d = 1 if X.ndim < 2 else X.shape[1]
    nfill = batch_size - (X.shape[0] % batch_size)
    return np.concatenate((X, np.zeros((nfill, d)).squeeze() ))

X_train = fill_samples(X_train, batch_size)
X_val =   fill_samples(X_val, batch_size)
X_test =  fill_samples(X_test, batch_size)
t_train = fill_samples(t_train, batch_size)
t_val =   fill_samples(t_val, batch_size)
t_test =  fill_samples(t_test, batch_size)

# Masks
m_train = np.any(X_train, 1)
m_val =   np.any(X_val, 1)
m_test =  np.any(X_test, 1)


#%% ONE-HOT ENCODING

T_train = toolsfeatures.onehotencoder(t_train.astype(np.int))
T_val   = toolsfeatures.onehotencoder(t_val.astype(np.int))
T_test  = toolsfeatures.onehotencoder(t_test.astype(np.int))


#%% CLASSIFIER MODEL DEFINITION

def create_net(batch_size):
    inputs = kls.Input(batch_shape=(batch_size, X_train.shape[1]))
    x = kls.Dense(512, activation='tanh')(inputs)
    x = kls.Dense(512, activation='tanh')(x)
    outputs = kls.Dense(T_train.shape[1], activation='softmax')(x)

    net = Model(inputs, outputs)
    return net

net = create_net(batch_size=batch_size)

net.compile(optimizer=optimizers.Adam(lr=0.01, decay=1e-6),
            loss='categorical_crossentropy',
            metrics=['acc'])


#%% TRAIN MODEL

print(':::: TRAINING FFNN ::::' )
time_start = time.time()

history = net.fit(x=X_train,y=T_train,
                  sample_weight=m_train,
                  validation_data=(X_val,T_val,m_val),
                  batch_size=batch_size,
                  epochs=1000,
                  callbacks=[EarlyStopping('val_loss',patience=12)],
                  verbose=1)

time_elapsed = time.time() - time_start
print('Training time: %.1f s' % time_elapsed)


#%% TEST MODEL

time_start = time.time()

# custom accuracy criterion
Y_train = net.predict(X_train, batch_size=batch_size)
Y_val = net.predict(X_val, batch_size=batch_size)
Y_test = net.predict(X_test, batch_size=batch_size)
time_elapsed = time.time() - time_start
y_train = Y_train.argmax(1)
y_val   = Y_val.argmax(1)
y_test  = Y_test.argmax(1)

# train
flt = PostProcessor(t_train[m_train], y_train[m_train])
flt = flt.filterLength(100).filterMerge().extendGestures().countGestures()
score_train = flt.score()
flt.checkLists()
print('Training:')
print('Total predicted gestures: %i' % flt.ls_pred_norest.shape[0] )
print('Total real gestures: %i' % flt.ls_true_norest.shape[0] )
# val
flt = PostProcessor(t_val[m_val], y_val[m_val])
flt = flt.filterLength(100).filterMerge().extendGestures().countGestures()
score_val = flt.score()
flt.checkLists()
print('Validation:')
print('Total predicted gestures: %i' % flt.ls_pred_norest.shape[0] )
print('Total real gestures: %i' % flt.ls_true_norest.shape[0] )
# test
flt = PostProcessor(t_test[m_test], y_test[m_test])
flt = flt.filterLength(100).filterMerge().extendGestures().countGestures()
score_test = flt.score()
print('Testing:')
print('Total predicted gestures: %i' % flt.ls_pred_norest.shape[0] )
print('Total real gestures: %i' % flt.ls_true_norest.shape[0] )

print('Testing time: %.1f s' % time_elapsed)

############

print('Accuracy (custom criterion):')
print('Train: %.2f' % (score_train['Mean']*100))
print('  Val: %.2f' % (score_val['Mean']*100))
print(' Test: %.2f' % (score_test['Mean']*100))

ers = np.zeros((3,4))
score_train['Types'] = np.array(score_train['Types'])
score_val['Types']   = np.array(score_val['Types'])
score_test['Types']  = np.array(score_test['Types'])
for i in range(4):
    ers[0,i] = sum(score_train['Types']==i)
    ers[1,i] = sum(score_val['Types']==i)
    ers[2,i] = sum(score_test['Types']==i)
print('Error types:')
for i in range(3):
    print('%s: %i,%i,%i,%i' % (['Train','  Val',' Test'][i], 
                              ers[i,0], ers[i,1], ers[i,2], ers[i,3]) )


##############


print('Accuracy:')
print('Train: %.2f' % (
        metrics.accuracy_score(t_train[m_train],y_train[m_train])*100))
print('  Val: %.2f' % (
        metrics.accuracy_score(t_val[m_val],y_val[m_val])*100))
print(' Test: %.2f' % (
        metrics.accuracy_score(t_test[m_test],y_test[m_test])*100))

#%% PLOT

import matplotlib.pyplot as plt


# Default configurations
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='small')
plt.rc('ytick', labelsize='small')
plt.rc('text', usetex=True)
plt.rc('legend', edgecolor=(0,0,0), fancybox=False)
plt.rc('lines', markeredgewidth=0.5, linewidth=0.5)

# Get 
Y_tmp = np.reshape(Y_test, (-1, 8))
T_tmp = np.reshape(T_test, (-1, 8))
y_tmp = np.reshape(y_test, (-1, 8))
I1 = np.arange(9000)
x = (I1 - I1[0]) / 200
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
#colors = ['w', 'b', 'g', 'r', 'c', 'm', 'y', 'k']

names = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8']
fig, ax = plt.subplots(1,1, figsize=(5.9,2))

for i in range(1,8):
    plt.plot(x, Y_tmp[I1,i], c=colors[i])
lg = plt.legend(names, ncol=8, mode='expand', bbox_to_anchor=(0, .25, 1, 1),
                borderaxespad=0)
for i in range(1,8):
    plt.plot(x, T_tmp[I1,i], c=colors[i], ls='--')
    

plt.ylabel('$p(y|z)$')
plt.xlabel('Time (s)')
plt.tight_layout()

plt.savefig('plot.pdf', bbox_extra_artists=(lg,), bbox_inches='tight')
    