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

from sklearn import preprocessing, metrics
from scipy import stats

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

import keras
from keras import Model
from keras.layers import Input, Dense, GaussianNoise, CuDNNGRU, CuDNNLSTM, LSTM, Dropout
from keras.callbacks import EarlyStopping

###############################################################################


#%% LOAD DATA

DataLoader = Loader()

sample_data, sample_target = DataLoader.load()
sample_data = np.concatenate( [sample.reshape((1,) + sample.shape) for sample in sample_data], axis=0 )
sample_target = np.array(sample_target)

# Data split
ind_train, ind_val, ind_test = DataLoader.split(sample_target)

## Data generation
#def generate_sequences(sample_data, sample_target, window_len=200, step=5):
#    X, T = SynteticSequences((sample_data, sample_target)).load_sequences()
#    X = np.concatenate([tts.window(seq, window_len, step) for seq in X])
#    T = np.concatenate([tts.window(seq, window_len, step) for seq in T])
#    T[T==-1] = 0
#    T = np.array([toolsfeatures.targetmode(seq)[0] for seq in T])
#    return X, T
#
#X1,T1 = generate_sequences(sample_data[ind_train], sample_target[ind_train], step=5)
#X2,T2 = generate_sequences(sample_data[ind_val], sample_target[ind_val], step=5)
#X3,T3 = generate_sequences(sample_data[ind_test], sample_target[ind_test], step=1)


#%% FEATURE EXTRACTION

# Data generation
def extractstd(sample_data, sample_target, span=200, step=1):
    X, T = SynteticSequences((sample_data, sample_target)).load_sequences()
#    X = X.reshape((-1,X.shape[-1]))
#    T = T.reshape((-1,))
    nsteps = (X.shape[1] - span) // step
    X2 = np.zeros((X.shape[0], nsteps, X.shape[2]))
    T2 = np.zeros((T.shape[0], nsteps, ))
    for i in range(len(X)):
        for j in range(nsteps):
            X2[i][j] = np.std( X[i][j*step:j*step+span], 0)
            T2[i][j] = stats.mode( T[i][j*step:j*step+span] )[0][0]
    T2[T2==-1] = 0
    
    return X2, T2

X1,T1 = extractstd(sample_data[ind_train], sample_target[ind_train], 100, step=1)
X2,T2 = extractstd(sample_data[ind_val], sample_target[ind_val], 100, step=1)
X3,T3 = extractstd(sample_data[ind_test], sample_target[ind_test], 100, step=1)

X1 = np.delete(X1, [0,1,10,11], axis=2)
X2 = np.delete(X2, [0,1,10,11], axis=2)
X3 = np.delete(X3, [0,1,10,11], axis=2)
#X1 = X1[:,:,:8]
#X2 = X2[:,:,:8]
#X3 = X3[:,:,:8]

# Feature extraction
#X_train = np.vstack([dualmyofeatures.extract_std(sample) for sample in X1])
#X_val = np.vstack([dualmyofeatures.extract_std(sample) for sample in X2])
#X_test = np.vstack([dualmyofeatures.extract_std(sample) for sample in X3])

#%% Feature scaling
shape_train = X1.shape
shape_val =   X2.shape
shape_test =  X3.shape
X1 = X1.reshape((-1,X1.shape[2]))
X2 = X2.reshape((-1,X2.shape[2]))
X3 = X3.reshape((-1,X3.shape[2]))
feature_scaler = preprocessing.StandardScaler().fit(X1)

X_train = feature_scaler.transform(X1).reshape(shape_train)
X_val =   feature_scaler.transform(X2).reshape(shape_val)
X_test =  feature_scaler.transform(X3).reshape(shape_test)

# Target processing
t_train = T1
t_val =   T2
t_test =  T3


#%% SHORTEN SEQUENCES / BATCHING

seqlen = 200
batchsize = 10
maxclasses = 8

def fill_samples(X, batch_size):
    d = 1 if X.ndim < 2 else X.shape[1]
    fill = batch_size - (X.shape[0] % batch_size)
    return np.concatenate((X, np.zeros((fill, d)).squeeze() ))

def batchfill(X, T, seqlen=200, batchsize=10):
    # Reshape into master sequence
    X = X.reshape((-1, X.shape[2]))
    T = T.reshape((-1,))
    # Fill zeros to allow sequences of seqlen
    X = fill_samples(X, seqlen).reshape((-1, seqlen, X.shape[1]))
    T = fill_samples(T, seqlen).reshape((-1, seqlen,))
    # Fill w/ zero-samples so that number of samples is divisible by batchsize
    nfill = batchsize - (X.shape[0] % batchsize)
    nfill = 0 if nfill == batchsize else nfill
    X = np.concatenate((X, np.zeros((nfill, X.shape[1], X.shape[2]))))
    T = np.concatenate((T, np.zeros((nfill, T.shape[1]))))
    m = np.any( np.any(X,axis=1), axis=1 )
    return X, T, m

X_train, t_train, m_train = batchfill(X_train, t_train, seqlen, batchsize)
X_val,   t_val,   m_val =   batchfill(X_val,   t_val,   seqlen, batchsize)
X_test,  t_test,  m_test =  batchfill(X_test,  t_test,  seqlen, batchsize)

T_train = tts.tsonehotencoder(t_train, maxclasses)
T_val =   tts.tsonehotencoder(t_val,   maxclasses)
T_test =  tts.tsonehotencoder(t_test,  maxclasses)

#%% BATCHING TIMESERIES
#batchsize = 10
#X_train, T_train = tts.tsroll(X_train, T_train, seqlen=200, batchsize=batchsize)
#X_val, T_val = tts.tsroll(X_val, T_val, seqlen=200, batchsize=batchsize)
#X_test, T_test = tts.tsroll(X_test, T_test, seqlen=200, batchsize=1)
#
## Masks
#m_train = np.array([np.any(seq) for seq in X_train], dtype='int')
#m_val = np.array([np.any(seq) for seq in X_val], dtype='int')
#
#maxclasses = 8
#T_train = tts.tsonehotencoder(T_train, maxclasses)
#T_val = tts.tsonehotencoder(T_val, maxclasses)
#T_test = tts.tsonehotencoder(T_test, maxclasses)


#%% CLASSIFIER MODEL DEFINITION

def create_net(batch_size):
    inputs = Input(batch_shape=(batch_size,X_train.shape[1],X_train.shape[2]))
    x = Dense(400, activation='tanh')(inputs)
    x = GaussianNoise(0.1)(x)
    x = CuDNNLSTM(256,stateful=True,return_sequences=True)(x)
    x = keras.layers.Activation('tanh')(x)
    outputs = Dense(T_train.shape[2], activation='softmax')(x)

    net = Model(inputs, outputs)
    return net

net = create_net(batch_size=batchsize)

#opt = keras.optimizers.SGD(lr = .001, momentum = 0.9)
opt = keras.optimizers.Adam(lr=0.001)

net.compile(optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['acc'])


# Show network summary:
net.summary()


try:
    # Train and time:
    timestart = time.time()
    history = net.fit(x=X_train,y=T_train,
                      sample_weight=m_train,
                      validation_data=(X_val, T_val, m_val),
                      epochs=1000,
                      batch_size=batchsize,
                      callbacks=[EarlyStopping('val_loss',patience=12)],
                      verbose=1)
except KeyboardInterrupt:
    print('\nTraining interrupted.')

print('Training time: %.1f seconds.' % (time.time() - timestart))

#%% PREDICT

X_temp = X_train.reshape((-1,X_train.shape[2]))
m_train = np.any(X_temp,1)
X_temp = X_val.reshape((-1,  X_val.shape[2]))
m_val = np.any(X_temp,1)
X_temp = X_test.reshape((-1, X_test.shape[2]))
m_test = np.any(X_temp,1)

# Predict
time_start = time.time()
Y_train = net.predict(X_train, batch_size=batchsize)
Y_val   = net.predict(X_val,   batch_size=batchsize)
Y_test  = net.predict(X_test,  batch_size=batchsize)
y_train = Y_train.argmax(2)
y_val   = Y_val.argmax(2)
y_test  = Y_test.argmax(2)

time_elapsed = time.time() - time_start
print('Testing time: %.1f s' % time_elapsed)

# Reshape
y_train = y_train.reshape((-1,))
y_val = y_val.reshape((-1,))
y_test = y_test.reshape((-1,))
t_train = t_train.reshape((-1,))
t_val = t_val.reshape((-1,))
t_test = t_test.reshape((-1,))

#%% SCORE

# train
flt = PostProcessor(t_train[m_train], y_train[m_train])
flt = flt.filterLength(100).filterMerge().extendGestures()
score_train = flt.score()
print('Training:')
print('Total predicted gestures: %i' % flt.ls_pred_norest.shape[0] )
print('Total real gestures: %i' % flt.ls_true_norest.shape[0] )
# val
flt = PostProcessor(t_val[m_val], y_val[m_val])
flt = flt.filterLength(100).filterMerge().extendGestures()
score_val = flt.score()
print('Validation:')
print('Total predicted gestures: %i' % flt.ls_pred_norest.shape[0] )
print('Total real gestures: %i' % flt.ls_true_norest.shape[0] )
# test
flt = PostProcessor(t_test[m_test], y_test[m_test])
flt = flt.filterLength(100).filterMerge().extendGestures()
score_test = flt.score()
print('Testing:')
print('Total predicted gestures: %i' % flt.ls_pred_norest.shape[0] )
print('Total real gestures: %i' % flt.ls_true_norest.shape[0] )



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
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
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
names = ['1', '2', '3', '4', '5', '6', '7', '8']
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