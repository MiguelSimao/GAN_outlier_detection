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

from sklearn import preprocessing

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
from keras.layers import Input, Dense, GaussianNoise, Masking, Dropout, LSTM, SimpleRNN
from keras.layers import Conv1D, MaxPooling1D, Flatten, BatchNormalization
import keras.optimizers
from keras.callbacks import EarlyStopping
from keras import utils

#%% LOAD DATA

DataLoader = Loader()

sample_data, sample_target = DataLoader.load()
sample_data = np.concatenate( [sample.reshape((1,) + sample.shape) for sample in sample_data], axis=0 )
sample_target = np.array(sample_target)

# Data split
ind_train, ind_val, ind_test = DataLoader.split(sample_target)

# Data generation
def generate_sequences(sample_data, sample_target, window_len=200, step=5):
    X, T = SynteticSequences((sample_data, sample_target)).load_sequences()
    X = np.concatenate([tts.window(seq, window_len, step) for seq in X])
    T = np.concatenate([tts.window(seq, window_len, step) for seq in T])
    T[T==-1] = 0
    T = np.array([toolsfeatures.targetmode(seq)[0] for seq in T])
    return X, T

X1,T1 = generate_sequences(sample_data[ind_train], sample_target[ind_train], step=5)
X2,T2 = generate_sequences(sample_data[ind_val], sample_target[ind_val], step=5)
X3,T3 = generate_sequences(sample_data[ind_test], sample_target[ind_test], step=1)


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


#%% BATCHING TIMESERIES
batchsize = 10
X_train, T_train = tts.tsroll(X_train, T_train, seqlen=200, batchsize=batchsize)
X_val, T_val = tts.tsroll(X_val, T_val, seqlen=200, batchsize=batchsize)
X_test, T_test = tts.tsroll(X_test, T_test, seqlen=200, batchsize=1)

# Masks
m_train = np.array([np.any(seq) for seq in X_train], dtype='int')
m_val = np.array([np.any(seq) for seq in X_val], dtype='int')

maxclasses = 8
T_train = tts.tsonehotencoder(T_train, maxclasses)
T_val = tts.tsonehotencoder(T_val, maxclasses)
T_test = tts.tsonehotencoder(T_test, maxclasses)


#%% CLASSIFIER MODEL DEFINITION

def create_net(batch_size):
    inputs = Input(batch_shape=(batch_size,X_train.shape[1],X_train.shape[2]))
    x = Dense(400, activation='relu')(inputs)
#    x = Dense(256, activation='relu')(x)
    x = CuDNNLSTM(256,stateful=True,return_sequences=True)(x)
    x = keras.layers.Activation('relu')(x)
    outputs = Dense(T_train.shape[2], activation='softmax')(x)

    net = Model(inputs, outputs)
    return net

def create_net_cnn(batch_size):
    inputs = Input(batch_shape=(batch_size,X_train.shape[1],X_train.shape[2]))
    x = Dense(512,activation='tanh')(inputs)
    x = Conv1D(100, 5, padding='same', activation='relu')(x)
    x = GaussianNoise(.2)(x)
    x = BatchNormalization()(x)
    x = Conv1D(100, 5, padding='same', activation='relu')(x)
    x = GaussianNoise(.2)(x)
    x = BatchNormalization()(x)
    outputs = Dense(T_train.shape[2], activation='softmax')(x)

    net = Model(inputs, outputs)
    return net

net = create_net(batch_size=batchsize)

opt = keras.optimizers.SGD(lr = .001, momentum = 0.9)

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
                      validation_data=(X_val, T_val,m_val),
                      epochs=1000,
                      batch_size=batchsize,
                      callbacks=[EarlyStopping('val_loss',patience=12)],
                      verbose=1)
except KeyboardInterrupt:
    print('\nTraining interrupted.')

print('Training time: %.1f seconds.' % (time.time() - timestart))

#%% SCORE

# TRAINING SET
y = net.predict(X_train, batch_size=batchsize)
yind = np.argmax(y,axis=2).reshape((-1,1)) 
tind = np.argmax(T_train, axis=2).reshape((-1,1))
flt = PostProcessor(tind, yind)
flt = flt.filterLength(20).filterMerge().extendGestures()
score_train = flt.score()

# VALIDATION SET
y = net.predict(X_val, batch_size=batchsize)
yind = np.argmax(y, axis=2).reshape((-1,1)) 
tind = np.argmax(T_val, axis=2).reshape((-1,1))
flt = PostProcessor(tind, yind)
flt = flt.filterLength(20).filterMerge().extendGestures()
score_val = flt.score()