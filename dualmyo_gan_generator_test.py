#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 17:19:33 2018

@author: simao
"""

import numpy as np
import time
import pickle
from dataset.dualmyo.utils import Loader, SynteticSequences
from dataset.dualmyo import dualmyofeatures
from tools import toolstimeseries as tts
from tools import toolsfeatures
from tools.postprocessing import PostProcessor

from sklearn import preprocessing
from keras import Model
from keras.optimizers import Adam
import keras
import keras.layers as kls

# ENSURE REPRODUCIBILITY ######################################################
import os
import random
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1337)
random.seed(12345)
###############################################################################

from classifiers import ACGAN


#%% LOAD DATA

#with open('dualmyo_dataset_generated.pkl','rb') as file:
#    data = pickle.load(file)
with open('dualmyo_dataset_generated7.pkl','rb') as file:
    data2 = pickle.load(file)

X_train = data2['X_train']
t_train = data2['t_train']
X_val = data2['X_val']
t_val = data2['t_val']
X_test = data2['X_test']
t_test = data2['t_test']
X_others = data2['X_others']
t_others = data2['t_others'] 
#X_train_gen = data2['X_train_gen']
#t_train_gen = data2['t_train_gen']

#%% LOAD GENERATOR

generator = keras.models.load_model('trainedGan_generator7.h5')
discriminator = keras.models.load_model('trainedGan_discriminator7.h5')
latent_dim = X_train.shape[1]


#%% PREPARE DATA

# Validation target 0
o_val = np.ones(X_val.shape[0])

# --- Use gesture 7 as outlier --- #

# Find gesture 7 in all sets:
#ind_train_gen7 = t_train_gen[:,7].astype('bool')
#ind_train7 = t_train[:,7].astype('bool')
#ind_val7 = t_val[:,7].astype('bool')
#ind_test7 = t_test[:,7].astype('bool')

# Remove samples of 7 from training and validation set
#X_val = X_val[np.logical_not(ind_val7)]
#t_val = t_val[np.logical_not(ind_val7)]
#o_val = o_val[np.logical_not(ind_val7)]

#X_train = X_train[np.logical_not(ind_train7)]
#t_train = t_train[np.logical_not(ind_train7)]
#X_val = X_val[np.logical_not(ind_val7)]
#t_val = t_val[np.logical_not(ind_val7)]
#X_test = X_test[np.logical_not(ind_test7)]
#t_test = t_test[np.logical_not(ind_test7)]

#X_train_gen = X_train_gen[np.logical_not(ind_train_gen7)]
#t_train_gen = t_train_gen[np.logical_not(ind_train_gen7)]

# Change index of 7 to 8 (others)
#t_test[ind_test7] = 0
#t_test[ind_test7, -1] = 1

# Generate outliers by style transfer
#t_train_gen = toolsfeatures.onehotencoder(np.tile((8,), (X_train.shape[0],) ))
t_train_gen = toolsfeatures.onehotnoise(np.random.randint(0, 7, (X_train.shape[0],)), 8, 0.5)
X_train_gen = generator.predict([np.random.normal(0, 1, (X_train.shape[0], 16)), t_train_gen])
t_train_gen = toolsfeatures.onehotencoder(np.tile((7,), (X_train.shape[0],) ))

X_train_new = np.concatenate((X_train, X_train_gen), axis=0)
t_train_new = np.concatenate((t_train, t_train_gen))
o_train_new = np.concatenate((np.ones(X_train.shape[0]),
                              np.zeros(X_train_gen.shape[0])), axis=0)


class_weight = [1,] * 9
class_weight[-1] = 0.2


#%% LOAD NETWORK

discriminator = net = keras.models.load_model('trainedGan_discriminator7.h5')
#generator = keras.models.load_model('trainedGan_generator.h5')

#discriminator.compile(loss=['binary_crossentropy', 'categorical_crossentropy'],
#                      loss_weights=[0.3, 0.7],
#                      optimizer=keras.optimizers.Adam(lr=0.01), # SGD(lr=1e-2, mom)
#                      metrics={'d_output_source': 'binary_crossentropy',
#                               'd_output_class': 'categorical_accuracy'})

#%% RETRAIN NETWORK
#try:
#    discriminator.fit(X_train_new,[o_train_new, t_train_new],
#                      validation_data=(X_val,[o_val, t_val]),
#                      class_weight=[1.,class_weight],
#                      callbacks=[keras.callbacks.EarlyStopping('d_output_class_categorical_accuracy',patience=12)],
#                      batch_size=None,
#                      epochs=1000,
#                      verbose=1)
#    print('\n\nTraining process finished.')
#except KeyboardInterrupt:
#    print('\n\nTraining process interrupted early.')

#%% EVALUATE MODEL
y = discriminator.predict(X_train)[1]
train_score = sum(y.argmax(axis=1)==t_train.argmax(axis=1))/len(y) * 100
y = discriminator.predict(X_val)[1]
val_score = sum(y.argmax(axis=1)==t_val.argmax(axis=1))/len(y) * 100
y = discriminator.predict(X_test)[1]
test_score = sum(y.argmax(axis=1)==t_test.argmax(axis=1))/len(y) * 100
y = discriminator.predict(X_others)[1]
others_score = sum(y.argmax(axis=1)==t_others.argmax(axis=1))/len(y) * 100

print('Class accuracies:')
print(' Train: %.2f%%' % (train_score))
print('   Val: %.2f%%' % (val_score))
print('  Test: %.2f%%' % (test_score))
print('Others: %.2f%%' % (others_score))

#%% CLASSIFICATION THRESHOLD

threshold = 0.60

# Train
y = discriminator.predict(X_train)[1]
yind = np.argmax(y, axis=1)
ymax = np.max(y, axis=1)
yind[ymax<threshold] = 7
train_score = sum(yind==t_train.argmax(axis=1))/len(y) * 100
# Val
y = discriminator.predict(X_val)[1]
yind = np.argmax(y, axis=1)
ymax = np.max(y, axis=1)
yind[ymax<threshold] = 7
val_score = sum(yind==t_val.argmax(axis=1))/len(y) * 100
# Test
y = discriminator.predict(X_test)[1]
yind = np.argmax(y, axis=1)
ymax = np.max(y, axis=1)
yind[ymax<threshold] = 7
test_score = sum(yind==t_test.argmax(axis=1))/len(y) * 100
# Others
y = discriminator.predict(X_others)[1]
yind = np.argmax(y, axis=1)
ymax = np.max(y, axis=1)
yind[ymax<threshold] = 7
others_score = sum(yind==t_others.argmax(axis=1))/len(y) * 100

print('With outlier threshold:')
print(' Train: %.2f%%' % (train_score))
print('   Val: %.2f%%' % (val_score))
print('  Test: %.2f%%' % (test_score))
print('Others: %.2f%%' % (others_score))

#%% Train baseline

discriminator.compile(loss=['binary_crossentropy', 'categorical_crossentropy'],
                      loss_weights=[0.1, 0.9],
                      optimizer=keras.optimizers.Adam(lr=0.01), # SGD(lr=1e-2, mom)
                      metrics={'d_output_source': 'binary_crossentropy',
                               'd_output_class': 'categorical_accuracy'})

#%% RETRAIN NETWORK
try:
    discriminator.fit(X_train,[np.ones(X_train.shape[0]), t_train],
                      validation_data=(X_val,[o_val, t_val]),
                      class_weight=[1.,class_weight],
                      callbacks=[keras.callbacks.EarlyStopping('d_output_class_categorical_accuracy',patience=12)],
                      batch_size=None,
                      epochs=200,
                      verbose=0)
    print('\n\nTraining process finished.')
except KeyboardInterrupt:
    print('\n\nTraining process interrupted early.')

#%% EVALUATE BASELINE
y = discriminator.predict(X_train)[1]
train_score = sum(y.argmax(axis=1)==t_train.argmax(axis=1))/len(y) * 100
y = discriminator.predict(X_val)[1]
val_score = sum(y.argmax(axis=1)==t_val.argmax(axis=1))/len(y) * 100
y = discriminator.predict(X_test)[1]
test_score = sum(y.argmax(axis=1)==t_test.argmax(axis=1))/len(y) * 100
y = discriminator.predict(X_others)[1]
others_score = sum(y.argmax(axis=1)==t_others.argmax(axis=1))/len(y) * 100

print('Baseline accuracies:')
print(' Train: %.2f%%' % (train_score))
print('   Val: %.2f%%' % (val_score))
print('  Test: %.2f%%' % (test_score))
print('Others: %.2f%%' % (others_score))


#%% BASELINE CLASSIFICATION THRESHOLD

# Train
y = discriminator.predict(X_train)[1]
yind = np.argmax(y, axis=1)
ymax = np.max(y, axis=1)
yind[ymax<threshold] = 7
train_score = sum(yind==t_train.argmax(axis=1))/len(y) * 100
# Val
y = discriminator.predict(X_val)[1]
yind = np.argmax(y, axis=1)
ymax = np.max(y, axis=1)
yind[ymax<threshold] = 7
val_score = sum(yind==t_val.argmax(axis=1))/len(y) * 100
# Test
y = discriminator.predict(X_test)[1]
yind = np.argmax(y, axis=1)
ymax = np.max(y, axis=1)
yind[ymax<threshold] = 7
test_score = sum(yind==t_test.argmax(axis=1))/len(y) * 100
# Others
y = discriminator.predict(X_others)[1]
yind = np.argmax(y, axis=1)
ymax = np.max(y, axis=1)
yind[ymax<threshold] = 7
others_score = sum(yind==t_others.argmax(axis=1))/len(y) * 100

print('With outlier threshold:')
print(' Train: %.2f%%' % (train_score))
print('   Val: %.2f%%' % (val_score))
print('  Test: %.2f%%' % (test_score))
print('Others: %.2f%%' % (others_score))