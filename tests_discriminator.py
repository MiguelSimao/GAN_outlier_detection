#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests_discriminator.py

Script to run the GAN discriminator tests for the UC2018 DualMyo data set.
Must change «id_test» value on line 44 to select test to run.

Author: Miguel Simão (miguel.simao@uc.pt)
"""

# imports
import numpy as np
import pickle
from tools import toolsfeatures

from sklearn import metrics
import matplotlib.pyplot as plt

# ENSURE REPRODUCIBILITY ######################################################
#Python/Numpy
import os
import random
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1337)
random.seed(12345)
#TF
import tensorflow as tf
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
tf.set_random_seed(123)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
###############################################################################

from keras import Model
from keras.optimizers import Adam
import keras.layers as kls
import keras

#%% TEST PARAMETERS:

id_test = 2

if id_test == 0:
    # Baseline test
    par_labelnoise = False
    par_ds = 'myo'          # myo or cg
    par_mode = 'baseline'  # baseline or generator
    par_retrain = True
elif id_test == 1:
    # Test 1 : Noisy labels
    par_labelnoise = True
    par_ds = 'myo'         
    par_mode = 'baseline'  
    par_retrain = True
elif id_test == 2:
    # Test 2 : GAN discriminator
    par_labelnoise = True
    par_ds = 'myo'         
    par_mode = 'generator'  
    par_retrain = False
elif id_test == 3:
    # Test 3 : GAN discriminator retrained
    par_labelnoise = True
    par_ds = 'myo'          
    par_mode = 'generator'  
    par_retrain = True    


#%% Load dataset

def load_data(ds='myo',mode='baseline',labelnoise=False):
    if ds=='myo':
        if mode=='baseline':
            with open('./data/dualmyo_base.pkl','rb') as file:
                data = pickle.load(file)
        elif mode=='generator' and not labelnoise:
            with open('./data/dualmyo_generated_noise.pkl','rb') as file:
                data = pickle.load(file)
        elif mode=='generator' and labelnoise:
            with open('./data/dualmyo_generated_noise.pkl','rb') as file:
                data = pickle.load(file)
        else:
            NotImplementedError    
    else:
        NotImplementedError
    return data

data = load_data(par_ds, par_mode, par_labelnoise)

    
    
#%% Process dataset (noisy labels or not)
# In case we want to add noise to labels

# Load data
X_train  = data['X_train']
t_train  = data['t_train']
X_val    = data['X_val']
t_val    = data['t_val']
X_test   = data['X_test']
t_test   = data['t_test']
X_others = data['X_others']
t_others = data['t_others']
o_train = np.ones((X_train.shape[0],1))
o_val = np.ones((X_val.shape[0],1))
o_test = np.ones((X_test.shape[0],1))

if par_mode == 'generator':
    X_train_gen = data['X_train_gen']
    t_train_gen = data['t_train_gen']
    o_train_gen = np.zeros((X_train_gen.shape[0],1))
    X_train = np.concatenate((X_train, X_train_gen), 0)
    t_train = np.concatenate((t_train, t_train_gen), 0)
    o_train = np.concatenate((o_train, o_train_gen), 0)
    
if par_labelnoise:
    t_train = toolsfeatures.label_noise(t_train, 0.8, 1.0)

    
#%% Load network

if par_mode == 'baseline':
    discriminator = keras.models.load_model('./nets/dualmyo_untrained_discriminator.h5')
elif par_mode == 'generator':
    if par_labelnoise:
        discriminator = keras.models.load_model('./nets/dualmyo_trainedGan_noise_discriminator.h5')
    else:
        discriminator = keras.models.load_model('./nets/dualmyo_trainedGan_noise_discriminator.h5')
else:
    NotImplementedError

discriminator.compile(loss=['binary_crossentropy', 'categorical_crossentropy'],
                      loss_weights=[0., 1.],
                      optimizer=keras.optimizers.Adam(lr=0.01), # SGD(lr=1e-2, mom)
                      metrics={'d_output_source': 'binary_crossentropy',
                               'd_output_class': 'categorical_accuracy'})
def build_discriminator():
        inputs = kls.Input(shape=(16,))
        
        x = kls.GaussianNoise(.4)(inputs)
        x = kls.Dense(300)(x)
        x = kls.Activation('relu')(x)
        
        x = kls.GaussianNoise(.4)(x)
        x = kls.Dense(300)(x)
        x = kls.Activation('relu')(x)
        x = kls.Dropout(0.3)(x)
        
        label = kls.Dense(8, activation="softmax", name='d_output_class')(x)
        
        discriminator = Model(inputs, label)
        
        discriminator.compile(loss='categorical_crossentropy',
                                   optimizer=Adam(0.0002, .5, decay=1e-7), #SGD(0.1, 0.1), #
                                   metrics=['accuracy'])
        
        return discriminator
    
#%% Train/retrain network if necessary

if par_retrain:

    try:
        discriminator.fit(X_train, [o_train, t_train],
                          validation_data=(X_val, [o_val, t_val]),
                          callbacks=[keras.callbacks.
                                     EarlyStopping('val_loss',
                                                   patience=12)],
                          batch_size=None,
                          epochs=1000,
                          verbose=1)
        print('\n\nTraining process finished.')
    except KeyboardInterrupt:
        print('\n\nTraining process interrupted early.')
else:
    Warning('Network not retrained')        

#%% Test

def target_denoise(t):
    t2 = np.zeros_like(t)
    n = t2.shape[0]
    t2[np.arange(n),np.argmax(t,1)] = 1
    return t2

t_train = target_denoise(t_train)
t_val   = target_denoise(t_val)
t_test  = target_denoise(t_test)

thresholds = np.arange(0.5,0.901,0.1)
perf_class = np.zeros((1,4))
perf_threshold = np.zeros((len(thresholds), 4))

def thresholding(y, threshold=0.8):
    for i in range(y.shape[0]):
        if np.max(y[i]) < threshold:
            y[i] = 0
            y[i][-1] = 1
    return y

# Aggregate 
X_test2 = np.concatenate((X_test, X_others), axis=0)
t_test2 = np.concatenate((t_test, t_others), axis=0)

# Discriminator outputs:
y_train = discriminator.predict(X_train)[1]
y_val = discriminator.predict(X_val)[1]
y_test = discriminator.predict(X_test2)[1]


# Target indexes:
t_train_ind = np.argmax(t_train, axis=1)
t_val_ind = np.argmax(t_val, axis=1)
t_test_ind = np.argmax(t_test2, axis=1)
#t_others_ind = np.argmax(t_others, axis=1)

# Predicted indexes:
y_train_ind = np.argmax(y_train, axis=1)
y_val_ind = np.argmax(y_val, axis=1)
y_test_ind = np.argmax(y_test, axis=1)
#y_others_ind = np.argmax(y_others, axis=1)

# Prediction probabilities
y_train_prob = np.max(y_train, axis=1)
y_val_prob = np.max(y_val, axis=1)
y_test_prob = np.max(y_test, axis=1)
#y_others_prob = np.max(y_others, axis=1)

# Accuracy:
accuracy_train = metrics.accuracy_score(t_train_ind, y_train_ind)
accuracy_val = metrics.accuracy_score(t_val_ind, y_val_ind)
accuracy_test = metrics.accuracy_score(t_test_ind, y_test_ind)
#accuracy_others = metrics.accuracy_score(t_others_ind, y_others_ind)

# Recall:
recall_train = metrics.recall_score(t_train_ind, y_train_ind, average=None)
recall_val = metrics.recall_score(t_val_ind, y_val_ind, average=None)
recall_test = metrics.recall_score(t_test_ind, y_test_ind, average=None)
#recall_others = metrics.recall_score(t_others_ind, y_others_ind, average=None)

# Precision:
precision_train = metrics.precision_score(t_train_ind, y_train_ind, average=None)
precision_val = metrics.precision_score(t_val_ind, y_val_ind, average=None)
precision_test = metrics.precision_score(t_test_ind, y_test_ind, average=None)
#precision_others = metrics.precision_score(t_others_ind, y_others_ind, average=None)

# Confusion matrix:
confusion_train = metrics.confusion_matrix(t_train_ind, y_train_ind)
confusion_val = metrics.confusion_matrix(t_val_ind, y_val_ind)
confusion_test = metrics.confusion_matrix(t_test_ind, y_test_ind)

#%% Binary classification

# Prediction probabilities
y_train_prob = np.max(y_train, axis=1)
y_val_prob = np.max(y_val, axis=1)
y_test_prob = np.max(y_test, axis=1)

# Rescale probabilities (minimum p being 1/7)
y_train_prob = (y_train_prob - 1/7) / (1 - 1/7)
y_val_prob = (y_val_prob - 1/7) / (1 - 1/7)
y_test_prob = (y_test_prob - 1/7) / (1 - 1/7)

# Binary prediction targets
t_train_prob = (t_train_ind!=7) + 0#np.any(t_train[:,:-1], axis=1) + 0
t_val_prob = (t_val_ind!=7) + 0
t_test_prob = (t_test_ind!=7) + 0

# ROC CURVE / AUC
fpr, tpr, thresholds = metrics.roc_curve(t_test_prob, y_test_prob)

plt.plot(fpr, tpr,
         label='ROC Curve (area = %.3f)' % metrics.auc(fpr, tpr))
plt.plot([0,1],[0,1], c='k', linestyle='--')
plt.xlim([-0.01,1])
plt.ylim([0,1.01])
plt.title('Receiver Operating Characteristic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.draw()

#%% FIND THRESHOLDS


threshold_star = [0.95, 0.90]

def accuracy_p(y, t, threshold):
    yind = np.argmax(y, axis=1)
    tind = np.argmax(t, 1)
    ymax = np.max(y, axis=1)
    yind[ymax<threshold] = 7
    return sum(yind==tind)/y.shape[0]

def optimize_threshold(y,t,p_star=0.90):
    # y is the discriminator's output
    # t is the target
    # p_star is the objective output accuracy
#    p_tol = 0.001
    threshold_list = np.arange(0.0,0.951,0.01)
    accuracy_main = np.zeros_like(threshold_list)
    accuracy_others = np.zeros_like(threshold_list)
    ind_main = np.argwhere(np.argmax(t,1)!=7).squeeze()
    ind_others = np.argwhere(np.argmax(t,1)==7).squeeze()
    for i in range(threshold_list.shape[0]):
        accuracy_main[i] = accuracy_p(y[ind_main], t[ind_main], threshold_list[i])
        accuracy_others[i] = accuracy_p(y[ind_others], t[ind_others], threshold_list[i])
    return accuracy_main, accuracy_others, threshold_list




acc_main, acc_others, thresholds = optimize_threshold(y_test, t_test2)
acc_mean = (acc_main * len(acc_main) + acc_others * len(acc_others)) / (len(acc_main) + len(acc_others))
threshold_best = thresholds[np.argmax(acc_main)]

istar095 = np.argwhere(acc_main>=0.95)[-1]
istar090 = np.argwhere(acc_main>=0.90)[-1]

#%% PLOTS

# Default plot configurations
plt.rc('font', family='serif')
plt.rc('text', usetex=True)
plt.rc('lines', markeredgewidth=0.5, linewidth=0.8)

plt.figure(figsize=(4,3))
plt.plot(thresholds, acc_main, label='classes')
plt.plot(thresholds, acc_others, label='others')
plt.plot(thresholds, acc_mean, label='mean')

# p=0.95 points
plt.plot([thresholds[istar095],thresholds[istar095]],[0.0, 1.0], c='k', ls='--')
#plt.scatter(thresholds[istar095], acc_main[istar095])
#plt.scatter(thresholds[istar095], acc_others[istar095])
#plt.scatter(thresholds[istar095], acc_mean[istar095])
# p=0.90 points
plt.plot([thresholds[istar090],thresholds[istar090]],[0.0, 1.0], c='k', ls='--')
#plt.scatter(thresholds[istar090], acc_main[istar090])
#plt.scatter(thresholds[istar090], acc_others[istar090])
#plt.scatter(thresholds[istar090], acc_mean[istar090])

plt.legend(loc='lower right', fancybox=False)
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.xticks(np.arange(0,1.01,0.2))

plt.savefig('accuracy-threshold.pdf', bbox_inches = 'tight')

print('Maximum performance: %.2f%%' % (np.max(acc_main)*100))
print('Performance:')
for value in (acc_main[0],
              acc_others[0],
              acc_mean[0],
              thresholds[0],
              acc_main[istar095],
              acc_others[istar095],
              acc_mean[istar095],
              thresholds[istar095],
              acc_main[istar090],
              acc_others[istar090],
              acc_mean[istar090],
              thresholds[istar090]):
    print('%.3f' % value)




#%% CONFUSION MATRIX

def confusion_matrix_full(cm):
    cmfull = np.empty((cm.shape[0] + 2,
                       cm.shape[1] + 2))
    cmfull[:-2,:-2] = cm
    
    tp = np.diag(cm)
    fp = np.sum(cm, 0) - tp
    fn = np.sum(cm, 1) - tp
    precision = tp / np.sum(cm, 0)
    recall = tp / np.sum(cm, 1)
    cmfull[-2,:-2] = fp
    cmfull[:-2,-2] = fn
    cmfull[-1,:-2] = precision
    cmfull[:-2,-1] = recall
    return cmfull

## No threshold:

# Output thresholding:
yt_test = thresholding(y_test, 0)
yt_test_ind = np.argmax(yt_test, 1)
#confusion_test = metrics.confusion_matrix(t_test_ind, yt_test_ind)
cm0 = confusion_matrix_full(confusion_test)
acc_000 = metrics.accuracy_score(t_test_ind, yt_test_ind)
prec_000 = metrics.precision_score(t_test_ind, yt_test_ind, average=None)
rec_000 = metrics.recall_score(t_test_ind, yt_test_ind, average=None) 

## Threshold (p=0.95)
yt_test = thresholding(y_test, thresholds[istar095])
yt_test_ind = np.argmax(yt_test, 1)
#confusion_test = metrics.confusion_matrix(t_test_ind, yt_test_ind)
cm095 = confusion_matrix_full(confusion_test)
acc_095 = metrics.accuracy_score(t_test_ind, yt_test_ind)
prec_095 = metrics.precision_score(t_test_ind, yt_test_ind, average=None)
rec_095 = metrics.recall_score(t_test_ind, yt_test_ind, average=None) 

## Threshold (p=0.90)
yt_test = thresholding(y_test, thresholds[istar090])
yt_test_ind = np.argmax(yt_test, 1)
#confusion_test = metrics.confusion_matrix(t_test_ind, yt_test_ind)
cm090 = confusion_matrix_full(confusion_test)
acc_090 = metrics.accuracy_score(t_test_ind, yt_test_ind)
prec_090 = metrics.precision_score(t_test_ind, yt_test_ind, average=None)
rec_090 = metrics.recall_score(t_test_ind, yt_test_ind, average=None) 

# Print results:
print('Precision','Recall','Precision','Recall','Precision','Recall')
for i in range(len(prec_000)):
    print(prec_000[i], rec_000[i],prec_095[i], rec_095[i],prec_090[i], rec_090[i])
