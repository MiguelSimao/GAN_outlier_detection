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

from sklearn import preprocessing
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
    X = np.concatenate([tts.window(seq, 100, 100) for seq in X])
    T = np.concatenate([tts.window(seq, 100, 100) for seq in T])
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

#%% CLASSIFIER MODEL DEFINITION

names = [
        'Nearest Neighbors',
        'SVM',
        'Decision Tree',
        'Random Forest',
        'ADABoost',
        'Naive-Bayes',
        'QDA',
        'LDA',
        'MLP',
        ]

classifiers = [
        KNeighborsClassifier(5, algorithm='auto'),
        SVC(),
        DecisionTreeClassifier(max_depth=30),
        RandomForestClassifier(max_depth=30, n_estimators=10),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
        LinearDiscriminantAnalysis(),
        MLPClassifier(hidden_layer_sizes=(100,100)),
        ]

scores = []
for name, clf in zip(names, classifiers):
    print(' ::: %s :::' % (name))
    time_start = time.time()
    clf.fit(X_train, T_train)
    time_elapsed = time.time() - time_start
    print('Training time: %.1f s' % time_elapsed)
    
    time_start = time.time()
    train_score = clf.score(X_train,T_train) * 100
    val_score = clf.score(X_val,T_val) * 100
    test_score = clf.score(X_test,T_test) * 100
    
    time_elapsed = time.time() - time_start
    
    scores.append((train_score,val_score,test_score))
    print('Testing time: %.1f s' % time_elapsed)
    
    print('Accuracies:')
    print('Train: %.1f' % (train_score))
    print('  Val: %.1f' % (val_score))
    print(' Test: %.1f' % (test_score))