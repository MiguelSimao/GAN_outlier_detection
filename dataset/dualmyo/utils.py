#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 18:05:21 2018

@author: simao
"""

import os
import pickle
import numpy as np
import random

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

# Ensure reproducibility:
os.environ['PYTHONHASHSEED'] = '0'

def restart_random():
    np.random.seed(1337)
    random.seed(12345)

class Loader():
    def __init__(self, relpath='./dualmyo_dataset.pkl') :
        self.ABSPATH = os.path.dirname(__file__) + os.path.sep + relpath
        if not os.path.exists(self.ABSPATH):
            raise FileNotFoundError
    
    def load(self):
        # Return a tuple with a list of samples and a list of targets
        with open(self.ABSPATH,'rb') as file :
            return pickle.load(file)
    
    def load_syntetic_sequences(self):
        restart_random()
        
    def split(self, targets, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
        restart_random()
        ind_all = np.arange(len(targets))
        targets = np.array(targets)
        # Split 1 : all -> train and rest
        ind_train, ind_test = train_test_split(ind_all,
                                       shuffle=True,
                                       stratify=targets[ind_all],
                                       test_size=val_ratio+test_ratio,
                                       random_state=42)
        
        # Split 2 : rest -> val and test
        ind_val, ind_test = train_test_split(ind_test,
                                shuffle=True,
                                stratify=targets[ind_test],
                                test_size=(test_ratio/(val_ratio + test_ratio)),
                                random_state=42)
        
        return ind_train, ind_val, ind_test

class SynteticSequences():
    def __init__(self, data):
        # Rest
        ind_rest = data[1] == 0
        data_rest = np.concatenate([sample for sample in data[0][ind_rest]], axis=0)
#        data_rest = np.delete(data_rest,[0,1,10,11], axis=1)
        self.rest_dist = normal_dist_par(data_rest)
        self.samples = (data[0][~ind_rest], data[1][~ind_rest])
        self.interval_dist = (3,0.5) # Mean / dist
        self.transition_dist = (0.2, 0.05)
        
    def load_sequences(self, n=8):
        restart_random()
        targets = self.samples[1]
        samples = self.samples[0]
        n_samples = samples.shape[0]
        
        # Shuffle samples
        ind_shuf = np.random.permutation( np.arange(targets.shape[0]) )
        ind_shuf = ind_shuf[:-(n_samples % n)] # Remove samples we can't use
        idxs = ind_shuf.reshape((-1,n))
        
        # Reshape samples into subsequences without dropping samples
#        sequence_targets = np.zeros((n_samples // n, n + int((n_samples % n) > 0) ))
#        for i in range(len(sequence_targets)):
#            if i < n_samples % n :
#                sequence_targets[i,:] = ind_shuf[i*(n+1):(i+1)*(n+1)]
#            else:
#                sequence_targets[i,:-1] = ind_shuf[i*n:(i+1)*n]
        
        # Add indexes (-1) for intervals between gestures:
        idxs2 = -1 * np.ones((idxs.shape[0], 2*n + 1))
        idxs2[:,1::2] = idxs
        idxs2 = idxs2.astype(np.int)
        
        master_seq = []
        master_tar = []
        # Iterate over sequences:
        for seq in idxs2:
            curr_seq = []
            curr_tar = []
            # Generate/get samples with intervals:
            for idx in seq:
                interval_len = int(sample_normal_dist(self.interval_dist) * 200) # 200 fps, sampled len is in seconds
                if idx == -1:
                    curr_seq.append( self.sample_interval(n_frames=interval_len) )
                    curr_tar.append(0)
                else:
                    curr_seq.append( samples[idx].reshape((1,)+samples[idx].shape) )
                    curr_tar.append(targets[idx])
            # Generate and add transitions
            curr_seq_trans = [curr_seq[0]]
            curr_tar_trans = [curr_tar[0]]
            for i in range(1,len(curr_seq)):
                frame1 = curr_seq[i-1][0,-1,:]
                frame2 = curr_seq[i][0,0,:]
                transition_len = int(sample_normal_dist(self.transition_dist) * 200)
                curr_seq_trans.append(self.sample_transition(frame1, frame2, transition_len))
                curr_seq_trans.append(curr_seq[i])
                curr_tar_trans.append(-1)
                curr_tar_trans.append(curr_tar[i])
            master_seq.append(curr_seq_trans)
            master_tar.append(curr_tar_trans)
        
        # Post-process targets
        T = []
        for seq, tar in zip(master_seq, master_tar):
            N = np.array([x.shape[1] for x in seq])
            T.append(np.concatenate([ np.tile(tar[i], N[i]) for i in range(len(N))] ))
        
        # Concatenate all samples
        X = [np.concatenate(seq, axis=1) for seq in master_seq]
        
        # Pad all sequences with rest to maximum sequence length
        N = np.array([x.shape[1] for x in X]).max()
        for i, seq in enumerate(X):
            n = seq.shape[1]
            X[i] = np.concatenate((seq,
                    self.sample_interval(n_frames=(N - seq.shape[1]))), axis=1)
            T[i] = np.concatenate( (T[i], np.zeros((N - n))) ).reshape((1,-1))
            
        X = np.concatenate(X, 0)
        T = np.concatenate(T, 0)
        return X, T
    
    
    def sample_interval(self, n_frames=400):
        mean, std = self.rest_dist
        sample = np.concatenate([
                np.random.normal(mean[i], std[i], (1, n_frames, 1)) for i in range(len(mean))],
                axis=2)
        sample = np.rint(sample)
        # Low/High limits
        cap_minmax(sample)
        return sample

    def sample_transition(self, startframe, endframe, n):
        transition = np.empty((1, n, len(startframe)))
        for i in range(len(startframe)):
            xp = np.linspace(-1,1,n)
            # Interpolate between points with a sigmoid curve
            transition[0,:,i] = scaled_sigmoid(xp, startframe[i], endframe[i], 0.8)
        # Add noise (pauses distribution)
        transition += np.random.normal(self.rest_dist[0], self.rest_dist[1], size=transition.shape)
        cap_minmax(transition)
        return transition

def cap_minmax(x, xmin=-127, xmax=128):
    x[x<xmin] = xmin
    x[x>xmax] = xmax

def sample_normal_dist(dist_param):
    return np.random.normal(dist_param[0],dist_param[1])
    
def normal_dist_par(data):
    return (np.mean(data, axis=0), np.std(data, axis=0))

def scaled_sigmoid(X, amin=0, amax=1, tau=1/4.6):
    return (amax - amin) / ( 1 + np.exp(-4.60*tau*X)) + amin
