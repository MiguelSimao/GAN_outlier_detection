#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 17:38:37 2018

@author: simao
"""
import numpy as np
from scipy import stats

def onehotencoder(tind, *args):
    if len(args) == 0:
        maxclasses = max(tind)+1
    elif len(args) == 1:
        maxclasses = args[0]
    else:
        raise NotImplementedError

    t = np.zeros((tind.shape[0], maxclasses))
    t[np.arange(tind.shape[0]),tind.astype(np.int).reshape((-1,))] = 1
    return t

def onehotnoise(tind, maxclasses, maxprob=0.5):
    tind = tind.astype('int')
    t = np.zeros((tind.shape[0], maxclasses))
    t = t + (1 - maxprob) / (maxclasses - 1)
    t[np.arange(tind.shape[0]), tind.reshape((-1,))] = maxprob
    return t

def label_noise(t, pmin=0.8, pmax=1.0):
    j = np.argmax(t, 1)
    n = t.shape[0]
    phigh = np.random.uniform(pmin, pmax, (n,))
    plow  = (1 - phigh) / (t.shape[1] - 1)
    for i in range(n):
        t[i] = plow[i]
        t[i,j[i]] = phigh[i]
    return t

def targetmode(tar_sequence):
    idx = stats.mode(tar_sequence)[0][0]
    return np.tile(idx, len(tar_sequence))


    
    
    