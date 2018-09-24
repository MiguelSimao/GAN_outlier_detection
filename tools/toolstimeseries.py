#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 17:06:30 2018

@author: simao
"""
import numpy as np
from tools import toolsfeatures


def window(sequence, span, step):
    # sequence is a [length x dimension] array
    nsteps = (len(sequence) - span) // step
    out = np.concatenate([
            sequence[i*step:i*step+span].reshape((1,span,-1)) for i in range(nsteps) ])
    return out

def tsonehotencoder(t,maxclasses):
    T = np.zeros((t.shape[0], t.shape[1], maxclasses,))
    for i, sample in enumerate(t):
        T[i] = toolsfeatures.onehotencoder(sample, maxclasses)
    return T

def tsroll(x, t, seqlen, batchsize):
    n = x.shape[0]
    
    n_seqs, pts_remain = n // seqlen, n % seqlen
    seqs_remain = batchsize - n_seqs % batchsize
    
    # Reshape into sequences of length seqlen (drop remaining frames (often zeros))
    x = x[:-pts_remain].reshape((-1, seqlen, x.shape[1]))
    t = t[:-pts_remain].reshape((-1, seqlen, 1))
    
    # Ensure output is divisible by the batch size
    x = np.concatenate((x, np.zeros( (seqs_remain, x.shape[1], x.shape[2]) )))
    t = np.concatenate((t, np.zeros( (seqs_remain, t.shape[1], t.shape[2]) )))
    return x, t