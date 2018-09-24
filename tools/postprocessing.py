#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 16:57:53 2018

@author: simao
"""

import numpy as np
from tools import toolstimeseries as tts
from enum import Enum


class PostProcessor():
    def __init__(self, tind, yind):
        self.window = 10
        self.tind = tind.copy()
        self.yind = yind.copy()
        self.ls_true = np.empty((0,0))
        self.ls_pred = np.empty((0,0))
        self.ls_true_norest = np.empty((0,0))
        self.ls_pred_norest = np.empty((0,0))
        
    def filterLength(self, minlen=100):
        self.checkLists()
        self.ls_pred = filterlength(self.ls_pred, minlen)
        self.ls_pred = extendgestures(self.ls_pred)
        return self
    
    def filterMerge(self):
        self.checkLists()
        self.ls_pred = mergegestures(self.ls_pred)
        self.ls_pred = extendgestures(self.ls_pred)
        return self
    
    def extendGestures(self):
        self.checkLists()
        self.ls_pred = extendgestures(self.ls_pred)
        return self
    
    def score(self):
        self.checkLists()
        return accuracycriterion(self.ls_true, self.ls_pred)
    
    def output(self):
        self.checkLists()
        return list2vector(self.ls_pred, self.yind.shape[0]).reshape((-1,))
    
    def defaultScore(self, minlen):
        self.checkLists()
        self.ls_pred = self.filterLength(minlen).filterMerge().extendGestures().ls_pred
        return accuracycriterion(self.ls_true, self.ls_pred)
    
    def checkLists(self):
        if self.ls_pred.shape[0] == 0:
            self.ls_true = vector2list(self.tind)
            self.ls_pred = vector2list(self.yind)
    
    def countGestures(self):
        self.ls_true_norest = self.ls_true[self.ls_true[:,0] != 0]
        self.ls_pred_norest = self.ls_pred[self.ls_pred[:,0] != 0]
        return self
    
    def resetLists(self):
        self.ls_true = np.empty((0,0))
        self.ls_pred = np.empty((0,0))
        self.checkLists()
        

class ClassificationType(Enum):
    TRUE_POSITIVE = 0
    MISSCLASSIFCATION = 1
    FALSE_POSITIVE = 2
    FALSE_NEGATIVE = 3
    
def mergegestures(ls):
    toDel = []
    for i in range(len(ls) - 1):
        if ls[i,0] == ls[i+1,0]:
            # when the same gesture shows twice consecutevely
            ls[i,2] == ls[i+1,2]
            toDel.append(i+1)
    return np.delete(ls, toDel, axis=0)

def extendgestures(ls):
    ls[:-1,2] = ls[1:,1] - 1
    return ls

def filterlength(ls, minlen=200):
    # gesture below minlen frames are discarded
    d = ls[:,2] - ls[:,1]
    return ls[d > minlen]

def vector2list(y):
    y = np.squeeze(y)
    r = np.zeros_like(y)
    r = y[:-1] != y[1:]
    r = np.concatenate( (np.ones((1,)), r), 0)
    ind = np.argwhere(r==1)
    l = []
    for i in ind:
        l.append((y[i],i))
    l = np.array(l, dtype=np.int).squeeze()
    l_end = np.append( l[1:,1]-1, y.shape[0] ) .reshape((-1,1))
    return np.hstack((l, l_end))

def list2vector(ls, *args):
    if len(args) == 0:
        vlen = ls[-1,-1]
    elif len(args) == 1:
        vlen = args[0]
    else:
        raise NotImplementedError

    vec = np.zeros((vlen,1))
    for gesture in ls:
        vec[gesture[1]:gesture[2]+1] = gesture[0]
    return vec

def ji_criterion(truth, prediction):
    maxclasses = 8
    
    ls_true = np.copy(truth)
    ls_pred = np.copy(prediction)
    # List of gestures for ground truth/prediction and frame activation
    vec_true = list2vector(ls_true)
    vec_pred = list2vector(ls_pred)
    vec_true = tts.tsonehotencoder(vec_true, maxclasses).squeeze()
    vec_pred = tts.tsonehotencoder(vec_pred, maxclasses).squeeze()
    
    # Get list of gestures
    trueGestures = np.unique(ls_true[ls_true[:,0] != 0, 0])
    predGestures = np.unique(ls_pred[ls_pred[:,0] != 0, 0])
    
    # Find false positives
    falsePos = np.setdiff1d(trueGestures, np.union1d(trueGestures, predGestures))
    
    # Get overlaps for each gestures
    overlaps = []
    for idx in trueGestures:
        intersec = sum(vec_true[:,idx] * vec_true[:,idx])
        aux = vec_true[:,idx] + vec_pred[:,idx]
        union = sum(aux > 0)
        ji = intersec/union
        overlaps.append(ji)
    return overlaps#sum(overlaps)/(len(overlaps) + len(falsePos))

def segmentation_accuracy(truth, prediction):
    maxclasses = 2
    ls_true = truth.copy()
    ls_pred = prediction.copy()
    
    ls_true[ls_true[:,0] != 0, 0] = 1
    ls_pred[ls_pred[:,0] != 0, 0] = 1
    # List of gestures for ground truth/prediction and frame activation
    vec_true = tts.tsonehotencoder(list2vector(ls_true), maxclasses).squeeze()
    vec_pred = tts.tsonehotencoder(list2vector(ls_pred), maxclasses).squeeze()
     # Get list of gestures
    trueGestures = np.unique(ls_true[:, 0])
    predGestures = np.unique(ls_pred[:, 0])    
    # Find false positives
    falsePos = np.setdiff1d(trueGestures, np.union1d(trueGestures, predGestures))    
    # Get overlaps for each gestures
    overlaps = []
    for idx in trueGestures:
        intersec = sum(vec_true[:,idx] * vec_true[:,idx])
        aux = vec_true[:,idx] + vec_pred[:,idx]
        union = sum(aux > 0)
        overlaps.append(intersec/union)
    return sum(overlaps)/(len(overlaps) + len(falsePos))

def accuracycriterion(list_truth, list_prediction):
    ls_true = list_truth.copy()
    ls_pred = list_prediction.copy()
    maxlen = ls_true[-1,2] # end frame of last gesture
    
    # Remove rest stance (default), does not influence score
    ls_true = ls_true[ls_true[:,0] != 0]
    ls_pred = ls_pred[ls_pred[:,0] != 0]
    
    # function to return list of predicted gestures that intersect with a true gesture
    def find_pred_intersection(true_gesture, pred_list):
        i1 = np.argwhere( np.logical_and(
                true_gesture[1] >= pred_list[:,1],
                true_gesture[1] <= pred_list[:,2]) )
        
        i2 = np.argwhere( np.logical_and(
                true_gesture[2] <= pred_list[:,2],
                true_gesture[2] >= pred_list[:,1]) )
        
        if len(i1) == 0:
            # predicted gesture starts AFTER true gesture
            i_start = np.argwhere(pred_list[:,1] >= true_gesture[1])
            if len(i_start) > 0:
                i_start = i_start[0]
            else: # no predicted gesture was found
                return np.empty((0,))
        else:
            # predicted gesture starts BEFORE true gesture
            i_start = np.argwhere(pred_list[:,1] <= true_gesture[1])[-1]

        if len(i2) == 0:
            # predicted gesture ends BEFORE true gesture
            i_stop = np.argwhere(pred_list[:,2] <= true_gesture[2])
            if len(i_stop) > 0:
                i_stop = i_stop[-1]
            else: # no predicted gesture was found
                return np.empty((0,))   
        else:
            # predicted gesture ends AFTER true gesture
            i_stop = np.argwhere(pred_list[:,2] >= true_gesture[2])[0]
        
        return np.arange(i_start, i_stop + 1)

    # Determine score for each true gesture (and false positives)
    ji = np.empty((0,))
    error_type = []
    last_pred_id = -1
    for i, true_gest in enumerate(ls_true):
#        print('%i/%i' % (i, len(ls_true)))
        # true_gest [0: id, 1: start_frame, 2: end_frame]
        # Vectorize true gesture subset:
        vec_true_gest = list2vector([true_gest], maxlen)
        # Find pred/truth intersections by index of the predicted gestures:
        pred_gest_inter = find_pred_intersection(true_gest, ls_pred)
        
        # Find false negatives (no intersection with prediciton):
        if len(pred_gest_inter) == 0:
            ji = np.append(ji,0.) # Score 0 for false negative
            error_type.append(3)  # Error type 3: false negative
            continue
        
        # Find false positives
        n_false_pos = pred_gest_inter[0] - (last_pred_id + 1)
        last_pred_id = pred_gest_inter[-1]
        for _ in range(n_false_pos) :
            ji = np.append(ji, 0.0) # Score 0 for false positive
            error_type.append(2) # Error type 2: false positive
        # Modified "Jaccard index" for the true gesture
#        tmp_ji = []
        for pred_gest in ls_pred[pred_gest_inter]:
#            missclassification = False
            # In case of a good classification, determine JI
            if pred_gest[0] == true_gest[0]:
                # Vectorize pred gesture subset:
                vec_pred_gest = list2vector([pred_gest], maxlen)
                # JI : intersection over union
                intersect = sum(np.logical_and(vec_pred_gest, vec_true_gest))/ \
                            sum(np.logical_or(vec_pred_gest, vec_true_gest))
                # If intersection is above a certain threshold,
                # consider the recognition successful (JI=1.0)
                if intersect >= 0.5:
                    ji = np.append(ji, 1.0)
                else:
                    ji = np.append(ji, float(intersect))
                error_type.append(0)
                
            else:
                # In case of a missclassication
                ji = np.append(ji, 0.0)
                error_type.append(1)
        
        
        # Remove already used classifications to simplify calculations
#        ls_pred = np.delete(ls_pred, pred_gest_inter, axis=0)
    
    # Add FALSE POSITIVES that occured AFTER the last prediction
    n_false_pos = ls_pred.shape[0] - (pred_gest_inter[-1] + 1)
    for _ in range(n_false_pos) :
            ji = np.append(ji, 0.0) # Score 0 for false positive
            error_type.append(2) # Error type 2: false positive
    
    return {'Mean': np.mean(ji), 'All': ji, 'Types': np.array(error_type), 'Predictions': ls_pred, 'Truth': ls_true}
    

def test():
    ls_true = np.array( [[1, 100, 300],
                         [1, 600, 850],
                         [2, 1350, 1750]])
    ls_pred = np.array( [[1, 650, 900],
                         [1, 1300, 1500],
                         [2, 1501, 1800],
                         [2, 2000, 2500] ])
    y_true = list2vector(ls_true, 3000)
    y_pred = list2vector(ls_pred, 3000)
    
    flt = PostProcessor(y_true, y_pred)
    score = flt.score()
    
    print('score: %.2f' % score['Mean'])
    for i in range(4):
        print('errors type %i: %i' % (i+1, sum(score['Types']==i)))
    
    