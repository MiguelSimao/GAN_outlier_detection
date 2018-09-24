#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 09:39:31 2018

@author: simao
"""

import numpy as np

def extract_std(rawdata):
    rawdata = np.delete(rawdata, [0,1,10,11], axis=1)
    return rawdata.std(axis=0)

def extract_ts(rawdata):
    return np.delete(rawdata, [0,1,10,11], axis=1)