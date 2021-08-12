#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 03:04:26 2021

@author: hamzakhalid
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

#Reading the CSV file using pandas
dataset1 = pd.read_csv('insurance.csv')

#Separating X and y from the dataset
X1 = dataset1.iloc[:, :-1].values
y1 = dataset1.iloc[:, 6:7].values

# Encoding categorical data
one_hot_encoder = OneHotEncoder()
X1[:,1:2] = np.array(np.array(one_hot_encoder.fit_transform(X1[:,1:2])))


# OrdinalEncoder is for converting features
# LabelEncoder is for converting target variable.

from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()

X = [['Good'], ['Bad'], ['Worst'], ['Worst'], ['Bad'] ]
X = enc.fit_transform(X)





