#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 03:04:56 2021

@author: hamzakhalid
"""

import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler

#Reading the CSV file using pandas
dataset = pd.read_csv('insurance.csv')

#Separating X and y from the dataset
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 6:7].values

#Splitting the dataset in training and testing sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, 
                                                         test_size=0.2,
                                                         train_size=0.8, 
                                                         random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train[:, [0,2,3]] = sc.fit_transform(X_train[:, [0,2,3]])
X_test[:, [0,2,3]] = sc.transform(X_test[:, [0,2,3]])
#Inspecting values
print(X_train[0])
print(X_train[1])
print(X_train[2])
print(X_train[3])