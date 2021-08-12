#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 02:58:00 2021

@author: hamzakhalid
"""

import pandas as pd
from sklearn import model_selection

#Reading the CSV file using pandas
dataset = pd.read_csv('insurance.csv')

#Separating X and y from the dataset
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 6:7]

#Splitting the dataset in training and testing sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, 
                                                         test_size=0.2,
                                                         train_size=0.8, 
                                                         random_state=42)