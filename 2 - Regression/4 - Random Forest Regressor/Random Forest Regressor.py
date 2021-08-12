#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 03:56:36 2021

@author: hamzakhalid
"""

#Data: https://www.kaggle.com/harlfoxem/housesalesprediction

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

#Reading the CSV file using pandas
dataset = pd.read_csv('kc_house_data.csv')
#Separating X and y from the dataset
X = dataset.iloc[:, 3:].values
y = dataset.iloc[:, 2:3].values

#Splitting the dataset in training and testing sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, 
                                                         test_size=0.2,
                                                         train_size=0.8, 
                                                         random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train[:, :] = sc.fit_transform(X_train[:, :])
X_test[:, :] = sc.transform(X_test[:, :])

#Loading Linear Regressor from Sklearn library
# Training the Random Forest Regression model on the whole dataset

regressor = RandomForestRegressor(n_estimators=10)
regressor.fit(X_train, y_train)


#Making predictions
y_pred = regressor.predict(X_test)


# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

