#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 02:13:49 2021

@author: hamzakhalid
"""

#Dataset: https://www.kaggle.com/mirichoi0218/insurance/
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

#Reading the CSV file using pandas
dataset = pd.read_csv('insurance.csv')
#Understanding the data types of our features
print(dataset.info())
#Separating X and y from the dataset
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 6:7].values
#Inspecting data
print(X)
print(y)

#Checking unique categorical variable
print(dataset['region'].unique())
print(dataset['sex'].unique())
print(dataset['smoker'].unique())
#Inspecting data
print(X[0])
print(X[1])
print(X[2])
print(X[3])

# Encoding categorical data
# One Hot Encoding categorical features that are not ordinal
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [5])], remainder='passthrough')
X = ct.fit_transform(X)

# Label Encoding the ordinal variables
le1 = LabelEncoder()
X[:,5] = le1.fit_transform(X[:,5])
le2 = LabelEncoder()
X[:,8] = le2.fit_transform(X[:,8])

#Splitting the dataset in training and testing sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, 
                                                         test_size=0.2,
                                                         train_size=0.8, 
                                                         random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train[:, [4,6,7]] = sc.fit_transform(X_train[:, [4,6,7]])
X_test[:, [4,6,7]] = sc.transform(X_test[:, [4,6,7]])


# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# The coefficients
print('Coefficients: \n', regressor.coef_)

# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))
