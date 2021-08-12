#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 03:56:36 2021

@author: hamzakhalid
"""
#Dataset: https://www.kaggle.com/indhusree/icecream

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn import model_selection
from sklearn.metrics import r2_score

#Reading the CSV file using pandas
dataset = pd.read_csv('icecream_sales.csv')
#Separating X and y from the dataset
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1:2].values

#Splitting the dataset in training and testing sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, 
                                                         test_size=0.2,
                                                         train_size=0.8, 
                                                         random_state=0)

#Loading Linear Regressor from Sklearn library
regressor = DecisionTreeRegressor(max_depth=5)
#Fitting training set
regressor.fit(X_train,y_train)

#Making predictions
y_pred = regressor.predict(X_test)

# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test, y_test, color = 'red')
plt.scatter(X_test, y_pred, color = 'green')
plt.plot(X_grid, regressor.predict(X_grid), color = 'black')
plt.title('Decision Tree Regression')
plt.xlabel('Temperature')
plt.ylabel('Revenue')
plt.show()