#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 23:53:00 2021

@author: hamzakhalid
"""

#Dataset: https://www.kaggle.com/akram24/position-salaries

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#Reading the CSV file using pandas
dataset = pd.read_csv('position_salaries.csv')
#Exploring the dataset
dataset.info()

#Separating X and y from the dataset
X = dataset.iloc[:,1:2].values.astype(float)
y = dataset.iloc[:,2:3].values.astype(float)

#Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# most important SVR parameter is Kernel type. It can be 
#linear,polynomial or gaussian SVR. We have a non-linear condition 
#so we can select polynomial or gaussian but here we select RBF(a 
#gaussian type) kernel.

regressor = SVR(kernel='rbf')
regressor.fit(X,y)


#6 Visualising the Support Vector Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Support Vector Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#Predicting a new result
y_pred = sc_y.inverse_transform ((regressor.predict (sc_X.transform(np.array([[10]])))))
print(y_pred)