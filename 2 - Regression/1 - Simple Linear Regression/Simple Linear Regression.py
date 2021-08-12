"""
Created on Sat Jul 31 22:39:38 2021

@author: hamzakhalid
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, r2_score

#Reading the CSV file using pandas
dataset = pd.read_csv('heights_and_weights.csv')
#Separating X and y from the dataset
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1:2].values

#Splitting the dataset in training and testing sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, 
                                                         test_size=0.2,
                                                         train_size=0.8, 
                                                         random_state=0)

#Loading Linear Regressor from Sklearn library
regressor = linear_model.LinearRegression()
#Fitting training set
regressor.fit(X_train,y_train)

#Making predictions
y_pred = regressor.predict(X_test)

# The coefficients
print('Coefficients: \n', regressor.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

print('\nTRAINING SET:')
# Plotting test set
plt.scatter(X_train, y_train,  color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue', linewidth=2)
plt.xlabel('Heights')
plt.ylabel('Weights')
plt.title('Heights vs Weights')
plt.show()

print('\nTEST SET:')
# Plotting test set
plt.scatter(X_test, y_test,  color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue', linewidth=2)
plt.xlabel('Heights')
plt.ylabel('Weights')
plt.title('Heights vs Weights')
plt.show()