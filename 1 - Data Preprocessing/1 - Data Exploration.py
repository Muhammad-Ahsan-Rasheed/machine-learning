#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 23:32:43 2021

@author: hamzakhalid
"""

#Dataset: https://www.kaggle.com/mirichoi0218/insurance/
import pandas as pd
import matplotlib.pyplot as plt

#Reading the CSV file using pandas
dataset = pd.read_csv('insurance.csv')
#Understanding the data types of our features
print(dataset.info())
#Checking unique categorical variable
print(dataset['region'].unique())
print(dataset['sex'].unique())
print(dataset['smoker'].unique())
#Counting region categories
print(dataset['region'].value_counts())
#Exploring the summary of dataset
print(dataset.describe())
#Inspecting head and tail
print(dataset.head(10))
print(dataset.tail(10))
#Exploring the dataset further with histograms
dataset.hist(bins=50,figsize=(10,10))
plt.show()
#Finding Correlation between every pair of attributes
correlation = dataset.corr()
print(correlation)
#How much each feature is correlated with the charges
#The correlation coefficient ranges from –1 to 1. 
#When it is close to 1, it means that there is a strong positive correlation; 
#When the coefficient is close to –1, it means that there is a strong negative correlation
#Coefficients close to 0 mean that there is no linear correlation
print(correlation["charges"].sort_values(ascending=False))
dataset.plot(kind="scatter", x="age", y="charges", alpha=0.1)