#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 01:30:30 2021

@author: hamzakhalid
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

#Reading the CSV file using pandas
dataset = pd.read_csv('insurance.csv')

#Transformation pipeline
onehot_attr = ["region", "sex", "smoker"]
scaler_attr = ["age","bmi","children"]

pipeline = ColumnTransformer(
                transformers = [
                                    ('standard_scaler', StandardScaler(), scaler_attr),
                                    ('one_hot_encoder', OneHotEncoder(), onehot_attr),
                                ], 
                remainder='passthrough')

data_prepared = np.array(pipeline.fit_transform(dataset))
print(data_prepared)


