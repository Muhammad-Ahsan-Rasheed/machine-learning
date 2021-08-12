#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 03:29:14 2021

@author: hamzakhalid
"""

#Source: https://scikit-learn.org/stable/modules/tree.html
#Dataset: https://en.wikipedia.org/wiki/Iris_flower_data_set
#Requirement: conda install python-graphviz

from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib.pyplot as plt

#Loading Iris dataset from sklearn library
iris = load_iris()
X, y = iris.data, iris.target

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()

#Rendering the graph and saving it in a pdf file
import graphviz 
dot_data = tree.export_graphviz(clf, out_file=None, 
                      feature_names=iris.feature_names,  
                      class_names=iris.target_names,  
                      filled=True, rounded=True,  
                      special_characters=True) 
graph = graphviz.Source(dot_data) 
graph.render("iris") 