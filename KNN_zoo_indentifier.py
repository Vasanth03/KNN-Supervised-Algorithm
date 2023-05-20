#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 10:54:28 2022

@author: vasanthdhanagopal
"""

########################### KNN - Glass Classifier ############################

# Import Libraries
import pandas as pd
import numpy as np

zoo = pd.read_csv("/copy path")
zoo.head()

zoo.describe()
zoo.info()
zoo.isnull().sum()
type(zoo)
zoo.dtypes
zoo.shape
zoo.columns
a = zoo['type'].unique()
a

# converting Type values to zoo animals Types
zoo['type'] = np.where(zoo['type'] == '1', 'ani1', zoo['type'])
zoo['type'] = np.where(zoo['type'] == '2', 'ani2', zoo['type'])
zoo['type'] = np.where(zoo['type'] == '3', 'ani3', zoo['type'])
zoo['type'] = np.where(zoo['type'] == '4', 'ani4', zoo['type'])
zoo['type'] = np.where(zoo['type'] == '5', 'ani5', zoo['type'])
zoo['type'] = np.where(zoo['type'] == '6', 'ani6', zoo['type'])
zoo['type'] = np.where(zoo['type'] == '7', 'ani7', zoo['type'])


zoo1 = zoo.iloc[:,1:17]
zoo_animal_name = zoo.iloc[:,0]


def scalerfun(i):
    x = (i - i.min()) / (i.max() - i.min())
    return(x)

# Normalized data frame (considering the numerical part of data)
zoo_n = scalerfun(zoo1.iloc[:, :])
zoo_n.describe()

X = np.array(zoo_n.iloc[:,:]) #Predictors
Y = np.array(zoo['type'])   #Target

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20)

# KNN Classifer
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
pred = knn.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, pred))
pd.crosstab(Y_test, pred, rownames = ['Actual'], colnames= ['Predictions']) 

# error on train data
pred_train = knn.predict(X_train)
print(accuracy_score(Y_train, pred_train))
pd.crosstab(Y_train, pred_train, rownames=['Actual'], colnames = ['Predictions']) 

# creating empty list variable 
acc = []

# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values

for i in range(1,31,2):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, Y_train)
    train_acc = np.mean(neigh.predict(X_train) == Y_train)
    test_acc = np.mean(neigh.predict(X_test) == Y_test)
    acc.append([train_acc, test_acc])


import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(1,31,2),[i[0] for i in acc],"ro-")

# test accuracy plot
plt.plot(np.arange(1,31,2),[i[1] for i in acc],"bo-")

plt.scatter(zoo['animal name'],zoo['type'])


###############################################################################

# also follow this https://www.kaggle.com/code/vazsumit/classification-of-zoo-animals









