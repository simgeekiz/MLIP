# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 11:10:20 2018

@author: Eric Buitinck
"""

#%% General: Packages

import numpy as np
import numpy.random as npr
import random as rn
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.cross_validation import train_test_split
from sklearn import datasets, svm, metrics

#%% General: Reading data

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

#%% General: Formatting & splitting data

xtrain = np.array(train_data.values[:,1:])
ytrain = np.array(train_data['label'])

xtest = np.array(test_data.values[:,1:])

X_train, X_test, y_train, y_test = train_test_split(xtrain, ytrain, test_size=0.2, random_state=42)

X_max = np.max(X_train)
train_size = np.size(y_train)
test_size = np.size(y_test)

#%% Noise: Label mutation

inertia = 0.5

labels = np.arange(10)
p = np.full(10, (1-inertia)/9)
p[0]= inertia
mutation = npr.choice(labels,size=train_size,  p=p)

y_train = np.mod(y_train+mutation,np.full(train_size,10))

#%% Noise: Pixel inversion mutation

inertia = 0.75

for i in range(train_size):
    for j in range(784):
        if rn.random()>inertia:
            if X_train[i][j]>X_max/2:
                X_train[i][j]=0
            else:
                X_train[i][j]=X_max

#%% Noise: Random layer

amplitude = 1

for picture in X_train:
    for pixel in picture:
            pixel = pixel + amplitude*X_max*rn.random()

#%% Noise: Flat layer
            
amplitude = 0.2

X_train = amplitude*X_max+X_train

#%% Noise: Resolution reduction



#%% Solution: K-nearest sklearn

knn = KNC(n_neighbors=5)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
print np.size(np.where((y_test-pred) == 0)[0])/float(np.size(pred))

#%% Solution: K-nearest own implementation

#for i in range(test_size): 
 #   for 
 
#%% Solution: SVM

X_train[X_train>0]=1
X_test[X_test>0]=1

classifier = svm.SVC(C=200,kernel='rbf',gamma=0.01,cache_size=8000,probability=False)
classifier.fit(X_train, y_train)

pred = classifier.predict(X_test)
print np.size(np.where((y_test-pred) == 0)[0])/float(np.size(pred))

 
 
 
 
 