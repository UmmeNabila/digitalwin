# -*- coding: utf-8 -*-
"""
Task: Linear regression for 2-group XS sensitivity analysis
Date: 10/12/2022
"""
import numpy as np   #numpy
import pandas as pd # Dataframes
from sklearn.model_selection import train_test_split
from sklearn import metrics

import keras
# from keras.models import Sequential
from keras.layers import Dense


#load data, save input and output into separate arrays X,Y
data=pd.read_csv('xs.csv')
X=data.iloc[:,0:8]   
Y=data.iloc[:,-1]   #-1 Python will start indexing from the end (keff)
print(data.head())

#This is a handy function to split into training/testing, can you think of an alternative way for splitting?
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state=42)


# make the model
learning_rate = 0.0009
batch_size = 8
epochs = 20
nn_model = keras.models.Sequential()
nn_model.add(keras.Input(shape=Xtrain.shape[1], batch_size=batch_size))
nn_model.add(Dense(100, activation='relu'))
nn_model.add(Dense(100, activation='relu'))
nn_model.add(Dense(100, activation='relu'))
nn_model.add(Dense(1, activation='linear'))

nn_model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                #  loss=keras.losses.BinaryCrossentropy(),
                #  loss='mean_absolute_error',
                 loss='mean_squared_error',
                #  metrics=['mean_absolute_error'],
                 )

nn_model.summary()

nn_model.fit(Xtrain, Ytrain, epochs=100, batch_size=batch_size, validation_split=.15)

Ynn_test = nn_model.predict(Xtest)
# Ynn_test = Ynn_test.reshape(Ynn_test.shape[0])
train_metrics= metrics.r2_score(Ytest, Ynn_test)
print('NN Test=', train_metrics)

Ynn_train = nn_model.predict(Xtrain)
# Ynn_train = Ynn_train.reshape(Ynn_train.shape[0])
train_metrics = metrics.r2_score(Ytrain, Ynn_train)
print('NN Train=', train_metrics)
# print(Ynn_train[0], Ytrain[0])

# NOTE: poor R2 performance with negative values (~ -.2). This is due
# to a nonlinear model being used.

