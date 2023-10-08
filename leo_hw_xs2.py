# -*- coding: utf-8 -*-
"""
Task: Linear regression for 2-group XS sensitivity analysis
Date: 10/12/2022
"""
import numpy as np   #numpy
import pandas as pd # Dataframes
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.model_selection import train_test_split

import keras
# from keras.models import Sequential
from keras.layers import Dense


#handy function to determine different validation metrics of the regression model
#Keep an eye on those metrics, we will use them very frequently
def calc_metric(Y, Yhat):
    MSE=np.mean((Y-Yhat)**2,0)
    RMSE= np.sqrt(MSE)
    MAE=np.mean(np.abs(Y-Yhat),0)
    Ybar=np.mean(Y,0)
    Q2=1-np.sum((Y-Yhat)**2,0)/np.sum((Y-Ybar)**2,0)    
    met=pd.DataFrame([MSE, RMSE, MAE, Q2],index=['MSE','RMSE','MAE', 'R2'])
    return(met)

#load data, save input and output into seperate arrays X,Y
data=pd.read_csv('xs.csv')
X=data.iloc[:,0:8]   
Y=data.iloc[:,-1]   #-1 Python will start indexing from the end (keff)
print(data.head())

#This is a handy function to split into training/testing, can you think of an alternative way for splitting?
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state=42)


# make the model
learning_rate = 0.0009
batch_size = 8
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

nn_model.fit(Xtrain, Ytrain, epochs=20, batch_size=batch_size, validation_split=.15)

Ynn_test = nn_model.predict(Xtest)
Ynn_test = Ynn_test.reshape(Ynn_test.shape[0])
train_metrics=calc_metric(Ytest, Ynn_test)
print('NN Test=', train_metrics)

Ynn_train = nn_model.predict(Xtrain)
Ynn_train = Ynn_train.reshape(Ynn_train.shape[0])
train_metrics=calc_metric(Ytrain, Ynn_train)
print('NN Train=', train_metrics)
print(Ynn_train[0], Ytrain[0])

# NOTE: poor R2 performance with negative values (~ -.2). This is due
# to a nonlinear model being used.



# make another model
learning_rate = 0.0009
batch_size = 8
nn_model = keras.models.Sequential()
nn_model.add(keras.Input(shape=Xtrain.shape[1], batch_size=batch_size))
nn_model.add(Dense(10, activation='linear'))
# nn_model.add(Dense(100, activation='linear'))
# nn_model.add(Dense(100, activation='linear'))
nn_model.add(Dense(1, activation='linear'))

nn_model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                #  loss=keras.losses.BinaryCrossentropy(),
                #  loss='mean_absolute_error',
                 loss='mean_squared_error',
                #  metrics=['mean_absolute_error'],
                 )

nn_model.summary()

nn_model.fit(Xtrain, Ytrain, epochs=20, batch_size=batch_size, validation_split=.15)

Ynn_test = nn_model.predict(Xtest)
Ynn_test = Ynn_test.reshape(Ynn_test.shape[0])
train_metrics=calc_metric(Ytest, Ynn_test)
print('NN Test=', train_metrics)

Ynn_train = nn_model.predict(Xtrain)
Ynn_train = Ynn_train.reshape(Ynn_train.shape[0])
train_metrics=calc_metric(Ytrain, Ynn_train)
print('NN Train=', train_metrics)
print(Ynn_train[0], Ytrain[0])

# NOTE: since the underlying data is best modeled with a linear model,
# fewer layers and linear activation actually works as well (poorly still) xD