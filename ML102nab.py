import numpy as np   #numpy
import pandas as pd # Dataframes
import tensorrt
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
# Import necessary modules
from sklearn.metrics import mean_squared_error
# Keras specific
import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

#handy function to determine different validation metrics of the regression model
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
print(data.shape)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state=1)

xscaler = MinMaxScaler()
Xtrain=xscaler.fit_transform(Xtrain)
Xtest=xscaler.transform(Xtest)

# Define model
learning_rate=0.0009
model = Sequential()
model.add(Dense(100, kernel_initializer='normal',input_dim = 8, activation= "relu"))
model.add(Dropout(0.5))
model.add(Dense(100, kernel_initializer='normal', activation= "relu"))
model.add(Dense(100, kernel_initializer='normal', activation= "relu"))
model.add(Dense(100, kernel_initializer='normal', activation= "relu"))
#Last layer (use linear activation and set nodes to number of Y columns/labels/outputs
model.add(Dense(1, kernel_initializer='normal', activation='linear'))


model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate), metrics=['mean_squared_error'])
model.summary()

checkpoint = ModelCheckpoint('mymodel.pkl', monitor='val_mean_absolute_error', save_best_only=True, mode='min', verbose=1)
history=model.fit(Xtrain, Ytrain, epochs=20, batch_size=8, validation_split = 0.15, callbacks=[checkpoint], verbose=True)


pred_train=model.predict(Xtrain)
df_train=pd.DataFrame(pred_train)
train_metrics_nn=calc_metric(Ytrain, df_train)
print('NN Train=', train_metrics_nn)

