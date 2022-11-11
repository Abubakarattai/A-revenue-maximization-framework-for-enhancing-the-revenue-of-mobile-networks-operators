# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 15:25:44 2020

@author: msamw
"""


import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import np_utils

import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#hyper parameters
LEARNING_RATE = 0.0009

dataset1 = pd.read_csv('RevMax_12Scs_ES_hyb.csv', header = None)
dataset2 = pd.read_csv('RevMax_12Scs_ES.csv', header = None)

# creating input features and target variables
#training dataset
X_train = dataset1.iloc[0:5000, 0:25]   # default value 50400 
Y_train = dataset1.iloc[0:5000, 37]
#y_train = np_utils.to_categorical(Y_train)
y_train = Y_train

#test dataset
X_test = dataset2.iloc[0:1008,0:25]
Y_test = dataset2.iloc[0:1008,37]
#y_test = np_utils.to_categorical(Y_test)
y_test = Y_test

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=123)


model = xgb.XGBClassifier()



model.fit(X_train, y_train, eval_metric="error", verbose=True)




# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))