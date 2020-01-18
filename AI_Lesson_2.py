# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 19:38:46 2019

@author: Scarc
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:, :-1]
Y = dataset.iloc[:, 4]

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X.iloc[: , 3] = labelEncoder_X.fit_transform(X.iloc[:,3])
ohe = OneHotEncoder(categorical_features = [3])
X = ohe.fit_transform(X).toarray()

#Avoiding the dummy var trap

X = X[:,1:]

from sklearn.model_selection import train_test_split as tts
X_train,X_test,Y_train,Y_test = tts(X,Y,test_size = 0.2,random_state = 0)

#Fitting Multiple Linear regression to the training set
from sklearn.linear_model import LinearRegression as lr
regressor = lr()
regressor.fit(X_train,Y_train)

#Predicting the test set results
Y_pred = regressor.predict(X_test)

#Building the optimal model using Backward Elimination 
import statsmodels.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = Y,exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = Y,exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = Y,exog = X_opt).fit()
regressor_OLS.summary()


X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = Y,exog = X_opt).fit()
regressor_OLS.summary()


X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog = Y,exog = X_opt).fit()
regressor_OLS.summary()