# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 01:00:26 2017

@author: user
"""

# evaluate ARIMA models with box-cox transformed time series
from pandas import Series
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from math import sqrt
from math import log
from math import exp
from scipy.stats import boxcox
# invert box-cox transform
def boxcox_inverse(value, lam):
 if lam == 0:
  return exp(value)
 return exp(log(lam * value + 1) / lam)
# load data
series = Series.from_csv('MonthlyData2.csv')
# prepare data
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.77)

train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
# transform
 transformed, lam = boxcox(history)
 if lam < -5:
  transformed, lam = history, 1
# predict
 model = ARIMA(transformed, order=(2,2,1))
 model_fit = model.fit(disp=0)
 yhat = model_fit.forecast()[0]
# invert transformed prediction
 yhat = boxcox_inverse(yhat, lam)
 predictions.append(yhat)
# observation
 obs = test[i]
 history.append(obs)
 print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)