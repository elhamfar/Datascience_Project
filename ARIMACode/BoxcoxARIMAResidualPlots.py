# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 13:49:38 2018

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 23:37:37 2017

@author: user
"""

# plot residual errors for ARIMA model
from pandas import Series
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from math import sqrt
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from math import log
from math import exp
from scipy.stats import boxcox
import numpy as np

def mean_absolute_percentage_error(y_true, y_pred): 
 y_true, y_pred = np.array(y_true), np.array(y_pred)
 return np.mean(np.abs((y_true - y_pred) / y_true)*100)
# invert box-cox transform
def boxcox_inverse(value, lam):
 if lam == 0:
  return exp(value)
 return exp(log(lam * value + 1) / lam)
# load data
series = Series.from_csv('mdata2.csv')
# prepare data
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.84)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()
bias=-9.738387
for i in range(len(test)):
# transform
 transformed, lam = boxcox(history)
 if lam < -5:
  transformed, lam = history, 1
# predict
 model = ARIMA(transformed, order=(3,2,0))
 model_fit = model.fit(disp=0)
 yhat = model_fit.forecast()[0]
# invert transformed prediction
 yhat = bias+boxcox_inverse(yhat, lam)
 predictions.append(yhat)

# observation
 obs = test[i]
 history.append(obs)
 print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
mape = mean_absolute_percentage_error(test, predictions)
print('RMSE: %.3f' % rmse)
print('MAPE: %.3f' % mape)
# errors
residuals = [test[i]-predictions[i] for i in range(len(test))]
residuals = DataFrame(residuals)

print(residuals.describe())
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
pyplot.show()
# density plot
pyplot.subplot(211)
residuals.hist(ax=pyplot.gca())
pyplot.subplot(212)
residuals.plot(kind='kde', ax=pyplot.gca())

pyplot.show()
pyplot.subplot(211)
plot_acf(residuals, ax=pyplot.gca())
pyplot.subplot(212)
plot_pacf(residuals, ax=pyplot.gca())
pyplot.show()