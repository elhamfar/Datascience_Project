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
# load data
series = Series.from_csv('WeeklyData2.csv')
# prepare data
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.77)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
 # predict
 model = ARIMA(history, order=(2,0,1))
 model_fit = model.fit(disp=0)
 yhat = model_fit.forecast()[0]
 predictions.append(yhat)
 # observation
 obs = test[i]
 history.append(obs)
 # evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# errors
residuals = [test[i]-predictions[i] for i in range(len(test))]
residuals = DataFrame(residuals)

print(residuals.describe())
# histogram plot

residuals.hist()
pyplot.show()
# density plot
residuals.plot(kind='kde')
pyplot.show()
pyplot.subplot(211)
plot_acf(residuals, ax=pyplot.gca())
pyplot.subplot(212)
plot_pacf(residuals, ax=pyplot.gca())
pyplot.show()