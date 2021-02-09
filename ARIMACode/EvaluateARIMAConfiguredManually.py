# -*- coding
# statistical test for the stationarity of the time series

"""
Created on Sun Oct  8 00:44:43 2017

@author: user

"""
#evaluate manually configured ARIMA model
from pandas import Series
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from math import sqrt
# load data
series = Series.from_csv('DailyData.csv')
# prepare data
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.66)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):

# predict
 model = ARIMA(history, order=(1,1,5))
 model_fit = model.fit(disp=0)
 yhat = model_fit.forecast()[0]
 predictions.append(yhat)
# observation
 obs = test[i]
 history.append(obs)
 print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)

