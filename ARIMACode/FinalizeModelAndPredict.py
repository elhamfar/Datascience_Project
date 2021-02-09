# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 09:06:29 2017

@author: user
"""

# save finalized model to file
from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
from scipy.stats import boxcox
import numpy
# monkey patch around bug in ARIMA class
def __getnewargs__(self):
 return ((self.endog),(self.k_lags, self.k_diff, self.k_ma))

ARIMA.__getnewargs__ = __getnewargs__

# load data
series = Series.from_csv('MonthlyData.csv')
# prepare data
X = series.values
X = X.astype('float32')
# transform data
transformed, lam = boxcox(X)
# fit model
model = ARIMA(transformed, order=(4,0,0))
model_fit = model.fit(disp=0)

# bias constant, could be calculated from in-sample mean residual
bias = 13.319375
# save model
model_fit.save('MModel.pkl')
numpy.save('MModel_lambda.npy', [lam])
numpy.save('MModel_bias.npy', [bias])


#load the finalized model and make a prediction
from pandas import Series
from statsmodels.tsa.arima_model import ARIMAResults
from math import exp
from math import log
import numpy
# invert box-cox transform
def boxcox_inverse(value, lam):
 if lam == 0:
  return exp(value)
 return exp(log(lam * value + 1) / lam)

model_fit = ARIMAResults.load('MModel.pkl')
lam = numpy.load('MModel_lambda.npy')
bias = numpy.load('MModel_bias.npy')
yhat, stderr, conf = model_fit.forecast()
yhat = bias + boxcox_inverse(yhat, lam)
print('Forecast: %.3f' % yhat)
print('Standard Error: %.3f' % stderr)
print('95%% Confidence Interval: %.3f to %.3f' % (conf[0][0], conf[0][1]))



# evaluate the finalized model on the validation dataset
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from scipy.stats import boxcox
from sklearn.metrics import mean_squared_error
from math import sqrt
from math import exp
from math import log
import numpy
# invert box-cox transform
def boxcox_inverse(value, lam):
 if lam == 0:
  return exp(value)
 return exp(log(lam * value + 1) / lam)
# load and prepare datasets
dataset = Series.from_csv('MonthlyData.csv')
X = dataset.values.astype('float32')
history = [x for x in X]
validation = Series.from_csv('ValidationMonthlyData.csv')
y = validation.values.astype('float32')
# load model
model_fit = ARIMAResults.load('Mmodel.pkl')
lam = numpy.load('Mmodel_lambda.npy')
bias = numpy.load('MModel_bias.npy')
# make first prediction
predictions = list()
yhat = model_fit.forecast()[0]
yhat = boxcox_inverse(yhat, lam)
predictions.append(yhat)
history.append(y[0])
print('>Predicted=%.3f, Expected=%3.f' % (yhat, y[0]))
# rolling forecasts
for i in range(1, len(y)):
# transform
 transformed, lam = boxcox(history)
 if lam < -5:
  transformed, lam = history, 1
# predict
 model = ARIMA(transformed, order=(4,0,0))
 model_fit = model.fit(disp=0)
 yhat = model_fit.forecast()[0]
# invert transformed prediction
 yhat = bias+boxcox_inverse(yhat, lam)
 predictions.append(yhat)
# observation

 obs = y[i]
 history.append(obs)
 print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
# report performance
rmse = sqrt(mean_squared_error(y, predictions))
print('RMSE: %.3f' % rmse)
pyplot.plot(y)
pyplot.plot(predictions, color='red')

# summarize the confidence interval on an ARIMA forecast
from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
# load dataset
series = Series.from_csv('MonthlyData.csv', header=0)
# split into train and test sets
X = series.values
X = X.astype('float32')
size = len(X) - 1
train, test = X[0:size], X[size:]
# fit an ARIMA model
model = ARIMA(train, order=(4,0,0))
model_fit = model.fit(disp=False)
# forecast
forecast, stderr, conf = model_fit.forecast()
# summarize forecast and confidence intervals
print('Expected: %.3f' % test[0])
print('Forecast: %.3f' % forecast)
print('Standard Error: %.3f' % stderr)
print('95%% Confidence Interval: %.3f to %.3f' % (conf[0][0], conf[0][1]))