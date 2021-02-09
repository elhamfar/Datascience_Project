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
series = Series.from_csv('mdata2.csv')
# prepare data
X = series.values
X = X.astype('float32')
# transform data
transformed, lam = boxcox(X)
# fit model
model = ARIMA(transformed, order=(3,2,0))
model_fit = model.fit(disp=0)

# bias constant, could be calculated from in-sample mean residual
bias = -10.526363
# save model
model_fit.save('mmodel.pkl')
numpy.save('mmodel_lambda.npy', [lam])
numpy.save('mmodel_bias.npy', [bias])


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

model_fit = ARIMAResults.load('bwmodel.pkl')
lam = numpy.load('bwmodel_lambda.npy')
bias = numpy.load('bwmodel_bias.npy')
yhat= model_fit.forecast()[0]
yhat = bias + boxcox_inverse(yhat, lam)
print('Predicted: %.3f' % yhat)


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
import numpy as np
# invert box-cox transform
def boxcox_inverse(value, lam):
 if lam == 0:
  return exp(value)
 return exp(log(lam * value + 1) / lam)

def mean_absolute_percentage_error(y_true, y_pred): 
 y_true, y_pred = np.array(y_true), np.array(y_pred)
 return np.mean(np.abs((y_true - y_pred) *100/ y_true))

# load and prepare datasets
dataset = Series.from_csv('mdata2.csv')
X = dataset.values.astype('float32')
history = [x for x in X]
validation = Series.from_csv('ValidationMonthlyData.csv')
y = validation.values.astype('float32')
# load model
model_fit = ARIMAResults.load('mmodel.pkl')
lam = numpy.load('mmodel_lambda.npy')
bias = numpy.load('mmodel_bias.npy')
# make first prediction
predictions = list()
yhat = model_fit.forecast()[0]
yhat =  bias+boxcox_inverse(yhat, lam)
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
 model = ARIMA(transformed, order=(3,2,0))
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
mape = mean_absolute_percentage_error(y, predictions)


print('Test RMSE: %.3f' % rmse)
print('Test MAPE: %.3f' % mape)


pyplot.plot(y)
pyplot.plot(predictions, color='red')
