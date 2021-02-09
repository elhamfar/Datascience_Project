# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 06:40:06 2018

@author: user
"""

# finalize model and save to file with workaround
from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
from scipy.stats import boxcox
import numpy
# monkey patch around bug in ARIMA class
def __getnewargs__(self):
 return ((self.endog),(self.k_lags, self.k_diff, self.k_ma))

ARIMA.__getnewargs__ = __getnewargs__
# load data
series = Series.from_csv('WeeklyData2.csv')
# prepare data
X = series.values
X = X.astype('float32')
# transform data
transformed, lam = boxcox(X)
# fit model
model = ARIMA(transformed, order=(2,0,1))
model_fit = model.fit(disp=0)
# save model
model_fit.save('Wmodel2.pkl')
numpy.save('Wmodel_lambda.npy', [lam])


# load the finalized model and make a prediction
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
model_fit = ARIMAResults.load('Wmodel2.pkl')
lam = numpy.load('Wmodel_lambda.npy')
yhat = model_fit.forecast()[0]
yhat = boxcox_inverse(yhat, lam)
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
# invert box-cox transform
def boxcox_inverse(value, lam):
 if lam == 0:
  return exp(value)
 return exp(log(lam * value + 1) / lam)
# load and prepare datasets
dataset = Series.from_csv('WeeklyData2.csv')
X = dataset.values.astype('float32')
history = [x for x in X]
validation = Series.from_csv('ValidationWeeklyData.csv')
y = validation.values.astype('float32')
# load model
model_fit = ARIMAResults.load('Wmodel2.pkl')
lam = numpy.load('Wmodel_lambda.npy')
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
 model = ARIMA(transformed, order=(2,0,1))
 model_fit = model.fit(disp=0)
 yhat = model_fit.forecast()[0]
# invert transformed prediction
 yhat = boxcox_inverse(yhat, lam)
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
pyplot.show()