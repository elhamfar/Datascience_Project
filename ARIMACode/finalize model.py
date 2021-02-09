# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 07:02:26 2018

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
series = Series.from_csv('WeeklyData2.csv')
# prepare data
X = series.values
X = X.astype('float32')
# fit model
model = ARIMA(X, order=(2,0,1))
model_fit = model.fit(trend='nc', disp=0)
# bias constant, could be calculated from in-sample mean residual
bias = -3.036627
# save model
model_fit.save('Wmodel.pkl')
numpy.save('Wmodel_bias.npy', [bias])








# load finalized model and make a prediction
from pandas import Series
from statsmodels.tsa.arima_model import ARIMAResults
import numpy
model_fit = ARIMAResults.load('Wmodel.pkl')
bias = numpy.load('Wmodel_bias.npy')
yhat = bias + float(model_fit.forecast()[0])
print('Predicted: %.3f' % yhat)





# load and evaluate the finalized model on the validation dataset
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy
# load and prepare datasets
dataset = Series.from_csv('DailyData2.csv')
X = dataset.values.astype('float32')
history = [x for x in X]
validation = Series.from_csv('ValidationDailyData.csv')
y = validation.values.astype('float32')
# load model
model_fit = ARIMAResults.load('Dmodel.pkl')
bias = numpy.load('Dmodel_bias.npy')
# make first prediction
predictions = list()
yhat = bias + float(model_fit.forecast()[0])
predictions.append(yhat)
history.append(y[0])
print('>Predicted=%.3f, Expected=%3.f' % (yhat, y[0]))
# rolling forecasts
for i in range(1, len(y)):
# predict
 model = ARIMA(history, order=(6,0,0))
 model_fit = model.fit(trend='nc', disp=0)
 yhat = bias + float(model_fit.forecast()[0])
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