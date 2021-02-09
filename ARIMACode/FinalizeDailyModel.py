# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 04:50:55 2018

@author: user
"""

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
# load and prepare datasets
dataset = Series.from_csv('ddata2.csv')
X = dataset.values.astype('float32')
history = [x for x in X]
validation = Series.from_csv('ValidationDailyData.csv')
y = validation.values.astype('float32')
# load model
model_fit = ARIMAResults.load('dmodel.pkl')
bias = numpy.load('dmodel_bias.npy')
#lam = numpy.load('dmodel_lambda.npy')
# make first prediction
predictions = list()
yhat =bias + model_fit.forecast()[0]
#yhat = boxcox_inverse(yhat, lam)
predictions.append(yhat)
history.append(y[0])
print('>Predicted=%.3f, Expected=%3.f' % (yhat, y[0]))
# rolling forecasts
for i in range(1, len(y)):
 # transform

# predict
 model = ARIMA(history, order=(4,0,0))
 model_fit = model.fit(disp=0)
 yhat =bias+ model_fit.forecast()[0]
# invert transformed prediction
 #yhat = boxcox_inverse(yhat, lam)
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