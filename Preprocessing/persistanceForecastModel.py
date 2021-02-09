# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 22:10:25 2017

@author: user
"""

# evaluate a persistence forecast model
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import statistics

# load dataset
def parser(x):
 return datetime.strptime(x, '%Y-%m-%d')

def mean_absolute_deviation(y_true, y_pred): 
 y_true, y_pred = np.array(y_true), np.array(y_pred)
 return np.mean(np.abs(y_true - y_pred))

def mean_absolute_deviation_over_mean(y_true, y_pred): 
 m = statistics.mean(series)
 return mean_absolute_deviation(y_true, y_pred) / m

def mean_absolute_percentage_error(y_true, y_pred): 
 y_true, y_pred = np.array(y_true), np.array(y_pred)
 return np.mean(np.abs((y_true - y_pred) / y_true))

series = read_csv('BiweeklyData.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# create lagged dataset
values = DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1)
dataframe.columns = ['t', 't+1']
print(dataframe.head(5))
# split into train and test sets
X = dataframe.values
train_size = int(len(X) * 0.86)
train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]
# persistence model
def model_persistence(x):
 return x
# walk-forward validation
predictions = list()
for x in test_X:
 yhat = model_persistence(x)
 predictions.append(yhat)
rmse = sqrt(mean_squared_error(test_y, predictions))
mad = mean_absolute_deviation(test_y, predictions)
mape = mean_absolute_percentage_error(test_y, predictions)
mapeovermean = mean_absolute_deviation_over_mean(test_y, predictions)

print('Test RMSE: %.3f' % rmse)
print('Test MAD: %.3f' % mad)
print('Test MAPE: %.3f' % mape)
print('Test MAPE/mean: %.3f' % mapeovermean)

# plot predictions and expected results
pyplot.plot(train_y)
pyplot.plot([None for i in train_y] + [x for x in test_y])
pyplot.plot([None for i in train_y] + [x for x in predictions])
pyplot.xlabel('index')
pyplot.ylabel('Demand')
pyplot.show()

# evaluate a persistence model
from pandas import Series
from sklearn.metrics import mean_squared_error
from math import sqrt
# load data
series = Series.from_csv('NorthLebanonMale.csv',header=0)
# prepare data
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.60)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
# predict
 yhat = history[-1]
 predictions.append(yhat)
# observation
 obs = test[i]
 history.append(obs)
 print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)
pyplot.plot(train)
pyplot.plot([None for i in train] + [x for x in test])
pyplot.plot([None for i in train] + [x for x in predictions])
pyplot.show()