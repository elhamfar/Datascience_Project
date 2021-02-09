# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 17:14:23 2018

@author: user
"""

from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot

		
def mean_absolute_percentage_error(y_true, y_pred): 
 y_true, y_pred = np.array(y_true), np.array(y_pred)
 return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# load dataset
def parser(x):
	return datetime.strptime(x, '%Y-%m-%d')
series = read_csv('mdata2.csv', header=None, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# split data into train and test
X = series.values
train_size = int(len(X) * 0.84)
train, test = X[0:-(len(X)-train_size)], X[-(len(X)-train_size):]
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	# make prediction
	predictions.append(history[-1])
	# observation
	history.append(test[i])
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)
mape = mean_absolute_percentage_error(test, predictions)
print('MAPE: %.3f' % mape)
# line plot of observed vs predicted
pyplot.plot(test)
pyplot.plot(predictions)
pyplot.show()