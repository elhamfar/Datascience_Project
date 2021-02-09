# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 23:05:42 2018

@author: user
"""

from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from math import sqrt

i=0.05
best_score = float("inf")
while i<= 0.95:

    # load dataset
    def parser(x):
     return datetime.strptime(x, '%Y-%m-%d')
    series = read_csv('BiWeeklyData.csv', header=0, parse_dates=[0], index_col=0,  
     squeeze=True, date_parser=parser)
    # create lagged dataset
    values = DataFrame(series.values)
    dataframe = concat([values.shift(1), values], axis=1)
    dataframe.columns = ['t', 't+1']
   # print(dataframe.head(5))
    # split into train and test sets
    X = dataframe.values
    train_size = int(len(X) * i)
    train, test = X[1:train_size], X[train_size:]
    train_X, train_y = train[:,0], train[:,1]
    test_X, test_y = test[:,0], test[:,1]
    # persistence model
    def model_persistence(x):
     return x
    # walk-forward validation
    predictions = list()
    i+=0.01
    for x in test_X:
     yhat = model_persistence(x)
     predictions.append(yhat)
    rmse = sqrt(mean_squared_error(test_y, predictions))
    if rmse < best_score:
      best_score = rmse
      print('Test RMSE: %.3f' % rmse)
      print('split: %.3f ' % i)