# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 22:39:01 2017

@author: user
"""

# grid search ARIMA parameters for time series
import warnings
from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from math import log
from math import exp
from scipy.stats import boxcox
from pandas import read_csv
from pandas import datetime
import numpy as np

def mean_absolute_percentage_error(y_true, y_pred): 
 y_true, y_pred = np.array(y_true), np.array(y_pred)
 return np.mean(np.abs((y_true - y_pred) / y_true)*100)

# evaluate an ARIMA model for a given order (p,d,q) and return RMSE
def evaluate_arima_model(X, arima_order):
# prepare training dataset
 X = X.astype('float32')
 train_size = int(len(X) * 0.84)
 train, test = X[0:train_size], X[train_size:]
 history = [x for x in train]
# make predictions
 predictions = list()
 for t in range(len(test)):
  model = ARIMA(history, order=arima_order)
  model_fit = model.fit(disp=0)
  yhat = model_fit.forecast()[0]
  predictions.append(yhat)
  history.append(test[t])
# calculate out of sample error
 rmse = sqrt(mean_squared_error(test, predictions))
 mape = mean_absolute_percentage_error(test, predictions)
#print('MAPE=%.3f' % (mape))
 return rmse,mape
# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
 dataset = dataset.astype('float32')
 best_score, best_cfg = float("inf"), None
 for p in p_values:
  for d in d_values:
   for q in q_values:
    order = (p,d,q)
    try:
     rmse,mape = evaluate_arima_model(dataset, order)
     if rmse < best_score:
      best_score, best_mape, best_cfg = rmse, mape, order
     print('ARIMA%s RMSE=%.3f MAPE=%.3f' % (order,rmse,mape))
     
    except:
     continue
 print('Best ARIMA%s RMSE=%.3f MAPE=%.3f' % (best_cfg, best_score, best_mape))
# load dataset
def parser(x):
 return datetime.strptime(x, '%Y-%m-%d')
series = read_csv('mdata2.csv', header=None, parse_dates=[0], index_col=0,
 squeeze=True, date_parser=parser)
# evaluate parameters
p_values = [0, 1, 2,3, 4]
d_values = range(0,3)
q_values = range(0,2)
warnings.filterwarnings("ignore")
evaluate_models(series.values, p_values, d_values, q_values)