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

# evaluate an ARIMA model for a given order (p,d,q) and return RMSE
def evaluate_arima_model(X, arima_order, i):
# prepare training dataset
 X = X.astype('float32')
 train_size = int(len(X) * i)
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
 return rmse
# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
 dataset = dataset.astype('float32')
 best_score, best_cfg = float("inf"), None
 best =  float("inf")                           
 i=0.05
 while i <=0.95 :                            
  for p in p_values:
   for d in d_values:
    for q in q_values:
     order = (p,d,q)
     try:
      rmse = evaluate_arima_model(dataset, order, i)
      if rmse < best_score:
       best_score, best_cfg = rmse, order
      # print('ARIMA%s RMSE=%.3f' % (order,rmse))
     except:
      continue
  if best_score < best:
   best = best_score    
   print('Best ARIMA%s RMSE=%.3f Split=%.3f' % (best_cfg, best_score, i))
  i+=0.01
# load dataset
def parser(x):
 return datetime.strptime(x, '%Y-%m-%d')
series = read_csv('MonthlyData2.csv', header=0, parse_dates=[0], index_col=0,
 squeeze=True, date_parser=parser)
# evaluate parameters
p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = range(0,3)
q_values = range(0,6)
warnings.filterwarnings("ignore")
evaluate_models(series.values, p_values, d_values, q_values)