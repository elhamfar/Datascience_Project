# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 20:36:37 2018

@author: user
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from dateutil.relativedelta import relativedelta
import seaborn as sns
import statsmodels.api as sm  
from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas import read_csv
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from math import sqrt
# evaluate manually configured ARIMA model
from pandas import Series
import statistics
import numpy as np

def mean_absolute_deviation(y_true, y_pred): 
 y_true, y_pred = np.array(y_true), np.array(y_pred)
 return np.mean(np.abs(y_true - y_pred))
		
def mean_absolute_percentage_error(y_true, y_pred): 
 y_true, y_pred = np.array(y_true), np.array(y_pred)
 return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mean_absolute_deviation_over_mean(y_true, y_pred): 
 m = statistics.mean(series)
 return mean_absolute_deviation(y_true, y_pred) / m


# load data
import numpy as np
import itertools
import sys
import timeit

start_time = timeit.default_timer()

series = Series.from_csv('wdata2.csv', header=0)
# prepare data
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()


p =q = range(0, 6)
d=  range(0, 2)
# generate all different combinations of p, d and q triplets
pdq = list(itertools.product(p, d, q))
 
# generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 3) for x in list(itertools.product(p, d, q))]

best_bic = np.inf
best_pdq = None
best_seasonal_pdq = None
tmp_model = None
best_mdl = None


for param in pdq:
 for param_seasonal in seasonal_pdq:
  try:
   warnings.filterwarnings('ignore')
   tmp_mdl = sm.tsa.statespace.SARIMAX(history,
                                       order = param,
                                       seasonal_order = param_seasonal,
                                       enforce_stationarity=True,
                                       enforce_invertibility=True)
   res = tmp_mdl.fit()
   if res.bic < best_bic:
    best_bic = res.bic
    best_pdq = param
    best_seasonal_pdq = param_seasonal
    best_mdl = tmp_mdl
    print("SARIMAX{}x{} model - BIC:{}".format(best_pdq, best_seasonal_pdq, best_bic))
    
    
  except:
   #print("Unexpected error:", sys.exc_info()[0])
   continue
 
 

print("Best SARIMAX{}x{} model - BIC:{}".format(best_pdq, best_seasonal_pdq, best_bic))

elapsed = timeit.default_timer() - start_time
print(elapsed)

