

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
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
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from math import sqrt
# load data

import numpy as np
		
def mean_absolute_percentage_error(y_true, y_pred): 
 y_true, y_pred = np.array(y_true), np.array(y_pred)
 return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



series = Series.from_csv('ddata2.csv', header=0)
# prepare data
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.94)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()
import warnings

for p in range(0,2):
 for q in range(0,2): 
  try:
   for i in range(len(test)):
    warnings.filterwarnings("ignore")
  
    model = sm.tsa.statespace.SARIMAX(history, trend='n', order=(p,1,q), seasonal_order=(0,1,1,90))
    results = model.fit()
    model_fit = model.fit(disp=0)
    yhat = model_fit.forecast()[0]
    predictions.append(yhat)
# observation
    obs = test[i]
    history.append(obs)
   rmse = sqrt(mean_squared_error(test, predictions))
   print('RMSE: %.3f' % rmse)
   mape = mean_absolute_percentage_error(test, predictions)
   print('MAPE: %.3f' % mape)   
   print('p: %.3f' % p)
   print('q: %.3f' % q)

  except :
   continue
  history = [x for x in train]
  predictions = list() 





