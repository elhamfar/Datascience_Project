# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 11:36:07 2018

@author: user
"""



from matplotlib import pyplot as plt
import numpy as np
import datetime
import statsmodels.api as sm  
from sklearn.metrics import mean_squared_error
from math import sqrt
# evaluate manually configured ARIMA model
from pandas import Series
# downsample to Weekly intervals
from pandas import read_csv

		
def mean_absolute_percentage_error(y_true, y_pred): 
 y_true, y_pred = np.array(y_true), np.array(y_pred)
 return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def parser(x):
 return datetime.datetime.strptime(x, '%Y-%m-%d')

series = Series.from_csv('BiWeeklyData.csv', header=0, parse_dates=[0], index_col=0)
# prepare data
X = series.values
X = X.astype('float32')
history = [x for x in X]
validation = read_csv('ValidationBiWeeklyData.csv', header= None, parse_dates=[0], index_col=0, 
   squeeze=True, date_parser=parser)
y = validation.values.astype('float32')
# walk-forward validation
predictions = list()
for i in range(0, len(y)):
 import warnings
 warnings.filterwarnings("ignore")
# predict
 model = sm.tsa.statespace.SARIMAX(history, trend='n', order=(0,1,0), seasonal_order=(0,0,1,6), enforce_invertibility=False)
 results = model.fit()
 model_fit = model.fit(disp=0)
 yhat = model_fit.forecast()[0]
 predictions.append(yhat)
# observation
 obs = y[i]
 history.append(obs)
 print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
 
rmse = sqrt(mean_squared_error(y, predictions))
print('RMSE: %.3f' % rmse)
mape = mean_absolute_percentage_error(y, predictions)
print('MAPE: %.3f' % mape)

fig=plt.figure()
ax=fig.add_subplot(111)
plt.plot(y)
plt.plot(predictions, color='red')
ax.set_xlabel('Week')
ax.set_ylabel('Demand') 



