# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 09:41:24 2018

@author: user
"""

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
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMAResults 
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from pandas import Series
from statsmodels.tsa.stattools import adfuller

series = Series.from_csv('ddata2.csv', header=None)

series.plot()
pyplot.show()

decomposition = seasonal_decompose(series,freq=4)  
fig = decomposition.plot()  
trend = decomposition.trend
seasonal = decomposition.seasonal 
 
decomposition = seasonal_decompose(series, freq=12) 
seasonal = decomposition.seasonal 
fig = plt.figure()  

fig = seasonal.plot()  
fig.set_size_inches(8, 4)

fig = plt.figure()  
fig = seasonal.plot()  
fig.set_size_inches(8, 4)


def test_stationarity(timeseries):

    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    plt.figure(figsize=(8,4))
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput )

test_stationarity(series)



series_log= series.apply(lambda x: np.log(x))  
test_stationarity(series_log)


first_difference =series - series.shift(1)  
test_stationarity(first_difference.dropna(inplace=False))



seasonal_difference = series - series.shift(12)  
test_stationarity(seasonal_difference.dropna(inplace=False))

seasonal_first_difference = first_difference - first_difference.shift(12)  
test_stationarity(seasonal_first_difference.dropna(inplace=False))

fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(series.iloc[13:], lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(series.iloc[13:], lags=40, ax=ax2)



fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(seasonal_first_difference.iloc[13:], lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(seasonal_first_difference.iloc[13:], lags=40, ax=ax2)

pyplot.figure()
pyplot.subplot(211)
plot_acf(first_difference.iloc[13:], ax=pyplot.gca())
pyplot.subplot(212)
plot_pacf(first_difference.iloc[13:], ax=pyplot.gca())
pyplot.show()

best_score = float("inf")
best =  float("inf")


p_values = range(1,13)
q_values = range(1,12)


for p in p_values:
 for q in q_values:
  mod = sm.tsa.statespace.SARIMAX(series, trend='n', order=(p,1,q), seasonal_order=(1,0,1,7))
  results = mod.fit()
  print(results.summary())
 


forecast= results.predict(start = 35)  
rmse = sqrt(mean_squared_error(series, forecast))
print('RMSE: %.3f' % rmse)
pyplot.plot(series)
pyplot.plot(forecast, color='red')