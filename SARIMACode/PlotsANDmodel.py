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

series = Series.from_csv('wdata2.csv')
f=7 #assign frequency
series.plot()
pyplot.show()

decomposition = seasonal_decompose(series,freq=12)  
fig = decomposition.plot()  
trend = decomposition.trend
seasonal = decomposition.seasonal 
 
decomposition = seasonal_decompose(series, freq=f) 
seasonal = decomposition.seasonal 
fig = plt.figure()  
fig = seasonal.plot()  



def test_stationarity(timeseries):

    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=f)
    rolstd = pd.rolling_std(timeseries, window=f)

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

first_difference2 =first_difference - first_difference.shift(1)  
test_stationarity(first_difference2.dropna(inplace=False))

seasonal_difference = series - series.shift(f)  
test_stationarity(seasonal_difference.dropna(inplace=False))


seasonal_first_difference = first_difference - first_difference.shift(f)  
test_stationarity(seasonal_first_difference.dropna(inplace=False))

fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(series.iloc[1:], lags=600, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(series.iloc[1:], lags=600, ax=ax2)



fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(seasonal_first_difference.iloc[1:], lags=5, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(seasonal_first_difference.iloc[1:], lags=5, ax=ax2)

fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(first_difference.iloc[1:], lags=30, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(first_difference.iloc[1:], lags=30, ax=ax2)



fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(seasonal_difference.iloc[f+1:], lags=6, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(seasonal_difference.iloc[f+1:], lags=6, ax=ax2)

