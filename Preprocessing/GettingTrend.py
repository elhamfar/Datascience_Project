# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:12:38 2019

@author: user
"""

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
series.plot()
pyplot.show()

decomposition = seasonal_decompose(series,freq=5)  
fig = decomposition.plot()  
trend = decomposition.trend
seasonal = decomposition.seasonal

print(trend) 
from pandas import DataFrame
df= DataFrame(trend)
df.to_excel('trend2.xlsx', sheet_name='sheet1', index=False)