# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 02:09:15 2017

@author: user
"""

# finalize model and save to file with workaround
from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
from scipy.stats import boxcox
import numpy
# monkey patch around bug in ARIMA class
def __getnewargs__(self):
 return ((self.endog),(self.k_lags, self.k_diff, self.k_ma))
ARIMA.__getnewargs__ = __getnewargs__
# load data
series = Series.from_csv('MonthlyData2.csv')
# prepare data
X = series.values
X = X.astype('float32')
# transform data
transformed, lam = boxcox(X)
# fit model
model = ARIMA(X, order=(2,2,1))
model_fit = model.fit(disp=0)
# save model
model_fit.save('monthlymodel2.pkl')
numpy.save('model_lambda2.npy', [lam])

# finalize model and save to file with workaround
from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
from scipy.stats import boxcox
import numpy
# monkey patch around bug in ARIMA class
def __getnewargs__(self):
 return ((self.endog),(self.k_lags, self.k_diff, self.k_ma))
ARIMA.__getnewargs__ = __getnewargs__
# load data
series = Series.from_csv('DailyData2.csv')
# prepare data
X = series.values
X = X.astype('float32')
# transform data
transformed, lam = boxcox(X)
# fit model
model = ARIMA(X, order=(6,0,0))
model_fit = model.fit(disp=0)
# save model
model_fit.save('dailymodel2.pkl')
numpy.save('dmodel_lambda2.npy', [lam])


# finalize model and save to file with workaround
from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
from scipy.stats import boxcox
import numpy
# monkey patch around bug in ARIMA class
def __getnewargs__(self):
 return ((self.endog),(self.k_lags, self.k_diff, self.k_ma))
ARIMA.__getnewargs__ = __getnewargs__
# load data
series = Series.from_csv('WeeklyData2.csv')
# prepare data
X = series.values
X = X.astype('float32')
# transform data
transformed, lam = boxcox(X)
# fit model
model = ARIMA(X, order=(2,0,1))
model_fit = model.fit(disp=0)
# save model
model_fit.save('weeklymodel2.pkl')
numpy.save('wmodel_lambda2.npy', [lam])
