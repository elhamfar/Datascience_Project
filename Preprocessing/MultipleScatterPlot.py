# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 13:38:24 2017

@author: user
"""
# square root transform a time series
from pandas import Series
from pandas import DataFrame
from numpy import sqrt
from matplotlib import pyplot
series = Series.from_csv('TimeSeriesDataset.csv', header=0)
dataframe = DataFrame(series.values)
dataframe.columns = ['Demand']
dataframe['Demand'] = sqrt(dataframe['Demand'])
pyplot.figure(1)
# line plot
pyplot.subplot(211)
pyplot.plot(dataframe['Demand'])
# histogram
pyplot.subplot(212)
pyplot.hist(dataframe['Demand'])
pyplot.show()

# load and plot a time series
from pandas import Series
from matplotlib import pyplot
series = Series.from_csv('TSData2.csv', header=0)
pyplot.figure(1)
# line plot
pyplot.subplot(211)
pyplot.plot(series)
# histogram
pyplot.subplot(212)
pyplot.hist(series)
pyplot.show()


# automatically box-cox transform a time series
from pandas import Series
from pandas import DataFrame
from scipy.stats import boxcox
from matplotlib import pyplot
series = Series.from_csv('TSData2.csv', header=0)
dataframe = DataFrame(series.values)
dataframe.columns = ['Demand']
dataframe['Demand'], lam = boxcox(dataframe['Demand'])
print('Lambda: %f' % lam)
pyplot.figure(1)
# line plot
pyplot.subplot(211)
pyplot.plot(dataframe['Demand'])
# histogram
pyplot.subplot(212)
pyplot.hist(dataframe['Demands'])
pyplot.show()


#/..log transform a time series
from pandas import Series
from pandas import DataFrame
from numpy import log
from matplotlib import pyplot
series = Series.from_csv('TSData2.csv', header=0)
dataframe = DataFrame(series.values)
dataframe.columns = ['Demand']
dataframe['Demand'] = log(dataframe['Demand'])
pyplot.figure(1)
# line plot
pyplot.subplot(211)
pyplot.plot(dataframe['Demand'])
# histogram
pyplot.subplot(212)
pyplot.hist(dataframe['Demand'])
pyplot.show()

# downsample to yearly intervals
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
def parser(x):
 return datetime.strptime(x, '%Y-%m')
series = read_csv('TimeSeriesDataset.csv', header=0, parse_dates=[0], index_col=0,
 squeeze=True, date_parser=parser)
resample = series.resample('A')
quarterly_mean_sales = resample.sum()
print(quarterly_mean_sales.head())
quarterly_mean_sales.plot()
pyplot.show()

#Autocorrelation Plot for Daily dataset
from pandas import Series
from matplotlib import pyplot
from pandas.tools.plotting import autocorrelation_plot
series = Series.from_csv('TSData2.csv', header=0)
autocorrelation_plot(series)
pyplot.show()

# create a scatter plot lag=1
from pandas import Series
from matplotlib import pyplot
from pandas.tools.plotting import lag_plot
series = Series.from_csv('TSData2.csv', header=0)
lag_plot(series)
pyplot.show()



# create multiple scatter plots different lags
from pandas import Series
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
from pandas.tools.plotting import scatter_matrix
series = Series.from_csv('TSData2.csv', header=0)
values = DataFrame(series.values)
lags = 7
columns = [values]
for i in range(1,(lags + 1)):
 columns.append(values.shift(i))
dataframe = concat(columns, axis=1)
columns = ['t+1']
for i in range(1,(lags + 1)):
 columns.append('t-' + str(i))
dataframe.columns = columns
pyplot.figure(1)
for i in range(1,(lags + 1)):
 ax = pyplot.subplot(240 + i)
 ax.set_title('t+1 vs t-' + str(i))
 pyplot.scatter(x=dataframe['t+1'].values, y=dataframe['t-'+str(i)].values)
pyplot.show()