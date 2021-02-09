# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 02:54:09 2017

@author: user
"""

# downsample to weekly intervals
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
def parser(x):
 return datetime.strptime(x, '%Y-%m-%d')
series = read_csv('DailyTimeSeriesTwoYears.csv', header=0, parse_dates=[0], index_col=0,
 squeeze=True, date_parser=parser)
resample = series.resample('W')
weekly_mean_sales = resample.mean()
print(weekly_mean_sales.head(60))
weekly_mean_sales.plot()
weekly_mean_sales.to_csv('out.csv')
pyplot.show()