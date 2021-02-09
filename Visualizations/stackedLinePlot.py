# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 12:57:17 2018

@author: user
"""

# create stacked line plots
from pandas import Series
from pandas import DataFrame
from pandas import TimeGrouper
from matplotlib import pyplot
from pandas import concat
series = Series.from_csv('NorthLebanonMale.csv', header=0, parse_dates=[0], index_col=0)
one_year = series['2013':'2016']
groups = one_year.groupby(TimeGrouper('A'))
years = concat([DataFrame(x[1].values) for x in groups], axis=1)
years = DataFrame(years)
years.columns = range(1,5)
years.plot(subplots=True, legend=False)
pyplot.xlabel('Months')
pyplot.ylabel('Demand')
pyplot.show()