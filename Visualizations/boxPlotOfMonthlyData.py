# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 02:25:33 2017

@author: user
"""

# create a boxplot of monthly data
from pandas import Series
from pandas import DataFrame
from pandas import TimeGrouper
from matplotlib import pyplot
from pandas import concat
series = Series.from_csv('TSData2.csv', header=0)
one_year = series['2017']
groups = one_year.groupby(TimeGrouper('M'))
months = concat([DataFrame(x[1].values) for x in groups], axis=1)
months = DataFrame(months)
months.columns = range(1,6)
months.boxplot()
pyplot.show()