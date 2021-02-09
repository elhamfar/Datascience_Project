# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 02:20:09 2017

@author: user
"""

# create a density plot
from pandas import Series
from matplotlib import pyplot
series = Series.from_csv('TSData2.csv', header=0)

series.plot(kind='kde')
pyplot.show()