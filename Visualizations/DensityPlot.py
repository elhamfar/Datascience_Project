# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 09:34:55 2017

@author: user
"""

# create a density plot
from pandas import Series
from matplotlib import pyplot
series = Series.from_csv('NorthLebanonMale.csv', header=0)
pyplot.xlabel('Demand')
series.plot(kind='kde')
pyplot.show()