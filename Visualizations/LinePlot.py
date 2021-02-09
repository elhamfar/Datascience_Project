# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 02:02:23 2017

@author: user
"""

# create a line plot
from pandas import Series
from matplotlib import pyplot
series = Series.from_csv('DeathsOfNonCivilians.csv', header=None)

pyplot.xlabel('Date')
pyplot.ylabel('DeathRate')

series.plot()
pyplot.show()