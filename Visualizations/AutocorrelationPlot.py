# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 09:28:30 2017

@author: user
"""

# create an autocorrelation plot
from pandas import Series
from matplotlib import pyplot
from pandas.tools.plotting import autocorrelation_plot
series = Series.from_csv('NorthLebanonMale.csv', header=0)
autocorrelation_plot(series)
pyplot.show()