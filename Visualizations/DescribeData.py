# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 01:59:11 2017

@author: user
"""

# calculate descriptive statistics
from pandas import Series
series = Series.from_csv('mdata2.csv')
print(series.describe())