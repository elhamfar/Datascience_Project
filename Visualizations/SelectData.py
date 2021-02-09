# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 01:57:07 2017

@author: user
"""

from pandas import Series
series = Series.from_csv('TimeSeriesDataset.csv', header=0)
print(series['Jan-15'])