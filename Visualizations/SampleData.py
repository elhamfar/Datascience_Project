# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 17:28:14 2018

@author: user
"""

from pandas import read_csv
from datetime import datetime
# load data
def parse(x):
	return datetime.strptime(x, '%Y %m %d')
dataset = read_csv('NorthLebanonMale.csv')

# summarize first 5 rows
print(dataset.head(5))


