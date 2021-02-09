# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 02:02:23 2017

@author: user
"""

# create a line plot
from pandas import Series
from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
series1 = Series.from_csv('AkkarSyriaChild.csv', header=0)
series2 = Series.from_csv('AkkarSyriaGeneralMedicine.csv', header=0)
series3 = Series.from_csv('AkkarSyriaPediatrics.csv', header=0)
series4 = Series.from_csv('MountLebanonLebanesePediatrics.csv', header=0)
series5 = Series.from_csv('MountLebanonSyriaVaccination.csv', header=0)
series6 = Series.from_csv('NorthLebanonChild.csv', header=0)
series7 = Series.from_csv('NorthLebanonFemale.csv', header=0)
series8 = Series.from_csv('NorthLebanonMale.csv', header=0)
series9 = Series.from_csv('NorthLebanonOBG.csv', header=0)
series10 = Series.from_csv('NorthLebanonPharmacy.csv', header=0)
series11= Series.from_csv('NorthSyriaChild.csv', header=0)
series12 = Series.from_csv('NorthSyriaFemale.csv', header=0)
series13 = Series.from_csv('NorthSyriaMale.csv', header=0)
pyplot.xlabel('Date')
pyplot.ylabel('Demand')
line1, = plt.plot(series1, label="AkkarSyrianChild")
line2, = plt.plot(series2, label="AkkarSyrianGM")
line3, = plt.plot(series3, label="AkkarSyrianPed")
line4, = plt.plot(series4, label="MLLebanesePed")
line5, = plt.plot(series5, label="MLSyrianVacc")
line6, = plt.plot(series6, label="NorthLebChild")
line7, = plt.plot(series7, label="NorthLebFemale")
line8, = plt.plot(series8, label="NorthLebMale")
line9, = plt.plot(series9, label="NorthLebOBG")
line10, = plt.plot(series10, label="NorthLebPharm")
line11 = plt.plot(series11, label="NorthSyrianChild")
line12, = plt.plot(series12, label="NorthSyrianFemale")
line13, = plt.plot(series13, label="NorthSyrianMale")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})

