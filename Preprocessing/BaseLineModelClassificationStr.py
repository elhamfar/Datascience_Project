# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 16:33:19 2019

@author: user
"""

# Load Libraries 
import numpy as np  

from pandas import read_csv
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

dataframe = read_csv('C:/Users/user/Desktop/cmps297/With Dr. Wassim/MultivariatePart/Weekly/TwoClasses/WMultiTestCSV.csv', header=0)
# separate into input and output variables
array = dataframe.values
X = array[:,0:-1]
y = array[:,-1]
names = dataframe.columns.values[0:-1]

from sklearn.dummy import DummyClassifier
dummy = DummyClassifier(strategy='stratified', random_state = 100, constant = None) 
dummy.fit(X, y)   

## Number of classes 
print ("Number of classes :{}".format(dummy.n_classes_)) ## Number of classes assigned to each tuple 
print ("Number of classes assigned to each tuple :{}".format(dummy.n_outputs_)) ### Prior distribution of the classes. 
print ("Prior distribution of the classes {}".format(dummy.class_prior_)) 

output = np.random.multinomial(1, [.33,.33,.33], size = X.shape[0]) 
predictions = output.argmax(axis=1) 
print (output[0] )
print (predictions[1])  


y_predicted = dummy.predict(X)  
#print (y_predicted) 
# Find model accuracy 
print ("Model accuracy = %0.2f"%(accuracy_score(y,y_predicted) * 100) + "%\n") 
# Confusion matrix 
print ("Confusion Matrix") 
print (confusion_matrix(y, y_predicted, labels=list(set(y))) )  

from sklearn.metrics import precision_score, recall_score, f1_score 
print ("Dummy Model 1, strategy: stratified, accuracy {0:.2f}, precision {0:.2f}, recall {0:.2f}, f1-score {0:.2f}"\
						.format(accuracy_score(y, y_predicted), precision_score(y, y_predicted), recall_score(y, y_predicted), f1_score(y, y_predicted)))

   