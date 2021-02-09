# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 11:18:36 2019

@author: user
"""
import numpy as np
import lime
import lime.lime_tabular
from pandas import read_csv
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
import pandas as pd

# Add your path
dataframe = read_csv('C:/Users/user/Desktop/cmps297/With Dr. Wassim/MultivariatePart/Weekly/TwoClasses/WMultiTestCSV.csv')
# separate into input and output variables
array = dataframe.values
X = array[:,0:-1]
y = array[:,-1]
names = dataframe.columns.values[0:-1]

print(X)
#### Extract the label column
train_target = y
train_df = X

# Split into training and validation set
validation_size = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X, y, test_size=validation_size, random_state=seed)



scoring = 'accuracy'
print(X_train)

# Model 1 - Logisitic Regression
model_logreg = LogisticRegression()
model_logreg.fit(X_train, Y_train)
ACC1=accuracy_score(Y_validation, model_logreg.predict(X_validation))
print(metrics.classification_report(Y_validation, model_logreg.predict(X_validation)))
print(metrics.confusion_matrix(Y_validation, model_logreg.predict(X_validation)))

print(ACC1)

# LIME SECTION
import lime
import lime.lime_tabular

predict_fn_logreg = lambda x: model_logreg.predict_proba(x).astype(float)
#predict_fn = lambda x: model.predict_proba(x).astype(float)


categorical_features = np.argwhere(
np.array([len(set(X[:,x]))
for x in range(X.shape[1])]) <= 10).flatten()

# Create the LIME Explainer
explainer = lime.lime_tabular.LimeTabularExplainer(X_train ,feature_names = names,class_names=['L','H'],
                                                   categorical_features=categorical_features, 
                                                   )

# Pick the observation in the validation set for which explanation is required
observation_1 = 2
print(X_validation[observation_1])
print(Y_validation[observation_1])
# Get the explanation for Logistic Regression
exp = explainer.explain_instance(X_validation[observation_1], predict_fn_logreg, num_features=9)
exp.show_in_notebook(show_all=False)

# Get the explanation for NB
#exp = explainer.explain_instance(X_validation[observation_1], predict_fn, num_features=9)
#exp.show_in_notebook(show_all=False)

