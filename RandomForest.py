import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.ensemble
import sklearn.preprocessing
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

##############
#Training on split labelled data and checking accuracy of model

print("Training on split labelled data and checking accuracy of model")

X_train = pd.read_csv('x_train.csv',header=0, index_col=0)
#print(X_train.head())


y_train = pd.read_csv('y_train.csv',header=0, index_col=0)
#print(y_train.head())

X_test = pd.read_csv('x_val.csv',header=0, index_col=0)
#print(X_test.head())

y_test = pd.read_csv('y_val.csv',header=0, index_col=0)
#print(y_test.head())

#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print(clf.predict_proba(X_test))

##############

#Training on all labelled data and predicting unlabelled cases

print("Training on all labelled data and predicting unlabelled cases")

train_data = pd.read_csv('train_mean_features_labelled.csv',header=0, index_col=0)

X_train = train_data.loc[:, train_data.columns != 'label']
y_train = train_data.loc[:, train_data.columns == 'label']

X_test = pd.read_csv('test_mean_features.csv',header=0, index_col=0)


#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

print("Number of positive cases identified in test data:")
print(sum(clf.predict(X_test)))
print("Number of total cases in test data:")
print(len(clf.predict(X_test)))
print("Prediction Scores:")
print(clf.predict_proba(X_test))
