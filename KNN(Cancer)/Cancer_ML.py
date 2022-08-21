"""
Created on Sun Oct  3 22:26:08 2021

@author: Ayaz Mehmood
"""

import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

Cancer_set=pd.read_csv("cancer_data.csv")

print(Cancer_set)

#checking for dublicate and if exist remove
#droping the diagnosis column to get the features data
X_train=Cancer_set.drop(['diagnosis'],axis=1)

print(X_train)
#column name is missing in Test set so fill the column
Test_set=pd.read_csv('Test_Set.csv', names=X_train.columns, header=None)

print(Test_set)

#labels data
labels=Cancer_set['diagnosis']

#Converting catogorical data to numerical data
Y_train=labels.replace(['B','M'],[0,1])
print(Y_train)

#using feature scaling (Normalization) to scale numeric features in the same scale or range
#normalization range of value [0,1]
#Using MinMaxScaling tranformation
min_max_scaler =MinMaxScaler()
min_max_scaler.fit_transform(X_train)
X_train= min_max_scaler.transform(X_train)
Test_set=min_max_scaler.transform(Test_set)

#using KNN algorithm to train the data
clf=KNeighborsClassifier(5)
clf.fit(X_train,Y_train)

#prediction of the model by using test set
predictions=clf.predict(Test_set)
print(predictions)
Test_res=pd.DataFrame(predictions)

#saving in a CSV file
prediction = pd.DataFrame(predictions, columns=['predictions'])
print(prediction)
prediction=prediction.replace([0,1],['B','M'])
prediction.to_csv('prediction.csv')
print(prediction)

