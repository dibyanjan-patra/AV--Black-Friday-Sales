# -*- coding: utf-8 -*-
"""
Created on Fri May 22 16:39:26 2020

@author: dibya
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")

test_original = pd.read_csv("D:\\Data Science\\Projects\\Black friday Sales\\test.csv")
train_original = pd.read_csv("D:\\Data Science\\Projects\\Black friday Sales\\train.csv")

#taking a copy of orignial data
train = train_original.copy()
test = test_original.copy()

######  Explarotary Data Analysis ####
train.columns
test.columns

train.describe() #description of all variables
train.info() #give details about data frame
train.dtypes   #give data types of variable
train.shape

#dropping user id and product id from test and train
train.drop(["User_ID","Product_ID"],axis=1,inplace=True)
test.drop(["User_ID","Product_ID"],axis=1,inplace=True)

#Removing the + sign
train['Age']=(train['Age'].str.strip('+'))
train['Stay_In_Current_City_Years']=(train['Stay_In_Current_City_Years'].str.strip('+').astype('float'))

#Removing for test sign
test['Age']=(test['Age'].str.strip('+'))
test['Stay_In_Current_City_Years']=(test['Stay_In_Current_City_Years'].str.strip('+').astype('float'))

#Graph Plotting// univariate analysis   #######
## For Gender
train['Gender'].value_counts(normalize=True)  #to be in percentage
sns.countplot(x='Gender',data=train,palette='hls')
#train['Gender'].value_counts(normalize=True).plot.bar(title='Gender') to show the percent of values
#For age
train['Age'].value_counts(normalize=True)  #to be in percentage
sns.countplot(x='Age',data=train,palette='hls')
#For city_category
train['City_Category'].value_counts(normalize=True)  #to be in percentage
sns.countplot(x='City_Category',data=train,palette='hls')
#For Stay_In_Current_City_Years
train['Stay_In_Current_City_Years'].value_counts(normalize=True)  #to be in percentage
sns.countplot(x='Stay_In_Current_City_Years',data=train,palette='hls')
#inferred that max people have stayed 1 yrs in city


#Heatmap to find co-relation
sns.heatmap(train.corr(),annot=True)
#positive correlation coefficientsare these three- occupation,Marital Status,stay_in_current_city_years

#Plotting distplot
sns.distplot(train['Product_Category_1'])
sns.distplot(train['Product_Category_2'])
sns.distplot(train['Product_Category_3'])

#Plotting boxplot for outliers
sns.boxplot(train['Product_Category_1'])  #very less outliers
sns.boxplot(train['Product_Category_2']) # no outliers such present
sns.boxplot(train['Product_Category_3'])   #no outliers



#Checking for isnull value
train.isnull().sum()  #product category 2 and 3 are having null values
#inference- Since no outliers are present hence applying mean outliers
#applying mean outliers
train = train.fillna({
    'Product_Category_2': np.mean(train["Product_Category_2"]),
    'Product_Category_3': np.mean(train["Product_Category_3"])
    })

train.isnull().sum()   #np null values not present

#checking for test dataset
test.isnull().sum()
#applying mean imputation
test = test.fillna({
    'Product_Category_2': np.mean(test["Product_Category_2"]),
    'Product_Category_3': np.mean(test["Product_Category_3"])
    })

#Creating dummy variable for catgorical values
y=train.Purchase
x=train.drop('Purchase',axis=1)

#Creating Dummies
x=pd.get_dummies(x)
train=pd.get_dummies(train)
test=pd.get_dummies(test)

#splitting into training and validation data
from sklearn.model_selection import train_test_split
x_train, x_test,y_tain, y_test = train_test_split(x,y,test_size=0.3)

#random forest model  ####
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

# find the best parameter for model making

"""from sklearn.model_selection import GridSearchCV
param_grid = {"n_estimators":[1, 5, 10, 50, 100, 150, 300, 500], \
              "max_depth":[1, 3, 5, 7, 9]}
grid_rf = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring="Root Mean Squared Error").fit(x,y)
print("Best parameter: {}".format(grid_rf.best_params_))
print("Best score: {:.2f}".format((-1*grid_rf.best_score_)**0.5))"""

regressor = RandomForestRegressor(n_estimators=500,max_depth=9,random_state=1)
regressor.fit(x_train,y_tain)

#predicting on train data
pred_test = regressor.predict(test)


submission = pd.read_csv("D:\\Data Science\\Projects\\Black friday Sales\\sample_submission_V9Inaty.csv")
submission['Purchase']=pred_test
submission['User_ID']=test['User_ID']
pd.DataFrame(submission, columns=['Purchase','User_ID','Product_ID']).to_csv('D:\\Data Science\\Projects\\Black friday Sales\\random_forest2.csv')

#### Linear model ###
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
model = LinearRegression()

model.fit(x_train,y_tain)

pred_lm= model.predict(test)

submission2 = pd.read_csv("D:\\Data Science\\Projects\\Black friday Sales\\sample_submission_V9Inaty.csv")
submission2['Purchase']=pred_lm
submission['User_ID']=test['User_ID']
pd.DataFrame(submission, columns=['Purchase','User_ID','Product_ID']).to_csv('D:\\Data Science\\Projects\\Black friday Sales\\linear_model.csv')


