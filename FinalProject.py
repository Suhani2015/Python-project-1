# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Dataset day.xlsx description
# instant: record index
# dteday : date
# season : season (1:springer, 2:summer, 3:fall, 4:winter)
# yr : year (0: 2011, 1:2012)
# mnth : month ( 1 to 12)
# hr : hour (0 to 23)
# holiday : weather day is holiday or not (extracted from [Web Link])
# weekday : day of the week
# workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
# weathersit : 
#  1: Clear, Few clouds, Partly cloudy, Partly cloudy
#  2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
#  3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
#  4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
# temp : Normalized temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-8, t_max=+39 (only in hourly scale)
# atemp: Normalized feeling temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-16, t_max=+50 (only in hourly scale)
# hum: Normalized humidity. The values are divided to 100 (max)
# windspeed: Normalized wind speed. The values are divided to 67 (max)
# casual: count of casual users
# registered: count of registered users
# cnt: count of total rental bikes including both casual and registered

#Our goal is build a model to forecast cnt
#the TARGET is cnt
#the poential predictors/features are: season, mnth, holiday, weathersit, atemp, hum, windspeed, casual,registered
#where the categorical predictors/feaures are: season, mnth, holiday, weathersit
#wher the continous predictors are: atemp, hum, windspeed, casual,registered


#---------------------------------------------------------------------------------------
#step1: load the packages needed for modeling 
#hint: you need to load RandomForestRegressor insteal of 
#RandomForestClassifier since we are modeling continous target cnt

#from sklearn.ensemble import RandomForestRegressor

import os
import numpy as np 
import xlrd
import pandas as pd 
import missingno as msno 
from collections import OrderedDict 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# i have imported all packages again as needed while completing
#the project 

#---------------------------------------------------------------------------------------
#step 2 load the dataset 


#2.1. load day.xlsx file into memory using pandas

# Import  operationg system(os) to handle the directory/path
import os

#Returns the current working directory.
os.getcwd

# Change the working directory 
os.chdir(r"/Users/ansh/desktop/python giniya/final project/FinalProjectDataset/")
# r"strings" which keep everything  in the string as it is

print(os.getcwd()) #to check if it changed the directory

# Import pandas for .csv and Excel files
import pandas as pd

# Load day.xlsx file into memory using pandas

day_xlsx = pd.read_excel(r"/Users/ansh/desktop/python giniya/final project/FinalProjectDataset/day.xlsx")

#pandas converts the data to a data frame

#2.2 printout the variable names/column names

day_xlsx.columns

#2.3 print out the size of the data

day_xlsx.shape

#2.4 summarize the data set using info and describle functions

day_xlsx.info()

#the key info(), helps us to deeply understand the data structure , before
# building any models 

#this particular data set is composed of datatypes - 1 datetime , 4 float
# and 11 intigers. We can see that there is no missing values in any of the
#columns. With the function describe we can see that the function count 731
#against 731 for all the columns . Indicating 731-731 = 0 , which means there
#is no missing value in any  coulmn


#2.5 check missing values

# we can use the package missingno which will allow us to display 
#completeness of data set

#https://www.youtube.com/watch?v=Z_Kxg-EYvxM (used this link to install 
#package using terminal in macbook and the imported it in spyder )

import missingno as msno
msno.matrix(day_xlsx)

#Output: <matplotlib.axes._subplots.AxesSubplot at 0x1a0e0484e0>

#since all the columns are black with no white lines , indicates that there
#are no missing values in the whole data set

#----------------------------------------------------------------------------------------


#3 Seperate the predictors/featues by using categorical variables groups and continous variables groups.
#note that 
#the categorical predictors/feaures are: season, mnth, holiday, weathersit
#the continous predictors are: atemp, hum, windspeed, casual,registered

categ =  ["season", "mnth", "holiday", "weathersit"]
conti = ["atemp", "hum", "windspeed", "casual","registered"]

#-------------------------------------------------------------------------------------

# 4 using the loops graph the categorical and continous variables 

# for visualization

import matplotlib.pyplot as plt
import seaborn as sns

#generate a figure object to hold the subplots

fig = plt.figure(figsize = (30,10))

for i in range (0,len(categ)):
    #we want to have  4 by 4 = 16 subplots
    ##determine the location to plot
    fig.add_subplot(4,4,i+1)
    # using i+1 here because index of subplot starts with 1 instead of 0 
    #sepcify the x (column name)
    sns.countplot(x=categ[i], data=day_xlsx)
    
    #plot continous variables
for k in conti:
    fig.add_subplot(4,4,i + 2)
    # here we have put i + 2, so that it does not overlab with previous subplot
    sns.distplot(day_xlsx[k])
    i = i+1
    
plt.show()
fig.clear()
    
#5: using loops countplot all the categrical variables except 
#holiday and set hue= 'holiday'

categ =  ["season", "mnth", "holiday", "weathersit"]


fig = plt.figure(figsize=(30, 10))
i = 1
for col in categ:
    if col != 'holiday':
        fig.add_subplot(3,3,i)
        #we look at the countplot by holiday; in otherwords, 
        #we have the detailed count of no holiday(0); day_xlsx[(1)
        sns.countplot(x=col, data=day_xlsx ,hue='holiday')
        i += 1


#6: swarmplot x = atemp, y = cnt, hue = season
        
fig.add_subplot(3,3,4)
#scatter plot by adding some noise and make sure there is no overlapping
sns.swarmplot(x="atemp", y="cnt", hue="season", data=day_xlsx)

#7: boxplot x = season, y = cnt

fig.add_subplot(3,3,5)
sns.boxplot(x="season", y="cnt", data=day_xlsx)

# 8: correlations with the new features
# you need to drop instant,	dteday
#plot the heatmap of the correlation

import numpy as np

corr = data=day_xlsx.drop(['instant','dteday'], axis=1).corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)
fig.add_subplot(3,3,6)
sns.heatmap(corr, mask=mask, cmap=cmap, cbar_kws={"shrink": .5})
plt.show()
fig.clear()



#9: set the Target to be cnt
#   set the features/predictors to be  'season', 'mnth', 'holiday', 'weathersit',
#           'atemp', 'hum', 'windspeed', 'casual','registered']
Target = "cnt"
Features = ['season', 'mnth', 'holiday', 'weathersit',
           'atemp', 'hum', 'windspeed', 'casual','registered']

# subseeting/filtering the data frame for Target

y = day_xlsx["cnt"]

#select the features by filtering the data frame
X = day_xlsx[Features]


#10  Create training and test sets by seting test_size to be 20% and random_state =100



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=123)

#11  Create a random Forest regressor model instance  and set n_estimators = 1000
#
from sklearn.ensemble import RandomForestRegressor

my_model_RF = RandomForestRegressor(n_estimators = 1000)

#11.1  Fit to the training data

my_model_RF.fit(X_train, y_train)

#11.2  Predict on the test data: y_pred
y_pred = my_model_RF.predict(X_test)

    
#11.3 print out  Score / Metrics

#using actual values and predicted values

from sklearn import metrics 
print(metrics.mean_absolute_error(y_test, y_pred))

#Mean Absolute Error = 71.06112925170069

# print result of Mean Squared Error

print(metrics.mean_squared_error(y_test, y_pred))


#Mean Absolute Error = 12840.11037076191

my_model_RF.score(X_test, y_test)

#0.9962372256853704


#12 Rank the importance of the Features by using the follwing given function
def FeaturesImportance(data,model):
    features=data.columns.tolist()
    fi=model.feature_importances_
    sorted_features={}
    for feature, imp in zip(features, fi):
        sorted_features[feature]=round(imp,3)

    # sort the dictionnary by value
    
    from collections import OrderedDict 
    sorted_features = OrderedDict(sorted(sorted_features.items(),reverse=True, key=lambda t: t[1]))

    for feature, imp in sorted_features.items():
        print(feature+" : ",imp)

    dfvi = pd.DataFrame(list(sorted_features.items()), columns=['Features', 'Importance'])
    dfvi.head()
    plt.figure(figsize=(15, 5))
    sns.barplot(x='Features', y='Importance', data=dfvi);
    plt.xticks(rotation=90) 
    plt.show()
    
#12.1 Features importance
    
FeaturesImportance(day_xlsx,my_model_RF)

##output

#FeaturesImportance(day_xlsx,my_model_RF)
#weathersit :  0.892
#workingday :  0.104
#dteday :  0.001
#mnth :  0.001
#holiday :  0.001
#weekday :  0.001
#instant :  0.0
#season :  0.0
#yr :  0.0

#weathersit  is the most important feature



