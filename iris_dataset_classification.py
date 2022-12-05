

#Importing all the essential libraries for the project

import pandas as pd
import sklearn
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

"""LOADING THE DATASET"""

columns={'Sepal length','Sepal width','Petal length','Petal width','target'} #List that contains the column name
#Load the data set
df=pd.read_csv("iris_dataset1.csv")
df.head(150)  #Gives the head part of the  data which means heading 
              #150 indicates that all data is showed by default 5 data will be printed

"""VISUALIZATION OF THE DATASET"""

df.describe()

#visualize the whole set
sns.pairplot(df,hue='target') #first argument is data and second argument is class label
                              #we will get different colour combination for each data

#From this we will easily get to know which has highest sepal length , sepal width and all
#Also Setosa can be differentiated easily than other
#Other two have too many overlapping parameters

"""SEPARATING THE  INPUT AND OUTPUT COLUMNS"""

#Separate features and target
data=df.values  #data is a matrix
#Slicing of matrix
X=data[:,0:4]  #Last column target is removed
Y=data[:,4]    #All four columns are removed except target
print(X)
print(Y)

"""SPLITTING THE DATA INTO TRAINING AND TESTING DATA"""

#SPLIT THE DATA INTO TRAIN AND TEST DATA
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)
print(X_train)
print(y_train)
print(X_test)
print(y_test)

#we get 120 training rows and 30 testing rows
#we get 120 rows of 4 column

"""MODEL1:SUPPORT VECTOR MACHINE"""

from sklearn.svm import SVC

model_svc=SVC()  #Model fitting
model_svc.fit(X_train,y_train)

#when we run this svc model will be trained

prediction1=model_svc.predict(X_test)
#calculate the accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,prediction1)*100)
for i in range(len(prediction1)):
  print(y_test[i],prediction1[i])

"""MODEL2:LOGISTIC REGRESSION"""

#Training the Logistic regression model
from sklearn.linear_model import LogisticRegression
model_LR=LogisticRegression()
model_LR.fit(X_train,y_train)

prediction2=model_LR.predict(X_test)
#calculate the accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,prediction2)*100)
for i in range(len(prediction2)):
  print(y_test[i],prediction2[i])

"""MODEL 3:DECISION TREE CLASSIFIER

"""

#fitting the model to the data
from sklearn.tree import DecisionTreeClassifier
model_DTC=DecisionTreeClassifier()
model_DTC.fit(X_train,y_train)



prediction3=model_svc.predict(X_test)
#calculating accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,prediction3)*100)
for i in range(len(prediction3)):
  print(y_test[i],prediction3[i])

"""DETAILED CLASSIFICATION REPORT"""

from sklearn.metrics import classification_report
print(classification_report(y_test,prediction2))

"""TESTING WITH MANUAL INPUT"""

X_new=np.array([[3,2,1,0.2],[5.3,2.5,4.6,1.9],[4.9,2.2,3.8,1.1]])
prediction = model_svc.predict(X_new)
print("Prediction of species : {}".format(prediction))