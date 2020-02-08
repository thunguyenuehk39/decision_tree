# Importing the required packages 
import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

def importdata():
    data = pd.read_csv("data.csv",sep = ',')
    # Assignment data
    District = {'Urban':1,'Suburban':2,'Rural':3}
    House_Type = {'Terrace':1,'Semi-detached':2,'Detached':3}
    Income  = {'High':1,'Low':2}
    Previous_Customer = {'No':1,'Yes':2}
    for i, j in District.items():
        data = data.replace(i,j)
    for a, b in House_Type.items():
        data = data.replace(a,b)
    for x, y in Income.items():
        data = data.replace(x,y)
    for c, d in Previous_Customer.items():
        data = data.replace(c,d)
    print(' District:',District,'\n','House Type:',House_Type, '\n','Income:',Income,'\n','Previous Customer:',Previous_Customer,'\n')
    # Printing the dataswet shape
    print("Dataset Length: ", len(data))
    print("Dataset Shape: ", data.shape)
    # Printing the dataset observations
    print("Dataset: \n",data)
    return data
def splitdataset(data):
    #  Separating the target variable
    X = data.values[:,0:4]
    Y = data.values[:,4]
    
    # Splitting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.3, random_state =100)
    
    return X, Y, X_train, X_test, y_train, y_test
def train_using_entropy(X_train,X_test,y_train):
    
    # Desision tree with entropy
    clf_entropy = DecisionTreeClassifier(criterion = "entropy",random_state = 100,max_depth = 3, min_samples_leaf = 5)
    
    # Performing training
    clf_entropy.fit(X_train,y_train)
    return clf_entropy
def prediction(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    print("Predicted values: ")
    print(y_pred)
    return y_pred
def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
    print("Accuracy: ", accuracy_score(y_test,y_pred)*100)
    print("Report: ",classification_report(y_test,y_pred))
def main():
    data = importdata()
    X,Y,X_train,X_test,y_train,y_test = splitdataset(data)
    clf_entropy = train_using_entropy(X_train,X_test,y_train)
    
    print("Results:")
    y_pred_entropy = prediction(X_test, clf_entropy)
    cal_accuracy(y_test,y_pred_entropy)
    
main()
