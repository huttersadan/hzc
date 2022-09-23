from minepy import MINE
import numpy as np
import scipy as sc
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X = pd.read_csv("feature_selection_X.txt",sep = '\t',header=None)
X = np.array(X)
X = X[:,0:-1]
Y = pd.read_csv("feature_selection_Y.txt",sep = '\t',header = None)
Y = np.array(Y)

train_x,test_x,train_y,test_y = train_test_split(X,Y,train_size=300/400)

#train_x,dev_x,train_y,dev_y = train_test_split(train_x,train_y,train_size=9/10)


num_features = 1000
def fisher(train_X,train_Y):
    J_F_W = []
    u_0 = 0
    u_1 = 0
    # print(train_X)
    # print(train_Y.reshape(1,-1).squeeze())
    train_x_0 = train_X[train_Y.reshape(1,-1).squeeze() == 0]
    train_x_1 = train_X[train_Y.reshape(1,-1).squeeze() == 1]
    for i in range(num_features):
        u_0 = (1/len(train_x_0))*np.sum(train_x_0[:,i])
        u_1 = (1/len(train_x_1))*np.sum(train_x_1[:,i])
        #print(u_0,u_1)
        s_0 = np.sum((train_x_0-train_x_0.mean())**2)
        s_1 = np.sum((train_x_1 - train_x_1.mean())**2)
        jfw = (u_0-u_1)**2
        jfw/=(s_1+s_0)
        J_F_W.append([jfw,i])

    return sorted(J_F_W,key=lambda x:x[0],reverse=True)

J_F_W = fisher(train_x,train_y)

features = [1,5,10,20,50,100]
for feature in features:
    indexs = []
    for i in range(feature):
        indexs.append(J_F_W[i][1])
    new_train_X = train_x[:,indexs]
    if feature == 10:
        print(indexs)

    model = LogisticRegression()
    model.fit(new_train_X,train_y.ravel())
    Y_pred = model.predict(new_train_X)
    acc = accuracy_score(train_y.ravel(),Y_pred)
    print("nums_of_features = {},train_acc:{:2f}".format(feature,acc))
    new_test_X = test_x[:,indexs]
    Y_pred = model.predict(new_test_X)
    acc = accuracy_score(test_y.ravel(),Y_pred)
    print("nums_of_features = {},test_acc:{:2f}".format(feature,acc))

model = LogisticRegression(penalty="none")
model.fit(train_x,train_y.ravel())
y_pred = model.predict(test_x)
acc = accuracy_score(test_y,y_pred)
print("全部特征,test_acc:{}".format(acc))

def info_max(train_x, train_y,index):
    mine = MINE()
    mine.compute_score(train_x[:,index].ravel(), train_y.ravel())
    var = mine.mic()
    return var

mine_vars = []
for index in range(num_features):
    mine_vars.append([info_max(train_x,train_y,index),index])
mine_vars = sorted(mine_vars,key = lambda x:x[0],reverse=True)


features = [1,5,10,20,50,100]
for feature in features:
    indexs = []
    for i in range(feature):
        indexs.append(mine_vars[i][1])
    if feature == 10:
        print(indexs)
    new_train_X = train_x[:,indexs]
    model = LogisticRegression(penalty="none")
    model.fit(new_train_X,train_y.ravel())
    Y_pred = model.predict(new_train_X)
    acc = accuracy_score(train_y.ravel(),Y_pred)
    print("nums_of_features = {},train_acc:{:2f}".format(feature,acc))
    new_test_X = test_x[:,indexs]
    Y_pred = model.predict(new_test_X)
    acc = accuracy_score(test_y.ravel(),Y_pred)
    print("nums_of_features = {},test_acc:{:2f}".format(feature,acc))

model = LogisticRegression(penalty="none")
model.fit(train_x,train_y.ravel())
y_pred = model.predict(test_x)
acc = accuracy_score(test_y,y_pred)
print("全部特征,test_acc:{}".format(acc))
