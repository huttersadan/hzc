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
import copy
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


X = pd.read_csv("feature_selection_X.txt",sep = '\t',header=None)
X = np.array(X)
X = X[:,0:-1]
Y = pd.read_csv("feature_selection_Y.txt",sep = '\t',header = None)
Y = np.array(Y)
nums_of_features = 1000

train_x,test_x,train_y,test_y = train_test_split(X,Y,train_size=300/400)
train_x,dev_x,train_y,dev_y = train_test_split(train_x,train_y,train_size=9/10)


epoch = 0
M_list = []#第一个是特征，第二个是r2
select_indexs = []
total_r_2_list = []
while(epoch < 10):
    if epoch == 0:
        r2_list = []
        for index in range(nums_of_features):
            new_train_x = train_x[:,[index]]
            model = LogisticRegression(penalty="none")
            model.fit(new_train_x,train_y.ravel())
            new_dev_x = dev_x[:,[index]]
            Y_pred = model.predict(new_dev_x)
            r2 = r2_score(dev_y.ravel(),Y_pred)
            r2_list.append([r2,index])
        r2_list = sorted(r2_list,key = lambda x:x[0],reverse=True)
        M_list.append([[r2_list[0][1]],r2_list[0][0]])
        select_indexs.append(r2_list[0][1])
        total_r_2_list.append(M_list[0][1])
        epoch+=1
    else:
        r2_list = []
        for index in range(nums_of_features):
            if index in select_indexs:
                continue
            else:
                temp_M_k_add_1 = copy.deepcopy(M_list[-1][0])
                temp_M_k_add_1.append(index)
                new_train_x = train_x[:,temp_M_k_add_1]
                #print(temp_M_k_add_1)
                new_dev_x = dev_x[:,temp_M_k_add_1]
                model = LogisticRegression(penalty="none")
                model.fit(new_train_x, train_y.ravel())
                Y_pred = model.predict(new_dev_x)
                r2 = r2_score(dev_y.ravel(), Y_pred)
                r2_list.append([r2, index])
        r2_list = sorted(r2_list, key=lambda x: x[0], reverse=True)
        temp_M_k_add_1 = copy.deepcopy(M_list[-1][0])
        temp_M_k_add_1.append(r2_list[0][1])
        M_list.append([temp_M_k_add_1,r2_list[0][0]])
        select_indexs.append(r2_list[0][1])
        total_r_2_list.append(M_list[-1][1])
        print(r2_list)
        epoch+=1

print(M_list)
nums = np.arange(len(M_list))
plt.scatter(nums, total_r_2_list)
plt.grid()
plt.show()