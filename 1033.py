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
from sklearn.tree import DecisionTreeClassifier
X = pd.read_csv("feature_selection_X.txt",sep = '\t',header=None)
X = np.array(X)
X = X[:,0:-1]
Y = pd.read_csv("feature_selection_Y.txt",sep = '\t',header = None)
Y = np.array(Y)
nums_of_features = 1000

train_x,test_x,train_y,test_y = train_test_split(X,Y,train_size=300/400)

model = DecisionTreeClassifier()
model.fit(train_x,train_y)
index = []

for i in range(len(model.feature_importances_)):
    if model.feature_importances_[i]!=0:
        index.append([i,model.feature_importances_[i]])
print(sorted(index,key=lambda x:x[1],reverse=True))