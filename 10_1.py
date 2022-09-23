import numpy as np
import scipy as sc
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import random
random.seed(535)

def get_x1_x2_Y():
    X_1 = []
    X_2 = []
    Y = []
    epsilon_1 = 0
    epsilon_2 = 0
    for i in range(1,21):
        X_1.append(i)
        epsilon_2 = random.normalvariate(0,np.sqrt(0.50))
        X_2.append(0.05*i+epsilon_2)
        epsilon_1 = random.normalvariate(0,2.5)
        Y.append(3*i+2+epsilon_1)
    return X_1,X_2,Y

def get_corr(X,Y):
    x_mean = np.mean(X)
    y_mean = np.mean(Y)
    up = np.sum((X-x_mean)*(Y-y_mean))
    X_sum = np.sum((X-x_mean)*(X-x_mean))
    Y_sum = np.sum((Y-y_mean)*(Y-y_mean))
    down = np.sqrt(X_sum*Y_sum)
    return up/down

# to_cor = 0
# for i in range(10000):
#     X_1,X_2,Y = get_x1_x2_Y()
#     X = np.array([X_1,X_2])
#     X_1 = np.array(X_1)
#     X_2 = np.array(X_2)
#     Y = np.array(Y)
#     cor = get_corr(X_1,X_2)
#     to_cor+=cor
#     print("x1,x2皮尔逊相关系数:{:2f}".format(cor))
# to_cor/=10000
# print("平均下来的x1,x2皮尔逊相关系数:{:2f}".format(to_cor))

N = 30

def regression(flag = 1):
    # flag == 1 linear
    # flag == 2 ridge
    # flag == 3 lasso
    betass = []
    beta1s = []
    beta0s = []
    for i in range(N):
        X_1, X_2, Y = get_x1_x2_Y()
        X = np.array([X_1, X_2])
        X_1 = np.array(X_1)
        X_2 = np.array(X_2)
        Y = np.array(Y)
        if flag == 1:
            model = LinearRegression()
        if flag == 2:
            model = Ridge(alpha=1)
        if flag == 3:
            model = Lasso(alpha=1)
        model.fit(X.T,Y)
        betas = model.coef_
        beta1s.append(betas[0])
        beta0s.append(betas[1])
        #print("betas:{},{}".format(betas[0],betas[1]))
    beta1 = np.array(beta1s)
    beta0 = np.array(beta0s)
    var_bata1 = (1/N) *np.sum((beta1 - beta1.mean())*(beta1 - beta1.mean()))
    var_bata0 = (1/N) *np.sum((beta0 - beta0.mean())*(beta0 - beta0.mean()))
    print("beta1的平均值:{:2f},beta0的平均值:{:2f}".format(beta1.mean(),beta0.mean()))
    print("beta1的方差:{:2f},beta0的方差:{:2f}".format(var_bata1,var_bata0))
regression(flag = 1)


regression(flag = 2)

regression(flag = 3)

