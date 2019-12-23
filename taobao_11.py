import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

# 2009年开始
data = [0.52,9.36,33.6,191,362,571,912.17,1207,1682,2135]
print(len(data))
x = [x for x in range(1,len(data)+1)]
print(x)

# plt.plot(x,data, label="Double 11’ Day")
# plt.legend()
# plt.xticks(fontsize=12)  #旋转x轴刻度,并设置字体大小
# plt.yticks(fontsize=15)
# plt.grid(alpha=0.6, linestyle=":")
# plt.xlabel("Year", fontsize=16)
# plt.ylabel("Pork Price", fontsize=16)
# plt.title("Pork Price Predict", fontsize=16)
# # plt.savefig("Pork Price Predict_second.png", dpi=200, bbox_inches='tight')
# plt.show()

def poly_linear(X, Y, degree, predict_value):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)
    quadratic_featurizer = PolynomialFeatures(degree=degree)
    X_train_quadratic = quadratic_featurizer.fit_transform(X_train)
    X_test_quadratic = quadratic_featurizer.transform(X_test)

    regressor_quadratic = LinearRegression()
    regressor_quadratic.fit(X_train_quadratic, y_train)

    xx_quadratic = quadratic_featurizer.transform(X.reshape(X.shape[0], 1))
    yy_quadratic = regressor_quadratic.predict(xx_quadratic)



    aa = quadratic_featurizer.fit_transform(predict_value)
    bb = regressor_quadratic.predict(aa)

    # aa = quadratic_featurizer.fit_transform([[367]])
    # bb = regressor_quadratic.predict(aa)
    print("多项式回归R方", regressor_quadratic.score(X_test_quadratic, y_test))
    print(regressor_quadratic.coef_)
    print(regressor_quadratic.intercept_)
    print("predict",bb)
    plt.plot(X, yy_quadratic, c='r', linestyle='--')
    plt.title("The Polynomial Linear Regression with Double 11’ Day")
    plt.xlabel("Time/Year")
    plt.ylabel("Price/Hundred million RMB")
    plt.grid(True)
    plt.scatter(X_train, y_train)
    plt.savefig("The Polynomial Regression with Double 11’ Day", dpi=200, bbox_inches='tight')
    plt.show()

X = np.arange(10)+1
X = np.array(X).reshape(10, -1)
poly_linear(X,data,2,[[11]])