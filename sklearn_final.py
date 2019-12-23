import numpy as np
import math
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm

def linear_regression_analysis():
    conn = sqlite3.connect('data.db')
    data_sql = pd.read_sql("select pork_price,date from pork_price where city='Zhuhai'", conn, parse_dates='date')
    Y = data_sql['pork_price']
    x = np.linspace(1, len(Y), len(Y))
    X = sm.add_constant(x)
    model = sm.OLS(Y,X)
    results = model.fit()
    print(results.params)
    print(results.summary())
    y_fitted = results.fittedvalues
    SSR = np.sum((y_fitted-Y.mean())**2)
    SSE = np.sum((Y-y_fitted)**2)
    F = (SSR/1)/(SSE/(len(Y)-2))
    print("F", F)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(x, Y, 'o', label='data')
    ax.plot(x, y_fitted, 'r--.', label='OLS')
    ax.legend(loc='best')
    plt.savefig("linear_regression_analysis", dpi=500, bbox_inches='tight')
    plt.show()


def polynomial_regression_analysis():
    conn = sqlite3.connect('data.db')
    data_sql = pd.read_sql("select pork_price,date from pork_price where city='Zhuhai'", conn, parse_dates='date')
    Y = data_sql['pork_price']
    x = np.linspace(1, len(Y), len(Y))
    X = np.column_stack((x, x ** 2))
    X = sm.add_constant(X)
    model = sm.OLS(Y,X)
    results = model.fit()
    print(results.params)
    print(results.summary())
    y_fitted = results.fittedvalues
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(x, Y, 'o', label='data')
    ax.plot(x, y_fitted, 'r--.', label='OLS')
    ax.legend(loc='best')
    plt.savefig("polynomial_regression_analysis111", dpi=500, bbox_inches='tight')
    plt.show()

# 绘出图像
def show_linear_line(X_parameter,Y_parameter):
    # 1. 构造回归对象
    regr = LinearRegression()
    regr.fit(X_parameter,Y_parameter)

    # 2. 绘出已知数据散点图
    plt.scatter(X_parameter,Y_parameter,color = 'blue')

    # 3. 绘出预测直线
    plt.plot(X_parameter,regr.predict(X_parameter),color = 'red',linewidth = 4)

    plt.title('The Pork Price Linear Regression')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.savefig("1.png")
    plt.show()

# 线性回归分析，其中predict_square_feet为要预测的数，函数返回对应的预测值
def linear_model_main(X_parameter,Y_parameter,predict_value,range_value):
    # 1. 构造回归对象
    regr = LinearRegression()
    regr.fit(X_parameter,Y_parameter)
    predict_outcome = []
    # 2. 获取预测值
    first_value = predict_value[0][0]
    for i in range(1, range_value):
        new_value = first_value+i
        predict = regr.predict([[new_value]])[0]
        predict_outcome.append(predict)

    # 3. 构造返回字典
    predictions = {}
    # 3.1 截距值
    predictions['intercept'] = regr.intercept_
    # 3.2 回归系数（斜率值）
    predictions['coefficient'] = regr.coef_
    # 3.3 预测值
    predictions['predict_value'] = predict_outcome
    return predictions

def poly_linear(X, Y, degree, predict_value, range_value):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)
    quadratic_featurizer = PolynomialFeatures(degree=degree)
    X_train_quadratic = quadratic_featurizer.fit_transform(X_train)
    X_test_quadratic = quadratic_featurizer.transform(X_test)

    regressor_quadratic = LinearRegression()
    regressor_quadratic.fit(X_train_quadratic, y_train)

    xx_quadratic = quadratic_featurizer.transform(X.reshape(X.shape[0], 1))
    yy_quadratic = regressor_quadratic.predict(xx_quadratic)

    predict_outcome = []
    first_value = predict_value[0][0]
    for i in range(1, range_value):
        new_value = first_value+i
        aa = quadratic_featurizer.fit_transform([[new_value]])
        bb = regressor_quadratic.predict(aa)[0]
        predict_outcome.append(bb)

    # aa = quadratic_featurizer.fit_transform([[367]])
    # bb = regressor_quadratic.predict(aa)
    print("多项式回归R方", regressor_quadratic.score(X_test_quadratic, y_test))
    print(regressor_quadratic.coef_)
    print(regressor_quadratic.intercept_)
    print("predict",predict_outcome)

    plt.plot(X, yy_quadratic, c='r', linestyle='--')
    plt.title("The Polynomial Regression with pork price")
    plt.xlabel("Time")
    plt.ylabel("Pork Price")
    plt.grid(True)
    plt.scatter(X_train, y_train)
    plt.savefig("2.png")
    plt.show()



conn = sqlite3.connect('data.db')
data_sql = pd.read_sql("select pork_price,date from pork_price where city='Zhuhai'", conn, parse_dates = 'date')
pork_price = data_sql['pork_price']
# print(pork_price)
# print(pork_price.describe())

predict_square_feet = [367]
# X的范围是从1到366天
X = np.arange(366)+1
Y = data_sql['pork_price']
X = np.array(X).reshape(366, -1)

# result = linear_model_main(X, Y, [predict_square_feet], 44)
# for key, value in result.items():
#     print ('{0}:{1}'.format(key, value))
# # 3. 绘图
# show_linear_line(X, Y)

# poly_linear(X,Y,2,[predict_square_feet], 44)
# linear_regression_analysis()
# polynomial_regression_analysis()
