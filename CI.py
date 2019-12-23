import numpy as np
import math
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures



def Ci_mean_known_variance_standard_normal_distribution(sample):
    print("Variance Known use standard distribution")
    n = len(sample)
    std = sample.std()
    x_hat = sample.mean()
    print("The mean of sample is: "+str(x_hat))
    Z = 1.96
    l = x_hat - ((Z*std) / math.sqrt(n))
    u = x_hat + ((Z*std) / math.sqrt(n))
    print(str(l)+" < mu <"+str(u))
    Z = 2.58
    l = x_hat - ((Z*std) / math.sqrt(n))
    u = x_hat + ((Z*std) / math.sqrt(n))
    print(str(l) +" < mu < "+ str(u))



def Ci_mean_unknown_variance_t_distribution(sample):
    print("Variance Unknown use t distribution")
    n = len(sample)
    std = sample.std()
    x_hat = sample.mean()
    print("The mean of sample is: "+str(x_hat))
    T = 2.064
    l = x_hat - ((T*std) / math.sqrt(n))
    u = x_hat + ((T*std) / math.sqrt(n))
    print(str(l)+" < mu <"+str(u))
    T = 2.797
    l = x_hat - ((T*std) / math.sqrt(n))
    u = x_hat + ((T*std) / math.sqrt(n))
    print(str(l) +" < mu < "+ str(u))

def Ci_variance_known_variance_chisquare_distribution(sample):
    print("CI of variance when variance known use chi-square distribution")
    n = len(sample)
    var = sample.var()
    # For 95% CI，n=51, n-1 = 50
    x_square_lowwr = 32.36
    x_square_upper = 71.42
    l = ((n-1)*var)/x_square_upper
    u = ((n-1)*var)/x_square_lowwr
    print(str(l)+" < sigma <"+str(u))
    x_square_lowwr = 27.99
    x_square_upper = 79.49
    l = ((n-1)*var)/x_square_upper
    u = ((n-1)*var)/x_square_lowwr
    print(str(l) +" <  sigma < "+ str(u))

def Ci_on_binomial_proportion(sample):
    higher_price_num = len(sample[sample>40])
    n = len(sample)
    p_hat = higher_price_num / n
    print("p_hat is "+str(p_hat))
    z = 1.96
    l = p_hat - z*(math.sqrt((p_hat*(1-p_hat))/n))
    u = p_hat + z*(math.sqrt((p_hat*(1-p_hat))/n))
    print(str(l)+" < p <"+str(u))
    z = 2.58
    l = p_hat - z*(math.sqrt((p_hat*(1-p_hat))/n))
    u = p_hat + z*(math.sqrt((p_hat*(1-p_hat))/n))
    print(str(l)+" < p <"+str(u))

def t_test(sample,mu):
    x_hat = sample.mean()
    S = sample.std()
    n = len(sample)
    t_0 = (x_hat - mu)/(S/math.sqrt(n))
    print("x_hat is " + str(x_hat))
    print("S is " + str(S))
    print("t_0 is " + str(t_0))


conn = sqlite3.connect('data.db')
data_sql = pd.read_sql("select pork_price,date from pork_price where city='Zhuhai'", conn, parse_dates = 'date')
pork_price = data_sql['pork_price'].round()
print(pork_price)
print(pork_price.describe())
# Ci_mean_known_variance_standard_normal_distribution(pork_price.sample(150))
# Ci_mean_unknown_variance_t_distribution(pork_price.sample(25))
# Ci_variance_known_variance_chisquare_distribution(pork_price.sample(51))
# Ci_on_binomial_proportion(pork_price.sample(300))
# t_test(pork_price.sample(31),15)

