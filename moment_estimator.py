import sqlite3
import pandas as pd

conn = sqlite3.connect('data.db')
data_sql = pd.read_sql("select pork_price,date from pork_price where city='Zhuhai'", conn, parse_dates = 'date', index_col = 'date')
pork_price = data_sql['pork_price'].round()
print(pork_price)
real_data_mean = pork_price.mean()
real_data_var = pork_price.var()
print(pork_price.describe())
print('Sample data mean is:', real_data_mean)
print('Sample data variance is:', real_data_var)

sample_data = pork_price.sample(200)
sample_data_mean = sample_data.mean()
sample_data_var = ((sample_data - sample_data_mean) ** 2).sum() / len(sample_data)
print('Estimator of mean is:', sample_data_mean)
print('Estimator of variance is:', sample_data_var)
print('Error of mean:', real_data_mean - sample_data_mean)
print('Error of variance:', real_data_var - sample_data_var)
