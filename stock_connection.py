import tushare as ts
import matplotlib.pylab as plt
import pandas as pd
import sqlite3
import datetime
from pylab import *
#支持中文
mpl.rcParams['font.sans-serif'] = ['SimHei']

conn = sqlite3.connect('12-13-data.db')
data = pd.read_sql("select pork_price,date from pork_price where city='Zhuhai'", conn, parse_dates = 'date', index_col = 'date')


gailian =ts.get_concept_classified()
pig_stock=[]
for i in gailian.T.to_dict().values():
    if u"猪肉" in i['c_name']:
        pig_stock.append(i)
print(pig_stock)

def get_stock(s):
    stock = ts.get_hist_data(s['code'])  # 一次性获取全部日k线数据
    stock['date'] = stock.index
    print(stock)
    stock['date1'] = pd.to_datetime(stock.date, format='%Y-%m-%d')
    stock['date2'] = pd.to_datetime(stock.date, format='%Y-%m-%d')
    stock.set_index('date1', inplace=True)
    # Sort the Data
    stock = stock.sort_values('date2')
    # Slice the Data
    From = '2018-12-13'
    To = '2019-12-13'
    # print(stock.loc[From:To, :])

    return stock.loc[From:To, :]

for s in pig_stock:
    stock = get_stock(s)
    x = [datetime.datetime.strptime(d, '%Y-%m-%d').date() for d in list(stock['date'])]
    stock_conv = pd.DataFrame()
    stock_conv['date'] = stock.index
    stock_conv['pricce'] = stock.close
    stock_conv[u'日期'] = stock.index
    stock_conv[u'价格（元/公斤）'] = stock.close
    diff = data['pork_price'].corr(stock['close'])
    name = s['name'] + '相关系数：' + str(round(diff,2))
    plt.plot(stock.index, stock.close, label=name)
    print(diff)

plt.legend(fontsize=5) # 显示图例
plt.title('Pork concept stock and pork price correlation coefficient',fontsize = 10)
plt.xlabel('Month')
plt.ylabel('Yuan')
plt.savefig('stock_latest.png',bbox_inches = 'tight',dpi = 300)
plt.show()