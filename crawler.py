import requests
from lxml import etree
import pandas as pd
import time
import datetime
import pprint

import sqlite3
conn = sqlite3.connect('12-13-data.db')

CREATE TABLE IF NOT EXISTS `pork_price`(
   `id` INTEGER PRIMARY KEY AUTOINCREMENT,
   `pork_price` REAL NOT NULL,
   `city` REAL NOT NULL,
	 `date` DATE
);

city_dict = {"440400":"Zhuhai","440300":"Shenzhen","440100":"Guangzhou","441300":"Huizhou",
             "440600":"Foshan","442000":"Zhongshan","441900":"Dongguan","441200":"Zhaoqing"}

for k,v in city_dict.items():
    url = 'http://zhujia.zhuwang.cc/index/api/chartData?areaId=%d&aa=1572527819963'%int(k)
    request = requests.get(url)
    data = request.json()
    print('data: ', data)
    print('pigprice: ', data['pigprice'])
    pigprice = data['pigprice']
    start_date_list = data['time'][3]
    start_date = datetime.datetime.strptime(('-').join(start_date_list),'%Y-%m-%d')
    print('start_date', start_date)
    cursor = conn.cursor()
    for index,value in enumerate(pigprice):
        if index != 0:
            start_date = start_date + datetime.timedelta(days=1)
        cursor.execute(
            "INSERT INTO pork_price(id, pork_price, city, date) VALUES (NULL, ?, ?, ?);",
            (pigprice[index], v, start_date))
cursor.close()
conn.commit()
conn.close()
