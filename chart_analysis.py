import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

conn = sqlite3.connect('12-13-data.db')

def plot_one_data():
    data = pd.read_sql("select pork_price,date from pork_price where city='Zhuhai'", conn, parse_dates='date',
                       index_col='date')
    plt.rcParams["figure.figsize"] = (8, 6)
    plt.plot(data)
    plt.xticks(rotation=45, fontsize=15)  #旋转x轴刻度,并设置字体大小
    plt.yticks(fontsize=15)
    plt.xlabel("Month", fontsize=16)
    plt.ylabel("Pork Price", fontsize=16)
    plt.title("Monthly Pork Price Trends in Zhuhai, 2018-2019", fontsize=16)
    plt.savefig("pork_price_trend.png", dpi=200, bbox_inches='tight')
    plt.show()

def plot_two_data(city):
    index = 0
    for i in city_dict:
        data = pd.read_sql("select pork_price,date from pork_price where city='%s'"%city_dict[i], conn, parse_dates='date',
                           index_col='date')
        plt.plot(data,label=city_dict[i])
        index = index + 1

    plt.legend()
    plt.xticks(rotation=45, fontsize=12)  #旋转x轴刻度,并设置字体大小
    plt.yticks(fontsize=15)
    plt.grid(alpha=0.6, linestyle=":")
    plt.xlabel("Month", fontsize=16)
    plt.ylabel("Pork Price", fontsize=16)
    plt.title("Pork Price Trends between ZH and GZ", fontsize=16)
    plt.savefig("%s_cities_pork_price_trend.png"%city_dict[i], dpi=200, bbox_inches='tight')
    plt.show()

# plot_one_data()
city_dict = {"440400": "Zhuhai", "440100":"Guangzhou"}
plot_two_data(city_dict)


