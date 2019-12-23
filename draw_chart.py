import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from itertools import groupby
from stemgraphic import stem_graphic

def sampleStatistics(sample):
    statistics = {}
    statistics['mean'] = sample.mean()
    statistics['mode'] = sample.mode()
    statistics['median'] = sample.median()
    q1 = sample.quantile(0.25)
    q3 = sample.quantile(0.75)
    statistics['IQR'] = q3 - q1
    mild_outliers_low_left = q1 - 3 * statistics['IQR']
    mild_outliers_low_right = q1 - 1.5 * statistics['IQR']
    mild_outliers_high_left = q3 + 1.5 * statistics['IQR']
    mild_outliers_high_right = q3 + 3 * statistics['IQR']
    extreme_outliers_low = q1 - 3 * statistics['IQR']
    extreme_outliers_high = q3 + 3 * statistics['IQR']
    mild_outliers = []
    extreme_outliers = []
    for i in sample:
        if mild_outliers_low_left < i < mild_outliers_low_right or mild_outliers_high_left < i < mild_outliers_high_right:
            mild_outliers.append(i)
        if i <= extreme_outliers_low or i >= extreme_outliers_high:
            extreme_outliers.append(i)
    statistics['mild_outliers'] = mild_outliers
    statistics['extreme_outliers'] = extreme_outliers
    statistics['std'] = sample.std()
    return statistics

def plotStemLeafDiagram(sample):
    for k, g in groupby(sorted(sample), key=lambda x: int(x) // 10):
        lst = map(str, [int(y) % 10 for y in list(g)])
        print (k, '|', ' '.join(lst))

conn = sqlite3.connect('data.db')
data_sql = pd.read_sql("select pork_price,date from pork_price where city='Zhuhai'", conn, parse_dates = 'date', index_col = 'date')


pork_price = data_sql['pork_price'].round()

print(pork_price)
print(sampleStatistics(pork_price))
# plotStemLeafDiagram(pork_price)


# hist = pork_price.hist(bins=3, grid=False)
# hist.plot()
# plt.xlabel("Pork price")
# plt.ylabel("Frequency")
# plt.title("Pork price histogram")
# plt.savefig('test2.png',dpi=500)
# plt.show()

#
pork_price.plot.box(title="Pork price in Zhuhai")
plt.savefig('box_plot.png',dpi=500)
plt.show()



