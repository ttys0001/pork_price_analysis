import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pandas.plotting import register_matplotlib_converters
import numpy as np
register_matplotlib_converters()

conn = sqlite3.connect('12-13-data.db')

def plot_two_data(predict_x,predict_value):
    data = pd.read_sql("select pork_price,date from pork_price where city='Zhuhai'", conn, parse_dates='date',
                       index_col='date')
    plt.plot(data,label="Zhuhai")
    plt.plot(predict_x,predict_value, label="Predict")
    plt.legend()
    plt.xticks(rotation=45, fontsize=12)  #旋转x轴刻度,并设置字体大小
    plt.yticks(fontsize=15)
    plt.grid(alpha=0.6, linestyle=":")
    plt.xlabel("Month", fontsize=16)
    plt.ylabel("Pork Price", fontsize=16)
    plt.title("Pork Price Predict", fontsize=16)
    plt.savefig("Pork Price Predict_second111.png", dpi=200, bbox_inches='tight')
    plt.show()

# plot_one_data()
city_dict = {"440400": "Zhuhai", "440100":"Guangzhou"}

first_day = datetime(2019, 10, 31)
predict_value_linear = [31.53156869212053, 31.59318529201859, 31.654801891916648, 31.716418491814707, 31.77803509171276, 31.83965169161082, 31.901268291508877, 31.962884891406937, 32.02450149130499, 32.08611809120305, 32.14773469110111, 32.209351290999166, 32.27096789089722, 32.33258449079528, 32.394201090693336, 32.455817690591395, 32.51743429048945, 32.57905089038751, 32.640667490285566, 32.70228409018362, 32.76390069008168, 32.825517289979736, 32.887133889877795, 32.94875048977585, 33.010367089673906, 33.071983689571965, 33.133600289470024, 33.195216889368076, 33.256833489266135, 33.318450089164195, 33.380066689062254, 33.441683288960306, 33.503299888858365, 33.564916488756424, 33.62653308865448, 33.688149688552535, 33.749766288450594, 33.81138288834865, 33.87299948824671, 33.934616088144764, 33.996232688042824, 34.05784928794088, 34.11946588783894]
predict_value_second =  [42.98891509807023, 43.23571708074304, 43.48352170413596, 43.732328968248964, 43.98213887308204, 44.232951418635224, 44.48476660490848, 44.73758443190183, 44.99140489961526, 45.24622800804878, 45.502053757202404, 45.758882147076086, 46.01671317766987, 46.275546848983744, 46.535383161017705, 46.79622211377174, 47.05806370724588, 47.32090794144009, 47.584754816354405, 47.84960433198879, 48.11545648834327, 48.382311285417835, 48.6501687232125, 48.91902880172723, 49.18889152096206, 49.45975688091698, 49.73162488159198, 50.00449552298708, 50.27836880510225, 50.55324472793752, 50.82912329149286, 51.106004495768296, 51.38388834076383, 51.66277482647944, 51.94266395291514, 52.22355572007093, 52.5054501279468, 52.78834717654275, 53.0722468658588, 53.35714919589494, 53.643054166651154, 53.92996177812747, 54.21787203032386]
predict_value_10 =  [49.54107003496087, 50.326784748693754, 51.14064491485682, 51.98349451692833, 52.85618849110382, 53.75959240774013, 54.69458213144448, 55.66204345908103, 56.66287173544755, 57.69797144553071, 58.76825578317279, 59.87464619513834, 61.01807190030435, 62.19946938302077, 63.41978186008113, 64.67995872075029, 65.98095493901259, 67.32373045734916, 68.7092495414769, 70.13848010527057, 71.61239300500951, 73.13196130241762, 74.69815949571843, 76.31196271769353, 77.9743459004318, 79.68628290548135, 81.4487456190624, 83.26270301121316, 85.12912015830138, 87.04895722770802, 89.02316842437052, 91.05270089791942, 93.13849360962253, 95.28147615848196, 97.48256756520335, 99.74267501355702, 102.06269254796702, 104.44349972649856, 106.88596022805054, 109.3909204134767, 111.95920783865644, 114.5916297196169, 117.2889713478295]
predict_8 =  [50.643412638450755, 51.64632178123669, 52.6993799183207, 53.804848947339124, 54.96506521990035, 56.18244128225582, 57.459467643605095, 58.79871457274807, 60.20283392295336, 61.674560985566714, 63.21671637234489, 64.8322079273154, 66.52403266788555, 68.29527875588487, 70.14912749842428, 72.08885537948903, 74.11783612174135, 76.23954277959514, 78.45754986325723, 80.77553549434619, 83.19728359348177, 85.7266860995684, 88.3677452218641, 91.12457572450025, 94.00140724396137, 97.00258664006346, 100.13258038011067, 103.39597695743818, 106.79748934359908, 110.34195747546131, 114.03435077677808, 117.8797707151631, 121.88345339400497, 126.05077218065948, 130.387240370134, 134.89851388548652, 139.59039401466237, 144.46883018427854, 149.5399227707627, 154.80992594906513, 160.28525057929312, 165.97246713138395, 171.8783086485879]

data = pd.read_sql("select pork_price,date from pork_price where city='Zhuhai'", conn, parse_dates='date',
                       index_col='date')
predict_x = []
for i in range(1,44):
    predict_x.append(first_day + timedelta(days=i))
print(predict_x)
plot_two_data(predict_x,predict_value_second)