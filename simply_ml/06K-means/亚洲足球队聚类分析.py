import numpy as np
import pandas as pd
#数据加载
data = pd.read_csv(r'footballdata.csv',encoding='utf-8')
train_x = data[['2019年国际排名','2018年世界杯','2015年亚洲杯']]
df = pd.DataFrame(train_x)
#聚类一般要做数据标准化处理，采用Min-max 规范化
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
train_x = min_max_scaler.fit_transform(train_x)
#sklearn内置k-means聚类
from sklearn.cluster import KMeans
#K值选3，将亚洲足球划分为3个梯队
kmeans = KMeans(n_clusters=3)
kmeans.fit(train_x)
predict_y = kmeans.predict(train_x)
#合并聚类结果，插入到原数据中
result = pd.concat((data,pd.DataFrame(predict_y)),axis = 1)
result.rename({0:'聚类'},axis = 1,inplace = True)
print(result)
