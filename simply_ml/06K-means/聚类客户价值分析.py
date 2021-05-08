import pandas as pd
import numpy as npy
import matplotlib.pylab as pyl
#忽略警告
import warnings
warnings.filterwarnings("ignore")
dataf = pd.read_csv('company.csv', encoding='utf-8')
x = dataf[['平均消费周期(天)', '平均每次消费金额']].values

# 导入聚类分析工具KMeans
from sklearn.cluster import KMeans
# 传入要分类的数目
kms = KMeans(n_clusters=3)
y = kms.fit_predict(x)
print(y)
#年龄-消费金额图，消费时间-消费金额图，年龄-消费时间图
for i in range(0,len(y)):
    if(y[i]==0):
        print(str(i)+"大众客户")
        pyl.subplot(2,3,1)
        #年龄-消费金额图
        pyl.xlabel('age')
        pyl.ylabel('amount')
        pyl.plot(dataf.iloc[i:i+1,0:1].values,dataf.iloc[i:i+1,1:2].values,"*r")
        pyl.subplot(2,3,2)
        #消费时间-消费金额图
        pyl.xlabel('period')
        pyl.ylabel('amount')
        pyl.plot(dataf.iloc[i:i+1,2:3].values,dataf.iloc[i:i+1,1:2].values,"*r")
        pyl.subplot(2,3,3)
        #年龄-消费时间图
        pyl.xlabel('age')
        pyl.ylabel('period')
        pyl.plot(dataf.iloc[i:i+1,0:1].values,dataf.iloc[i:i+1,2:3].values,"*r")
    elif(y[i]==1):
        print(str(i)+"超级VIP客户")
        pyl.subplot(2,3,1)
        pyl.xlabel('age')
        pyl.ylabel('amount')
        #年龄-消费金额图
        pyl.plot(dataf.iloc[i:i+1,0:1].values,dataf.iloc[i:i+1,1:2].values,"sy")
        pyl.subplot(2,3,2)
        pyl.xlabel('period')
        pyl.ylabel('amount')
        #消费时间-消费金额图
        pyl.plot(dataf.iloc[i:i+1,2:3].values,dataf.iloc[i:i+1,1:2].values,"sy")
        pyl.subplot(2,3,3)
        #年龄-消费时间图
        pyl.xlabel('age')
        pyl.ylabel('period')
        pyl.plot(dataf.iloc[i:i+1,0:1].values,dataf.iloc[i:i+1,2:3].values,"sy")
    elif(y[i]==2):
        print(str(i)+"VIP客户")
        pyl.subplot(2,3,1)
        #年龄-消费金额图
        pyl.xlabel('age')
        pyl.ylabel('amount')
        pyl.plot(dataf.iloc[i:i+1,0:1].values,dataf.iloc[i:i+1,1:2].values,"pb")
        pyl.subplot(2,3,2)
        #消费时间-消费金额图
        pyl.xlabel('period')
        pyl.ylabel('amount')
        pyl.plot(dataf.iloc[i:i+1,2:3].values,dataf.iloc[i:i+1,1:2].values,"pb")
        pyl.subplot(2,3,3)
        #年龄-消费时间图
        pyl.xlabel('age')
        pyl.ylabel('period')
        pyl.plot(dataf.iloc[i:i+1,0:1].values,dataf.iloc[i:i+1,2:3].values,"pb")
pyl.show()
