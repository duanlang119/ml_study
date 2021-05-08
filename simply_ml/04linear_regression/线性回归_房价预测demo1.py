import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv('boston.csv')
# print(df.head())
# seaborn中pairplot函数可视化探索数据特征间的关系
sns.set(context = 'notebook')
cols = ['LSTAT', 'RM','MEDV']
sns.pairplot(df[cols], height=2)
plt.show()
# 可视化相关系数矩阵，理论：皮尔逊相关系数
#在统计学中，皮尔逊相关系数( Pearson correlation coefficient），又称皮尔逊积矩相关系数
# 是用于度量两个变量X和Y之间的相关（线性相关）
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size':15},
                 yticklabels=cols,
                 xticklabels=cols)
plt.show()

from sklearn.linear_model import LinearRegression
sk_model = LinearRegression()
X = df[['RM']].values
y = df['MEDV'].values
sk_model.fit(X, y)
print('Slope: %.3f' % sk_model.coef_[0]) #斜率
print('Intercept : %.3f' % sk_model.intercept_) #截距
plt.show()

def Regression_plot(X, y, model):
    plt.scatter(X, y, c='blue')
    plt.plot(X, model.predict(X), color='red')
    return None
Regression_plot(X, y, sk_model)
plt.xlabel('RM')
plt.ylabel('House price')
plt.show()

