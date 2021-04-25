import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine
wine=load_wine()
x=wine.data
y=wine.target
print("红酒数据集的数据结构:")
print(wine.data.shape)
print("红酒数据集的特征:")
print(wine.feature_names)
print("红酒数据集的标签:")
print(wine.target_names)
import pandas as pd
sample=pd.concat([pd.DataFrame(x),pd.DataFrame(y)],axis=1)
print("将样本数据和标签，连接起来，展示表格头5行数据")
print(sample.head())
#标准化
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(x)
print('打印做标准化处理后的数据形态')
print(X_train_std.shape)
#导入PCA模块
from sklearn.decomposition import PCA
#设置主成分数量为2
pca = PCA(n_components=2)
reduced_x = pca.fit_transform(X_train_std)
print('打印主成分提取后的数据形态')
print(reduced_x.shape)
# 经过PCA降维的数据可视化情况
red_x,red_y=[],[]
blue_x,blue_y=[],[]
green_x,green_y=[],[]
for i in range(len(reduced_x)):
    if y[i] ==0:
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])
    elif y[i]==1:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])
    else:
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])
plt.scatter(red_x,red_y,c='r',marker='x')
plt.scatter(blue_x,blue_y,c='b',marker='D')
plt.scatter(green_x,green_y,c='g',marker='.')
plt.title("wine-standard-PCA")
plt.xlabel('Dimension1')
plt.ylabel('Dimension2')
plt.legend(wine.target_names,loc='best')
plt.show()
#使用主成分绘制热度图
plt.matshow(pca.components_,cmap='plasma')
#纵轴为主成分数
plt.yticks([0,1],['Dimension1','Dimension2'])
plt.colorbar()
#横轴为原始特征数量 
plt.xticks(range(len(wine.feature_names)),wine.feature_names,rotation=60,ha='left')
plt.show()

