#葡萄酒数据集+PCA
import matplotlib.pyplot as plt#画图工具
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
data=datasets.load_wine()
X=data['data']
y=data['target']
###########
print("红酒数据集的数据结构:")
print(data.data.shape)
print("红酒数据集的特征:")
print(data.feature_names)
print("红酒数据集的标签:")
print(data.target_names)
#选取三个特征查看wine数据分布
def wine_data_feature():
    ax = Axes3D(plt.figure())
    for c, i, target_name in zip('>o*', [0, 1, 2], data.target_names):
        ax.scatter(X[y == i, 0], X[y == i, 1], X[y == i, 2], marker=c, label=target_name)
    ax.set_xlabel(data.feature_names[0])
    ax.set_ylabel(data.feature_names[1])
    ax.set_zlabel(data.feature_names[2])
    ax.set_title("wine")
    plt.legend()
    plt.show()


#选取两个特征查看IRIS数据分布
def IRIS_show():
    ax = plt.figure()
    for c, i, target_name in zip("rgb", [0, 1, 2], data.target_names):
        plt.scatter(X[y == i, 0], X[y == i, 1], c=c, label=target_name)
    plt.xlabel(data.feature_names[0])
    plt.ylabel(data.feature_names[1])
    plt.title("wine")
    plt.legend()
    plt.show()
'''
PCA：

通过坐标轴转换，寻找数据分布的最优子空间。用协方差矩阵前N个最大特征值对应的特征向量构成映射矩阵，
然后原始矩阵左乘映射矩阵实现降维。特征向量可以理解为坐标转换中新坐标轴的方向，特征值表示对应特征向量上的方差。

LDA：

   将带标签的数据通过投影降低维度，使投影不同类距离远，同类点分散程度小。

 
'''
#标准化后做PCA
def PCA_after_std():
    from sklearn.preprocessing import StandardScaler
    X_std = StandardScaler().fit(X).transform(X)
    from sklearn.decomposition import PCA
    # 降维后，主成分数为2
    pca = PCA(n_components=2)
    X_p = pca.fit(X_std).transform(X_std)
    ax = plt.figure()
    for c, i, target_name in zip("rgb", [0, 1, 2], data.target_names):
        plt.scatter(X_p[y == i, 0], X_p[y == i, 1], c=c, label=target_name)
    plt.xlabel('Dimension1')
    plt.ylabel('Dimension2')
    plt.title("wine-standard-PCA")
    plt.legend()
    plt.show()
    # 使用主成分绘制热力图
    plt.matshow(pca.components_,cmap='plasma')
    # 纵轴为主成分数
    plt.yticks([0,1],['Dimension1','Dimension2'])
    plt.colorbar()
    #横轴为原始特征数量
    print("红酒数据集的特征:----")
    print(type(data))
    wine_data_feature_list=data.feature_names
    print(wine_data_feature_list)
    plt.xticks(range(len(list(wine_data_feature_list))),wine_data_feature_list,rotation=60,ha='left')
    plt.show()
#不需要标准化的LDA有监督降维
def PCA_without_LDA():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    lda = LDA(n_components=2)
    X_r = lda.fit(X, y).transform(X)
    ax = plt.figure()
    for c, i, target_name in zip("rgb", [0, 1, 2], data.target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=target_name)
    plt.xlabel('Dimension1')
    plt.ylabel('Dimension2')
    plt.title("LDA")
    plt.legend()
    plt.show()

# wine_data_feature()
# IRIS_show()
PCA_after_std()
# PCA_without_LDA()