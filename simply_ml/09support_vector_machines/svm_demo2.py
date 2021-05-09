import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib as mpl
from matplotlib import colors
import matplotlib.pyplot as plt
def iris_type(s):
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return it[s]
path = 'iris.data'
data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
# 将Iris分为训练集与测试集
# split(数据，分割位置，轴=1（水平分割） or 0（垂直分割）)
x, y = np.split(data, (4,), axis=1)
# x取样本X的所有行和前两列，进行特征向量训练
x = x[:, :2]
# train_test_split(train_data,train_target,test_size=数字, random_state=0)
# train_data：所要划分的样本特征集
# train_target：所要划分的样本结果
# test_size：样本占比，如果是整数的话就是样本的数量
# random_state：是随机数的种子
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
# 训练svm分类器
# kernel='linear'时，为线性核，C越大分类效果越好，但有可能会过拟合（defaul C=1）。
# kernel='rbf'时（default），为高斯核，gamma值越小，分类界面越连续
# gamma值越大，分类界面越“散”，分类效果越好，但有可能会过拟合。
# ecision_function_shape='ovr'时，为one v rest
# 即一个类别与其他类别进行划分，
# decision_function_shape='ovo'时，为one v one
# 即将类别两两之间进行划分，用二分类的方法模拟多分类的结果。
clf = svm.SVC(kernel='rbf', gamma=0.1, decision_function_shape='ovo', C=0.8)
clf.fit(x_train, y_train.ravel())

# 计算svc分类器的准确率
print('训练数据集得分：', clf.score(x_train, y_train))
print('测试数据集得分：', clf.score(x_test, y_test))

x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0列的范围
x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1列的范围
x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # 生成网格采样点
grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
grid_hat = clf.predict(grid_test)   # 预测分类值
grid_hat = grid_hat.reshape(x1.shape)
# 指定默认字体
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
# 绘制
cm_light = colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark = colors.ListedColormap(['g', 'r', 'b'])
plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
plt.scatter(x[:, 0], x[:, 1], c=np.squeeze(y), edgecolors='k', s=50, cmap=cm_dark)  # 样本
plt.scatter(x_test[:, 0], x_test[:, 1], s=120, facecolors='none', zorder=10)  # 圈中测试集样本
plt.xlabel(u'花萼长度', fontsize=13)
plt.ylabel(u'花萼宽度', fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title(u'鸢尾花SVM二特征分类', fontsize=15)
plt.show()
