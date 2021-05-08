import csv
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn import tree

filename="film.csv"
film_data = open(filename, 'rt')
#  从csv文件里读取数据
reader = csv.reader(film_data)
headers=next(reader)
# 表头信息
print(headers)

feature_list = [] # 特征值feature
result_list = []  # 结果result

for row in reader:
    #结果集只要保留最后一列，这里取每一行的最后一个元素
    result_list.append(row[-1])
    # 去掉首尾两列，特征集中只保留'type', 'country', 'gross'
    # 通过zip函数，把两个list合成一个字典取  从下标为1的元素
    feature_list.append(dict(zip(headers[1:-1], row[1:-1])))

print(result_list)
#
print(feature_list)
#这个list里每一项是一个字典，
'''
将原始数据转换成包含有字典的List
将建好的包含字典的list用DictVectorizer对象转换成0-1 特征提取和量化
'''
# 初始化字典特征抽取器
vec = DictVectorizer()

#sk-learn所有输入都要numpy array
#DictVectorizer对非数字化的处理方式是，借助原特征的名称，组合成新的特征，
#并采用0/1的方式进行量化，而数值型的特征转化比较方便，一般情况维持原值即可。
#输出转化后的特征矩阵,#  toarray方法将dict类型的list数据，转换成numpy array
dummyX = vec.fit_transform(feature_list).toarray()
print(dummyX)

#打印结果
# country     |,gross|,type
# 0，0，0 , 0 |0，0，|0 , 0，0
# 注意，dummyX是按首字母排序的 'country','gross','type'
#输出各个维度的特征含义
print(vec.get_feature_names())
dummyY = preprocessing.LabelBinarizer().fit_transform(result_list)

print(dummyY)
'''
 1.构建分类器——决策树模型
 2.使用数据训练决策树模型
'''
clf = tree.DecisionTreeClassifier(criterion='entropy', random_state=1)
clf = clf.fit(dummyX, dummyY)
"""
fit()可以说是调用的通用方法，fit(X,Y)为监督学习算法
线性模型的fit其实一个进行学习的过程，根据数据和标签进行学习。
fit就是开始学习（如果数据量大，可以发现需要执行很长时间）
"""
# print("clf: " + str(clf))
import pydotplus
import os
#根据你安装Graphviz路径修改
os.environ["PATH"] += os.pathsep + 'D:/Graphviz2.38/bin'
dot_data = tree.export_graphviz(clf,
                                feature_names=vec.get_feature_names(),
                                filled=True, rounded=True,
                                special_characters=True,
                                out_file=None)

graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("film.pdf")

# # 开始预测，测试集数据
A = ([[0, 0, 0, 1, 0, 1, 0, 1, 0]]) # 日本(4)-低票房(2)-动画片(3)
B = ([[0, 0, 1, 0, 0, 1, 0, 1, 0]]) # 法国(4)-低票房(2)-动画片(3)
C = ([[1, 0, 0, 0, 1, 0, 1, 0, 0]]) # 美国(4)-高票房(2)-动作片(3)
# predict则是基于fit之后形成的模型，来决定指定的数据对应于标签的值。
# predict则是根据fit形成的体系来判断指定值对应的计算结果。
predict_resultA = clf.predict(A)
predict_resultB = clf.predict(B)
predict_resultC = clf.predict(C)
print("日本低票房的动画片，想看吗？ " + str(predict_resultA))
print("法国低票房的动画片，想看吗？ " + str(predict_resultB))
print("美国高票房的动作片，想看吗？ " + str(predict_resultC))

